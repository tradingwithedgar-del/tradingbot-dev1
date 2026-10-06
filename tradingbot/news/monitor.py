"""Keeps the agent aware of news: what's scheduled, what just broke, and what it means per symbol.

For each symbol the agent gets a small news "state" that becomes part of every trade's
features, so the learner can find out which news situations are worth trading:
    news_state  clear | pre_event | post_event | breaking | earnings_ahead
    news_kind   cpi, fomc, nfp, oil_inventory, earnings, geopolitics, central_bank, ...
    news_dir    +1 news points up for this symbol, -1 down, 0 unknown (scheduled data before release)
    news_impact 0-3
    news_age_min minutes since the release/headline (negative = minutes until a scheduled event)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd

from ..config import Settings
from ..journal import Journal
from .calendar import ScheduledEvent, fetch_calendar, load_earnings
from .classify import Assessment, KeywordClassifier, make_classifier
from .headlines import Headline, fetch_headlines

log = logging.getLogger(__name__)


@dataclass
class BreakingItem:
    time: pd.Timestamp
    title: str
    category: str
    impact: int
    effects: dict[str, int] = field(default_factory=dict)
    summary: str = ""


class NewsMonitor:
    def __init__(self, settings: Settings, journal: Journal, classifier=None, fetch_calendar_fn=None,
                 fetch_headlines_fn=None, load_earnings_fn=None) -> None:
        self.s = settings
        self.cfg = settings.news
        self.j = journal
        self.classifier = classifier or make_classifier(self.cfg.model)
        self._fetch_calendar = fetch_calendar_fn or fetch_calendar
        self._fetch_headlines = fetch_headlines_fn or fetch_headlines
        self._load_earnings = load_earnings_fn or load_earnings
        self.events: list[ScheduledEvent] = []
        self.breaking: list[BreakingItem] = []
        self._seen: set[str] = set(journal.get("news_seen", []) or [])
        self._last_calendar: pd.Timestamp | None = None
        self._last_headlines: pd.Timestamp | None = None

    @property
    def reader(self) -> str:
        return getattr(self.classifier, "name", "keywords")

    # --- refreshing ---------------------------------------------------------------------
    def refresh(self, now: pd.Timestamp) -> None:
        if self._last_calendar is None or now - self._last_calendar >= pd.Timedelta(minutes=self.cfg.calendar_refresh_minutes):
            self._last_calendar = now
            events: list[ScheduledEvent] = []
            try:
                events += self._fetch_calendar(self.s.symbols)
            except Exception as e:
                log.warning("economic calendar unavailable: %s", e)
                events = [e for e in self.events if e.source == "calendar"]
            try:
                events += self._load_earnings(self.s.symbols, self.s.db_path.parent)
            except Exception as e:
                log.warning("earnings calendar unavailable: %s", e)
            self.events = sorted(events, key=lambda e: e.time)
            for e in self.events:
                self._record(e.key, "event", e.time, e.title, e.kind, e.impact, {s: 0 for s in e.symbols}, e.title, e.source)
        if self._last_headlines is None or (now - self._last_headlines).total_seconds() >= self.cfg.headline_poll_seconds:
            self._last_headlines = now
            try:
                heads = self._fetch_headlines(self.cfg.feeds, self.cfg.max_headline_age_minutes, now)
            except Exception as e:
                log.warning("headline feeds unavailable: %s", e)
                heads = []
            new = [h for h in heads if h.key not in self._seen]
            for i in range(0, len(new), self.cfg.max_headlines_per_call):
                self._classify(new[i:i + self.cfg.max_headlines_per_call])
        cutoff = now - pd.Timedelta(minutes=self.cfg.post_event_window_minutes)
        self.breaking = [b for b in self.breaking if b.time >= cutoff]

    def _classify(self, batch: list[Headline]) -> None:
        try:
            results = self.classifier.classify(batch, self.s.symbols)
        except Exception as e:
            log.warning("news classifier failed (%s) - keyword fallback", e)
            results = KeywordClassifier().classify(batch, self.s.symbols)
        for h, a in zip(batch, results):
            self._seen.add(h.key)
            self._record(h.key, "headline", h.time, h.title, a.category, a.impact, a.effects, a.summary, h.source,
                         link=h.link, reader=a.source)
            if a.impact >= self.cfg.min_impact and a.effects:
                self.breaking.append(BreakingItem(h.time, h.title, a.category, a.impact, a.effects, a.summary))
                self.j.event("news", f"[impact {a.impact}] {h.title} -> "
                             + ", ".join(f"{s} {'up' if d > 0 else 'down'}" for s, d in a.effects.items()),
                             {"category": a.category, "effects": a.effects, "source": h.source})
        self.j.set("news_seen", sorted(self._seen)[-3000:])

    def _record(self, key, kind, t, title, category, impact, effects, summary, source, link="", reader="") -> None:
        with self.j._lock:
            self.j.conn.execute(
                "INSERT OR IGNORE INTO news (key, ts, kind, title, category, impact, effects, summary, source, link, reader) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (key, t.isoformat(), kind, title, category, int(impact), json.dumps(effects), summary, source, link, reader))
            self.j.conn.commit()

    # --- what the agent asks ---------------------------------------------------------------
    def state(self, symbol: str, now: pd.Timestamp) -> dict:
        clear = {"news_state": "clear", "news_kind": "none", "news_dir": 0, "news_impact": 0, "news_age_min": None,
                 "news_title": ""}
        candidates = []
        for e in self.events:
            if symbol not in e.symbols:
                continue
            mins = (now - e.time).total_seconds() / 60
            if e.kind == "earnings" and -self.cfg.earnings_block_hours * 60 <= mins < 0:
                candidates.append((2, {"news_state": "earnings_ahead", "news_kind": "earnings", "news_dir": 0,
                                       "news_impact": 3, "news_age_min": round(mins), "news_title": e.title}))
            elif -self.cfg.pre_event_block_minutes * 2 <= mins < 0:
                candidates.append((3, {"news_state": "pre_event", "news_kind": e.kind, "news_dir": 0,
                                       "news_impact": e.impact, "news_age_min": round(mins), "news_title": e.title}))
            elif 0 <= mins <= self.cfg.post_event_window_minutes:
                candidates.append((4, {"news_state": "post_event", "news_kind": e.kind, "news_dir": 0,
                                       "news_impact": e.impact, "news_age_min": round(mins), "news_title": e.title}))
        for b in self.breaking:
            if symbol in b.effects:
                mins = (now - b.time).total_seconds() / 60
                if 0 <= mins <= self.cfg.post_event_window_minutes:
                    candidates.append((5 + b.impact, {"news_state": "breaking", "news_kind": b.category,
                                                      "news_dir": b.effects[symbol], "news_impact": b.impact,
                                                      "news_age_min": round(mins), "news_title": b.title}))
        if not candidates:
            return clear
        # most important first; among equals the most recent
        candidates.sort(key=lambda c: (c[0], -(abs(c[1]["news_age_min"] or 0))), reverse=True)
        return candidates[0][1]

    def blackout(self, symbol: str, now: pd.Timestamp) -> str | None:
        """Reason not to open a new trade right now, or None."""
        for e in self.events:
            if symbol not in e.symbols:
                continue
            mins = (now - e.time).total_seconds() / 60
            if e.kind == "earnings":
                if -self.cfg.earnings_block_hours * 60 <= mins <= 60:
                    return f"{e.title} {'in' if mins < 0 else 'just released'} - no new trades"
            elif -self.cfg.pre_event_block_minutes <= mins <= self.cfg.post_event_block_minutes:
                return f"high-impact news window: {e.title}"
        return None

    def must_flatten(self, symbol: str, now: pd.Timestamp) -> str | None:
        for e in self.events:
            if e.kind == "earnings" and symbol in e.symbols:
                mins = (e.time - now).total_seconds() / 60
                if 0 <= mins <= self.cfg.flatten_before_earnings_minutes:
                    return f"closing before {e.title}"
        return None

    def hot_symbols(self, now: pd.Timestamp) -> list[str]:
        return [s for s in self.s.symbols if self.state(s, now)["news_state"] in ("post_event", "breaking")]

    def upcoming(self, now: pd.Timestamp, hours: int = 48) -> list[ScheduledEvent]:
        return [e for e in self.events if now - pd.Timedelta(hours=1) <= e.time <= now + pd.Timedelta(hours=hours)]
