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
import threading
from concurrent.futures import Future, ThreadPoolExecutor
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
        self._last_classify: pd.Timestamp | None = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="news-reader")
        self._pending: Future | None = None

    @property
    def reader(self) -> str:
        return getattr(self.classifier, "name", "keywords")

    @property
    def breaking_threshold(self) -> int:
        """Headlines only drive decisions when Claude reads them. The keyword reader is too crude:
        its ratings are stored for context, but never put a symbol into 'breaking news' mode."""
        return self.cfg.min_impact if self.reader == "claude" else 99

    def reset_headlines(self) -> int:
        with self.j._lock:
            n = self.j.conn.execute("DELETE FROM news WHERE kind = 'headline'").rowcount
            self.j.conn.commit()
        self._seen.clear()
        self.breaking.clear()
        self.j.set("news_seen", [])
        return n

    # --- refreshing ---------------------------------------------------------------------
    def refresh(self, now: pd.Timestamp) -> None:
        if self._last_calendar is None or now - self._last_calendar >= pd.Timedelta(minutes=self.cfg.calendar_refresh_minutes):
            self._last_calendar = now
            events: list[ScheduledEvent] = []
            try:
                fetched = self._fetch_calendar(self.s.symbols)
                events += fetched
                self.j.set("calendar_cache", [{"time": e.time.isoformat(), "title": e.title, "kind": e.kind,
                                               "impact": e.impact, "symbols": e.symbols} for e in fetched])
            except Exception as e:
                # The free calendar rate-limits; fall back to the last copy and try again in 10 minutes.
                events = [ev for ev in self.events if ev.source == "calendar"] or self._cached_calendar()
                log.warning("economic calendar unavailable (%s) - using saved copy with %d events, retrying in 10 min",
                            e, len(events))
                self._last_calendar = now - pd.Timedelta(minutes=self.cfg.calendar_refresh_minutes - 10)
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
            self._dispatch([h for h in heads if h.key not in self._seen], now)
        cutoff = now - pd.Timedelta(minutes=self.cfg.post_event_window_minutes)
        with self._lock:
            self.breaking = [b for b in self.breaking if b.time >= cutoff]

    def _dispatch(self, new: list[Headline], now: pd.Timestamp) -> None:
        """Send new headlines to the reader without ever holding up trading.

        Claude reads only the freshest batch (one call at a time, at most every N seconds, in the
        background). Anything older or beyond that batch is skimmed by the keyword reader, which
        is instant. This keeps memory/CPU low on small servers and protects plan usage."""
        if not new:
            return
        slow = getattr(self.classifier, "background", False)
        if not slow:
            for i in range(0, len(new), self.cfg.max_headlines_per_call):
                self._classify(new[i:i + self.cfg.max_headlines_per_call])
            return
        busy = self._pending is not None and not self._pending.done()
        gap = getattr(self.classifier, "min_interval_seconds", 0)
        due = self._last_classify is None or (now - self._last_classify).total_seconds() >= gap
        fresh_cut = now - pd.Timedelta(minutes=self.cfg.post_event_window_minutes)
        fresh = sorted((h for h in new if h.time >= fresh_cut), key=lambda h: h.time, reverse=True)
        batch = fresh[: self.cfg.max_headlines_per_call] if (due and not busy) else []
        rest = [h for h in new if h not in batch]
        stale = [h for h in rest if h.time < fresh_cut]
        if stale:   # too old to trade on - skim them so they aren't queued forever
            self._classify(stale, KeywordClassifier())
        if batch:
            self._last_classify = now
            for h in batch:
                self._seen.add(h.key)
            self._pending = self._executor.submit(self._classify, batch)

    def wait(self, timeout: float = 300) -> None:
        """Block until a background read finishes (used by the `news` command)."""
        if self._pending is not None:
            try:
                self._pending.result(timeout=timeout)
            except Exception:
                log.exception("background news reading failed")

    def _cached_calendar(self) -> list[ScheduledEvent]:
        out = []
        for d in self.j.get("calendar_cache", []) or []:
            syms = [x for x in d["symbols"] if x in self.s.symbols]
            if syms:
                out.append(ScheduledEvent(pd.Timestamp(d["time"]), d["title"], d["kind"], d["impact"], syms))
        return out

    def _classify(self, batch: list[Headline], reader=None) -> None:
        reader = reader or self.classifier
        try:
            results = reader.classify(batch, self.s.symbols)
        except Exception as e:
            log.warning("news classifier failed (%s) - keyword fallback", e)
            results = KeywordClassifier().classify(batch, self.s.symbols)
        for h, a in zip(batch, results):
            self._seen.add(h.key)
            self._record(h.key, "headline", h.time, h.title, a.category, a.impact, a.effects, a.summary, h.source,
                         link=h.link, reader=a.source)
            if a.impact >= self.breaking_threshold and a.effects and a.source != "keywords":
                with self._lock:
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

    def blackout(self, symbol: str, now: pd.Timestamp, earnings_strategy: bool = False) -> str | None:
        """Reason not to open a new trade right now, or None. Earnings strategies may trade the days
        before a release, but not in the last stretch before it (TIIM closes them before the release)."""
        for e in self.events:
            if symbol not in e.symbols:
                continue
            mins = (now - e.time).total_seconds() / 60
            if e.kind == "earnings":
                start = (self.cfg.flatten_before_earnings_minutes + 120) if earnings_strategy else self.cfg.earnings_block_hours * 60
                if -start <= mins <= 60:
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
