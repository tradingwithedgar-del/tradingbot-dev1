"""How each stock has behaved around its past earnings, measured from price action.

For every past release TIIM measures (in daily ATRs, so stocks are comparable):
    pre_drift   how far price ran in the 5 sessions before the release
    gap         the reaction gap on the first session after the release
    post_drift  how far price kept going over the next 5 sessions, in the direction of the gap

Past release dates come from data/earnings.json and Finnhub when available. When fewer than four
are known, TIIM finds them in the chart itself: the largest overnight gaps, roughly one per quarter.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import requests

from ..analysis.indicators import atr as atr_fn
from .assets import asset_class
from .calendar import FINNHUB_EARNINGS_URL, HEADERS

log = logging.getLogger(__name__)


@dataclass
class EarningsReaction:
    session: str          # date of the first trading session after the release
    pre_drift: float      # close[T-1] - close[T-6], in ATR
    gap: float            # open[T] - close[T-1], in ATR
    day: float            # close[T] - close[T-1], in ATR
    post_drift: float     # (close[T+5] - close[T]) * sign(gap), in ATR - positive = the move continued
    detected: bool        # True = found from a gap, False = known release date


def reaction_sessions(daily: pd.DataFrame, releases: list[tuple[str, str]]) -> list[int]:
    """Map (date, bmo|amc) releases to the index of the first session that could react."""
    idx = daily.index.normalize()
    out = []
    for date, hour in releases:
        d = pd.Timestamp(date, tz="UTC").normalize()
        pos = idx.searchsorted(d, side="right" if hour == "amc" else "left")
        if 6 <= pos < len(daily):
            out.append(int(pos))
    return sorted(set(out))


def detect_from_gaps(daily: pd.DataFrame, min_gap_atr: float = 2.0, spacing: int = 45) -> list[int]:
    """Largest overnight gaps, at most one per `spacing` sessions - a proxy for earnings days."""
    a = atr_fn(daily).shift(1)
    gap = ((daily["open"] - daily["close"].shift(1)).abs() / a).to_numpy()
    order = np.argsort(-np.nan_to_num(gap))
    chosen: list[int] = []
    for i in order:
        if not np.isfinite(gap[i]) or gap[i] < min_gap_atr:
            break
        if i >= 6 and all(abs(i - c) >= spacing for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


def measure(daily: pd.DataFrame, sessions: list[int], detected: bool) -> list[EarningsReaction]:
    a = atr_fn(daily).shift(1).to_numpy()
    o, c = daily["open"].to_numpy(), daily["close"].to_numpy()
    out = []
    for t in sessions:
        if t < 6 or not np.isfinite(a[t]) or a[t] <= 0:
            continue
        gap = (o[t] - c[t - 1]) / a[t]
        post = ((c[min(t + 5, len(c) - 1)] - c[t]) / a[t]) * (1 if gap >= 0 else -1) if t + 1 < len(c) else 0.0
        out.append(EarningsReaction(daily.index[t].strftime("%Y-%m-%d"), round((c[t - 1] - c[t - 6]) / a[t], 3),
                                    round(gap, 3), round((c[t] - c[t - 1]) / a[t], 3), round(post, 3), detected))
    return out


def profile(reactions: list[EarningsReaction]) -> dict:
    """Summary TIIM uses to decide whether there's a repeatable earnings pattern."""
    n = len(reactions)
    if not n:
        return {"n": 0}
    pre = np.array([r.pre_drift for r in reactions])
    gaps = np.array([r.gap for r in reactions])
    post = np.array([r.post_drift for r in reactions])
    return {
        "n": n,
        "pre_up_rate": round(float((pre > 0).mean()), 3),        # how often price ran up into earnings
        "avg_pre_drift": round(float(pre.mean()), 3),
        "gap_up_rate": round(float((gaps > 0).mean()), 3),
        "avg_abs_gap": round(float(np.abs(gaps).mean()), 3),
        "continuation_rate": round(float((post > 0).mean()), 3),  # how often the gap direction kept going
        "avg_post_drift": round(float(post.mean()), 3),
        "pre_predicts_gap": round(float((np.sign(pre) == np.sign(gaps)).mean()), 3),
        "detected_dates": int(sum(r.detected for r in reactions)),
        "last_session": reactions[-1].session,
    }


def known_releases(symbol: str, data_dir, lookback_days: int = 1100) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    path = data_dir / "earnings.json"
    if path.exists():
        for d in json.loads(path.read_text()).get(symbol, []):
            date, _, hour = d.partition(" ")
            out.append((date, hour or "amc"))
    key = os.getenv("FINNHUB_API_KEY", "")
    if key:
        today = pd.Timestamp.now(tz="UTC").normalize()
        try:
            r = requests.get(FINNHUB_EARNINGS_URL, headers=HEADERS, timeout=15, params={
                "symbol": symbol, "from": (today - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d"),
                "to": today.strftime("%Y-%m-%d"), "token": key})
            r.raise_for_status()
            out += [(e["date"], e.get("hour") or "amc") for e in r.json().get("earningsCalendar", [])]
        except Exception as e:
            log.warning("earnings history lookup failed for %s: %s", symbol, e)
    today_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    return sorted({d for d in out if d[0] < today_str})


def build(symbol: str, daily: pd.DataFrame, releases: list[tuple[str, str]]) -> tuple[dict, list[EarningsReaction]]:
    sessions = reaction_sessions(daily, releases)
    reactions = measure(daily, sessions, detected=False)
    if len(reactions) < 4:
        found = [i for i in detect_from_gaps(daily) if all(abs(i - s) > 10 for s in sessions)]
        reactions = sorted(reactions + measure(daily, found, detected=True), key=lambda r: r.session)
    return profile(reactions), reactions


class EarningsDesk:
    """Keeps each stock's earnings profile fresh and tells strategies where they are in the earnings cycle."""

    def __init__(self, settings, broker, journal, news_monitor=None) -> None:
        self.s = settings
        self.broker = broker
        self.j = journal
        self.news = news_monitor
        self.stocks = [s for s in settings.symbols if asset_class(s) == "stock"]

    def refresh(self, symbol: str, now: pd.Timestamp) -> dict | None:
        key = f"earnings_profile:{symbol}"
        cached = self.j.get(key)
        if cached and now - pd.Timestamp(cached["built"]) < pd.Timedelta(days=1):
            return cached
        try:
            daily = self.broker.get_bars(symbol, "1D", 800)
        except Exception as e:
            log.warning("daily history for %s failed: %s", symbol, e)
            return cached
        if len(daily) < 120:
            return cached
        prof, reactions = build(symbol, daily, known_releases(symbol, self.s.db_path.parent))
        data = {"built": now.isoformat(), "profile": prof, "reactions": [asdict(r) for r in reactions],
                "daily_atr": float(atr_fn(daily).iloc[-1])}
        self.j.set(key, data)
        if prof.get("n"):
            self.j.event("earnings", f"{symbol} earnings profile: {prof['n']} releases, ran up into earnings "
                                     f"{prof['pre_up_rate']:.0%} of the time, move continued after "
                                     f"{prof['continuation_rate']:.0%}", prof)
        return data

    def context(self, symbol: str, now: pd.Timestamp) -> dict:
        """Empty unless the stock is within ~5 days before or ~2 days after a release."""
        if symbol not in self.stocks:
            return {}
        data = self.refresh(symbol, now)
        if not data or not data["profile"].get("n"):
            return {}
        out = {"profile": data["profile"]}
        nxt = None
        if self.news:
            nxt = next((e for e in self.news.events if e.kind == "earnings" and symbol in e.symbols and e.time > now), None)
        if nxt is not None:
            hours_to = (nxt.time - now).total_seconds() / 3600
            if hours_to <= 24 * 7:
                out.update(next_time=nxt.time.isoformat(), hours_to=round(hours_to, 2))
        last_event = None
        if self.news:
            past = [e for e in self.news.events if e.kind == "earnings" and symbol in e.symbols and e.time <= now]
            last_event = past[-1].time if past else None
        if last_event is not None:
            opened = _reaction_open(last_event)
        elif data["reactions"]:
            opened = pd.Timestamp(data["reactions"][-1]["session"], tz="UTC") + pd.Timedelta(hours=13, minutes=30)
        else:
            opened = None
        if opened is not None:
            hours_since = (now - opened).total_seconds() / 3600
            if 0 <= hours_since <= 48:
                out.update(last_reaction_open=opened.isoformat(), hours_since=round(hours_since, 2))
        out["daily_atr"] = data.get("daily_atr")
        return out if ("hours_to" in out or "hours_since" in out) else {}


def _reaction_open(release: pd.Timestamp) -> pd.Timestamp:
    """First US cash-session open (13:30 UTC, approximate) at or after a release time, skipping weekends."""
    t = release.normalize() + pd.Timedelta(hours=13, minutes=30)
    if t < release:
        t += pd.Timedelta(days=1)
    while t.weekday() >= 5:
        t += pd.Timedelta(days=1)
    return t
