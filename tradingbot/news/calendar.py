"""Scheduled market-moving events: the economic calendar and company earnings."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from .assets import asset_class, usd_driven

log = logging.getLogger(__name__)

# Free weekly economic calendar (ForexFactory export). No key needed.
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"
HEADERS = {"User-Agent": "Mozilla/5.0 (tradingbot news monitor)"}

EVENT_KINDS = [
    ("fomc", r"FOMC|Federal Funds Rate|Fed Chair|Powell"),
    ("cpi", r"\bCPI\b"),
    ("nfp", r"Non-Farm|NFP"),
    ("pce", r"\bPCE\b"),
    ("ppi", r"\bPPI\b"),
    ("gdp", r"\bGDP\b"),
    ("jobs", r"Unemployment|Jobless|JOLTS|ADP"),
    ("retail_sales", r"Retail Sales"),
    ("ism", r"\bISM\b|PMI"),
    ("oil_inventory", r"Crude Oil Inventories|Crude Inventories"),
]


@dataclass
class ScheduledEvent:
    time: pd.Timestamp
    title: str
    kind: str
    impact: int                 # 1 low, 2 medium, 3 high
    symbols: list[str] = field(default_factory=list)
    source: str = "calendar"

    @property
    def key(self) -> str:
        return f"{self.source}|{self.time.isoformat()}|{self.title}"


def event_kind(title: str) -> str:
    for kind, pattern in EVENT_KINDS:
        if re.search(pattern, title, re.I):
            return kind
    return "macro"


def parse_calendar(rows: list[dict], symbols: list[str]) -> list[ScheduledEvent]:
    """Keep USD events that move the traded symbols. Oil inventories only matter to oil."""
    impact_map = {"high": 3, "medium": 2, "low": 1}
    out = []
    for r in rows:
        if str(r.get("country", "")).upper() != "USD":
            continue
        title = str(r.get("title", ""))
        kind = event_kind(title)
        impact = impact_map.get(str(r.get("impact", "")).lower(), 0)
        if kind == "oil_inventory":
            affected = [s for s in symbols if asset_class(s) == "oil"]
            impact = max(impact, 3) if affected else impact
        else:
            affected = [s for s in symbols if usd_driven(s)]
        if impact < 3 or not affected:
            continue
        try:
            t = pd.Timestamp(r["date"]).tz_convert("UTC")
        except Exception:
            continue
        out.append(ScheduledEvent(t, title, kind, impact, affected))
    return out


def fetch_calendar(symbols: list[str]) -> list[ScheduledEvent]:
    resp = requests.get(CALENDAR_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return parse_calendar(resp.json(), symbols)


def _earnings_time(date: str, hour: str) -> pd.Timestamp:
    # bmo = before market open, amc = after market close (US/Eastern -> UTC, approximate)
    d = pd.Timestamp(date, tz="UTC")
    return d + (pd.Timedelta(hours=12) if hour.lower() == "bmo" else pd.Timedelta(hours=20, minutes=15))


def load_earnings(symbols: list[str], data_dir: Path) -> list[ScheduledEvent]:
    """Earnings dates from Finnhub (free key in FINNHUB_API_KEY) plus data/earnings.json:
        {"NVDA": ["2026-11-19 amc"], "AAPL": ["2026-10-29 amc"]}
    """
    stocks = [s for s in symbols if asset_class(s) == "stock"]
    out: list[ScheduledEvent] = []
    path = data_dir / "earnings.json"
    if path.exists():
        for sym, dates in json.loads(path.read_text()).items():
            if sym in stocks:
                for d in dates:
                    date, _, hour = d.partition(" ")
                    out.append(ScheduledEvent(_earnings_time(date, hour or "amc"), f"{sym} earnings", "earnings", 3,
                                              [sym], "earnings"))
    key = os.getenv("FINNHUB_API_KEY", "")
    if key and stocks:
        today = pd.Timestamp.now(tz="UTC").normalize()
        for sym in stocks:
            try:
                r = requests.get(FINNHUB_EARNINGS_URL, headers=HEADERS, timeout=15, params={
                    "symbol": sym, "from": today.strftime("%Y-%m-%d"),
                    "to": (today + pd.Timedelta(days=60)).strftime("%Y-%m-%d"), "token": key})
                r.raise_for_status()
                for e in r.json().get("earningsCalendar", []):
                    out.append(ScheduledEvent(_earnings_time(e["date"], e.get("hour") or "amc"), f"{sym} earnings",
                                              "earnings", 3, [sym], "earnings"))
            except Exception as ex:
                log.warning("earnings lookup failed for %s: %s", sym, ex)
    uniq = {e.key: e for e in out}
    return sorted(uniq.values(), key=lambda e: e.time)
