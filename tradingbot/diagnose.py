"""`tiim why`: a plain-English summary of what TIIM did recently and why it did or didn't trade."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import Settings
from .journal import Journal
from .learning import CANDIDATE

# How a shadow decision's last reason maps to a human explanation.
REASON_GROUPS = [
    ("not positive", "TIIM's estimate for this setup was not profitable enough (still learning; it explores on demo)"),
    ("learned filter", "TIIM learned this setup loses in these conditions"),
    ("major news points the other way", "major news pointed against the trade"),
    ("news window", "high-impact news release window"),
    ("earnings", "earnings coming up"),
    ("spread", "spread too wide / off-market quote"),
    ("misquote", "quote looked like a misquote/spike"),
    ("stale quote", "price quote was stale"),
    ("stop too tight", "stop would have been too tight"),
    ("already in a real trade", "already had a trade on that symbol"),
    ("max open trades", "already at the maximum number of open trades"),
    ("correlated", "already enough trades in that market group (e.g. several US indices the same way)"),
    ("open risk", "already at the maximum total risk"),
    ("daily loss limit", "daily loss limit reached"),
    ("drawdown", "drawdown limit reached (halted)"),
    ("halted", "TIIM is halted"),
    ("STOP file", "paused with `tiim stop`"),
    ("position too small", "position size below the broker minimum"),
    ("instrument spec failed", "could not read the instrument's contract details"),
    ("order failed", "the broker rejected the order"),
    ("not approved for live", "strategy not approved for the live account"),
]


def _why(reasons: list[str]) -> str:
    last = " ".join(reasons[-2:]) if reasons else ""
    for key, text in REASON_GROUPS:
        if key.lower() in last.lower():
            return text
    return reasons[-1] if reasons else "no reason recorded"


def report(j: Journal, s: Settings, now: pd.Timestamp | None = None, hours: int = 24) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    since = (now - pd.Timedelta(hours=hours)).isoformat()
    out: list[str] = [f"TIIM over the last {hours} hours ({s.mode} account)", ""]

    halted = j.get(f"halted:{s.mode}")
    if halted:
        out.append(f"!! HALTED: {halted}")
    if s.stop_file.exists():
        out.append("!! Paused: new trades are off (`tiim stop`). Run `tiim resume` to turn them back on.")

    last_bar = j.get(f"last_bar:{s.mode}", {}) or {}
    out.append("Markets checked (last candle analysed):")
    for sym in s.symbols:
        t = last_bar.get(sym)
        if not t:
            out.append(f"  {sym:<8} never - no price data yet (market closed or symbol problem)")
            continue
        age = (now - pd.Timestamp(t)).total_seconds() / 3600
        flag = "" if age < 1 else ("  (market closed?)" if age < 72 else "  (NOT UPDATING - check `journalctl -u tiim`)")
        out.append(f"  {sym:<8} {t[5:16].replace('T', ' ')} UTC, {age:.1f} h ago{flag}")

    rows = j.trades("mode = ? AND opened_at >= ?", (s.mode, since))
    real = [t for t in rows if not t["shadow"]]
    shadow = [t for t in rows if t["shadow"]]
    by_design = [t for t in shadow if t["experimental"] or CANDIDATE in t["strategy"]]
    blocked = [t for t in shadow if t not in by_design]
    out += ["", f"Setups found: {len(rows)}",
            f"  real trades opened:            {len(real)}",
            f"  blocked / passed on (virtual): {len(blocked)}",
            f"  experiments & tweak tests:     {len(by_design)}  (always virtual - that's by design)"]
    for t in real:
        out.append(f"  REAL #{t['id']} {t['symbol']} {t['side']} {t['strategy']} at {t['opened_at'][11:16]} UTC - {t['reason'][:70]}")
    if blocked:
        out += ["", "Why setups were passed on:"]
        for text, n in Counter(_why((t.get("decision") or {}).get("why", [])) for t in blocked).most_common():
            out.append(f"  {n:>3} x {text}")
        out.append("  by strategy: " + ", ".join(f"{k} {v}" for k, v in Counter(t["strategy"] for t in blocked).most_common()))
    if not rows:
        out += ["", "No setup met any strategy's rules. That's normal on quiet days: TIIM's strategies are",
                "strict on purpose (fresh zone + rejection, confirmed divergence, clean pullback)."]

    errors = [e for e in j.events(500, s.mode) if e["kind"] == "error" and e["ts"] >= since]
    if errors:
        out += ["", f"Errors: {len(errors)} (latest first)"]
        out += [f"  {e['ts'][5:16].replace('T', ' ')} {e['message'][:110]}" for e in errors[:5]]
    lessons = [e for e in j.events(500, s.mode) if e["kind"] in ("lesson", "tuning", "news", "manage") and e["ts"] >= since]
    if lessons:
        out += ["", "Notable:"] + [f"  {e['ts'][5:16].replace('T', ' ')} [{e['kind']}] {e['message'][:100]}" for e in lessons[:8]]
    return "\n".join(out)
