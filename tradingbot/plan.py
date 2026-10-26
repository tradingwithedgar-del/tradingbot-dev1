"""The weekly plan: before the week opens (Sunday 18:00 UTC) TIIM reads every market top-down
(Monthly -> Weekly -> Daily -> 4H), marks the key levels and the chart patterns that are forming, and writes
the plan it will trade that week:

  direction   buy / sell (with the higher-timeframe bias) or both (no clear bias - only clean breaks)
  scenarios   the concrete setups it is waiting for (pullback into a zone, a pattern breakout, ...)
  invalidation the daily level that proves the plan wrong; a daily close beyond it re-plans that market

During the week TIIM only takes real trades in the planned direction, unless the 4H or Daily structure
confirms a reversal (a close through the last swing, a break of structure or a chart-pattern breakout).
Price merely touching a zone or a reversal candle is not enough.
"""
from __future__ import annotations

import pandas as pd

from .analysis import htf
from .analysis.patterns import TITLES


def week_key(now) -> str:
    """Trading week label. From Sunday 18:00 UTC the coming week is planned."""
    t = pd.Timestamp(now)
    t = (t if t.tzinfo else t.tz_localize("UTC")) + pd.Timedelta(hours=6)
    y, w, _ = t.isocalendar()
    return f"{y}-W{w:02d}"


def _fmt(x: float) -> str:
    return f"{x:,.5g}" if abs(x) < 1000 else f"{x:,.1f}"


def market_plan(symbol: str, frames: dict, price: float, now) -> dict:
    b, score = htf.bias(frames)
    direction = {"bull": "buy", "bear": "sell"}.get(b, "both")
    day = frames.get("1D")
    zones = [(tf, z) for tf in ("1W", "1D", "4H") for z in (frames[tf].zones if tf in frames else [])]
    above = sorted([(tf, z) for tf, z in zones if z["kind"] == "supply" and z["bottom"] > price], key=lambda x: x[1]["bottom"])
    below = sorted([(tf, z) for tf, z in zones if z["kind"] == "demand" and z["top"] < price], key=lambda x: -x[1]["top"])
    scenarios = []
    if direction in ("buy", "both") and below:
        tf, z = below[0]
        scenarios.append({"side": "buy", "kind": "pullback",
                          "text": f"Buy a pullback into the {htf.SHORT[tf]} demand {_fmt(z['bottom'])}-{_fmt(z['top'])} "
                                  f"on a 15m rejection"})
    if direction in ("sell", "both") and above:
        tf, z = above[0]
        scenarios.append({"side": "sell", "kind": "pullback",
                          "text": f"Sell a rally into the {htf.SHORT[tf]} supply {_fmt(z['bottom'])}-{_fmt(z['top'])} "
                                  f"on a 15m rejection"})
    for tf in ("1D", "4H"):
        for p in (frames[tf].forming if tf in frames else [])[:3]:
            side = "buy" if p["dir"] > 0 else "sell"
            with_plan = direction in ("both", side)
            scenarios.append({
                "side": side, "kind": "pattern", "pattern": p["type"], "tf": tf, "trigger": p["trigger"],
                "target": p["target"], "with_plan": with_plan,
                "text": f"{'' if with_plan else 'Counter-plan: '}{side.capitalize()} if the {htf.NAMES[tf]} "
                        f"{TITLES[p['type']].lower()} breaks: {htf.NAMES[tf]} close {'above' if p['dir'] > 0 else 'below'} "
                        f"{_fmt(p['trigger'])}, target {_fmt(p['target'])}, wrong beyond {_fmt(p['invalid'])}"})
    if direction == "buy" and day and day.last_low is not None and day.last_low < price:
        invalid = day.last_low
    elif direction == "sell" and day and day.last_high is not None and day.last_high > price:
        invalid = day.last_high
    else:
        invalid = None
    if direction == "both":
        summary = f"No clear higher-timeframe direction (score {score:+d}/8): only clean breakouts, both ways."
    else:
        summary = (f"{'Bullish' if direction == 'buy' else 'Bearish'} week (score {score:+d}/8): "
                   f"{direction} setups only" + (f"; plan is wrong on a daily close {'below' if direction == 'buy' else 'above'} "
                                                 f"{_fmt(invalid)}" if invalid is not None else "") + ".")
    return {
        "symbol": symbol, "made": pd.Timestamp(now).isoformat(), "price": price, "bias": b, "score": score,
        "direction": direction, "invalidation": invalid, "status": "active", "summary": summary,
        "frames": {htf.SHORT[tf]: frames[tf].trend for tf in htf.FRAMES if tf in frames},
        "resistance": [{"tf": htf.SHORT[tf], "bottom": z["bottom"], "top": z["top"]} for tf, z in above[:2]],
        "support": [{"tf": htf.SHORT[tf], "bottom": z["bottom"], "top": z["top"]} for tf, z in below[:2]],
        "scenarios": scenarios[:6],
    }


def invalidated(p: dict, daily_close: float) -> bool:
    lvl = p.get("invalidation")
    if p.get("status") != "active" or lvl is None:
        return False
    return daily_close < lvl if p["direction"] == "buy" else (daily_close > lvl if p["direction"] == "sell" else False)


def against(p: dict | None, side: str, features: dict) -> str | None:
    """Why a real trade on this side breaks the weekly plan, or None."""
    if not p or p.get("status") != "active" or p.get("direction") not in ("buy", "sell") or side == p["direction"]:
        return None
    if features.get("htf_reversal"):
        return None          # the 4H/Daily structure broke the other way: a confirmed reversal, not a guess
    return f"against this week's plan for {p['symbol']} ({p['summary']})"


def text(book: dict) -> str:
    if not book:
        return "No weekly plan yet (it is made Sunday 18:00 UTC, or as soon as TIIM starts)."
    out = [f"Weekly plan {book['week']} (made {book['created'][:16].replace('T', ' ')} UTC)"]
    if book.get("briefing"):
        out.append(book["briefing"])
    for sym, p in sorted(book.get("markets", {}).items()):
        tag = "" if p["status"] == "active" else f" [{p['status'].upper()}]"
        out.append(f"\n{sym}{tag}: {p['summary']}")
        if book.get("notes", {}).get(sym):
            out.append(f"  desk: {book['notes'][sym]}")
        out += [f"  - {s['text']}" for s in p["scenarios"]]
    from .config import branded

    return branded("\n".join(out))
