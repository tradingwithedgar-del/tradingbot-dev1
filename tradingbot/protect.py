"""Profit protection: once a trade is in profit, the stop only ever moves toward profit, never away.

The original stop (the "how much am I willing to lose to be wrong" decision) stays on record for measuring R.
The stop actually working at the broker is `stop_now` in the trade's decision. It moves in steps, at the
agent's discretion:

  1. break-even plus costs, once the trade has shown `be_at` R of profit (default 1R; a strategy that
     keeps giving profits back gets it earlier, one whose cut-short trades later hit target gets it later)
  2. small locked profits as the trade runs: +0.5R at ~1.5R, +1R at ~2R, +1.5R at ~2.5R
  3. behind the latest swing the trade has made since entry (a real structure low/high), when that is
     tighter than the step above and still leaves room for normal noise

`scale` is TIIM's learned discretion per strategy (see Learner.protection_scale): above 1 it waits longer
before protecting, below 1 it protects earlier. Short-target trades (scalps) use proportionally earlier steps.
"""
from __future__ import annotations

MIN_STEP_R = 0.1          # don't move the stop for less than this: no broker spam
ROOM_ATR = 0.35           # keep the stop at least this far from price (normal noise)
LADDER = [(1.0, None), (1.5, 0.5), (2.0, 1.0), (2.5, 1.5)]   # (mfe R needed, locked profit R); None = break-even+


def plan(t: dict, dec: dict, ctx, scale: float = 1.0, be_at: float | None = None,
         swing_after: int | None = None) -> tuple[float, str] | None:
    """The stop to move to now, with a short reason, or None."""
    entry, orig = float(t["entry"]), float(t["stop"])
    risk = abs(entry - orig)
    if risk <= 0:
        return None
    d = 1 if t["side"] == "buy" else -1
    mfe = float(t.get("mfe_r") or 0.0)
    target_r = abs(float(t["take_profit"]) - entry) / risk
    cur = float(dec.get("stop_now", orig))
    price, atr = float(ctx.close), float(ctx.atr_now)
    cost_r = (float(ctx.spread) * 1.5) / risk

    # steps, scaled by the learned discretion and kept proportional for short targets
    first = (be_at or LADDER[0][0]) * scale
    squeeze = min(1.0, (0.55 * target_r) / first) if target_r < 2.5 else 1.0
    best_lock, label = None, ""
    for i, (need, lock) in enumerate(LADDER):
        trigger = (first if i == 0 else need * scale) * squeeze
        trigger = min(trigger, 0.85 * target_r)
        if mfe >= trigger:
            locked = (cost_r + 0.05) if lock is None else max(cost_r + 0.05, lock * squeeze)
            locked = min(locked, max(0.0, mfe - 0.5))          # always leave the trade room
            best_lock = locked
            label = "break-even plus costs" if lock is None else f"+{locked:.1f}R locked"
    cand = None
    if best_lock is not None:
        cand = entry + d * best_lock * risk
    # behind the latest swing since entry, once at least +1.5R was reached and a step is already on
    if best_lock is not None and mfe >= 1.5 * scale * squeeze and swing_after is not None:
        sw = [s for s in ctx.swings if s.idx > swing_after and s.confirmed_at <= ctx.i
              and ((s.kind == "L") if d > 0 else (s.kind == "H"))]
        if sw:
            lvl = sw[-1].price - d * 0.2 * atr
            if (lvl - entry) * d > best_lock * risk and (price - lvl) * d >= ROOM_ATR * atr:
                cand, label = lvl, "behind the latest swing"
    if cand is None:
        return None
    # keep the noise room, never loosen, ignore tiny steps
    cand = min(cand, price - d * ROOM_ATR * atr) if d > 0 else max(cand, price + ROOM_ATR * atr)
    if (cand - cur) * d < MIN_STEP_R * risk or (cand - entry) * d < 0:
        return None          # a protective stop is never worse than break-even; wait for price to give room
    return cand, label
