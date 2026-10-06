"""Earnings strategies for single stocks. Price action decides; earnings history only sets the bias.

earnings_runup  In the days before a release, if this stock has usually drifted one way into
                earnings, trade that way - but only when the chart agrees (right side of the
                50 EMA, structure not against it, candle in the trade direction). TIIM closes the
                trade before the release, so it never gambles on the gap.
earnings_drift  After the release, if this stock's earnings moves have usually kept going, trade
                in the direction of the reaction once price breaks the first hour's range that way.
"""
from __future__ import annotations

import pandas as pd

from ..analysis.context import MarketContext
from ..analysis.structure import last_of
from .base import Param, Signal, Strategy


def _bounded_stop(ctx: MarketContext, side: str, raw_stop: float, lo_atr: float = 0.8, hi_atr: float = 3.0) -> float:
    a = ctx.atr_now
    dist = min(max(abs(ctx.close - raw_stop), lo_atr * a), hi_atr * a)
    return ctx.close - dist if side == "buy" else ctx.close + dist


class EarningsRunup(Strategy):
    name = "earnings_runup"
    title = "Pre-earnings run-up"
    description = ("Uses the stock's past earnings to see if it usually drifts up or down into the release, then "
                   "trades that drift only when the chart agrees. Closed before the release - no gap gamble.")
    markets = ("stock",)
    needs = ("earnings",)
    stricter = ("min_consistency", 0.05)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "min_events": Param(4, 3, 8),
            "min_consistency": Param(0.65, 0.55, 0.9),   # share of past releases with the same pre-earnings drift
            "min_hours_to": Param(3, 1, 24),             # don't enter too close to the release
            "max_hours_to": Param(120, 24, 168),
            "stop_atr_buffer": Param(0.3, 0.1, 1.5),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        e = ctx.earnings or {}
        prof = e.get("profile") or {}
        hours_to = e.get("hours_to")
        if hours_to is None or not self.p("min_hours_to") <= hours_to <= self.p("max_hours_to"):
            return None
        if prof.get("n", 0) < self.p("min_events"):
            return None
        up = prof["pre_up_rate"]
        if up >= self.p("min_consistency"):
            side = "buy"
        elif up <= 1 - self.p("min_consistency"):
            side = "sell"
        else:
            return None
        # price action has the final say
        bar = ctx.bar
        ema50 = float(ctx.series("ema", 50).iloc[-1])
        if side == "buy" and not (ctx.close > ema50 and bar["close"] > bar["open"] and not ctx.trend.startswith("down")):
            return None
        if side == "sell" and not (ctx.close < ema50 and bar["close"] < bar["open"] and not ctx.trend.startswith("up")):
            return None
        sw = last_of(ctx.swings, "L" if side == "buy" else "H")
        buf = self.p("stop_atr_buffer") * ctx.atr_now
        raw = (sw.price - buf if side == "buy" else sw.price + buf) if sw else (ctx.close - 1.5 * ctx.atr_now
                                                                            if side == "buy" else ctx.close + 1.5 * ctx.atr_now)
        stop = _bounded_stop(ctx, side, raw)
        return self._signal(ctx, side, stop,
                            f"pre-earnings drift: {up:.0%} of {prof['n']} past releases ran "
                            f"{'up' if side == 'buy' else 'down'} into earnings; chart agrees",
                            earnings_phase="pre", earnings_consistency=round(max(up, 1 - up), 2))


class EarningsDrift(Strategy):
    name = "earnings_drift"
    title = "Post-earnings drift"
    description = ("After the release, if this stock's earnings moves have usually kept going, joins the reaction "
                   "once price breaks the first hour's range in the gap direction. Stop at the other side of that range.")
    markets = ("stock",)
    needs = ("earnings",)
    stricter = ("min_continuation", 0.05)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "min_events": Param(4, 3, 8),
            "min_continuation": Param(0.6, 0.5, 0.9),
            "min_gap_atr": Param(0.8, 0.3, 3.0),      # reaction size vs the stock's daily ATR
            "range_minutes": Param(60, 15, 120),       # the opening range after the release
            "max_hours_since": Param(30, 4, 48),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        e = ctx.earnings or {}
        prof = e.get("profile") or {}
        hs = e.get("hours_since")
        opened = e.get("last_reaction_open")
        if hs is None or opened is None or hs > self.p("max_hours_since") or hs * 60 < self.p("range_minutes"):
            return None
        if prof.get("n", 0) < self.p("min_events") or prof["continuation_rate"] < self.p("min_continuation"):
            return None
        t0 = pd.Timestamp(opened)
        before = ctx.df[ctx.df.index < t0]
        after = ctx.df[ctx.df.index >= t0]
        if before.empty or after.empty:
            return None
        rng = after[after.index < t0 + pd.Timedelta(minutes=self.p("range_minutes"))]
        if rng.empty or len(after) <= len(rng):
            return None
        prev_close = float(before["close"].iloc[-1])
        gap = float(after["open"].iloc[0]) - prev_close
        daily_atr = float(e.get("daily_atr") or 0) or ctx.atr_now * 6
        if abs(gap) < self.p("min_gap_atr") * daily_atr:
            return None
        hi, lo = float(rng["high"].max()), float(rng["low"].min())
        prev_bar_close = float(ctx.df["close"].iloc[-2])
        if gap > 0 and ctx.close > hi >= prev_bar_close:            # fresh break of the range, up
            side, raw = "buy", lo
        elif gap < 0 and ctx.close < lo <= prev_bar_close:          # fresh break, down
            side, raw = "sell", hi
        else:
            return None
        stop = _bounded_stop(ctx, side, raw, 0.8, 4.0)
        return self._signal(ctx, side, stop,
                            f"post-earnings drift: gapped {'up' if gap > 0 else 'down'} {abs(gap) / daily_atr:.1f} daily ATR, "
                            f"moves kept going after {prof['continuation_rate']:.0%} of {prof['n']} releases; range broken",
                            earnings_phase="post", earnings_gap_atr=round(gap / daily_atr, 2))
