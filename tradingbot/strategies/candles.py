"""Reversal candlestick patterns (the 50-pattern list) traded the way the guide says to use them: only at a key
level, after a move into it, and best with confirmation.

  * at a key level: price is at a higher-timeframe (Weekly/Daily/4H) zone or a fresh 15m zone that supports the
    trade (demand for buys, supply for sells)
  * after a move into it: built into every pattern rule (analysis/candles.py)
  * confirmation: weak patterns (spinning top, harami, homing pigeon, inverted hammer) need the next candle to
    close beyond them; medium and strong ones may be traded on the pattern candle itself
Stop beyond the pattern's extreme. TIIM learns per pattern which ones actually pay (condition 'candle').
"""
from __future__ import annotations

from ..analysis import candles as cdl
from ..analysis import htf
from ..analysis.context import MarketContext
from ..analysis.zones import zone_at_price
from .base import Param, Signal, Strategy


class CandleReversal(Strategy):
    name = "candle_reversal"
    title = "Reversal candlestick at a key level"
    description = ("Hammer, engulfing, morning/evening star, pin bar, tweezers, three soldiers/crows and the rest of "
                   "the 50 reversal patterns, on the 15m chart, taken only at a higher-timeframe or fresh 15m "
                   "supply/demand zone after a move into it. Weak patterns wait for a confirmation candle. Stop beyond "
                   "the pattern.")
    author = "TIIM (from your Alchemy Markets patterns)"
    stricter = ("min_strength", 1)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "min_strength": Param(2, 1, 3),         # 1 weak / 2 medium / 3 strong patterns allowed unconfirmed
            "stop_atr_buffer": Param(0.2, 0.05, 1.0),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        if len(ctx.df) < 60:
            return None
        found = cdl.detect(ctx.df, ctx.atr.to_numpy(), lookback=2)
        for p in found:
            fresh = p["bars_ago"] == 0
            if fresh and p["strength"] < self.p("min_strength"):
                continue                      # weak pattern: wait for its confirmation candle
            if not fresh and not p["confirmed"]:
                continue
            side = "buy" if p["dir"] > 0 else "sell"
            kind = "demand" if side == "buy" else "supply"
            at = htf.zone_at(ctx.htf, p["low"], p["high"]) if ctx.htf else None
            level = f"{htf.SHORT[at[0]]} {at[1]}" if at and at[1] == kind else None
            if not level:
                z = zone_at_price(ctx.zones, kind, p["low"], p["high"])
                level = f"15m {kind}" if z is not None and z.touches <= 1 else None
            if not level:
                continue
            a = ctx.atr_now
            buf = self.p("stop_atr_buffer") * a
            stop = p["low"] - buf if side == "buy" else p["high"] + buf
            if abs(ctx.close - stop) < 0.6 * a:
                stop = ctx.close - 0.6 * a if side == "buy" else ctx.close + 0.6 * a
            return self._signal(
                ctx, side, stop, f"{cdl.describe(p)} at a {level} zone",
                candle_pattern=p["type"], candle_strength=p["strength"], candle_confirmed=p["confirmed"],
                candle_level=level,
            )
        return None
