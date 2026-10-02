"""The foundation strategies: supply/demand reversals, divergences, and HH/HL - LH/LL structure."""
from __future__ import annotations

from ..analysis.context import MarketContext
from ..analysis.divergence import recent_divergence
from ..analysis.structure import last_of
from ..analysis.zones import zone_at_price
from .base import Param, Signal, Strategy


def _rejection(bar, side: str, min_wick_ratio: float) -> bool:
    """Candle closed in the direction of the trade and left a rejection wick."""
    rng = float(bar["high"] - bar["low"])
    if rng <= 0:
        return False
    if side == "buy":
        lower_wick = float(min(bar["open"], bar["close"]) - bar["low"])
        return bar["close"] > bar["open"] or lower_wick / rng >= min_wick_ratio
    upper_wick = float(bar["high"] - max(bar["open"], bar["close"]))
    return bar["close"] < bar["open"] or upper_wick / rng >= min_wick_ratio


class SupplyDemandReversal(Strategy):
    """Buy a fresh demand zone / sell a fresh supply zone on a rejection candle."""

    name = "sd_reversal"

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "stop_atr_buffer": Param(0.3, 0.1, 1.5),
            "min_impulse_atr": Param(1.5, 1.0, 3.0),
            "max_touches": Param(1, 0, 3),
            "min_wick_ratio": Param(0.4, 0.2, 0.7),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        bar = ctx.bar
        a = ctx.atr_now
        for kind, side in (("demand", "buy"), ("supply", "sell")):
            z = zone_at_price(ctx.zones, kind, float(bar["low"]), float(bar["high"]))
            if z is None or z.impulse_atr < self.p("min_impulse_atr") or z.touches > self.p("max_touches"):
                continue
            if z.created_idx >= ctx.i - 1:
                continue  # zone just formed, this is not a retest
            if not _rejection(bar, side, self.p("min_wick_ratio")):
                continue
            buf = self.p("stop_atr_buffer") * a
            stop = z.bottom - buf if side == "buy" else z.top + buf
            return self._signal(
                ctx, side, stop, f"{kind} zone retest ({z.bottom:.5f}-{z.top:.5f}), rejection candle",
                zone_fresh=z.fresh, zone_strength=round(z.impulse_atr, 2),
                with_structure=(side == "buy" and ctx.trend.startswith("up")) or (side == "sell" and ctx.trend.startswith("down")),
            )
        return None


class DivergenceReversal(Strategy):
    """Regular RSI/MACD divergence at a swing, confirmed by a candle in the trade direction."""

    name = "divergence"

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "stop_atr_buffer": Param(0.3, 0.1, 1.5),
            "max_bars_since": Param(4, 1, 10),
            "min_osc_gap": Param(2.0, 0.0, 10.0),   # min RSI point difference between the two swings
            "allow_hidden": Param(1, 0, 1),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        d = recent_divergence(ctx.divergences, ctx.i, int(self.p("max_bars_since")))
        if d is None:
            return None
        if d.kind.startswith("hidden") and not self.p("allow_hidden"):
            return None
        if d.oscillator == "rsi" and abs(d.osc_second - d.osc_first) < self.p("min_osc_gap"):
            return None
        side = d.side
        bar = ctx.bar
        if side == "buy" and bar["close"] <= bar["open"]:
            return None
        if side == "sell" and bar["close"] >= bar["open"]:
            return None
        buf = self.p("stop_atr_buffer") * ctx.atr_now
        extreme = d.second.price
        if side == "buy":
            extreme = min(extreme, float(ctx.df["low"].iloc[d.second.idx:].min()))
            stop = extreme - buf
        else:
            extreme = max(extreme, float(ctx.df["high"].iloc[d.second.idx:].max()))
            stop = extreme + buf
        return self._signal(ctx, side, stop, f"{d.kind} {d.oscillator} divergence", divergence=d.kind, oscillator=d.oscillator)


class StructureTrend(Strategy):
    """Trade pullbacks in an established HH/HL uptrend or LH/LL downtrend."""

    name = "structure_trend"

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "stop_atr_buffer": Param(0.3, 0.1, 1.5),
            "max_pullback_atr": Param(1.0, 0.3, 2.5),  # how close to the last HL/LH price must be
            "require_bias": Param(1, 0, 1),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        bar = ctx.bar
        a = ctx.atr_now
        buf = self.p("stop_atr_buffer") * a
        if ctx.trend == "up":
            hl = last_of(ctx.swings, "L")
            if self.p("require_bias") and ctx.bias == "bear":
                return None
            if hl and 0 <= ctx.close - hl.price <= self.p("max_pullback_atr") * a and bar["close"] > bar["open"]:
                return self._signal(ctx, "buy", hl.price - buf, "uptrend (HH/HL) pullback to last HL")
        elif ctx.trend == "down":
            lh = last_of(ctx.swings, "H")
            if self.p("require_bias") and ctx.bias == "bull":
                return None
            if lh and 0 <= lh.price - ctx.close <= self.p("max_pullback_atr") * a and bar["close"] < bar["open"]:
                return self._signal(ctx, "sell", lh.price + buf, "downtrend (LH/LL) pullback to last LH")
        return None


def core_strategies() -> list[Strategy]:
    return [SupplyDemandReversal(), DivergenceReversal(), StructureTrend()]
