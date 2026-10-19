"""Break & retest: the bigger chart gives the signal, the 15m gives the entry.

Built from your USTECH example:
  1. On the 4H or 1H chart, a lower high is followed by a close below the last higher low (break of structure),
     or a support level that rejected price two or more times finally closes through.
  2. On the 15m chart, price pulls back a little toward the broken level without reclaiming it, and a candle
     rejects the pullback.
  3. Sell (or buy, mirrored) on that rejection, early in the move, with the stop just above the pullback high.

The early entry is the point: the stop is small and close to where the idea is proven wrong, instead of
chasing the move after seven red candles.
"""
from __future__ import annotations

from ..analysis.context import MarketContext
from .base import Param, Signal, Strategy

from ..analysis.patterns import PATTERN_TYPES, TITLES

MAX_BARS_SINCE = {"1H": 12, "4H": 6, "1D": 3}   # how recent the break must be on each timeframe


class BreakRetest(Strategy):
    name = "break_retest"
    title = "Break of structure / support break, 15m retest"
    description = ("4H or 1H: a lower high then a close below the last higher low, or a support that held 2+ times "
                   "finally breaks (mirrored for buys). 15m: small pullback toward the broken level, rejection "
                   "candle, enter. Stop just beyond the pullback.")
    author = "TIIM (from your USTECH chart)"
    stricter = ("min_wick_ratio", 0.05)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "max_dist_atr": Param(1.0, 0.3, 2.5),      # entry no further than this from the broken level (HTF ATR)
            "reclaim_atr": Param(0.3, 0.0, 1.0),       # the pullback may poke this far back through the level
            "pullback_bars": Param(8, 3, 20),          # 15m candles in which to look for the pullback
            "min_pullback_atr": Param(0.5, 0.2, 2.0),  # the pullback must be at least this big (15m ATR)
            "stop_atr_buffer": Param(0.2, 0.05, 1.0),
            "min_wick_ratio": Param(0.4, 0.2, 0.7),
        }

    frames = ("1H", "4H")
    break_types = {"bos", "level"}

    def _latest_break(self, ctx: MarketContext) -> tuple[str, dict, object] | None:
        best = None
        for tf in self.frames:
            f = ctx.htf.get(tf)
            mine = [b for b in (f.breaks if f else []) if b["type"] in self.break_types]
            if not mine:
                continue
            b = mine[-1]
            if b["bars_ago"] > MAX_BARS_SINCE[tf]:
                continue
            if best is None or b["time"] > best[1]["time"]:
                best = (tf, b, f)
        return best

    def generate(self, ctx: MarketContext) -> Signal | None:
        if not ctx.htf or len(ctx.df) < 60:
            return None
        found = self._latest_break(ctx)
        if not found:
            return None
        tf, brk, frame = found
        d = brk["dir"]
        side = "buy" if d > 0 else "sell"
        level, htf_atr, a = brk["level"], frame.atr, ctx.atr_now
        close = ctx.close
        # still close to the break: not late
        dist = (close - level) * d / (htf_atr or 1e-12)
        if dist > self.p("max_dist_atr") or dist < -self.p("reclaim_atr"):
            return None
        # 15m pullback against the break, then a rejection candle
        n = int(self.p("pullback_bars"))
        win = ctx.df.iloc[-n - 1:-1]
        if len(win) < 3:
            return None
        if side == "sell":
            pb_extreme = float(win["high"].max())
            start = float(ctx.df["low"].iloc[-3 * n:-n].min()) if len(ctx.df) > 3 * n else float(win["low"].min())
            pullback = pb_extreme - min(start, float(win["low"].min()))
            reclaimed = pb_extreme > level + self.p("reclaim_atr") * htf_atr
        else:
            pb_extreme = float(win["low"].min())
            start = float(ctx.df["high"].iloc[-3 * n:-n].max()) if len(ctx.df) > 3 * n else float(win["high"].max())
            pullback = max(start, float(win["high"].max())) - pb_extreme
            reclaimed = pb_extreme < level - self.p("reclaim_atr") * htf_atr
        if reclaimed or pullback < self.p("min_pullback_atr") * a:
            return None
        bar, prev = ctx.bar, ctx.df.iloc[-2]
        rng = float(bar["high"] - bar["low"]) or 1e-12
        if side == "sell":
            with_trade = bar["close"] < bar["open"]
            turned = bar["close"] < prev["low"]
            wick = float(bar["high"] - bar["open"]) / rng
        else:
            with_trade = bar["close"] > bar["open"]
            turned = bar["close"] > prev["high"]
            wick = float(bar["open"] - bar["low"]) / rng
        # rejection: a candle in the trade's direction that either broke the previous candle or left a wick
        if not with_trade or not (turned or wick >= self.p("min_wick_ratio")):
            return None
        buf = self.p("stop_atr_buffer") * a
        stop = pb_extreme + buf if side == "sell" else pb_extreme - buf
        if abs(close - stop) < 0.6 * a:                       # never a stop tighter than the compliance floor
            stop = close + 0.6 * a if side == "sell" else close - 0.6 * a
        return self._make(ctx, side, stop, tf, brk, level, dist)

    def _make(self, ctx, side, stop, tf, brk, level, dist) -> Signal | None:
        d = brk["dir"]
        what = ("break of structure (lower high, then close below the higher low)" if brk["type"] == "bos" and d < 0 else
                "break of structure (higher low, then close above the lower high)" if brk["type"] == "bos" else
                f"{'support' if d < 0 else 'resistance'} that held {brk.get('touches', 2)} times broke")
        return self._signal(
            ctx, side, stop,
            f"{tf} {what} at {level:.5g} {brk['bars_ago']} candle(s) ago; 15m pullback rejected",
            break_tf=tf, break_type=brk["type"], break_level=level, break_bars_ago=brk["bars_ago"],
            break_dist_atr=round(dist, 2),
        )


class ChartPattern(BreakRetest):
    """Classic chart patterns on the Daily / 4H / 1H chart, entered on the 15m retest of the breakout."""

    name = "chart_pattern"
    title = "Chart patterns (H&S, double/triple tops, triangles, wedges, flags, cups...)"
    description = ("Daily, 4H or 1H: head & shoulders, double/triple top or bottom, triangles, wedges, channels, "
                   "flags, pennants, rounding bottom, cup & handle, bump & run. After the breakout, waits for the "
                   "15m pullback toward the breakout line and a rejection candle. Target = the pattern's measured "
                   "move; stop beyond the pullback. TIIM learns which patterns work on which market.")
    author = "TIIM (from your pattern sheet)"
    frames = ("1D", "4H", "1H")
    break_types = PATTERN_TYPES
    MIN_RR, MAX_RR = 1.5, 4.0

    def _make(self, ctx, side, stop, tf, brk, level, dist) -> Signal | None:
        risk = abs(ctx.close - stop) or 1e-12
        rr = abs(brk["target"] - ctx.close) / risk
        if (brk["target"] - ctx.close) * brk["dir"] <= 0 or rr < self.MIN_RR:
            return None                               # the measured move doesn't leave enough room
        sig = self._signal(
            ctx, side, stop,
            f"{tf} {TITLES[brk['type']].lower()} broke at {level:.5g} {brk['bars_ago']} candle(s) ago "
            f"(measured target {brk['target']:.5g}); 15m pullback rejected",
            pattern=brk["type"], break_tf=tf, break_type=brk["type"], break_level=level,
            break_bars_ago=brk["bars_ago"], break_dist_atr=round(dist, 2), pattern_target=brk["target"],
        )
        if sig is not None:
            sig.reward_multiple = round(min(self.MAX_RR, rr), 2)
        return sig
