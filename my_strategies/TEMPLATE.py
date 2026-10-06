"""Template for your own TIIM strategy. Copy this file, rename it (e.g. inside_bar_breakout.py),
and fill in generate(). This template file itself is ignored.

What generate() gets (ctx):
    ctx.symbol, ctx.df (closed candles: open/high/low/close), ctx.bar (last candle), ctx.close, ctx.atr_now
    ctx.trend      "up" | "down" | "range" | "up_break" | "down_break"   (HH/HL - LH/LL structure)
    ctx.swings     swing highs/lows with labels HH/HL/LH/LL
    ctx.zones      supply/demand zones (kind, top, bottom, fresh, touches)
    ctx.divergences, ctx.bias ("bull"/"bear" from EMA50 vs EMA200), ctx.regime
    ctx.series("rsi", 14), ctx.series("ema", 50), ctx.series("macd"), ctx.series("bollinger", 20, 2.0) ...
    ctx.news       news state (only for reinforcement - decide on price action)
What it returns: self._signal(ctx, "buy" | "sell", stop_price, "why") or None.
TIIM sets the target (3R), the position size (5% risk) and runs all safety checks.
"""
from tradingbot.strategies.base import Param, Strategy


class InsideBarBreakout(Strategy):
    name = "inside_bar_breakout"          # unique, lowercase, no spaces
    title = "Inside bar breakout"
    description = "Trades the break of an inside bar in the direction of the HH/HL or LH/LL trend."
    author = "Edgar"
    markets = ("index", "gold", "crypto")  # any of: index stock gold metal oil gas crypto fx
    stricter = ("min_mother_atr", 0.1)     # how TIIM makes it pickier if trades keep failing early

    def __init__(self):
        super().__init__()
        # Knobs TIIM may tune, with the range it's allowed to move them in: Param(start, low, high)
        self.params = {
            "min_mother_atr": Param(1.0, 0.5, 2.5),
            "stop_atr_buffer": Param(0.2, 0.05, 1.0),
        }

    def generate(self, ctx):
        if len(ctx.df) < 50:
            return None
        mother, inside, now = ctx.df.iloc[-3], ctx.df.iloc[-2], ctx.bar
        if not (inside["high"] < mother["high"] and inside["low"] > mother["low"]):
            return None
        if mother["high"] - mother["low"] < self.p("min_mother_atr") * ctx.atr_now:
            return None
        buf = self.p("stop_atr_buffer") * ctx.atr_now
        if ctx.trend.startswith("up") and now["close"] > mother["high"]:
            return self._signal(ctx, "buy", inside["low"] - buf, "inside bar broken upward in an uptrend")
        if ctx.trend.startswith("down") and now["close"] < mother["low"]:
            return self._signal(ctx, "sell", inside["high"] + buf, "inside bar broken downward in a downtrend")
        return None
