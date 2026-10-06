"""News strategies. They only act while news is driving a symbol, and only after price confirms.

news_momentum: a high-impact release or breaking headline produced a strong impulse candle.
               Ride it, as long as the move agrees with what the news implies (when that is known).
news_fade:     the news spike over-extended and the next candle rejected it. Trade the snap-back.

Every trade records the news type (cpi, fomc, geopolitics, earnings, ...), so the learner works
out which kinds of news are worth trading on which symbols, and switches off the ones that aren't.
"""
from __future__ import annotations

from ..analysis.context import MarketContext
from .base import Param, Signal, Strategy

ACTIVE = ("post_event", "breaking")


def _news_ok(ctx: MarketContext, max_age: float) -> bool:
    n = ctx.news or {}
    age = n.get("news_age_min")
    return n.get("news_state") in ACTIVE and age is not None and 0 <= age <= max_age


class NewsMomentum(Strategy):
    name = "news_momentum"
    title = "News momentum"
    description = ("After high-impact news, rides a strong impulse candle - only when price confirms the news "
                   "direction. Stop beyond the impulse candle.")
    needs = ("news",)
    markets = ("index", "stock", "gold", "metal", "oil", "crypto")
    stricter = ("min_impulse_atr", 0.1)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "min_impulse_atr": Param(1.2, 0.8, 3.0),   # impulse candle body vs pre-news ATR
            "max_age_min": Param(45, 10, 90),
            "require_agree": Param(1, 0, 1),           # skip moves that contradict the headline
            "stop_atr_buffer": Param(0.2, 0.05, 1.0),
            "max_stop_atr": Param(2.5, 1.0, 4.0),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        if len(ctx.df) < 30 or not _news_ok(ctx, self.p("max_age_min")):
            return None
        bar = ctx.bar
        pre_atr = float(ctx.atr.iloc[-2])
        body = float(bar["close"] - bar["open"])
        if pre_atr <= 0 or abs(body) < self.p("min_impulse_atr") * pre_atr:
            return None
        side = "buy" if body > 0 else "sell"
        news_dir = int(ctx.news.get("news_dir", 0))
        if self.p("require_agree") and news_dir and (news_dir > 0) != (side == "buy"):
            return None
        buf = self.p("stop_atr_buffer") * pre_atr
        stop = float(bar["low"]) - buf if side == "buy" else float(bar["high"]) + buf
        max_dist = self.p("max_stop_atr") * pre_atr
        if abs(ctx.close - stop) > max_dist:   # very large candle: stop at its midpoint area instead
            stop = ctx.close - max_dist if side == "buy" else ctx.close + max_dist
        return self._signal(ctx, side, stop,
                            f"news momentum: {ctx.news.get('news_kind')} - {ctx.news.get('news_title', '')[:80]}",
                            impulse_atr=round(abs(body) / pre_atr, 2))


class NewsFade(Strategy):
    name = "news_fade"
    title = "News spike fade"
    description = ("When a news spike over-extends and the next candle rejects it, trades the snap-back. Stop beyond "
                   "the spike extreme.")
    needs = ("news",)
    markets = ("index", "stock", "gold", "metal", "oil", "crypto")
    stricter = ("min_spike_atr", 0.25)

    def __init__(self) -> None:
        super().__init__()
        self.params = {
            "min_spike_atr": Param(2.5, 1.5, 5.0),     # how far the news spike must have run
            "min_retrace": Param(0.38, 0.2, 0.7),      # how much of the spike the rejection must give back
            "max_age_min": Param(60, 10, 120),
            "stop_atr_buffer": Param(0.3, 0.1, 1.0),
        }

    def generate(self, ctx: MarketContext) -> Signal | None:
        if len(ctx.df) < 30 or not _news_ok(ctx, self.p("max_age_min")):
            return None
        recent = ctx.df.iloc[-4:-1]           # the spike happened in the previous 1-3 bars
        pre_atr = float(ctx.atr.iloc[-5])
        if pre_atr <= 0:
            return None
        start = float(recent["open"].iloc[0])
        hi, lo = float(recent["high"].max()), float(recent["low"].min())
        bar = ctx.bar
        buf = self.p("stop_atr_buffer") * pre_atr
        if hi - start >= self.p("min_spike_atr") * pre_atr:          # spiked up -> fade down
            spike = hi - start
            if bar["close"] < bar["open"] and hi - ctx.close >= self.p("min_retrace") * spike:
                return self._signal(ctx, "sell", max(hi, float(bar["high"])) + buf,
                                    f"news fade: spike up {spike / pre_atr:.1f} ATR rejected", spike_atr=round(spike / pre_atr, 2))
        if start - lo >= self.p("min_spike_atr") * pre_atr:          # spiked down -> fade up
            spike = start - lo
            if bar["close"] > bar["open"] and ctx.close - lo >= self.p("min_retrace") * spike:
                return self._signal(ctx, "buy", min(lo, float(bar["low"])) - buf,
                                    f"news fade: spike down {spike / pre_atr:.1f} ATR rejected", spike_atr=round(spike / pre_atr, 2))
        return None


def news_strategies() -> list[Strategy]:
    return [NewsMomentum(), NewsFade()]
