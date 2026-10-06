"""Builds one snapshot of everything the strategies need for the latest closed bar."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import pandas as pd

from . import indicators as ind
from .divergence import Divergence, find_divergences
from .regime import classify, session
from .structure import Swing, find_swings, structure_sequence, trend_state
from .zones import Zone, find_zones


@dataclass
class MarketContext:
    symbol: str
    df: pd.DataFrame                 # closed bars only, oldest first
    spread: float = 0.0
    news: dict = field(default_factory=dict)   # news state from NewsMonitor (empty in backtests)
    earnings: dict = field(default_factory=dict)  # upcoming/recent earnings + this stock's earnings history
    _cache: dict = field(default_factory=dict, repr=False)

    # --- basics -----------------------------------------------------------
    @property
    def i(self) -> int:
        return len(self.df) - 1

    @property
    def bar(self) -> pd.Series:
        return self.df.iloc[-1]

    @property
    def close(self) -> float:
        return float(self.df["close"].iloc[-1])

    @property
    def time(self) -> pd.Timestamp:
        return self.df.index[-1]

    def series(self, name: str, *args):
        """Cached indicator access, e.g. ctx.series('rsi', 14)."""
        key = (name, args)
        if key not in self._cache:
            fn = getattr(ind, name)
            src = self.df["close"] if name in {"ema", "sma", "rsi", "macd", "bollinger"} else self.df
            self._cache[key] = fn(src, *args)
        return self._cache[key]

    @cached_property
    def atr(self) -> pd.Series:
        return self.series("atr", 14)

    @property
    def atr_now(self) -> float:
        return float(self.atr.iloc[-1])

    # --- structure ------------------------------------------------------------
    @cached_property
    def swings(self) -> list[Swing]:
        return find_swings(self.df, 3, 3)

    @cached_property
    def trend(self) -> str:
        return trend_state(self.swings, self.close)

    @cached_property
    def zones(self) -> list[Zone]:
        return find_zones(self.df, self.atr)

    @cached_property
    def divergences(self) -> list[Divergence]:
        return find_divergences(self.swings, self.series("rsi", 14), "rsi") + find_divergences(
            self.swings, self.series("macd")["hist"], "macd"
        )

    @cached_property
    def bias(self) -> str:
        """Higher-timeframe proxy: EMA50 vs EMA200."""
        e50 = float(self.series("ema", 50).iloc[-1])
        e200 = float(self.series("ema", 200).iloc[-1])
        if len(self.df) < 200:
            return "neutral"
        return "bull" if e50 > e200 else "bear"

    @cached_property
    def regime(self) -> dict:
        return classify(float(self.series("adx", 14).iloc[-1]), self.atr)

    def features(self) -> dict:
        """Snapshot stored with every trade so the learner can explain wins and losses."""
        r = float(self.series("rsi", 14).iloc[-1])
        return {
            "regime": self.regime["regime"],
            "trend_strength": self.regime["trend_strength"],
            "volatility": self.regime["volatility"],
            "atr_percentile": self.regime["atr_percentile"],
            "structure": self.trend,
            "structure_seq": structure_sequence(self.swings),
            "bias": self.bias,
            "session": session(self.time),
            "rsi": round(r, 1),
            "rsi_zone": "oversold" if r < 30 else ("overbought" if r > 70 else "mid"),
            "atr": self.atr_now,
            "spread": self.spread,
            "weekday": int(self.time.weekday()),
            "news_state": self.news.get("news_state", "clear"),
            "news_kind": self.news.get("news_kind", "none"),
            "news_impact": self.news.get("news_impact", 0),
            "news_dir": self.news.get("news_dir", 0),
            "news_title": self.news.get("news_title", ""),
        }
