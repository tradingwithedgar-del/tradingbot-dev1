"""Market regime: trend strength x volatility. The learner tracks every strategy per
regime, so a strategy that only works in quiet ranging markets gets used there and
not in a fast trend."""
from __future__ import annotations

import pandas as pd


def classify(adx_value: float, atr_series: pd.Series, lookback: int = 200) -> dict:
    trend = "trending" if adx_value >= 25 else ("weak" if adx_value >= 18 else "ranging")
    window = atr_series.dropna().iloc[-lookback:]
    pct = float((window <= window.iloc[-1]).mean()) if len(window) else 0.5
    vol = "high_vol" if pct >= 0.8 else ("low_vol" if pct <= 0.2 else "normal_vol")
    return {"regime": f"{trend}_{vol}", "trend_strength": trend, "volatility": vol, "atr_percentile": round(pct, 3)}


def session(ts: pd.Timestamp) -> str:
    """Trading session by UTC hour."""
    h = ts.hour
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "london_ny_overlap"
    if 16 <= h < 21:
        return "new_york"
    return "asia"
