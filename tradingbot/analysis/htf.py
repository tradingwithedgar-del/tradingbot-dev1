"""Top-down analysis: Monthly -> Weekly -> Daily -> 4H decide the direction; the 15m chart only times
the entry.

For each higher timeframe TIIM reads the swing structure (HH/HL = up, LH/LL = down) and the unbroken
supply/demand zones. Together they give:

  * htf_bias   bull / bear / mixed - weighted vote of the four timeframes (Daily counts most)
  * htf_align  with / against / mixed - the 15m setup's direction versus that bias
  * htf_zone   the higher-timeframe zone price is sitting in, if any (e.g. "D demand")
  * htf_room_r how far the nearest higher-timeframe zone in the way of the trade is, in R

A 15m sell that makes a lower high inside a Daily/Weekly uptrend is "against". TIIM doesn't take
those for real unless price is reacting at a higher-timeframe supply zone (a genuine reversal spot).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import atr as atr_fn
from .indicators import ema
from .structure import find_swings, structure_sequence, trend_state
from .zones import find_zones

FRAMES = ("1M", "1W", "1D", "4H")
NAMES = {"1M": "Monthly", "1W": "Weekly", "1D": "Daily", "4H": "4H"}
SHORT = {"1M": "M", "1W": "W", "1D": "D", "4H": "4H"}
WEIGHTS = {"1M": 1, "1W": 2, "1D": 3, "4H": 2}
BIAS_THRESHOLD = 3          # weighted score needed for a clear bull/bear bias (max is 8)


@dataclass
class Frame:
    tf: str
    trend: str = "unknown"          # up / down / range / up_break / down_break / unknown
    sequence: str = ""
    ema_side: str = ""              # price above/below the 20-period EMA of that timeframe
    zones: list = field(default_factory=list)   # [{kind, top, bottom, fresh}]
    bars: int = 0

    @property
    def direction(self) -> int:
        return {"up": 1, "up_break": 1, "down": -1, "down_break": -1}.get(self.trend, 0)

    def to_dict(self) -> dict:
        return {"tf": self.tf, "name": NAMES[self.tf], "trend": self.trend, "sequence": self.sequence,
                "ema_side": self.ema_side, "zones": self.zones, "bars": self.bars}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Higher-timeframe candles from lower ones. The last (still forming) candle is dropped."""
    if df.empty:
        return df
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
    return out.iloc[:-1]


def analyze_frame(tf: str, df: pd.DataFrame, max_zones: int = 4) -> Frame:
    n = len(df)
    if n < 12:
        return Frame(tf, bars=n)
    side = 2 if n < 60 else 3
    swings = find_swings(df, left=side, right=side)
    close = float(df["close"].iloc[-1])
    e = float(ema(df["close"], 20).iloc[-1])
    zones = []
    if n >= 30:
        for z in find_zones(df, atr_fn(df, 14))[-12:]:
            zones.append({"kind": z.kind, "top": z.top, "bottom": z.bottom, "fresh": z.fresh,
                          "from": df.index[z.base_start].isoformat()})
        zones = zones[-max_zones:]
    return Frame(tf, trend_state(swings, close), structure_sequence(swings, 4), "above" if close > e else "below",
                 zones, n)


def build(daily: pd.DataFrame, h4: pd.DataFrame) -> dict[str, Frame]:
    """All four frames from daily and 4H candles (weekly and monthly are built from the daily ones)."""
    out = {}
    if daily is not None and len(daily):
        out["1M"] = analyze_frame("1M", resample(daily, "MS"))
        out["1W"] = analyze_frame("1W", resample(daily, "W-MON"))
        out["1D"] = analyze_frame("1D", daily)
    if h4 is not None and len(h4):
        out["4H"] = analyze_frame("4H", h4)
    return out


def bias(frames: dict[str, Frame]) -> tuple[str, int]:
    score = sum(WEIGHTS[tf] * f.direction for tf, f in frames.items())
    known = sum(1 for f in frames.values() if f.trend != "unknown")
    if known < 2:
        return "unknown", score
    return ("bull" if score >= BIAS_THRESHOLD else "bear" if score <= -BIAS_THRESHOLD else "mixed"), score


def zone_at(frames: dict[str, Frame], low: float, high: float) -> tuple[str, str] | None:
    """(timeframe, kind) of the biggest-timeframe zone the current candle is touching."""
    for tf in FRAMES[1:]:              # weekly, daily, 4H zones (monthly zones are too wide to act on)
        f = frames.get(tf)
        for z in reversed(f.zones if f else []):
            if low <= z["top"] and high >= z["bottom"]:
                return tf, z["kind"]
    return None


def room_r(frames: dict[str, Frame], side: str, entry: float, stop: float) -> float | None:
    """Distance from entry to the nearest higher-timeframe zone in the trade's way, in R."""
    risk = abs(entry - stop) or 1e-12
    best = None
    for tf in ("1W", "1D", "4H"):
        f = frames.get(tf)
        for z in f.zones if f else []:
            if side == "buy" and z["kind"] == "supply" and z["bottom"] > entry:
                d = (z["bottom"] - entry) / risk
            elif side == "sell" and z["kind"] == "demand" and z["top"] < entry:
                d = (entry - z["top"]) / risk
            else:
                continue
            best = d if best is None else min(best, d)
    return None if best is None else round(best, 2)


def features(frames: dict[str, Frame], side: str, low: float, high: float, entry: float, stop: float) -> dict:
    if not frames:
        return {"htf_align": "unknown"}
    b, score = bias(frames)
    want = "bull" if side == "buy" else "bear"
    align = "unknown" if b == "unknown" else ("mixed" if b == "mixed" else ("with" if b == want else "against"))
    at = zone_at(frames, low, high)
    feats = {f"htf_{SHORT[tf]}": f.trend for tf, f in frames.items()}
    feats.update({
        "htf_bias": b, "htf_score": score, "htf_align": align,
        "htf_zone": f"{SHORT[at[0]]} {at[1]}" if at else "none",
        # price reacting at a higher-timeframe zone that supports this direction = a real reversal spot
        "htf_reversal_spot": bool(at and at[1] == ("demand" if side == "buy" else "supply")),
        "htf_room_r": room_r(frames, side, entry, stop),
    })
    return feats


def describe(frames: dict[str, Frame]) -> str:
    if not frames:
        return "higher timeframes: not available"
    parts = [f"{NAMES[tf]} {frames[tf].trend}" + (f" ({frames[tf].sequence})" if frames[tf].sequence else "")
             for tf in FRAMES if tf in frames]
    b, score = bias(frames)
    return "; ".join(parts) + f" -> bias {b} (score {score:+d} of 8)"
