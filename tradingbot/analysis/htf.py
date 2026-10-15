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

It also measures how stretched the move already is (`extension`), so TIIM doesn't chase: selling after
seven red 4H candles and a -3% day usually means selling right before the pullback that takes the stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .indicators import atr as atr_fn
from .indicators import ema, rsi
from .structure import find_swings, structure_sequence, trend_state
from .zones import find_zones

FRAMES = ("1M", "1W", "1D", "4H")
NAMES = {"1M": "Monthly", "1W": "Weekly", "1D": "Daily", "4H": "4H", "1H": "1H"}
SHORT = {"1M": "M", "1W": "W", "1D": "D", "4H": "4H", "1H": "1H"}
PATTERN_FRAMES = ("4H", "1H")   # where breaks of structure / support breaks are looked for (1H isn't in the bias)
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
    close: float = 0.0
    atr: float = 0.0
    streak: int = 0                 # consecutive candles of one colour at the end: +5 = five green, -7 = seven red
    stretch_atr: float = 0.0        # (close - EMA20) / ATR: how far price has run from its average
    rsi: float = 50.0
    breaks: list = field(default_factory=list)  # recent structure / level breaks, newest last (see find_breaks)

    @property
    def direction(self) -> int:
        return {"up": 1, "up_break": 1, "down": -1, "down_break": -1}.get(self.trend, 0)

    def to_dict(self) -> dict:
        return {"tf": self.tf, "name": NAMES[self.tf], "trend": self.trend, "sequence": self.sequence,
                "ema_side": self.ema_side, "zones": self.zones, "bars": self.bars, "streak": self.streak,
                "stretch_atr": round(self.stretch_atr, 2), "rsi": round(self.rsi, 1), "breaks": self.breaks}


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
    a = float(atr_fn(df, 14).iloc[-1]) or 1e-12
    body = (df["close"] - df["open"]).to_numpy()
    streak = 0
    for b in body[::-1]:
        sign = 1 if b > 0 else -1 if b < 0 else 0
        if sign == 0 or (streak and (sign > 0) != (streak > 0)):
            break
        streak += sign
    r = float(rsi(df["close"], 14).iloc[-1])
    breaks = find_breaks(df, swings, a) if tf in PATTERN_FRAMES else []
    return Frame(tf, trend_state(swings, close), structure_sequence(swings, 4), "above" if close > e else "below",
                 zones, n, close, a, streak, (close - e) / a, r if r == r else 50.0, breaks)


def find_breaks(df: pd.DataFrame, swings, atr: float, lookback: int = 30, level_tol_atr: float = 0.35,
                min_touches: int = 2) -> list[dict]:
    """Recent breaks on this timeframe, each the first close through the level:

      bos   - break of structure: after a lower high, a close below the last higher low (bearish), or after a
              higher low, a close above the last lower high (bullish)
      level - a support/resistance level that rejected price 2+ times finally closes through
    """
    c = df["close"].to_numpy()
    n = len(c)
    out: list[dict] = []
    tol = level_tol_atr * atr
    for k in range(max(2, n - lookback), n):
        known = [s for s in swings if s.confirmed_at < k]
        hs = [s for s in known if s.kind == "H"]
        ls = [s for s in known if s.kind == "L"]
        if hs and ls:
            h, lo = hs[-1], ls[-1]
            if h.label == "LH" and c[k] < lo.price <= c[k - 1]:
                out.append({"type": "bos", "dir": -1, "level": lo.price, "swing": h.price, "k": k})
            if lo.label == "HL" and c[k] > h.price >= c[k - 1]:
                out.append({"type": "bos", "dir": 1, "level": h.price, "swing": lo.price, "k": k})
        for kind, d in (("L", -1), ("H", 1)):
            pts = [s for s in (ls if kind == "L" else hs) if k - s.idx <= 150]
            if len(pts) < min_touches:
                continue
            ref = pts[-1].price
            cluster = [s for s in pts if abs(s.price - ref) <= tol]
            if len(cluster) < min_touches:
                continue
            level = (min if d < 0 else max)(s.price for s in cluster)
            first = min(s.idx for s in cluster)
            held = c[first:k]
            if d < 0 and c[k] < level - 0.1 * atr and held.min() >= level - 0.1 * atr:
                out.append({"type": "level", "dir": -1, "level": level, "touches": len(cluster), "k": k,
                            "swing": max(s.price for s in hs[-3:]) if hs else level})
            if d > 0 and c[k] > level + 0.1 * atr and held.max() <= level + 0.1 * atr:
                out.append({"type": "level", "dir": 1, "level": level, "touches": len(cluster), "k": k,
                            "swing": min(s.price for s in ls[-3:]) if ls else level})
    for b in out:
        b["bars_ago"] = n - 1 - b.pop("k")
        b["time"] = df.index[n - 1 - b["bars_ago"]].isoformat()
    return out[-6:]


def build(daily: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame | None = None) -> dict[str, Frame]:
    """All frames from daily, 4H (and 1H) candles. Weekly and monthly are built from the daily ones."""
    out = {}
    if daily is not None and len(daily):
        out["1M"] = analyze_frame("1M", resample(daily, "MS"))
        out["1W"] = analyze_frame("1W", resample(daily, "W-MON"))
        out["1D"] = analyze_frame("1D", daily)
    if h4 is not None and len(h4):
        out["4H"] = analyze_frame("4H", h4)
    if h1 is not None and len(h1):
        out["1H"] = analyze_frame("1H", h1)
    return out


def bias(frames: dict[str, Frame]) -> tuple[str, int]:
    score = sum(WEIGHTS.get(tf, 0) * f.direction for tf, f in frames.items())
    known = sum(1 for tf, f in frames.items() if tf in WEIGHTS and f.trend != "unknown")
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


# A move counts as stretched in the trade's direction when (each adds one point):
EXT_STREAK = 5          # 5+ 4H candles in a row the trade's way
EXT_STRETCH = 2.0       # price 2+ 4H ATRs from the 4H 20-EMA, the trade's way
EXT_DAY_MOVE = 1.2      # today already moved 1.2+ daily ATRs the trade's way
EXT_RSI = 30            # 4H RSI at/below 30 for a sell, at/above 70 for a buy
EXHAUSTED = 2           # points at which TIIM won't chase


def extension(frames: dict[str, Frame], side: str, price: float) -> dict:
    """How far the move has already run in the trade's direction, from the 4H and Daily candles."""
    d = 1 if side == "buy" else -1
    h4, day = frames.get("4H"), frames.get("1D")
    if h4 is None or h4.bars < 20:
        return {"extended": "unknown"}
    streak = max(0, h4.streak * d)
    stretch = h4.stretch_atr * d
    day_move = ((price - day.close) / day.atr * d) if day and day.atr else 0.0
    rsi_hit = h4.rsi >= 100 - EXT_RSI if d > 0 else h4.rsi <= EXT_RSI
    score = int(streak >= EXT_STREAK) + int(stretch >= EXT_STRETCH) + int(day_move >= EXT_DAY_MOVE) + int(rsi_hit)
    return {"ext_streak_4h": streak, "ext_stretch_4h": round(stretch, 2), "ext_day_move": round(day_move, 2),
            "ext_rsi_4h": round(h4.rsi, 1), "ext_score": score,
            "extended": "exhausted" if score >= EXHAUSTED else ("stretched" if score else "no")}


def extension_text(f: dict, side: str) -> str:
    colour = "green" if side == "buy" else "red"
    parts = []
    if f.get("ext_streak_4h", 0) >= EXT_STREAK:
        parts.append(f"{f['ext_streak_4h']} {colour} 4H candles in a row")
    if f.get("ext_stretch_4h", 0) >= EXT_STRETCH:
        parts.append(f"{f['ext_stretch_4h']:.1f} ATR {'above' if side == 'buy' else 'below'} the 4H average")
    if f.get("ext_day_move", 0) >= EXT_DAY_MOVE:
        parts.append(f"today already {f['ext_day_move']:.1f} daily ATRs {'up' if side == 'buy' else 'down'}")
    rsi_v = f.get("ext_rsi_4h")
    if rsi_v is not None and (rsi_v >= 100 - EXT_RSI if side == "buy" else rsi_v <= EXT_RSI):
        parts.append(f"4H RSI {rsi_v:.0f}")
    return ", ".join(parts) or "not stretched"


def features(frames: dict[str, Frame], side: str, low: float, high: float, entry: float, stop: float) -> dict:
    if not frames:
        return {"htf_align": "unknown"}
    b, score = bias(frames)
    want = "bull" if side == "buy" else "bear"
    align = "unknown" if b == "unknown" else ("mixed" if b == "mixed" else ("with" if b == want else "against"))
    at = zone_at(frames, low, high)
    feats = {f"htf_{SHORT[tf]}": f.trend for tf, f in frames.items() if tf in WEIGHTS}
    feats.update({
        "htf_bias": b, "htf_score": score, "htf_align": align,
        "htf_zone": f"{SHORT[at[0]]} {at[1]}" if at else "none",
        # price reacting at a higher-timeframe zone that supports this direction = a real reversal spot
        "htf_reversal_spot": bool(at and at[1] == ("demand" if side == "buy" else "supply")),
        "htf_room_r": room_r(frames, side, entry, stop),
    })
    feats.update(extension(frames, side, entry))
    return feats


def describe(frames: dict[str, Frame]) -> str:
    if not frames:
        return "higher timeframes: not available"
    parts = [f"{NAMES[tf]} {frames[tf].trend}" + (f" ({frames[tf].sequence})" if frames[tf].sequence else "")
             for tf in FRAMES if tf in frames]
    b, score = bias(frames)
    return "; ".join(parts) + f" -> bias {b} (score {score:+d} of 8)"
