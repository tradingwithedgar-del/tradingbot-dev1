"""The 50 reversal candlestick patterns (Alchemy Markets list) as rules TIIM can check on any timeframe.

#1-21 bullish candle patterns and #26-46 their bearish counterparts are detected here. Each bearish pattern is
the bullish rule applied to the price chart turned upside down (a shooting star is an upside-down hammer,
dark cloud cover an upside-down piercing line, ...), so both sides follow exactly the same logic.
#22-25 (double/triple bottom, inverse head & shoulders, falling wedge) and #47-50 (double/triple top, head &
shoulders, rising wedge) are chart patterns: see analysis/patterns.py.

What the page stresses, and TIIM applies:
  * a reversal pattern only means something after a move into it (downtrend for bullish, uptrend for bearish),
    ideally at a support/resistance level -> `trend` check here, key-level check in the candle_reversal strategy
  * most patterns need a confirmation candle -> `confirmed` when the next candle closes beyond the pattern's
    last candle in the pattern's direction
  * higher volume strengthens the signal -> `volume_up` (CFD volume is tick volume, so it is a hint only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name: (title, candles, strength 1 weak / 2 medium / 3 strong, number on the page)
BULLISH = {
    "hammer": ("Hammer", 1, 2, 1),
    "inverted_hammer": ("Inverted hammer", 1, 1, 2),
    "dragonfly_doji": ("Dragonfly doji", 1, 2, 3),
    "bullish_pin_bar": ("Bullish pin bar", 1, 2, 4),
    "bullish_spinning_top": ("Bullish spinning top", 1, 1, 5),
    "tweezer_bottoms": ("Tweezer bottoms", 2, 2, 6),
    "piercing_line": ("Piercing line", 2, 2, 7),
    "bullish_harami": ("Bullish harami", 2, 1, 8),
    "bullish_engulfing": ("Bullish engulfing", 2, 3, 9),
    "bullish_kicker": ("Bullish kicker", 2, 3, 10),
    "bullish_belt_hold": ("Bullish belt hold", 1, 2, 11),
    "bullish_counterattack": ("Bullish counterattack line", 2, 2, 12),
    "homing_pigeon": ("Bullish homing pigeon", 2, 1, 13),
    "morning_star": ("Morning star", 3, 3, 14),
    "morning_doji_star": ("Morning doji star", 3, 3, 15),
    "three_white_soldiers": ("Three white soldiers", 3, 3, 16),
    "bullish_abandoned_baby": ("Bullish abandoned baby", 3, 3, 17),
    "three_inside_up": ("Three inside up", 3, 3, 18),
    "three_outside_up": ("Three outside up", 3, 3, 19),
    "bullish_stick_sandwich": ("Bullish stick sandwich", 3, 2, 20),
    "bullish_tri_star": ("Bullish tri star", 3, 2, 21),
}
MIRROR = {
    "hammer": ("shooting_star", "Shooting star", 26), "inverted_hammer": ("hanging_man", "Hanging man", 27),
    "dragonfly_doji": ("gravestone_doji", "Gravestone doji", 28),
    "bullish_pin_bar": ("bearish_pin_bar", "Bearish pin bar", 29),
    "bullish_spinning_top": ("bearish_spinning_top", "Bearish spinning top", 30),
    "tweezer_bottoms": ("tweezer_tops", "Tweezer tops", 31), "piercing_line": ("dark_cloud_cover", "Dark cloud cover", 32),
    "bullish_harami": ("bearish_harami", "Bearish harami", 33),
    "bullish_engulfing": ("bearish_engulfing", "Bearish engulfing", 34),
    "bullish_kicker": ("bearish_kicker", "Bearish kicker", 35),
    "bullish_belt_hold": ("bearish_belt_hold", "Bearish belt hold", 36),
    "bullish_counterattack": ("bearish_counterattack", "Bearish counterattack line", 37),
    "homing_pigeon": ("descending_hawk", "Bearish descending hawk", 38),
    "morning_star": ("evening_star", "Evening star", 39), "morning_doji_star": ("evening_doji_star", "Evening doji star", 40),
    "three_white_soldiers": ("three_black_crows", "Three black crows", 41),
    "bullish_abandoned_baby": ("bearish_abandoned_baby", "Bearish abandoned baby", 42),
    "three_inside_up": ("three_inside_down", "Three inside down", 43),
    "three_outside_up": ("three_outside_down", "Three outside down", 44),
    "bullish_stick_sandwich": ("bearish_stick_sandwich", "Bearish stick sandwich", 45),
    "bullish_tri_star": ("bearish_tri_star", "Bearish tri star", 46),
}
CATALOG: dict[str, dict] = {}
for _k, (_t, _n, _s, _no) in BULLISH.items():
    CATALOG[_k] = {"title": _t, "dir": 1, "candles": _n, "strength": _s, "no": _no}
    _bk, _bt, _bno = MIRROR[_k]
    CATALOG[_bk] = {"title": _bt, "dir": -1, "candles": _n, "strength": _s, "no": _bno}
PRIORITY = sorted(CATALOG, key=lambda k: (-CATALOG[k]["candles"], -CATALOG[k]["strength"]))


class _Bars:
    def __init__(self, o, h, l, c, a):
        self.o, self.h, self.l, self.c, self.a = o, h, l, c, a

    def body(self, i):
        return abs(self.c[i] - self.o[i])

    def rng(self, i):
        return max(self.h[i] - self.l[i], 1e-12)

    def upper(self, i):
        return self.h[i] - max(self.o[i], self.c[i])

    def lower(self, i):
        return min(self.o[i], self.c[i]) - self.l[i]

    def green(self, i):
        return self.c[i] > self.o[i]

    def red(self, i):
        return self.c[i] < self.o[i]

    def long(self, i):
        return self.body(i) >= 0.6 * self.a[i] and self.body(i) >= 0.5 * self.rng(i)

    def small(self, i):
        return self.body(i) <= 0.35 * self.rng(i) or self.body(i) <= 0.25 * self.a[i]

    def doji(self, i):
        return self.body(i) <= 0.1 * self.rng(i)

    def btop(self, i):
        return max(self.o[i], self.c[i])

    def bbot(self, i):
        return min(self.o[i], self.c[i])

    def downtrend(self, start, end):
        """A move down into the pattern: falling closes before it and the pattern near the recent low."""
        if start < 11:
            return False
        a = self.a[start]
        moved = self.c[start - 1] - self.c[start - 11] <= -0.8 * a
        low = min(self.l[start:end + 1])
        return moved and low <= min(self.l[start - 10:end + 1]) + 0.3 * a


def _bullish(b: _Bars, i: int) -> list[str]:
    """Bullish patterns whose last candle is bar i."""
    out = []
    a = b.a[i]
    # one candle
    if b.downtrend(i, i):
        r, body, up, lo = b.rng(i), b.body(i), b.upper(i), b.lower(i)
        if b.doji(i):
            if lo >= 0.7 * r and up <= 0.1 * r:
                out.append("dragonfly_doji")
        else:
            if lo >= 2 * body and up <= 0.15 * r and body >= 0.05 * r:
                out.append("hammer")
            if up >= 2 * body and lo <= 0.15 * r and body >= 0.05 * r:
                out.append("inverted_hammer")
            if body <= 0.3 * r and up >= 0.3 * r and lo >= 0.3 * r and r >= 0.8 * a:
                out.append("bullish_spinning_top")
        if lo >= 0.6 * r and body <= 0.33 * r and b.c[i] >= b.l[i] + 0.66 * r and b.l[i] < b.l[i - 1] and r >= 0.8 * a:
            out.append("bullish_pin_bar")
        if b.green(i) and b.long(i) and lo <= 0.05 * r and body >= 0.7 * r:
            out.append("bullish_belt_hold")
    # two candles
    p = i - 1
    if p >= 0 and b.downtrend(p, i):
        tol = 0.1 * a
        if b.red(p) and b.green(i) and abs(b.l[p] - b.l[i]) <= tol:
            out.append("tweezer_bottoms")
        if b.red(p) and b.green(i) and b.o[i] <= b.c[p] and b.c[i] >= b.o[p] and b.body(i) > b.body(p) \
                and b.body(i) >= 0.4 * a:
            out.append("bullish_engulfing")
        if b.red(p) and b.long(p) and b.green(i):
            mid = (b.o[p] + b.c[p]) / 2
            if b.o[i] <= b.c[p] and mid < b.c[i] < b.o[p]:
                out.append("piercing_line")
            if b.body(i) <= 0.5 * b.body(p) and b.btop(i) <= b.o[p] and b.bbot(i) >= b.c[p]:
                out.append("bullish_harami")
            if b.long(i) and b.o[i] < b.c[p] - 0.3 * a and abs(b.c[i] - b.c[p]) <= tol:
                out.append("bullish_counterattack")
        if b.red(p) and b.long(p) and b.green(i) and b.long(i) and b.o[i] >= b.o[p] + 0.05 * a and b.l[i] >= b.o[p] - 0.05 * a:
            out.append("bullish_kicker")
        if b.red(p) and b.long(p) and b.red(i) and b.body(i) < b.body(p) and b.btop(i) <= b.o[p] and b.bbot(i) >= b.c[p]:
            out.append("homing_pigeon")
    # three candles
    s = i - 2
    if s >= 0 and b.downtrend(s, i):
        m = s + 1
        mid_s = (b.o[s] + b.c[s]) / 2
        if b.red(s) and b.long(s) and b.green(i) and b.c[i] > mid_s and b.btop(m) <= b.c[s] + 0.1 * a and b.btop(m) < b.o[i] + 0.3 * a:
            if b.doji(m):
                out.append("morning_doji_star")
            elif b.small(m):
                out.append("morning_star")
            if b.doji(m) and b.h[m] < b.l[s] and b.l[i] > b.h[m]:
                out.append("bullish_abandoned_baby")
        if all(b.green(k) for k in (s, m, i)) and b.c[s] < b.c[m] < b.c[i] \
                and b.bbot(s) <= b.o[m] <= b.btop(s) and b.bbot(m) <= b.o[i] <= b.btop(m) \
                and all(b.body(k) >= 0.5 * a and b.upper(k) <= 0.4 * b.body(k) for k in (s, m, i)):
            out.append("three_white_soldiers")
        if b.red(s) and b.long(s) and b.green(m) and b.body(m) <= 0.5 * b.body(s) and b.btop(m) <= b.o[s] and b.bbot(m) >= b.c[s] \
                and b.green(i) and b.c[i] > b.o[s]:
            out.append("three_inside_up")
        if b.red(s) and b.green(m) and b.o[m] <= b.c[s] and b.c[m] >= b.o[s] and b.body(m) > b.body(s) \
                and b.green(i) and b.c[i] > b.c[m]:
            out.append("three_outside_up")
        if b.red(s) and b.green(m) and b.red(i) and b.c[m] > b.o[s] - 0.1 * a and abs(b.c[i] - b.c[s]) <= 0.05 * a:
            out.append("bullish_stick_sandwich")
        if all(b.doji(k) for k in (s, m, i)) and b.h[m] < min(b.l[s], b.l[i]) + 0.2 * a \
                and (b.o[m] + b.c[m]) / 2 < min((b.o[s] + b.c[s]) / 2, (b.o[i] + b.c[i]) / 2):
            out.append("bullish_tri_star")
    return out


def _bars(df: pd.DataFrame, atr) -> tuple[_Bars, _Bars]:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if np.isscalar(atr):
        a = np.full(len(c), float(atr))
    else:
        a = np.asarray(atr, float)
        a = np.where(np.isfinite(a) & (a > 0), a, np.nanmedian(a[a > 0]) if np.any(a > 0) else 1.0)
    return _Bars(o, h, l, c, a), _Bars(-o, -l, -h, -c, a)


def _at(df, up: _Bars, dn: _Bars, i: int, vol) -> list[dict]:
    found = [(k, 1) for k in _bullish(up, i)] + [(MIRROR[k][0], -1) for k in _bullish(dn, i)]
    out = []
    for name, d in found:
        info = CATALOG[name]
        s = i - info["candles"] + 1
        out.append({"type": name, "title": info["title"], "dir": d, "strength": info["strength"], "no": info["no"],
                    "end": i, "low": float(min(up.l[s:i + 1])), "high": float(max(up.h[s:i + 1])),
                    "volume_up": bool(vol is not None and i >= 20 and vol[i] > 1.2 * np.mean(vol[i - 20:i]) > 0)})
    return out


def detect(df: pd.DataFrame, atr, lookback: int = 2) -> list[dict]:
    """Patterns ending on the last `lookback` candles, strongest first. Each: type, title, dir (+1/-1),
    strength (1-3), bars_ago, low/high of the pattern, confirmed (next candle closed beyond it), volume_up."""
    n = len(df)
    if n < 15:
        return []
    up, dn = _bars(df, atr)
    vol = df["volume"].to_numpy(float) if "volume" in df and df["volume"].sum() > 0 else None
    out = []
    for i in range(max(12, n - lookback), n):
        for p in _at(df, up, dn, i, vol):
            ago = n - 1 - i
            p["bars_ago"] = ago
            last = n - 1
            if ago >= 1:
                ref = up.h[i] if p["dir"] > 0 else up.l[i]
                p["confirmed"] = bool((up.c[last] - ref) * p["dir"] > 0)
            else:
                p["confirmed"] = False
            p["time"] = df.index[i].isoformat()
            del p["end"]
            out.append(p)
    out.sort(key=lambda p: (p["bars_ago"], -p["strength"], PRIORITY.index(p["type"])))
    return out


def scan(df: pd.DataFrame, atr, bars: int = 160) -> list[dict]:
    """Every pattern in the last `bars` candles (for the chart)."""
    n = len(df)
    if n < 15:
        return []
    up, dn = _bars(df, atr)
    out = []
    for i in range(max(12, n - bars), n):
        best = sorted(_at(df, up, dn, i, None), key=lambda p: (-p["strength"], PRIORITY.index(p["type"])))
        if best:
            p = best[0]
            out.append({"t": df.index[i].isoformat(), "type": p["type"], "title": p["title"], "dir": p["dir"],
                        "strength": p["strength"]})
    return out


def best(found: list[dict], side: str | None = None, min_strength: int = 1) -> dict | None:
    d = None if side is None else (1 if side == "buy" else -1)
    for p in found:
        if (d is None or p["dir"] == d) and p["strength"] >= min_strength:
            return p
    return None


def describe(p: dict | None) -> str:
    if not p:
        return "none"
    when = "on the last candle" if p["bars_ago"] == 0 else f"{p['bars_ago']} candle(s) ago"
    return f"{p['title']} {when}" + (", confirmed" if p.get("confirmed") else "") + (", on rising volume" if p.get("volume_up") else "")
