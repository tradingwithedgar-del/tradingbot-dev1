"""Supply and demand zones.

A zone is a small "base" (1-3 tight candles) followed by an impulsive move away.
- Demand: base then a strong bullish leave (drop-base-rally / rally-base-rally).
- Supply: base then a strong bearish leave (rally-base-drop / drop-base-drop).
A zone is invalidated once price closes through it, and loses freshness with every touch.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import atr as atr_fn


@dataclass
class Zone:
    kind: str          # "demand" or "supply"
    top: float
    bottom: float
    created_idx: int   # index of the impulse candle (zone known after this bar closes)
    base_start: int
    impulse_atr: float  # strength of the leave, in ATRs
    touches: int = 0
    broken: bool = False

    @property
    def fresh(self) -> bool:
        return self.touches == 0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_zones(
    df: pd.DataFrame,
    atr_series: pd.Series | None = None,
    base_max_body_atr: float = 0.6,
    impulse_min_atr: float = 1.5,
    max_base_len: int = 3,
) -> list[Zone]:
    if atr_series is None:
        atr_series = atr_fn(df)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    a = atr_series.to_numpy()
    n = len(df)
    zones: list[Zone] = []

    i = 15
    while i < n:
        body = abs(c[i] - o[i])
        if a[i - 1] <= 0 or body < impulse_min_atr * a[i - 1]:
            i += 1
            continue
        # i is an impulse candle; walk back over the base candles
        j = i - 1
        while j >= 0 and i - j <= max_base_len and abs(c[j] - o[j]) <= base_max_body_atr * a[j]:
            j -= 1
        base_start = j + 1
        if base_start == i:
            i += 1
            continue
        base_hi = h[base_start:i].max()
        base_lo = lo[base_start:i].min()
        strength = body / a[i - 1]
        if c[i] > o[i]:
            zones.append(Zone("demand", float(base_hi), float(base_lo), i, base_start, float(strength)))
        else:
            zones.append(Zone("supply", float(base_hi), float(base_lo), i, base_start, float(strength)))
        i += 1

    # Replay price after each zone was created: count touches, mark broken zones.
    # A touch is a fresh visit: a bar inside the zone whose previous bar was outside it.
    # The current (last) bar is not counted so a first retest still reads as fresh.
    for z in zones:
        inside_prev = True  # the impulse bar itself starts at the zone
        for k in range(z.created_idx + 1, n):
            if z.kind == "demand":
                if c[k] < z.bottom:
                    z.broken = True
                    break
                inside = lo[k] <= z.top
            else:
                if c[k] > z.top:
                    z.broken = True
                    break
                inside = h[k] >= z.bottom
            if inside and not inside_prev and k < n - 1:
                z.touches += 1
            inside_prev = inside
    return [z for z in zones if not z.broken]


def zone_at_price(zones: list[Zone], kind: str, low: float, high: float) -> Zone | None:
    """Most recent unbroken zone of `kind` that the current bar's range overlaps."""
    for z in reversed(zones):
        if z.kind == kind and low <= z.top and high >= z.bottom:
            return z
    return None
