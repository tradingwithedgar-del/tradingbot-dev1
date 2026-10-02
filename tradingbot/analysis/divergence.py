"""RSI / MACD divergences measured between confirmed price swings.

Regular bullish : price LL, oscillator HL  -> reversal up
Regular bearish : price HH, oscillator LH  -> reversal down
Hidden bullish  : price HL, oscillator LL  -> continuation up
Hidden bearish  : price LH, oscillator HH  -> continuation down
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .structure import Swing


@dataclass
class Divergence:
    kind: str        # regular_bull / regular_bear / hidden_bull / hidden_bear
    oscillator: str
    first: Swing
    second: Swing
    osc_first: float
    osc_second: float

    @property
    def side(self) -> str:
        return "buy" if self.kind.endswith("bull") else "sell"


def find_divergences(
    swings: list[Swing],
    osc: pd.Series,
    name: str = "rsi",
    max_gap: int = 60,
    min_gap: int = 4,
) -> list[Divergence]:
    vals = osc.to_numpy()
    out: list[Divergence] = []
    for kind in ("H", "L"):
        pts = [s for s in swings if s.kind == kind]
        for a, b in zip(pts, pts[1:]):
            gap = b.idx - a.idx
            if gap < min_gap or gap > max_gap:
                continue
            oa, ob = float(vals[a.idx]), float(vals[b.idx])
            if kind == "L":
                if b.price < a.price and ob > oa:
                    out.append(Divergence("regular_bull", name, a, b, oa, ob))
                elif b.price > a.price and ob < oa:
                    out.append(Divergence("hidden_bull", name, a, b, oa, ob))
            else:
                if b.price > a.price and ob < oa:
                    out.append(Divergence("regular_bear", name, a, b, oa, ob))
                elif b.price < a.price and ob > oa:
                    out.append(Divergence("hidden_bear", name, a, b, oa, ob))
    out.sort(key=lambda d: d.second.confirmed_at)
    return out


def recent_divergence(divs: list[Divergence], current_idx: int, within: int = 5) -> Divergence | None:
    """Latest divergence whose second swing was confirmed within `within` bars."""
    for d in reversed(divs):
        if 0 <= current_idx - d.second.confirmed_at <= within:
            return d
    return None
