"""Market structure: swing points, HH/HL/LH/LL labelling and trend state.

A swing high at bar i is only *confirmed* `right` bars later, so a swing is never
used before it could have been known in real time (no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Swing:
    idx: int          # bar index of the swing extreme
    confirmed_at: int  # first bar index at which the swing is known
    price: float
    kind: str         # "H" or "L"
    label: str = ""   # HH / LH for highs, HL / LL for lows (empty for the first of each kind)


def find_swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Swing]:
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)
    swings: list[Swing] = []
    for i in range(left, n - right):
        hwin = highs[i - left: i + right + 1]
        lwin = lows[i - left: i + right + 1]
        if highs[i] == hwin.max() and np.argmax(hwin) == left:
            swings.append(Swing(i, i + right, float(highs[i]), "H"))
        if lows[i] == lwin.min() and np.argmin(lwin) == left:
            swings.append(Swing(i, i + right, float(lows[i]), "L"))
    swings.sort(key=lambda s: (s.idx, s.kind))
    return label_swings(swings)


def label_swings(swings: list[Swing]) -> list[Swing]:
    last_h: Swing | None = None
    last_l: Swing | None = None
    for s in swings:
        if s.kind == "H":
            if last_h is not None:
                s.label = "HH" if s.price > last_h.price else "LH"
            last_h = s
        else:
            if last_l is not None:
                s.label = "HL" if s.price > last_l.price else "LL"
            last_l = s
    return swings


def last_of(swings: list[Swing], kind: str) -> Swing | None:
    for s in reversed(swings):
        if s.kind == kind:
            return s
    return None


def trend_state(swings: list[Swing], last_close: float | None = None) -> str:
    """'up' = HH + HL, 'down' = LH + LL, otherwise 'range'.

    A close beyond the most recent opposing swing (break of structure) overrides
    the label, because structure has already shifted even if the next swing isn't
    confirmed yet.
    """
    h = last_of(swings, "H")
    lo = last_of(swings, "L")
    if h is None or lo is None:
        return "range"
    state = "range"
    if h.label == "HH" and lo.label == "HL":
        state = "up"
    elif h.label == "LH" and lo.label == "LL":
        state = "down"
    if last_close is not None:
        if state == "up" and last_close < lo.price:
            return "down_break"
        if state == "down" and last_close > h.price:
            return "up_break"
    return state


def structure_sequence(swings: list[Swing], n: int = 6) -> str:
    return " ".join(s.label for s in swings[-n:] if s.label)
