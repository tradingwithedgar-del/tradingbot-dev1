"""Classic chart patterns, found from swing points on the Daily, 4H and 1H charts.

Every pattern has a breakout line (neckline, triangle side, rim, trendline), a direction, a measured-move
target (the pattern's height projected from the breakout) and an invalidation level (where the idea is
wrong). TIIM reports each one either as `forming` (breakout line known, not broken yet - used by the
weekly plan) or as a break event (first close through the line - traded by the chart_pattern strategy on
a 15m pullback, like break_retest).

Patterns: double/triple top and bottom, head & shoulders (and inverse), ascending/descending/symmetrical
triangle, rising/falling wedge, price channel, flag, pennant, rounding bottom/top, cup & handle and
bump-and-run. Only swing points confirmed by the current candle are used (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TITLES = {
    "double_top": "Double top", "double_bottom": "Double bottom",
    "triple_top": "Triple top", "triple_bottom": "Triple bottom",
    "head_shoulders": "Head & shoulders", "inverse_head_shoulders": "Inverse head & shoulders",
    "ascending_triangle": "Ascending triangle", "descending_triangle": "Descending triangle",
    "symmetrical_triangle": "Symmetrical triangle", "rising_wedge": "Rising wedge", "falling_wedge": "Falling wedge",
    "channel": "Price channel", "flag": "Flag", "pennant": "Pennant",
    "rounding_bottom": "Rounding bottom", "rounding_top": "Rounding top", "cup_handle": "Cup & handle",
    "bump_and_run": "Bump & run", "bump_and_run_bottom": "Bump & run (bottom)",
}
PATTERN_TYPES = set(TITLES)
# When several patterns break the same way on the same candle, the more specific one is kept.
PRIORITY = ["head_shoulders", "inverse_head_shoulders", "flag", "pennant", "triple_top", "triple_bottom",
            "double_top", "double_bottom", "cup_handle", "falling_wedge", "rising_wedge", "ascending_triangle",
            "descending_triangle", "symmetrical_triangle", "bump_and_run", "bump_and_run_bottom",
            "rounding_bottom", "rounding_top", "channel"]


def _alternate(swings) -> list:
    """Swing points alternating high/low (consecutive highs keep the highest, lows the lowest)."""
    out = []
    for s in swings:
        if out and out[-1].kind == s.kind:
            if (s.kind == "H" and s.price >= out[-1].price) or (s.kind == "L" and s.price <= out[-1].price):
                out[-1] = s
            continue
        out.append(s)
    return out


def _flat(v):
    return lambda k: v


def _line(i1, p1, i2, p2):
    slope = (p2 - p1) / ((i2 - i1) or 1)
    return lambda k: p1 + slope * (k - i1)


def _cand(name, d, line, height, invalid, pivots, start):
    return {"type": name, "dir": d, "line": line, "height": abs(height), "invalid": invalid, "pivots": pivots,
            "start": start}


def _tops_bottoms(a, tol, atr) -> list[dict]:
    out = []
    n = len(a)
    for end in range(max(0, n - 2), n):           # pattern must end on one of the last two pivots
        s = a[end]
        top = s.kind == "H"
        d = -1 if top else 1
        # triple: X1 Y1 X2 Y2 X3
        if end >= 4:
            x1, y1, x2, y2, x3 = a[end - 4:end + 1]
            xs = [x1.price, x2.price, x3.price]
            if max(xs) - min(xs) <= tol:
                neck = min(y1.price, y2.price) if top else max(y1.price, y2.price)
                h = (max(xs) - neck) if top else (neck - min(xs))
                if h >= 2 * atr:
                    out.append(_cand("triple_top" if top else "triple_bottom", d, _flat(neck), h,
                                     max(xs) if top else min(xs), [x1, y1, x2, y2, x3], x3))
                    continue
            # head & shoulders
            head = x2.price
            sh = (x1.price, x3.price)
            bigger = (head > max(sh) + tol) if top else (head < min(sh) - tol)
            neck_line = _line(y1.idx, y1.price, y2.idx, y2.price)
            hgt = abs(head - neck_line(x2.idx))
            if bigger and abs(sh[0] - sh[1]) <= max(tol, 0.35 * hgt) and hgt >= 2 * atr:
                out.append(_cand("head_shoulders" if top else "inverse_head_shoulders", d, neck_line, hgt, x3.price,
                                 [x1, y1, x2, y2, x3], x3))
                continue
        if end >= 2:
            x1, y1, x2 = a[end - 2:end + 1]
            if abs(x1.price - x2.price) <= tol:
                neck = y1.price
                h = (max(x1.price, x2.price) - neck) if top else (neck - min(x1.price, x2.price))
                if h >= 2 * atr:
                    out.append(_cand("double_top" if top else "double_bottom", d, _flat(neck), h,
                                     max(x1.price, x2.price) if top else min(x1.price, x2.price), [x1, y1, x2], x2))
    return out


def _fit(points):
    x = np.array([p.idx for p in points], float)
    y = np.array([p.price for p in points], float)
    slope, icept = np.polyfit(x, y, 1)
    resid = float(np.abs(y - (slope * x + icept)).max())
    return slope, icept, resid


def _lines(a, tol, atr, c) -> list[dict]:
    """Triangles, wedges, channels, flags and pennants from the last 4-6 alternating pivots."""
    out = []
    for size in (6, 5, 4):
        if len(a) < size:
            continue
        pts = a[-size:]
        hs = [p for p in pts if p.kind == "H"]
        ls = [p for p in pts if p.kind == "L"]
        if len(hs) < 2 or len(ls) < 2:
            continue
        sh, ih, rh = _fit(hs)
        sl, il, rl = _fit(ls)
        if rh > tol or rl > tol:
            continue
        span = pts[-1].idx - pts[0].idx
        if span < 8:
            continue
        up_line = lambda k, s=sh, i=ih: s * k + i
        lo_line = lambda k, s=sl, i=il: s * k + i
        width0 = up_line(pts[0].idx) - lo_line(pts[0].idx)
        width1 = up_line(pts[-1].idx) - lo_line(pts[-1].idx)
        if width1 <= 0 or width0 <= 0:
            continue
        rise_h, rise_l = sh * span, sl * span
        flat_h, flat_l = abs(rise_h) <= tol, abs(rise_l) <= tol
        converging = width1 < 0.75 * width0
        parallel = abs(width1 - width0) <= max(tol, 0.25 * width0)
        height = width0
        # a pole right before the pattern makes it a flag / pennant
        first = pts[0].idx
        p0 = max(0, first - 15)
        pole = c[first] - c[p0] if first > 0 else 0.0
        pole_dir = 1 if pole > 0 else -1
        has_pole = abs(pole) >= 4 * atr and width0 <= 0.6 * abs(pole) and span <= 30
        name = None
        dirs: tuple = ()
        if has_pole and parallel and (rise_h * pole_dir < 0):
            name, dirs, height = "flag", (pole_dir,), abs(pole)
        elif has_pole and converging and rise_h < 0 < rise_l:
            name, dirs, height = "pennant", (pole_dir,), abs(pole)
        elif flat_h and rise_l > tol and converging:
            name, dirs = "ascending_triangle", (1,)
        elif flat_l and rise_h < -tol and converging:
            name, dirs = "descending_triangle", (-1,)
        elif rise_h < -tol and rise_l > tol and converging:
            name, dirs = "symmetrical_triangle", (1, -1)
        elif rise_h > tol and rise_l > tol and converging:
            name, dirs = "rising_wedge", (-1,)
        elif rise_h < -tol and rise_l < -tol and converging:
            name, dirs = "falling_wedge", (1,)
        elif parallel and (abs(rise_h) > tol or abs(rise_l) > tol):
            name, dirs = "channel", (1, -1)
        if not name:
            continue
        for d in dirs:
            line = up_line if d > 0 else lo_line
            invalid = lo_line(pts[-1].idx) if d > 0 else up_line(pts[-1].idx)
            out.append(_cand(name, d, line, height, invalid, pts, pts[-1]))
        break
    return out


def _rounding(df, atr) -> list[dict]:
    """Rounding bottom / cup & handle (and rounding top): a U-shaped base between two rims at about the same
    level. The breakout line is the rim; a small dip under the right rim before the break is the handle."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(c)
    for d in (1, -1):          # 1 = rounding bottom / cup, -1 = rounding top
        for w in range(min(120, n - 5), 29, -5):
            s = n - w
            seg = c[s:]
            x = np.arange(w, dtype=float)
            q = max(3, int(w * 0.15))
            left = (h if d > 0 else lo)[s:s + q]
            rim = float(left.max() if d > 0 else left.min())
            mid = slice(s + int(0.25 * w), s + int(0.75 * w))
            extreme = float(lo[mid].min() if d > 0 else h[mid].max())
            depth = (rim - extreme) * d
            if depth < 3 * atr:
                continue
            # right side climbs back to the rim
            right = np.arange(s + int(0.6 * w), n)
            near = right[((h[right] >= rim - atr) if d > 0 else (lo[right] <= rim + atr))]
            if not len(near):
                continue
            core = slice(0, near[0] - s + 1)
            a2, b2, c2 = np.polyfit(x[core], seg[core], 2)
            fit = a2 * x[core] ** 2 + b2 * x[core] + c2
            ss = float(((seg[core] - seg[core].mean()) ** 2).sum()) or 1e-12
            r2 = 1 - float(((seg[core] - fit) ** 2).sum()) / ss
            vertex = -b2 / (2 * a2) if a2 else -1
            if a2 * d <= 0 or r2 < 0.8 or not (0.3 * w <= vertex <= 0.7 * w) or core.stop < 0.6 * w:
                continue
            name = "rounding_bottom" if d > 0 else "rounding_top"
            if d > 0:
                after = np.arange(near[0], n)
                broke = after[c[after] > rim + 0.1 * atr]
                stop_at = broke[0] if len(broke) else n
                dip = rim - float(lo[near[0]:stop_at].min()) if stop_at > near[0] else 0.0
                if 0.15 * depth <= dip <= 0.5 * depth:
                    name = "cup_handle"
            vi = s + int(max(0, min(w - 1, vertex)))
            start = type("P", (), {"idx": int(near[0]), "confirmed_at": int(near[0]), "price": extreme,
                                   "kind": "L" if d > 0 else "H"})()
            first = type("P", (), {"idx": s, "price": rim})()
            bottom = type("P", (), {"idx": vi, "price": extreme})()
            out = [_cand(name, d, _flat(rim), depth, extreme, [first, bottom, start], start)]
            out[0]["pivots"][-1].price = rim
            return out
    return []


def _bump(df, atr) -> list[dict]:
    c = df["close"].to_numpy()
    n = len(c)
    if n < 80:
        return []
    out = []
    lead = slice(n - 80, n - 32)
    x = np.arange(n - 80, n - 32, dtype=float)
    for d in (-1, 1):                     # -1: bump & run top (breaks down), 1: bottom version
        src = df["low"].to_numpy() if d < 0 else df["high"].to_numpy()
        slope, icept = np.polyfit(x, src[lead], 1)
        if slope * -d <= 0:              # top needs a rising lead-in, bottom a falling one
            continue
        line = lambda k, s=slope, i=icept: s * k + i
        recent = df["high"].to_numpy()[n - 32:] if d < 0 else df["low"].to_numpy()[n - 32:]
        ext_i = int(recent.argmax() if d < 0 else recent.argmin()) + n - 32
        dist = (c[ext_i] - line(ext_i)) * -d
        bump_slope = (c[ext_i] - c[n - 32]) / max(1, ext_i - (n - 32))
        if dist < 3 * atr or abs(bump_slope) < 2 * abs(slope):
            continue
        start = type("P", (), {"idx": ext_i, "confirmed_at": ext_i + 2, "price": float(c[ext_i]),
                               "kind": "H" if d < 0 else "L"})()
        out.append(_cand("bump_and_run" if d < 0 else "bump_and_run_bottom", d, line, dist, float(c[ext_i]),
                         [start], start))
    return out


def detect(df: pd.DataFrame, swings, atr: float, lookback: int = 30) -> tuple[list[dict], list[dict]]:
    """(breaks, forming): pattern breakouts in the last `lookback` candles, and patterns still waiting."""
    if len(df) < 40 or not atr or atr <= 0:
        return [], []
    c = df["close"].to_numpy()
    n = len(c)
    a = _alternate([s for s in swings if s.confirmed_at <= n - 1])
    tol = 0.5 * atr
    cands = _tops_bottoms(a, 0.8 * atr, atr) + _lines(a, tol, atr, c) + _rounding(df, atr) + _bump(df, atr)
    breaks, forming, seen = [], [], set()
    for cd in cands:
        key = (cd["type"], cd["dir"])
        if key in seen:
            continue
        seen.add(key)
        d, line = cd["dir"], cd["line"]
        begin = cd["start"].confirmed_at + 1
        hit = None
        for k in range(max(begin, 1), n):
            beyond = (c[k] - line(k)) * d > 0.1 * atr
            if beyond:
                hit = k
                break
            if (c[k] - cd["invalid"]) * d < 0 and cd["type"] not in ("channel", "symmetrical_triangle"):
                break      # price went the other way through the invalidation level: pattern failed
        pts = [[df.index[p.idx].isoformat(), float(p.price)] for p in cd["pivots"] if 0 <= p.idx < n]
        base = {"type": cd["type"], "title": TITLES[cd["type"]], "dir": d, "height": round(cd["height"], 6),
                "invalid": float(cd["invalid"]), "points": pts}
        if hit is not None:
            ago = n - 1 - hit
            if ago <= lookback:
                lvl = float(line(hit))
                breaks.append(dict(base, level=lvl, target=lvl + d * cd["height"], bars_ago=ago,
                                   time=df.index[hit].isoformat()))
        elif n - 1 - cd["start"].idx <= 3 * lookback:
            lvl = float(line(n - 1))
            forming.append(dict(base, trigger=lvl, target=lvl + d * cd["height"]))
    best: dict = {}
    for b in breaks:
        key = (b["bars_ago"], b["dir"])
        if key not in best or PRIORITY.index(b["type"]) < PRIORITY.index(best[key]["type"]):
            best[key] = b
    breaks = sorted(best.values(), key=lambda b: -b["bars_ago"])
    forming.sort(key=lambda f: PRIORITY.index(f["type"]))
    return breaks, forming
