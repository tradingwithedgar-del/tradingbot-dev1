import numpy as np
import pandas as pd
import pytest

from tradingbot.analysis import candles as C
from tradingbot.analysis.patterns import TITLES as CHART_PATTERNS

P = 106.0
CASES = {
    "hammer": [(P, P + 0.05, P - 2.0, P + 0.3)],
    "inverted_hammer": [(P, P + 2.0, P - 0.05, P + 0.3)],
    "dragonfly_doji": [(P, P + 0.02, P - 2.0, P + 0.01)],
    "bullish_pin_bar": [(P - 0.1, P + 0.1, P - 2.2, P + 0.05)],
    "bullish_spinning_top": [(P, P + 0.7, P - 0.6, P + 0.25)],
    "bullish_belt_hold": [(P - 0.5, P + 1.0, P - 0.52, P + 0.95)],
    "tweezer_bottoms": [(P, P + 0.1, P - 1.5, P - 1.2), (P - 1.2, P, P - 1.52, P - 0.1)],
    "piercing_line": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.7, P - 0.3, P - 1.8, P - 0.5)],
    "bullish_engulfing": [(P, P + 0.1, P - 0.9, P - 0.8), (P - 0.9, P + 0.4, P - 1.0, P + 0.3)],
    "bullish_harami": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.0, P - 0.4, P - 1.1, P - 0.6)],
    "bullish_kicker": [(P, P + 0.1, P - 1.6, P - 1.5), (P + 0.2, P + 1.5, P + 0.15, P + 1.4)],
    "bullish_counterattack": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 2.6, P - 1.4, P - 2.7, P - 1.5)],
    "homing_pigeon": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 0.5, P - 0.4, P - 1.1, P - 1.0)],
    "morning_star": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.7, P - 1.4, P - 2.0, P - 1.6), (P - 1.5, P + 0.1, P - 1.6, P)],
    "morning_doji_star": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.7, P - 1.4, P - 2.0, P - 1.71), (P - 1.5, P + 0.1, P - 1.6, P)],
    "three_white_soldiers": [(P, P + 1.1, P - 0.1, P + 1.0), (P + 0.6, P + 1.7, P + 0.5, P + 1.6), (P + 1.2, P + 2.3, P + 1.1, P + 2.2)],
    "bullish_abandoned_baby": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 2.0, P - 1.8, P - 2.2, P - 2.01), (P - 1.6, P, P - 1.7, P - 0.1)],
    "three_inside_up": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.1, P - 0.5, P - 1.2, P - 0.7), (P - 0.6, P + 0.4, P - 0.7, P + 0.3)],
    "three_outside_up": [(P, P + 0.1, P - 0.9, P - 0.8), (P - 0.9, P + 0.4, P - 1.0, P + 0.3), (P + 0.3, P + 1.0, P + 0.2, P + 0.9)],
    "bullish_stick_sandwich": [(P, P + 0.1, P - 1.6, P - 1.5), (P - 1.3, P + 0.2, P - 1.4, P + 0.1), (P + 0.1, P + 0.2, P - 1.6, P - 1.5)],
    "bullish_tri_star": [(P - 0.6, P - 0.3, P - 0.9, P - 0.6), (P - 1.6, P - 1.3, P - 1.9, P - 1.61), (P - 0.7, P - 0.4, P - 1.0, P - 0.69)],
}


def frame(tail):
    rows, p = [], 120.0
    for _ in range(14):                      # the downtrend into the pattern
        rows.append((p, p + 0.2, p - 1.2, p - 1.0))
        p -= 1.0
    rows += tail
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 0.0
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="15min", tz="UTC")
    return df


def mirror(df):
    m = df.copy()
    m["open"], m["high"], m["low"], m["close"] = 300 - df["open"], 300 - df["low"], 300 - df["high"], 300 - df["close"]
    return m


def test_catalog_has_all_50_patterns():
    assert len(C.CATALOG) == 42                           # 21 bullish + 21 bearish candlestick patterns
    numbers = {v["no"] for v in C.CATALOG.values()}
    assert numbers == set(range(1, 22)) | set(range(26, 47))
    for chart in ("double_bottom", "triple_bottom", "inverse_head_shoulders", "falling_wedge",
                  "double_top", "triple_top", "head_shoulders", "rising_wedge"):
        assert chart in CHART_PATTERNS                     # #22-25 and #47-50 are chart patterns


@pytest.mark.parametrize("name", sorted(CASES))
def test_bullish_pattern_and_its_bearish_mirror(name):
    df = frame(CASES[name])
    got = [p for p in C.detect(df, 1.0, lookback=1) if p["type"] == name]
    assert got and got[0]["dir"] == 1, f"{name}: {[p['type'] for p in C.detect(df, 1.0, lookback=1)]}"
    bear = C.MIRROR[name][0]
    mgot = [p for p in C.detect(mirror(df), 1.0, lookback=1) if p["type"] == bear]
    assert mgot and mgot[0]["dir"] == -1 and mgot[0]["title"] == C.MIRROR[name][1]


def test_no_reversal_without_a_move_into_it():
    rows = [(100 + (k % 2) * 0.1, 100.5, 99.5, 100 + ((k + 1) % 2) * 0.1) for k in range(14)]
    rows.append((100, 100.05, 98.0, 100.3))               # hammer shape, but in a flat market
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["volume"] = 0.0
    df.index = pd.date_range("2026-01-01", periods=len(df), freq="15min", tz="UTC")
    assert not [p for p in C.detect(df, 1.0, lookback=1) if p["type"] == "hammer"]


def test_confirmation_candle():
    df = frame(CASES["hammer"] + [(P + 0.3, P + 1.2, P + 0.2, P + 1.1)])
    h = [p for p in C.detect(df, 1.0, lookback=2) if p["type"] == "hammer"]
    assert h and h[0]["bars_ago"] == 1 and h[0]["confirmed"]
    assert "confirmed" in C.describe(h[0])


def test_candle_reversal_strategy_needs_a_key_level():
    from tradingbot.analysis.context import MarketContext
    from tradingbot.analysis.htf import Frame
    from tradingbot.strategies.candles import CandleReversal

    rows = [(130 + np.sin(k / 3), 131 + np.sin(k / 3), 129 + np.sin(k / 3), 130 + np.sin(k / 3)) for k in range(60)]
    df = frame(CASES["bullish_engulfing"])
    pre = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    pre["volume"] = 0.0
    pre.index = pd.date_range("2025-12-31", periods=60, freq="15min", tz="UTC")
    df = pd.concat([pre, df])
    no_level = CandleReversal().generate(MarketContext("US30", df, htf={}))
    assert no_level is None or "15m demand" in no_level.reason
    day = Frame("1D", zones=[{"kind": "demand", "top": P - 0.5, "bottom": P - 3, "fresh": True}])
    sig = CandleReversal().generate(MarketContext("US30", df, htf={"1D": day}))
    assert sig is not None and sig.side == "buy" and "Bullish engulfing" in sig.reason and "D demand" in sig.reason
    assert sig.stop < P - 1.0
