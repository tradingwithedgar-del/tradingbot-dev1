import numpy as np
import pandas as pd
import pytest

from tradingbot.analysis import patterns
from tradingbot.analysis.indicators import atr
from tradingbot.analysis.structure import find_swings


def make(pts, noise=0.15, seed=1, freq="4h", start="2026-09-01"):
    x = np.arange(pts[-1][0] + 1)
    y = np.interp(x, [p[0] for p in pts], [p[1] for p in pts]) + np.random.default_rng(seed).normal(0, noise, len(x))
    df = pd.DataFrame({"open": np.r_[y[0], y[:-1]], "close": y},
                      index=pd.date_range(start, periods=len(y), freq=freq, tz="UTC"))
    df["high"] = df[["open", "close"]].max(axis=1) + 0.2
    df["low"] = df[["open", "close"]].min(axis=1) - 0.2
    df["volume"] = 1.0
    return df


CASES = {
    "double_top": [(0, 90), (20, 100), (30, 110), (40, 102), (50, 110.2), (60, 101)],
    "double_bottom": [(0, 120), (20, 110), (30, 100), (40, 108), (50, 99.8), (60, 109)],
    "triple_top": [(0, 90), (20, 100), (30, 110), (38, 103), (46, 110.3), (54, 103.2), (62, 109.8), (72, 101)],
    "head_shoulders": [(0, 90), (20, 100), (30, 106), (38, 100), (48, 114), (58, 100.3), (66, 106.2), (76, 98)],
    "inverse_head_shoulders": [(0, 120), (20, 110), (30, 104), (38, 110), (48, 96), (58, 110.2), (66, 104.3), (76, 112)],
    "ascending_triangle": [(0, 90), (15, 100), (25, 110), (33, 101), (41, 110.1), (49, 104), (57, 110), (63, 106.5), (70, 113)],
    "descending_triangle": [(0, 120), (15, 110), (25, 100), (33, 109), (41, 99.9), (49, 106), (57, 100.1), (63, 103.5), (70, 97)],
    "symmetrical_triangle": [(0, 120), (15, 100), (25, 112), (33, 101.5), (41, 110), (49, 103), (57, 108), (63, 104.5), (70, 99)],
    "rising_wedge": [(0, 90), (10, 100), (20, 108), (28, 104), (36, 111), (44, 108), (52, 113.3), (60, 111.5), (68, 106)],
    "falling_wedge": [(0, 120), (10, 110), (20, 102), (28, 106), (36, 99), (44, 102), (52, 96.7), (60, 98.5), (68, 104)],
    "flag": [(0, 90), (8, 91), (18, 112), (24, 108), (30, 111), (36, 106.5), (42, 109.5), (48, 105.5), (54, 112), (60, 115)],
    "cup_handle": [(0, 100), (5, 110), (20, 100), (35, 96), (50, 101), (62, 110), (68, 106.5), (74, 113)],
}
BEARISH = {"double_top", "triple_top", "head_shoulders", "descending_triangle", "rising_wedge"}


@pytest.mark.parametrize("name", sorted(CASES))
def test_each_pattern_is_recognised_with_direction_target_and_stop(name):
    df = make(CASES[name])
    a = float(atr(df, 14).iloc[-1])
    breaks, _ = patterns.detect(df, find_swings(df, 3, 3), a)
    hit = [b for b in breaks if b["type"] == name]
    assert hit, f"{name} not found: {[b['type'] for b in breaks]}"
    b = hit[-1]
    d = -1 if name in BEARISH or (name == "symmetrical_triangle") else 1
    assert b["dir"] == d
    assert (b["target"] - b["level"]) * d > 0 and b["height"] > 0     # measured move in the breakout direction
    assert b["points"] and b["title"] == patterns.TITLES[name]


def test_forming_pattern_reports_its_trigger_before_the_break():
    df = make(CASES["double_top"][:-1] + [(56, 104)])                 # second top done, neckline not broken yet
    a = float(atr(df, 14).iloc[-1])
    breaks, forming = patterns.detect(df, find_swings(df, 3, 3), a)
    dt = [f for f in forming if f["type"] == "double_top"]
    assert dt and dt[0]["trigger"] < 104 and dt[0]["target"] < dt[0]["trigger"]
    assert not [b for b in breaks if b["type"] == "double_top"]


def test_chart_pattern_strategy_sells_the_retest_with_the_measured_target():
    from tradingbot.analysis import htf
    from tradingbot.analysis.context import MarketContext
    from tradingbot.strategies.breakout import ChartPattern

    frame = htf.analyze_frame("4H", make(CASES["double_top"]))
    brk = [b for b in frame.breaks if b["type"] == "double_top"][-1]
    neck = brk["level"]
    frame.breaks = [dict(brk, bars_ago=1)]
    closes = list(np.linspace(neck + 1.5, neck - 0.6, 250)) + [neck - 1.0, neck - 1.2, neck - 0.9, neck - 0.6,
                                                               neck - 0.4, neck - 0.3, neck - 0.35, neck - 0.8]
    t = pd.date_range("2026-10-10", periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame({"close": closes}, index=t)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0
    sig = ChartPattern().generate(MarketContext("US30", df, htf={"4H": frame}))
    assert sig is not None and sig.side == "sell" and sig.features["pattern"] == "double_top"
    assert 1.5 <= sig.reward_multiple <= 4.0
    assert "double top" in sig.reason


def test_weekly_plan_direction_invalidation_and_gate():
    from tradingbot import plan
    from tradingbot.analysis import htf

    up = make([(i * 10, 100 + i * 4 + (6 if i % 2 else 0)) for i in range(90)], freq="1D", start="2024-01-01")
    h4 = make([(i * 8, 100 + i * 2 + (4 if i % 2 else 0)) for i in range(50)], freq="4h")
    frames = htf.build(up, h4)
    price = float(up["close"].iloc[-1])
    p = plan.market_plan("US500", frames, price, pd.Timestamp("2026-10-18 18:30", tz="UTC"))
    assert p["direction"] == "buy" and p["status"] == "active" and p["invalidation"] < price
    assert plan.against(p, "sell", {}) and "plan" in plan.against(p, "sell", {})
    assert plan.against(p, "buy", {}) is None
    assert plan.against(p, "sell", {"htf_reversal_spot": True}) is None          # reaction at a HTF zone
    assert plan.against(p, "sell", {"pattern": "head_shoulders", "break_tf": "1D"}) is None
    assert plan.invalidated(p, p["invalidation"] - 1) and not plan.invalidated(p, price)
    # Sunday 18:00 UTC starts planning the coming week
    assert plan.week_key(pd.Timestamp("2026-10-18 17:59", tz="UTC")) != plan.week_key(pd.Timestamp("2026-10-18 18:00", tz="UTC"))
    assert plan.week_key(pd.Timestamp("2026-10-18 18:00", tz="UTC")) == plan.week_key(pd.Timestamp("2026-10-21 12:00", tz="UTC"))
    assert "US500" in plan.text({"week": "2026-W43", "created": "2026-10-18T18:00", "markets": {"US500": p}})
