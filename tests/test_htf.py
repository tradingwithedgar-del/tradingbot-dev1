
import numpy as np
import pandas as pd

from tradingbot.analysis import htf
from tradingbot.backtest import run_backtest, synthetic
from tradingbot.config import Settings


def zigzag(n, slope, freq, start="2020-01-01"):
    i = np.arange(n)
    c = 100 + slope * i + 6 * np.sin(i / 3.0)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}, index=idx)


def test_top_down_read_and_alignment():
    frames = htf.build(zigzag(900, 0.5, "1D"), zigzag(300, 0.2, "4h"))
    assert {"1M", "1W", "1D", "4H"} <= set(frames)
    assert frames["1D"].direction == 1 and frames["4H"].direction == 1
    b, score = htf.bias(frames)
    assert b == "bull" and score >= 3
    px = float(zigzag(900, 0.5, "1D")["close"].iloc[-1])
    sell = htf.features(frames, "sell", px - 1, px + 1, px, px + 2)
    buy = htf.features(frames, "buy", px - 1, px + 1, px, px - 2)
    assert sell["htf_align"] == "against" and buy["htf_align"] == "with"
    assert "Daily up" in htf.describe(frames)


def test_sell_against_uptrend_is_blocked_unless_at_htf_supply():
    from tradingbot.agent import Agent
    from tradingbot.broker.sim import SimBroker
    from tradingbot.journal import Journal
    from tradingbot.strategies import Signal

    s = Settings()
    s.mode = "backtest"
    data = {"BTCUSD": synthetic(600, 100, seed=2)}
    import tempfile
    from pathlib import Path
    j = Journal(Path(tempfile.mkdtemp()) / "b.db", mode="backtest")
    agent = Agent(s, SimBroker(data, warmup=300), j)
    sig = Signal("BTCUSD", "structure_trend", "sell", 100.0, 100.5, 98.5, features={
        "htf_align": "against", "htf_reversal_spot": False, "htf_M": "up", "htf_W": "up", "htf_D": "up", "htf_4H": "up",
        "htf_bias": "bull"})
    q = agent.broker.get_quote("BTCUSD")
    ctx = type("C", (), {"close": 100.0, "atr_now": 1.0})()
    why = agent._entry_blocker(sig, ctx, q, [], set())
    assert why and "against the higher timeframes" in why and "D up" in why
    sig.features["htf_reversal_spot"] = True          # sell at a Daily/Weekly supply zone: allowed through this rule
    why = agent._entry_blocker(sig, ctx, q, [], set())
    assert not why or "against the higher timeframes" not in why
    s.htf_filter = False
    sig.features["htf_reversal_spot"] = False
    why = agent._entry_blocker(sig, ctx, q, [], set())
    assert not why or "against the higher timeframes" not in why


def test_backtest_records_top_down_features(tmp_path):
    j = run_backtest({"A": synthetic(2000, 1.10, seed=5)}, Settings(), db_path=tmp_path / "h.db", seed=1)
    trades = j.trades("mode='backtest'")
    assert trades
    late = [t for t in trades if t["opened_at"] > "2025-01-18"]
    assert late and all(t["features"].get("htf_align") in ("with", "against", "mixed", "unknown") for t in late)
    assert any(t["features"].get("htf_D") for t in late)
    # no real trade went against the top-down bias without being at a higher-timeframe zone
    for t in trades:
        f = t["features"]
        if not t["shadow"]:
            assert not (f.get("htf_align") == "against" and not f.get("htf_reversal_spot"))


def test_exhausted_move_is_not_chased():
    from tradingbot.agent import Agent
    from tradingbot.broker.sim import SimBroker
    from tradingbot.journal import Journal
    from tradingbot.strategies import Signal
    import tempfile
    from pathlib import Path

    # 4H: steady chop, then 7 red candles falling hard (like USTECH -3.4% with 7 red 4H candles)
    base = zigzag(120, 0.0, "4h")
    c = list(base["close"].to_numpy()[:-7]) + [100 - 3 * k for k in range(1, 8)]
    h4 = pd.DataFrame({"close": c}, index=base.index)
    h4["open"] = h4["close"].shift(1).fillna(h4["close"].iloc[0]) + 0.0
    h4["high"], h4["low"], h4["volume"] = h4[["open", "close"]].max(axis=1) + 0.5, h4[["open", "close"]].min(axis=1) - 0.5, 1.0
    frames = htf.build(zigzag(900, 0.0, "1D"), h4)
    price = float(h4["close"].iloc[-1])
    sell = htf.features(frames, "sell", price - 1, price + 1, price, price + 2)
    buy = htf.features(frames, "buy", price - 1, price + 1, price, price - 2)
    assert sell["ext_streak_4h"] == 7 and sell["extended"] == "exhausted"
    assert buy["extended"] in ("no", "stretched") and buy["ext_streak_4h"] == 0
    assert "7 red 4H candles in a row" in htf.extension_text(sell, "sell")

    s = Settings()
    s.mode = "backtest"
    j = Journal(Path(tempfile.mkdtemp()) / "x.db", mode="backtest")
    agent = Agent(s, SimBroker({"BTCUSD": synthetic(600, 100, seed=2)}, warmup=300), j)
    sig = Signal("BTCUSD", "structure_trend", "sell", 100.0, 100.5, 98.5, features=sell)
    ctx = type("C", (), {"close": 100.0, "atr_now": 1.0})()
    why = agent._entry_blocker(sig, ctx, agent.broker.get_quote("BTCUSD"), [], set())
    assert why and "chasing an exhausted move" in why


def test_break_retest_sells_the_first_pullback_after_a_support_break():
    from tradingbot.analysis.context import MarketContext
    from tradingbot.strategies.breakout import BreakRetest

    # 1H: support at ~103.7 rejected twice, a lower high, then a close below it (like the USTECH chart)
    pts = [(0, 100), (10, 110), (20, 104), (30, 112), (40, 104.2), (50, 109), (60, 103.5), (66, 101)]
    x = np.arange(67)
    y = np.interp(x, [p[0] for p in pts], [p[1] for p in pts])
    idx = pd.date_range("2026-10-01", periods=67, freq="1h", tz="UTC")
    h1 = pd.DataFrame({"open": np.r_[y[0], y[:-1]], "close": y}, index=idx)
    h1["high"], h1["low"], h1["volume"] = h1[["open", "close"]].max(axis=1) + 0.3, h1[["open", "close"]].min(axis=1) - 0.3, 1.0
    frame = htf.analyze_frame("1H", h1)
    kinds = {(b["type"], b["dir"]) for b in frame.breaks}
    assert ("bos", -1) in kinds and ("level", -1) in kinds

    # 15m: drifted below the level, small pullback up toward it, then a red candle breaking the previous low
    closes = list(np.linspace(104.5, 103.2, 250)) + [102.8, 102.6, 102.7, 102.95, 103.15, 103.3, 103.25, 102.9]
    t = pd.date_range("2026-10-03T12:00", periods=len(closes), freq="15min", tz="UTC")
    df = pd.DataFrame({"close": closes}, index=t)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) + 0.05
    df["low"] = df[["open", "close"]].min(axis=1) - 0.05
    df["volume"] = 1.0
    brk = [b for b in frame.breaks if b["dir"] == -1 and b["type"] in ("bos", "level")][-1]
    frame.breaks = [dict(brk, bars_ago=2)]                     # the break happened 2 hours ago
    ctx = MarketContext("US30", df, htf={"1H": frame})
    sig = BreakRetest().generate(ctx)
    assert sig is not None and sig.side == "sell"
    assert sig.stop > max(closes[-8:-1])                       # stop just above the pullback high
    assert sig.stop - sig.entry < 1.5                          # small risk: entered early, near the break
    assert "1H" in sig.reason and sig.features["break_tf"] == "1H"
    frame.breaks = [dict(brk, bars_ago=30)]                    # break too long ago: too late, no trade
    assert BreakRetest().generate(MarketContext("US30", df, htf={"1H": frame})) is None


def test_backtest_makes_a_weekly_plan_and_keeps_to_it(tmp_path):
    j = run_backtest({"A": synthetic(1500, 1.10, seed=8)}, Settings(), db_path=tmp_path / "p.db", seed=1)
    book = j.get("plan:backtest")
    assert book and "A" in book["markets"] and book["markets"]["A"]["summary"]
    assert any(e["kind"] == "plan" for e in j.events(500, "backtest"))
    for t in j.trades("mode='backtest' AND shadow=0"):
        assert t["features"].get("plan_agree") in ("with", "both", "none", "against")
