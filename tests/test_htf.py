
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
    j = run_backtest({"A": synthetic(3000, 1.10, seed=5)}, Settings(), db_path=tmp_path / "h.db", seed=1)
    trades = j.trades("mode='backtest'")
    assert trades
    late = [t for t in trades if t["opened_at"] > "2025-01-25"]
    assert late and all(t["features"].get("htf_align") in ("with", "against", "mixed", "unknown") for t in late)
    assert any(t["features"].get("htf_D") for t in late)
    # no real trade went against the top-down bias without being at a higher-timeframe zone
    for t in trades:
        f = t["features"]
        if not t["shadow"]:
            assert not (f.get("htf_align") == "against" and not f.get("htf_reversal_spot"))
