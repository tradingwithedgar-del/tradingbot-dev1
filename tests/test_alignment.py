import pandas as pd

from tradingbot import alignment as A
from tradingbot.backtest import synthetic
from tradingbot.config import Settings


def test_crypto_follows_btc_and_waits_until_btc_decides():
    for sym in ("ETHUSD", "SOLUSD", "XRPUSD", "LTCUSD", "ETHEUR", "BTCEUR"):
        assert A.check(sym, "buy", {"BTCUSD": 1}, {}) is None
        assert "against its group" in A.check(sym, "sell", {"BTCUSD": 1}, {})
        assert "against its group" in A.check(sym, "buy", {"BTCUSD": -1}, {})
        assert "waiting" in A.check(sym, "buy", {}, {})                              # BTC has no direction yet
        assert "waiting" in A.check(sym, "sell", {"BTCUSD": 0}, {})
    assert A.check("BTCUSD", "sell", {"BTCUSD": 1}, {}) is None                      # the leader follows nobody
    # BTC's open trade is its position: it outweighs a trend that disagrees
    assert A.check("ETHUSD", "buy", {"BTCUSD": -1}, {"BTCUSD": ["buy"]}) is None
    assert A.check("ETHUSD", "sell", {"BTCUSD": 1}, {"BTCUSD": ["sell"]}) is None


def test_indices_move_as_regional_groups():
    up = {"US30": 1, "USTECH": 1, "RUSS2000": 1}
    assert "US indices are going up" in A.check("US500", "sell", up, {})
    assert A.check("US500", "buy", up, {}) is None
    assert A.check("US500", "sell", {"US30": 1, "USTECH": -1, "RUSS2000": 0}, {}) is None   # split: no consensus
    assert A.check("US500", "sell", {"US30": 1, "USTECH": 1, "RUSS2000": -1}, {})            # 2 of 3 is a consensus
    assert "European indices" in A.check("DE40", "buy", {"F40": -1, "UK100": -1, "ES35": -1}, {})
    assert "Asia-Pacific" in A.check("JP225", "sell", {"HK50": 1, "CHINA50": 1}, {})
    assert A.check("DE40", "buy", {"US30": -1, "US500": -1, "USTECH": -1}, {}) is None       # regions are separate


def test_stocks_follow_the_indices_of_their_industry():
    ups = {"USTECH": 1, "US500": 1, "US30": 1}
    assert "technology" in A.check("NVDA", "sell", ups, {})
    assert A.check("NVDA", "buy", ups, {}) is None
    assert "financials" in A.check("JPM", "sell", ups, {})
    assert "energy" in A.check("XOM", "buy", {"XTIUSD": -1, "US500": -1}, {})
    assert A.check("XOM", "buy", {"XTIUSD": -1, "US500": 1}, {}) is None                     # oil and S&P disagree
    assert A.check("UNH", "buy", {"US500": -1, "US30": -1}, {})                              # healthcare
    assert A.check("TSLA", "sell", {"USTECH": -1, "US500": -1}, {}) is None                  # with the group: fine
    assert A.check("EURUSD", "sell", ups, {}) is None and A.check("XAUUSD", "buy", ups, {}) is None
    covered = {s for _, syms in A.SECTORS.values() for s in syms.split()}
    assert {"AAPL", "NVDA", "TSLA", "JPM", "XOM", "BA", "UNH", "KO", "BABA"} <= covered


def test_agent_applies_the_group_rule_and_records_the_reason(tmp_path):
    from tradingbot.agent import Agent
    from tradingbot.broker.sim import SimBroker
    from tradingbot.journal import Journal
    from tradingbot.strategies import Signal

    s = Settings()
    s.mode = "backtest"
    sim = SimBroker({"ETHUSD": synthetic(600, 100, seed=2)}, warmup=300)
    a = Agent(s, sim, Journal(tmp_path / "a.db", mode="backtest"))
    now = pd.Timestamp(a.now())
    a._dir["BTCUSD"] = (1, now)
    sig = Signal("ETHUSD", "structure_trend", "sell", 100.0, 101.0, 97.0, features={"htf_align": "with"})
    ctx = type("C", (), {"close": 100.0, "atr_now": 1.0})()
    q = sim.get_quote("ETHUSD")
    assert "against its group" in a._entry_blocker(sig, ctx, q, [], set())
    sig.side = "buy"
    why = a._entry_blocker(sig, ctx, q, [], set())
    assert not why or "group" not in why
    a._dir["BTCUSD"] = (0, now)
    assert "waiting" in a._entry_blocker(sig, ctx, q, [], set())
    s.alignment_filter = False
    why = a._entry_blocker(sig, ctx, q, [], set())
    assert not why or "waiting" not in why
    # the weekly plan stands in for a market that hasn't been read yet this run
    s.alignment_filter = True
    a._dir.clear()
    a.j.set("plan:backtest", {"week": "w", "markets": {"BTCUSD": {"direction": "sell", "status": "active"}}})
    assert a._group_dirs()["BTCUSD"] == -1
