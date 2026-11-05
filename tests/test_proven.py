import random

from tradingbot.news.assets import tradeable


def test_tradeable_markets():
    for s in ["AAPL", "TSLA", "US500", "USTECH", "XAUUSD", "XAGUSD", "XTIUSD", "BTCUSD", "ETHUSD"]:
        assert tradeable(s), s
    for s in ["EURUSD", "GBPJPY", "SOLUSD", "XRPUSD", "LTCUSD"]:
        assert not tradeable(s), s


def test_unproven_strategies_stay_virtual(tmp_path, monkeypatch):
    from tradingbot.config import Settings
    from tradingbot.journal import Journal
    from tradingbot.learning import Learner
    from tradingbot.strategies.base import Strategy

    monkeypatch.setenv("PROVEN_ONLY", "1")
    s = Settings()
    assert s.proven_only and s.experiments_allowed
    l = Learner(Journal(tmp_path / "j.db", mode="demo"), s.learning, s.risk, live=False, rng=random.Random(1))

    class S(Strategy):
        name = "x"
        def generate(self, ctx):
            return None

    d = l.decide(S(), {"regime": "trend"}, "AAPL", [])
    assert d.action == "shadow" and "not proven" in d.reasons[0]
