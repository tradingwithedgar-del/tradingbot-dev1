from tradingbot.news.assets import tradeable


def test_tradeable_markets():
    for s in ["AAPL", "TSLA", "US500", "USTECH", "XAUUSD", "XAGUSD", "XTIUSD", "BTCUSD", "ETHUSD"]:
        assert tradeable(s), s
    for s in ["EURUSD", "GBPJPY", "SOLUSD", "XRPUSD", "LTCUSD"]:
        assert not tradeable(s), s


def test_proven_only_blocks_experiments(monkeypatch):
    from tradingbot.config import Settings
    monkeypatch.setenv("PROVEN_ONLY", "1")
    s = Settings()
    assert s.proven_only and not s.experiments_allowed
    monkeypatch.setenv("PROVEN_ONLY", "0")
    assert Settings().experiments_allowed
