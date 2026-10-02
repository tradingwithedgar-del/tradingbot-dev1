from tradingbot.backtest import run_backtest, synthetic
from tradingbot.config import Settings


def test_backtest_end_to_end_respects_risk_rules(tmp_path):
    data = {"A": synthetic(700, 1.10, seed=3), "B": synthetic(700, 1.30, seed=5)}
    s = Settings()
    s.history_bars = 300
    j = run_backtest(data, s, db_path=tmp_path / "bt.db", seed=2)
    trades = j.trades("mode='backtest'")
    assert trades, "agent should at least shadow-trade on 900 bars"
    real = [t for t in trades if not t["shadow"]]
    for t in real:
        assert t["risk_pct"] <= s.risk.risk_per_trade + 1e-9
    # never two real trades open on the same symbol at the same time
    for a in real:
        for b in real:
            if a["id"] < b["id"] and a["symbol"] == b["symbol"]:
                assert (a["closed_at"] or "9") <= b["opened_at"]
    # every closed trade has a post-mortem
    for t in trades:
        if t["status"] == "closed":
            assert t["postmortem"] is not None
    assert j.equity_curve("backtest")


def test_targets_are_3r(tmp_path):
    data = {"A": synthetic(600, 1.10, seed=9)}
    j = run_backtest(data, Settings(), db_path=tmp_path / "bt.db", seed=4)
    # Target is 3R from the signal price: |TP - SL| == 4R. Shadow entries are shifted by half the
    # spread (realistic cost), so compare the planned R with the recorded entry within that spread.
    trades = j.trades("mode='backtest' AND shadow=1")
    assert trades
    for t in trades:
        planned_r = abs(t["take_profit"] - t["stop"]) / 4
        assert abs(abs(t["entry"] - t["stop"]) - planned_r) <= t["entry"] * 1e-4
