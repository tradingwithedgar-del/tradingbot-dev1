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


def test_real_trades_keep_entry_snapshot_and_full_candle_history(tmp_path):
    import pandas as pd

    from tradingbot.agent import CHART_MAX_BARS, CHART_PRE_BARS, _trim_chart

    rows = [[str(i)] for i in range(CHART_MAX_BARS + 500)]
    kept = _trim_chart(rows)
    assert len(kept) == CHART_MAX_BARS and kept[:CHART_PRE_BARS] == rows[:CHART_PRE_BARS] and kept[-1] == rows[-1]

    data = {"A": synthetic(1500, 18000, seed=11)}
    j = run_backtest(data, Settings(), db_path=tmp_path / "bt.db", seed=3)
    real = [t for t in j.trades("mode='backtest' AND shadow=0 AND status='closed'")]
    assert real
    for t in real:
        assert t["entry_view"] and t["entry_view"]["t"][-1] <= t["opened_at"]   # snapshot is from entry time
        times = [pd.Timestamp(r[0]) for r in t["chart"]]
        assert all((b - a).total_seconds() == 900 for a, b in zip(times, times[1:]))  # no missing candles
        assert any(r[0] == t["closed_at"] for r in t["chart"])                        # exit candle recorded


def test_why_report_explains_blocked_setups(tmp_path):
    import pandas as pd

    from tradingbot.diagnose import report
    from tradingbot.journal import Journal

    s = Settings()
    s.mode = "demo"
    s.symbols = ["USTECH", "NVDA"]
    s.db_path = tmp_path / "w.db"
    j = Journal(s.db_path, mode="demo")
    now = pd.Timestamp("2026-10-07 20:00", tz="UTC")
    j.set("last_bar:demo", {"USTECH": "2026-10-07T19:30:00+00:00"})
    common = dict(shadow=1, symbol="USTECH", side="buy", opened_at="2026-10-07T15:00:00+00:00", entry=1, stop=0.9,
                  take_profit=1.3, qty=0, risk_amount=0, risk_pct=0, reason="x", features={})
    j.open_trade(strategy="sd_reversal", decision={"why": ["demo: Thompson-sampled win rate", "expected +0.20R",
                                                           "major news points the other way: Iran strikes"]}, **common)
    j.open_trade(strategy="divergence", decision={"why": ["demo: Thompson-sampled win rate", "expected -0.10R not positive"]}, **common)
    j.open_trade(strategy="exp_1234abcd", experimental=1, decision={"why": ["experimental genome in shadow stage"]}, **common)
    j.event("error", "order failed USTECH buy: rejected", ts="2026-10-07T16:00:00+00:00")
    text = report(j, s, now)
    assert "real trades opened:            0" in text and "blocked / passed on (virtual): 2" in text
    assert "major news pointed against the trade" in text and "not profitable enough" in text
    assert "NVDA     never" in text and "order failed" in text
