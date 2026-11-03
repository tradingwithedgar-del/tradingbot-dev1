from tradingbot.config import Settings
from tradingbot.diagnose import autopsy
from tradingbot.journal import Journal, now_iso


def test_autopsy_runs_on_empty_and_filled(tmp_path):
    j = Journal(tmp_path / "j.db", mode="demo")
    s = Settings()
    assert "No equity" in autopsy(j, s)
    j.record_equity(1000, 1000, 1000)
    for i, r in enumerate([-1, -1, 3, -2.5]):
        tid = j.open_trade(mode="demo", shadow=0, symbol="EURUSD", strategy="x", side="buy", status="open",
                           opened_at=now_iso(), entry=1.1, stop=1.09, qty=1, risk_amount=10, risk_pct=0.01,
                           features={"spread": 0.001, "session": "london", "weekday": 1}, decision={})
        j.update_trade(tid, status="closed", closed_at=now_iso(), r_multiple=r, pnl=r * 10, outcome="loss" if r < 0 else "win",
                       decision={"close_reason": "stop"})
    j.event("resume", "human resumed")
    txt = autopsy(j, s, 30)
    assert "ALL REAL" in txt and "By strategy" in txt and "Resumes by you: 1" in txt
