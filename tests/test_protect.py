import numpy as np
import pandas as pd

from tradingbot import protect
from tradingbot.backtest import run_backtest, synthetic
from tradingbot.config import Settings


class Ctx:
    def __init__(self, close, atr=1.0, spread=0.02, swings=(), i=100):
        self.close, self.atr_now, self.spread, self.swings, self.i = close, atr, spread, list(swings), i


class Sw:
    def __init__(self, idx, price, kind, confirmed_at=None):
        self.idx, self.price, self.kind, self.confirmed_at = idx, price, kind, confirmed_at or idx + 3


def trade(side="buy", mfe=0.0, entry=100.0, stop=98.0, tp=106.0):
    return {"side": side, "entry": entry, "stop": stop, "take_profit": tp, "mfe_r": mfe}


def test_ladder_break_even_then_locked_profit_never_loosens():
    t = trade(mfe=0.6)
    assert protect.plan(t, {}, Ctx(101.2)) is None                        # not enough profit yet
    t["mfe_r"] = 1.05
    stop, why = protect.plan(t, {}, Ctx(102.2))
    assert 100.0 < stop < 100.4 and "break-even" in why                   # entry plus costs, small profit
    dec = {"stop_now": stop}
    assert protect.plan(t, dec, Ctx(102.2)) is None                       # already there: no spam
    assert protect.plan(trade(mfe=1.05), {}, Ctx(100.3)) is None          # price too close to entry to leave noise room
    t["mfe_r"] = 2.1
    stop2, why2 = protect.plan(t, dec, Ctx(104.3))
    assert stop2 > stop and stop2 >= 100.0 + 2.0 * 0.9 and "locked" in why2        # ~ +1R locked at 2R
    assert protect.plan(t, {"stop_now": stop2}, Ctx(100.5)) is None       # price fell back: the stop never loosens
    # a sell mirrors it
    s = trade("sell", mfe=2.1, entry=100.0, stop=102.0, tp=94.0)
    st, _ = protect.plan(s, {}, Ctx(95.7))
    assert st < 100.0 - 2.0 * 0.9


def test_stop_keeps_noise_room_and_trails_behind_swing():
    t = trade(mfe=2.6)
    stop, why = protect.plan(t, {}, Ctx(105.0, atr=1.0), swing_after=50)
    assert stop <= 105.0 - 0.35                                           # never closer than normal noise to price
    sw = [Sw(120, 103.4, "L")]
    stop2, why2 = protect.plan(t, {}, Ctx(105.4, atr=1.0, swings=sw, i=130), swing_after=50)
    assert why2 == "behind the latest swing" and abs(stop2 - (103.4 - 0.2)) < 1e-9
    # a swing from before the entry doesn't count
    assert protect.plan(t, {}, Ctx(105.4, swings=[Sw(40, 103.4, "L")], i=130), swing_after=50)[1] != "behind the latest swing"


def test_discretion_scale_and_short_targets():
    t = trade(mfe=1.1)
    assert protect.plan(t, {}, Ctx(102.3), scale=0.7) is not None         # protects sooner
    assert protect.plan(t, {}, Ctx(102.3), scale=1.5) is None              # waits longer (needs 1.5R)
    scalp = trade(mfe=0.8, tp=102.6)                                       # target only 1.3R away
    assert protect.plan(scalp, {}, Ctx(101.5)) is not None                 # steps are proportional to the target


def test_learner_scale_from_cut_winners_and_give_backs(tmp_path):
    from tradingbot.journal import Journal
    from tradingbot.learning import Learner

    s = Settings(mode="demo")
    j = Journal(tmp_path / "p.db", mode="demo")
    L = Learner(j, s.learning, s.risk, live=False)
    assert L.protection_scale("x") == 1.0                                  # no evidence: neutral

    def add(cut):
        tid = j.open_trade(shadow=0, symbol="A", strategy="x", side="buy", opened_at="2026-10-01T10:00", entry=1, stop=0.9,
                           take_profit=1.3, qty=1, risk_amount=1, risk_pct=0.01, reason="", features={},
                           decision={"close_reason": "protected", "why": []})
        j.update_trade(tid, status="closed", r_multiple=0.1, mfe_r=1.2, postmortem={"after_exit": {"hit_target": cut}})
    for _ in range(12):
        add(True)                                                          # every protected exit later hit target
    L._prot_cache.clear()
    assert L.protection_scale("x") > 1.3                                    # be more patient


def test_backtest_protects_real_and_virtual_trades(tmp_path):
    j = run_backtest({"A": synthetic(3000, 1.10, seed=21)}, Settings(), db_path=tmp_path / "pp.db", seed=2)
    closed = j.trades("mode='backtest' AND status='closed'")
    moved = [t for t in closed if (t.get("decision") or {}).get("stop_now") is not None]
    assert moved, "some trades should have reached the protection step"
    for t in moved:
        d = t["decision"]
        assert (t["mfe_r"] or 0) >= 0.5
        # the moved stop is on the profit side of the original stop and never worse than it
        assert (d["stop_now"] - t["stop"]) * (1 if t["side"] == "buy" else -1) > 0
    protected = [t for t in closed if t["decision"].get("close_reason") == "protected"]
    for t in protected:
        assert t["r_multiple"] > -0.2                                      # stopped near break-even or in profit
        assert "protected_exit" in t["postmortem"]["tags"]
    assert any(e["kind"] == "manage" and "secured" in e["message"] for e in j.events(500, "backtest"))
