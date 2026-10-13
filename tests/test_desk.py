import pandas as pd

from tradingbot.backtest import run_backtest, synthetic
from tradingbot.config import LabConfig, Settings
from tradingbot.llm import Reply


class FakeLLM:
    """Stands in for Claude: the judge always gives `action`."""
    name = "fake"

    def __init__(self, action="skip"):
        self.action, self.calls = action, []

    def ask(self, system, prompt, schema, model, timeout=0):
        self.calls.append(model)
        props = schema["properties"]
        if "action" in props:
            return Reply({"action": self.action, "confidence": 70, "reason": f"test says {self.action}"})
        if "rebuttal" in props:
            return Reply({"rebuttal": "no", "conviction": 4})
        if "case" in props:
            return Reply({"case": "because", "conviction": 7 if "BULL" in prompt[-200:] else 3, "key_risk": "x"})
        if "markets" in props:
            return Reply({"markets": [{"symbol": "A", "bias": "bull", "confidence": 60, "drivers": "d", "risks": "r"}],
                          "overview": "calm"})
        return Reply(None, "unknown schema")


def _desk(llm, daily=10_000):
    from tradingbot.desk import Desk

    def make(settings, journal):
        settings.desk.daily_calls = daily
        return Desk(settings, journal, llm, background=False)
    return make


def test_desk_veto_turns_real_trades_into_graded_virtual_ones(tmp_path):
    llm = FakeLLM("skip")
    j = run_backtest({"A": synthetic(1200, 1.10, seed=3)}, Settings(), db_path=tmp_path / "d.db", seed=2,
                     desk_factory=_desk(llm))
    real = j.trades("mode='backtest' AND shadow=0")
    debated = [t for t in j.trades("mode='backtest'") if (t.get("decision") or {}).get("desk")]
    assert debated, "the desk should have debated some setups"
    assert not real, "every setup was vetoed, so no real trade"
    for t in debated:
        d = t["decision"]["desk"]
        assert d["action"] == "skip" and d["bull"]["conviction"] == 4 and "desk veto" in " ".join(t["decision"]["why"])
    assert {"opus", "sonnet"} <= set(llm.calls)


def test_desk_half_size_halves_risk_and_never_raises_it(tmp_path):
    s = Settings()
    j = run_backtest({"A": synthetic(1200, 1.10, seed=3)}, s, db_path=tmp_path / "h.db", seed=2,
                     desk_factory=_desk(FakeLLM("half")))
    real = j.trades("mode='backtest' AND shadow=0")
    assert real
    for t in real:
        assert t["decision"]["desk"]["action"] == "half"
        assert t["risk_pct"] <= s.risk.risk_per_trade / 2 + 1e-6


def test_desk_steps_aside_when_budget_is_used_up(tmp_path):
    llm = FakeLLM("skip")
    j = run_backtest({"A": synthetic(1200, 1.10, seed=3)}, Settings(), db_path=tmp_path / "b.db", seed=2,
                     desk_factory=_desk(llm, daily=0))
    assert not llm.calls
    real = j.trades("mode='backtest' AND shadow=0")
    assert real and all("TIIM decided alone" in " ".join(t["decision"]["why"]) for t in real)


def test_scorekeeper_suspends_veto_power_when_vetoes_cost_money(tmp_path):
    from tradingbot.desk import scorecard
    from tradingbot.journal import Journal

    s = Settings()
    s.mode = "demo"
    j = Journal(tmp_path / "s.db", mode="demo")
    for i in range(25):
        tid = j.open_trade(shadow=1, symbol="A", strategy="x", side="buy", opened_at=f"2026-10-0{1 + i % 9}T10:00",
                           entry=1, stop=0.9, take_profit=1.3, qty=0, risk_amount=0, risk_pct=0, reason="", features={},
                           decision={"why": [], "desk": {"action": "skip", "bull": {"conviction": 7}, "bear": {"conviction": 3}}})
        j.update_trade(tid, status="closed", r_multiple=3.0 if i % 2 else -1.0)
    card = scorecard(j, s)
    assert card["judge"]["vetoes"] == 25 and card["judge"]["veto_avg_r"] > 0.15
    assert card["judge"]["veto_power"] is False
    assert card["debate"]["graded"] == 25


def test_research_brief_is_stored_and_used(tmp_path):
    from tradingbot.desk import Desk
    from tradingbot.journal import Journal

    s = Settings()
    j = Journal(tmp_path / "r.db", mode="demo")
    desk = Desk(s, j, FakeLLM(), background=False)
    now = pd.Timestamp.now(tz="UTC")
    assert desk.research_tick(now, {"A": {"price": 1.0, "chg_1d": 0.5, "chg_5d": 1.0, "trend": "up"}})
    assert desk.brief_for("A")["bias"] == "bull"
    assert not desk.research_due(now)          # fresh brief: not again yet


def test_lab_gate_uses_win_rate_on_unseen_data_and_rejects_money_losers():
    from tradingbot.lab import Result, passes

    cfg = LabConfig()
    cfg.min_trades = 100
    ok, _ = passes(Result(120, 90, 20.0), Result(40, 30, 5.0), cfg)
    assert ok
    assert not passes(Result(120, 90, 20.0), Result(40, 24, 5.0), cfg)[0]       # 60% on unseen data
    assert not passes(Result(120, 90, -8.0), Result(40, 30, 1.0), cfg)[0]       # wins often, loses money
    assert not passes(Result(50, 45, 9.0), Result(40, 35, 5.0), cfg)[0]         # too few trades
    cfg.require_profit = False
    assert passes(Result(120, 90, -8.0), Result(40, 30, 1.0), cfg)[0]


def test_lab_run_and_merge_into_learner(tmp_path):
    from tradingbot.journal import Journal
    from tradingbot.lab import clean_idea, run_lab
    from tradingbot.learning import Learner

    cfg = LabConfig()
    cfg.candidates_per_night, cfg.min_trades, cfg.min_win_rate, cfg.require_profit = 30, 20, 0.3, False
    data = {s: synthetic(3000, 100 + i, seed=i) for i, s in enumerate(["US30", "BTCUSD"])}
    rep = run_lab(data, cfg, {}, seed=4)
    assert rep["tested"] >= 25 and rep["approved"]
    s = Settings()
    s.mode = "demo"
    j = Journal(tmp_path / "l.db", mode="demo")
    learner = Learner(j, s.learning, s.risk, live=False)
    notes = learner.merge_lab(rep, "2026-10-13T04:30:00+00:00")
    assert notes and all(learner.population[a["gid"]]["status"] == "demo" for a in rep["approved"])
    assert not learner.merge_lab(rep, "later")                                   # not added twice
    gid = rep["approved"][0]["gid"]
    assert learner.merge_lab({"approved": [], "withdrawn": [gid]}, "later")
    assert learner.population[gid]["status"] == "retired"
    # Claude's ideas are clipped to the menu; anything off-menu is dropped
    g = clean_idea({"conditions": [{"name": "rsi_extreme", "params": {"period": 99, "level": 5}},
                                   {"name": "os.system", "params": {}}, {"name": "adx_strong", "params": {"level": 20}}],
                    "stop_atr": 9, "target_r": 0.1, "max_bars": 500})
    assert g and g.conditions[0][1] == {"period": 21, "level": 20} and len(g.conditions) == 2
    assert (g.stop_atr, g.target_r, g.max_bars) == (3.0, 0.3, 48)
