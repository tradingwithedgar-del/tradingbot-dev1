import random

import numpy as np
import pandas as pd
import pytest

from tradingbot import postmortem as pm
from tradingbot.analysis import indicators as ind
from tradingbot.analysis.divergence import find_divergences
from tradingbot.analysis.structure import Swing, find_swings, label_swings, trend_state
from tradingbot.analysis.zones import find_zones
from tradingbot.broker.base import InstrumentSpec, Position, Quote
from tradingbot.broker.sim import SimBroker
from tradingbot.compliance import Compliance
from tradingbot.config import ComplianceConfig, LearningConfig, RiskConfig, Settings
from tradingbot.journal import Journal
from tradingbot.learning import EdgeStat, Learner
from tradingbot.risk import RiskManager
from tradingbot.strategies import ExperimentalStrategy, StructureTrend, random_genome


def frame(closes, spread=0.2):
    idx = pd.date_range("2025-01-06", periods=len(closes), freq="15min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": 1.0}, index=idx)


def zigzag(points, steps=6):
    out = []
    for a, b in zip(points, points[1:]):
        out += list(np.linspace(a, b, steps, endpoint=False))
    return out + [points[-1]]


# --- indicators / structure -------------------------------------------------------
def test_rsi_bounds_and_atr_positive():
    df = frame(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 300)))
    r = ind.rsi(df["close"]).dropna()
    assert r.between(0, 100).all()
    assert (ind.atr(df).iloc[20:] > 0).all()


def test_uptrend_structure_labels():
    df = frame(zigzag([100, 110, 105, 115, 108, 120, 112, 125, 118]))
    swings = find_swings(df, 3, 3)
    labels = [s.label for s in swings if s.label]
    assert "HH" in labels and "HL" in labels and "LL" not in labels
    assert trend_state(swings) == "up"


def test_downtrend_structure_and_break():
    df = frame(zigzag([125, 115, 120, 110, 116, 105, 111, 100, 104]))
    swings = find_swings(df, 3, 3)
    assert trend_state(swings) == "down"
    last_high = max(s.price for s in swings if s.kind == "H" and s.idx == max(x.idx for x in swings if x.kind == "H"))
    assert trend_state(swings, last_close=last_high + 1) == "up_break"


def test_swings_never_confirmed_in_the_future():
    df = frame(zigzag([100, 110, 105, 115, 108, 120]))
    for s in find_swings(df, 3, 3):
        assert s.confirmed_at == s.idx + 3 and s.confirmed_at < len(df)


def test_demand_zone_found_and_broken():
    closes = [100 + 0.01 * i for i in range(30)] + [100.3, 100.32, 100.31] + [103.5, 104, 104.5, 105] + [104.8] * 5
    df = frame(closes, spread=0.05)
    zones = find_zones(df, impulse_min_atr=1.5)
    assert any(z.kind == "demand" for z in zones)
    broken = frame(closes + [99.0], spread=0.05)
    assert not any(z.kind == "demand" and z.bottom > 99.5 for z in find_zones(broken, impulse_min_atr=1.5))


def test_regular_bullish_divergence():
    swings = label_swings([Swing(10, 13, 100.0, "L"), Swing(30, 33, 98.0, "L")])
    osc = pd.Series(np.full(40, 50.0))
    osc[10], osc[30] = 25.0, 35.0
    divs = find_divergences(swings, osc)
    assert divs and divs[0].kind == "regular_bull" and divs[0].side == "buy"


# --- risk ---------------------------------------------------------------------------
def test_size_risks_exactly_5_percent():
    rm = RiskManager(RiskConfig())
    spec = InstrumentSpec(value_per_point=100_000, qty_step=0.01, min_qty=0.01)
    d = rm.size(10_000, 10_000, entry=1.1000, stop=1.0950, spec=spec, open_risk_pct=0)
    assert d.ok and d.qty == pytest.approx(1.0)
    assert d.risk_pct == pytest.approx(0.05, abs=1e-6)


def test_risk_scales_down_in_drawdown_never_up():
    rm = RiskManager(RiskConfig())
    assert rm.current_risk_pct(10_000, 10_000) == pytest.approx(0.05)
    assert rm.current_risk_pct(8_500, 10_000) < 0.05
    assert rm.current_risk_pct(7_000, 10_000) == pytest.approx(0.01)
    assert rm.current_risk_pct(12_000, 10_000) <= 0.05


def test_limits():
    rm = RiskManager(RiskConfig())
    assert "drawdown" in rm.check_limits(6_900, 10_000, 7_000, 0, 0)
    assert "daily" in rm.check_limits(8_900, 10_000, 10_000, 0, 0)
    assert "open trades" in rm.check_limits(10_000, 10_000, 10_000, 0.05, 2)
    assert rm.check_limits(10_000, 10_000, 10_000, 0.05, 1) is None
    assert "open risk" in rm.check_limits(10_000, 10_000, 10_000, 0.095, 1)


def test_live_needs_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "NO")
    s = Settings(mode="live", tl_environment="https://live.tradelocker.com")
    with pytest.raises(RuntimeError):
        s.validate()
    s = Settings(mode="demo", tl_environment="https://live.tradelocker.com")
    with pytest.raises(RuntimeError):
        s.validate()
    assert not Settings(mode="live").experiments_allowed


# --- compliance ---------------------------------------------------------------------
def test_compliance_blocks_abusive_conditions():
    c = Compliance(ComplianceConfig())
    t = pd.Timestamp.now(tz="UTC")
    good = Quote(1.1000, 1.1001, t)
    assert c.check("buy", "EURUSD", 1.1001, 1.0980, 1.1000, 0.002, good, [], t) is None
    assert "spread" in c.check("buy", "EURUSD", 1.1001, 1.0980, 1.1000, 0.002, Quote(1.1, 1.102, t), [], t)
    assert "misquote" in c.check("buy", "EURUSD", 1.12, 1.10, 1.1000, 0.002, Quote(1.12, 1.1201, t), [], t)
    assert "tight" in c.check("buy", "EURUSD", 1.1001, 1.0999, 1.1000, 0.002, good, [], t)
    opp = [Position("1", "EURUSD", "sell", 1, 1.1)]
    assert "hedging" in c.check("buy", "EURUSD", 1.1001, 1.0980, 1.1000, 0.002, good, opp, t)
    stale = Quote(1.1000, 1.1001, t - pd.Timedelta(minutes=10))
    assert "stale" in c.check("buy", "EURUSD", 1.1001, 1.0980, 1.1000, 0.002, stale, [], t)


# --- learning -------------------------------------------------------------------------
def make_learner(tmp_path, mode="demo", live=False):
    j = Journal(tmp_path / "t.db", mode=mode)
    return Learner(j, LearningConfig(), RiskConfig(), live=live, rng=random.Random(0)), j


def test_edge_stat_decay_and_expectancy():
    s = EdgeStat()
    cfg = LearningConfig()
    for _ in range(30):
        s.update(3.0, 1.0, 0.97)
    assert s.p_mean(cfg) > 0.8 and s.expected_r(s.p_mean(cfg), 3.0) > 1.5
    for _ in range(60):
        s.update(-1.0, 1.0, 0.97)
    assert s.p_mean(cfg) < 0.3  # recent losses dominate: the agent adapts


def test_learned_filter_blocks_losing_condition(tmp_path):
    learner, j = make_learner(tmp_path)
    feats = {"regime": "ranging_high_vol", "session": "asia"}
    for _ in range(20):
        learner.record({"strategy": "structure_trend", "symbol": "EURUSD", "r_multiple": -1.0, "features": feats})
    assert "regime=ranging_high_vol" in learner.filters["structure_trend"]
    d = learner.decide(StructureTrend(), feats, "EURUSD", [])
    assert d.action == "shadow" and "learned filter" in d.reasons[0]
    assert any(e["kind"] == "lesson" for e in j.events())


def test_live_only_trades_approved_strategies_and_no_experiments(tmp_path):
    learner, _ = make_learner(tmp_path, mode="live", live=True)
    assert learner.decide(StructureTrend(), {}, "EURUSD", []).action == "shadow"
    exp = ExperimentalStrategy(random_genome(random.Random(1)))
    assert learner.decide(exp, {}, "EURUSD", [exp.name]).action == "skip"


def test_evolution_fills_population_and_retires_losers(tmp_path):
    learner, j = make_learner(tmp_path)
    learner.evolve("2025-01-01")
    active = [g for g, i in learner.population.items() if i["status"] != "retired"]
    assert len(active) == learner.cfg.population_size
    loser = active[0]
    for _ in range(25):
        learner.record({"strategy": loser, "symbol": "X", "r_multiple": -1.0, "features": {}, "shadow": 1})
    learner.evolve("2025-01-02")
    assert learner.population[loser]["status"] == "retired"
    assert len([g for g, i in learner.population.items() if i["status"] != "retired"]) == learner.cfg.population_size


def test_tuning_widens_stop_after_stop_hunts(tmp_path):
    learner, _ = make_learner(tmp_path)
    strat = StructureTrend()
    before = strat.p("stop_atr_buffer")
    recent = [{"r_multiple": -1.0, "postmortem": {"tags": ["stop_too_tight"]}}] * 15 + [{"r_multiple": 3.0, "postmortem": {"tags": []}}] * 5
    changes = []
    for _ in range(15):
        changes += learner.tune(strat, recent)
    assert strat.p("stop_atr_buffer") > before and changes


# --- post-mortem -------------------------------------------------------------------------
def test_postmortem_stop_too_tight():
    t = {"side": "buy", "entry": 100.0, "stop": 99.0, "take_profit": 103.0, "r_multiple": -1.0, "mfe_r": 0.1, "mae_r": 1.0,
         "features": {"bias": "bear", "structure": "range", "volatility": "normal_vol"}}
    tags = pm.initial_tags(t)
    assert "immediately_wrong" in tags and "against_htf_bias" in tags
    t["postmortem"] = {"tags": tags}
    w = pm.new_watch(t, atr=0.5)
    pm.update_watch(t, w, high=101.0, low=98.8)
    pm.update_watch(t, w, high=103.2, low=100.5)
    assert "stop_too_tight" in pm.final_tags(t, w)


# --- simulated broker ----------------------------------------------------------------------
def test_sim_fills_next_open_and_stop_first():
    df = frame([100] * 5 + [100, 100, 100], spread=0.1)
    df.iloc[6, df.columns.get_loc("high")] = 104   # same bar hits both stop and target
    df.iloc[6, df.columns.get_loc("low")] = 98
    b = SimBroker({"X": df}, spread_frac=0.0, warmup=4)
    pid = b.place_market("X", "buy", 1.0, stop=99.0, take_profit=103.0, tag="t")
    b.step()
    assert b.open[pid]["entry"] == pytest.approx(df["open"].iloc[5])
    b.step()
    assert b.closed[pid].exit_price == pytest.approx(99.0)  # pessimistic: stop assumed first


def test_sim_keeps_symbols_with_different_hours_aligned():
    full = frame(np.linspace(100, 110, 96))                       # 24h instrument
    stock = full.between_time("13:30", "19:45")                    # US cash session only
    stock = stock.assign(close=stock["close"] + 1000, open=stock["open"] + 1000,
                         high=stock["high"] + 1000, low=stock["low"] + 1000)
    b = SimBroker({"IDX": full, "STK": stock}, spread_frac=0.0, warmup=10)
    while b.step():
        now = b.now()
        for sym, df in (("IDX", full), ("STK", stock)):
            bars = b.get_bars(sym, "15m", 5)
            if len(bars):
                assert bars.index[-1] <= now   # never sees the future
                assert bars.index[-1] == df.index[df.index <= now][-1]


def test_symbol_classification():
    from tradingbot.broker.sim import default_contract, is_fx_pair

    assert is_fx_pair("EURUSD") and not is_fx_pair("USOIL") and not is_fx_pair("NAS100") and not is_fx_pair("XAUUSD")
    assert default_contract("XAUUSD") == 100 and default_contract("USOIL") == 1000 and default_contract("XTIUSD") == 1000
    assert default_contract("NVDA") == 1 and default_contract("US30") == 1


def test_default_symbols_are_the_requested_markets(monkeypatch):
    monkeypatch.delenv("BOT_SYMBOLS", raising=False)
    assert Settings().symbols == ["US30", "US500", "USTECH", "XAUUSD", "NVDA", "AAPL", "TSLA", "XTIUSD",
                                  "BTCUSD", "ETHUSD", "SOLUSD"]


def test_open_trade_limits_demo_live_and_env(monkeypatch):
    from tradingbot.config import Settings

    for k in ("MAX_OPEN_TRADES", "MAX_OPEN_RISK", "MAX_SAME_GROUP", "RISK_PER_TRADE"):
        monkeypatch.delenv(k, raising=False)
    assert (Settings(mode="demo").risk.max_open_trades, Settings(mode="demo").risk.max_total_open_risk) == (20, 0.40)
    assert (Settings(mode="live").risk.max_open_trades, Settings(mode="live").risk.max_total_open_risk) == (2, 0.10)
    monkeypatch.setenv("MAX_OPEN_TRADES", "4")
    monkeypatch.setenv("MAX_OPEN_RISK", "15")
    s = Settings(mode="live")
    assert (s.risk.max_open_trades, s.risk.max_total_open_risk) == (4, 0.15)
    monkeypatch.setenv("RISK_PER_TRADE", "2")
    assert Settings(mode="demo").risk.risk_per_trade == 0.02


def test_correlated_trades_are_capped(tmp_path):
    from tradingbot.backtest import run_backtest, synthetic
    from tradingbot.config import Settings
    from tradingbot.news.assets import asset_class

    s = Settings(mode="backtest")
    s.risk.max_open_trades, s.risk.max_total_open_risk, s.risk.max_same_group = 6, 0.30, 1
    data = {k: synthetic(900, 18000 + i * 1000, seed=20 + i) for i, k in enumerate(["US30", "US500", "USTECH"])}
    j = run_backtest(data, s, db_path=tmp_path / "c.db", seed=5)
    real = j.trades("mode='backtest' AND shadow=0")
    for a in real:
        overlap = [b for b in real if b["id"] != a["id"] and b["side"] == a["side"]
                   and asset_class(b["symbol"]) == asset_class(a["symbol"])
                   and b["opened_at"] <= a["opened_at"] < (b["closed_at"] or "9")]
        assert not overlap


def test_gap_protection_sessions():
    import pandas as pd

    from tradingbot.agent import Agent
    from tradingbot.sessions import entry_block, flatten_reason

    utc = lambda s: pd.Timestamp(s, tz="UTC")
    # Oct 2026 is EDT (UTC-4): stock close 16:00 ET = 20:00 UTC
    assert flatten_reason("TSLA", utc("2026-10-08 19:50")) and not flatten_reason("TSLA", utc("2026-10-08 19:40"))
    assert entry_block("TSLA", utc("2026-10-08 19:05")) and not entry_block("TSLA", utc("2026-10-08 18:55"))
    assert not flatten_reason("US30", utc("2026-10-08 19:50"))              # indices trade overnight
    assert flatten_reason("US30", utc("2026-10-09 20:10"))                  # ...but not over the weekend (Friday)
    assert flatten_reason("XAUUSD", utc("2026-10-09 20:10")) and entry_block("XTIUSD", utc("2026-10-09 19:30"))
    assert not flatten_reason("BTCUSD", utc("2026-10-09 20:10")) and not entry_block("BTCUSD", utc("2026-10-09 20:10"))
    # a fill beyond the stop is a gapped stop, not a manual close
    t = {"side": "sell", "stop": 100.0, "take_profit": 85.0, "decision": {}}
    assert Agent._close_reason(t, 114.0, 95.0, estimated=False) == "stop"
    assert Agent._close_reason(t, 97.0, 95.0, estimated=False) == "manual"


def test_subscription_token_in_api_key_slot_is_moved(monkeypatch):
    import importlib

    import tradingbot.config as config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-oat01-example")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    importlib.reload(config)
    import os
    assert os.getenv("CLAUDE_CODE_OAUTH_TOKEN") == "sk-ant-oat01-example" and not os.getenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    importlib.reload(config)


def test_new_agent_keeps_the_knowledge_and_renames_everywhere(tmp_path, monkeypatch):
    import argparse

    import tradingbot.__main__ as cli
    import tradingbot.config as config
    from tradingbot.config import Settings
    from tradingbot.journal import Journal

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "Settings", lambda: _settings_at(tmp_path))
    monkeypatch.setenv("AGENT_NAME", "TIIM")

    def _settings_at(p):
        s = Settings()
        s.db_path = p / "bot.db"
        return s

    j = Journal(tmp_path / "bot.db", mode="demo")
    j.set("learner:demo", {"stats": {"sd_reversal": {"wins": 5}}})
    (tmp_path / ".env").write_text("TL_EMAIL=x\nAGENT_NAME=TIIM\n")
    monkeypatch.setattr("tradingbot.config.ROOT", tmp_path)
    cli.cmd_new_agent(argparse.Namespace(name="Nova!"))
    assert "AGENT_NAME=Nova\n" in (tmp_path / ".env").read_text() and "TL_EMAIL=x" in (tmp_path / ".env").read_text()
    backup = next((tmp_path / "backups").glob("tiim-*.db"))
    assert Journal(backup, mode="demo").get("learner:demo") == {"stats": {"sd_reversal": {"wins": 5}}}
    assert Journal(tmp_path / "bot.db", mode="demo").get("learner:demo")             # knowledge untouched
    assert any(e["kind"] == "handover" and "Nova" in e["message"] for e in j.events(10, "demo"))
    monkeypatch.setenv("AGENT_NAME", "Nova")
    assert Settings().agent_name == "Nova"


def test_dashboard_and_desk_use_the_agent_name(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tradingbot.dashboard.app import create_app
    from tradingbot.desk import _Named
    from tradingbot.journal import Journal

    Journal(tmp_path / "x.db", mode="demo")
    monkeypatch.setenv("AGENT_NAME", "Nova")
    html = TestClient(create_app(tmp_path / "x.db")).get("/").text
    assert "<title>Nova Dashboard</title>" in html and "TIIM" not in html

    seen = {}

    class Fake:
        name = "fake"

        def ask(self, system, prompt, schema, model, **kw):
            seen.update(system=system, prompt=prompt)

    monkeypatch.setenv("AGENT_NAME", "Nova")
    _Named(Fake(), "Nova").ask("You are TIIM, the judge.", "TIIM's odds", {}, "opus")
    assert seen == {"system": "You are Nova, the judge.", "prompt": "Nova's odds"}
