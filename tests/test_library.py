import json
import random

import numpy as np
import pandas as pd

from tradingbot.agent import Agent
from tradingbot.analysis.context import MarketContext
from tradingbot.backtest import synthetic
from tradingbot.broker.sim import SimBroker
from tradingbot.config import LearningConfig, RiskConfig, Settings
from tradingbot.journal import Journal
from tradingbot.learning import CANDIDATE, Learner
from tradingbot.news.earnings_history import build, detect_from_gaps, profile
from tradingbot.strategies import EarningsDrift, EarningsRunup, StructureTrend, discover, load_library

GOOD = '''
from tradingbot.strategies.base import Param, Strategy

class MyBreakout(Strategy):
    name = "my_breakout"
    title = "My breakout"
    description = "test"
    markets = ("index",)
    def __init__(self):
        super().__init__()
        self.params = {"x": Param(1.0, 0.5, 2.0)}
    def generate(self, ctx):
        return None
'''


def test_library_loads_user_strategies_and_survives_broken_files(tmp_path):
    user = tmp_path / "my_strategies"
    user.mkdir()
    (user / "good.py").write_text(GOOD)
    (user / "broken.py").write_text("this is not python(")
    (user / "TEMPLATE.py").write_text(GOOD.replace("my_breakout", "template_one"))
    (user / "dupe.py").write_text(GOOD.replace("my_breakout", "divergence"))
    classes, problems = discover(user)
    names = [c.name for c in classes]
    assert "my_breakout" in names and "template_one" not in names
    assert names.count("divergence") == 1
    assert any("broken.py" in p for p in problems) and any("duplicate" in p for p in problems)
    mine = next(c for c in classes if c.name == "my_breakout")
    assert mine.author == "you"
    (tmp_path / "strategies.json").write_text(json.dumps({"disabled": ["news_fade", "my_breakout"]}))
    lib = load_library(tmp_path, user)
    assert {"news_fade", "my_breakout"}.isdisjoint({s.name for s in lib}) and len(lib) >= 6


def test_strategies_only_apply_where_they_fit():
    df = synthetic(300, 100, seed=1)
    assert not EarningsRunup().applies(MarketContext("USTECH", df))                 # index: stocks only
    assert not EarningsRunup().applies(MarketContext("NVDA", df))                   # no earnings data
    assert EarningsRunup().applies(MarketContext("NVDA", df, earnings={"hours_to": 30}))
    assert StructureTrend().applies(MarketContext("BTCUSD", df))


def test_challenger_adopted_only_when_clearly_better(tmp_path):
    j = Journal(tmp_path / "l.db", mode="demo")
    learner = Learner(j, LearningConfig(), RiskConfig(), live=False, rng=random.Random(3))
    strat = StructureTrend()
    cand = learner.challenger_for(strat)
    assert cand is not None and cand.name == "structure_trend" + CANDIDATE
    changed = learner.challengers["structure_trend"]["changed"]
    for _ in range(30):
        learner.record({"strategy": "structure_trend", "symbol": "X", "r_multiple": -1.0, "features": {}})
        learner.record({"strategy": cand.name, "symbol": "X", "r_multiple": 3.0 if random.random() < 0.6 else -1.0,
                        "features": {}, "shadow": 1})
    verdict = learner.judge_challenger(strat)
    assert verdict and "improved" in verdict
    for k, v in changed.items():
        assert strat.p(k) == v
    assert "structure_trend" not in learner.challengers
    # live accounts never test tweaks
    live = Learner(Journal(tmp_path / "live.db", mode="live"), LearningConfig(), RiskConfig(), live=True)
    assert live.challenger_for(StructureTrend()) is None


def _daily_with_earnings(drift_up: bool, continue_after: bool, n_quarters: int = 8):
    rng = np.random.default_rng(4)
    n = 63 * n_quarters + 30
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    events = [40 + 63 * q for q in range(n_quarters)]
    for t in events:
        close[t - 6:t] += np.linspace(0, 3 if drift_up else -3, 6)        # run into the release
        gap = 6 * (1 if rng.random() < 0.5 else -1)
        close[t:] += gap
        cont = 2 * np.sign(gap) * (1 if continue_after else -1)
        close[t + 1:t + 6] += np.linspace(0, cont, 5)
        close[t + 6:] += cont
    idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    o = np.r_[close[0], close[:-1]]
    for t in events:   # make the gap show up as an opening gap
        o[t] = close[t] - 0.2
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + 0.3, "low": np.minimum(o, close) - 0.3,
                       "close": close, "volume": 1.0}, index=idx)
    return df, events


def test_earnings_history_from_gaps_and_known_dates():
    df, events = _daily_with_earnings(drift_up=True, continue_after=True)
    found = detect_from_gaps(df)
    assert len(set(found) & set(events)) >= 6
    prof, reactions = build("NVDA", df, [])
    assert prof["n"] >= 6 and prof["pre_up_rate"] >= 0.8 and prof["continuation_rate"] >= 0.8
    known = [(df.index[t - 1].strftime("%Y-%m-%d"), "amc") for t in events]
    prof2, r2 = build("NVDA", df, known)
    assert prof2["n"] == len(events) and prof2["detected_dates"] == 0
    assert profile([])["n"] == 0


def _uptrend_15m(n=300):
    df = synthetic(n, 180, seed=7)
    drift = np.linspace(0, 12, n)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] + drift
    df.iloc[-1, df.columns.get_loc("close")] = df["open"].iloc[-1] + 0.3     # bullish last candle
    df.iloc[-1, df.columns.get_loc("high")] = max(df["high"].iloc[-1], df["close"].iloc[-1])
    return df


def test_earnings_runup_needs_history_and_chart_agreement():
    prof = {"n": 8, "pre_up_rate": 0.875, "continuation_rate": 0.5}
    ctx = MarketContext("NVDA", _uptrend_15m(), earnings={"profile": prof, "hours_to": 40})
    sig = EarningsRunup().generate(ctx)
    assert sig and sig.side == "buy" and sig.stop < sig.entry and "into earnings" in sig.reason
    mixed = dict(prof, pre_up_rate=0.5)
    assert EarningsRunup().generate(MarketContext("NVDA", _uptrend_15m(), earnings={"profile": mixed, "hours_to": 40})) is None
    down_hist = dict(prof, pre_up_rate=0.1)   # history says down but the chart is going up -> no trade
    assert EarningsRunup().generate(MarketContext("NVDA", _uptrend_15m(), earnings={"profile": down_hist, "hours_to": 40})) is None
    assert EarningsRunup().generate(MarketContext("NVDA", _uptrend_15m(), earnings={"profile": prof, "hours_to": 1})) is None


def test_earnings_drift_joins_confirmed_reaction():
    n = 100
    idx = pd.date_range("2026-10-29 14:00", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    t0 = pd.Timestamp("2026-10-30 13:30", tz="UTC")
    after = idx >= t0
    close[after] += 8                                             # gap up on the release
    k = np.argmax(after)
    close[k:k + 4] = close[k] + np.array([0.0, 0.3, -0.2, 0.1])   # first hour range
    close[k + 4:] = close[k] + 0.2 + np.linspace(0, 0.1, n - k - 4)
    o = np.r_[close[0], close[:-1]]
    o[k] = close[k] - 0.1                                         # the release candle opens with the gap
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + 0.05, "low": np.minimum(o, close) - 0.05,
                       "close": close, "volume": 1.0}, index=idx)
    rng_hi = df[(df.index >= t0) & (df.index < t0 + pd.Timedelta(minutes=60))]["high"].max()
    df.iloc[-2, df.columns.get_loc("close")] = rng_hi - 0.01
    df.iloc[-1, df.columns.get_loc("close")] = rng_hi + 0.4       # fresh break of the first-hour range
    df.iloc[-1, df.columns.get_loc("high")] = rng_hi + 0.45
    e = {"profile": {"n": 6, "continuation_rate": 0.83}, "last_reaction_open": t0.isoformat(),
         "hours_since": (df.index[-1] - t0).total_seconds() / 3600, "daily_atr": 3.0}
    sig = EarningsDrift().generate(MarketContext("NVDA", df, earnings=e))
    assert sig and sig.side == "buy"
    e2 = dict(e, profile={"n": 6, "continuation_rate": 0.33})
    assert EarningsDrift().generate(MarketContext("NVDA", df, earnings=e2)) is None


class FakeNews:
    def __init__(self, state):
        self._state = state
        self.events = []

    def refresh(self, now):
        pass

    def state(self, symbol, now):
        return self._state

    def blackout(self, symbol, now, earnings_strategy=False):
        return None

    def must_flatten(self, symbol, now):
        return None

    def hot_symbols(self, now):
        return []


def _agent_with_open_trade(tmp_path, news_state, profit_r):
    data = {"USTECH": synthetic(400, 18000, seed=5)}
    s = Settings()
    s.symbols = ["USTECH"]
    s.mode = "backtest"
    s.db_path = tmp_path / "g.db"
    j = Journal(s.db_path, mode="backtest")
    broker = SimBroker(data, warmup=300, spread_frac=0)
    agent = Agent(s, broker, j, rng=random.Random(1), news=FakeNews(news_state))
    nxt = data["USTECH"].iloc[301]
    risk = 50.0
    entry = float(nxt["close"]) - profit_r * risk           # so the next close shows `profit_r` R of profit
    pid = broker.place_market("USTECH", "buy", 1.0, entry - risk, entry + 3 * risk, "structure_trend")
    tid = j.open_trade(shadow=0, symbol="USTECH", strategy="structure_trend", side="buy",
                       opened_at=broker.now().isoformat(), entry=entry, stop=entry - risk, take_profit=entry + 3 * risk,
                       qty=1.0, risk_amount=1000, risk_pct=0.05, features={}, decision={"filled": True}, broker_id=pid)
    broker.step()
    broker.open[pid]["entry"] = entry
    broker.open[pid]["stop"], broker.open[pid]["tp"] = entry - risk * 5, entry + risk * 10   # keep it open
    return agent, j, broker, tid, pid


BAD_NEWS = {"news_state": "breaking", "news_kind": "geopolitics", "news_dir": -1, "news_impact": 3,
            "news_age_min": 2, "news_title": "Missile strike"}


def test_news_guard_takes_profit_or_protects(tmp_path):
    agent, j, broker, tid, pid = _agent_with_open_trade(tmp_path, BAD_NEWS, profit_r=2.0)
    agent.run_cycle()
    assert pid not in broker.open                                   # profit taken
    assert j.trade(tid)["decision"]["closed_by_agent"].startswith("took profit")
    agent, j, broker, tid, pid = _agent_with_open_trade(tmp_path / "b", BAD_NEWS, profit_r=0.6)
    agent.run_cycle()
    assert pid in broker.open and broker.open[pid]["stop"] == j.trade(tid)["entry"]   # stop to break-even
    good = dict(BAD_NEWS, news_dir=1)
    agent, j, broker, tid, pid = _agent_with_open_trade(tmp_path / "c", good, profit_r=2.0)
    agent.run_cycle()
    assert pid in broker.open and not j.trade(tid)["decision"].get("news_guard")       # news agrees: leave it
