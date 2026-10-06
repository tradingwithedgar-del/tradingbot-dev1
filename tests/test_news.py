import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tradingbot.agent import Agent
from tradingbot.analysis.context import MarketContext
from tradingbot.analysis.view import chart_view
from tradingbot.backtest import synthetic
from tradingbot.broker.sim import SimBroker
from tradingbot.config import Settings
from tradingbot.journal import Journal
from tradingbot.news.assets import asset_class
from tradingbot.news.calendar import ScheduledEvent, parse_calendar
from tradingbot.news.classify import ClaudeClassifier, KeywordClassifier
from tradingbot.news.headlines import Headline, parse_rss
from tradingbot.news.monitor import NewsMonitor
from tradingbot.strategies import NewsFade, NewsMomentum

SYMS = ["US30", "US500", "USTECH", "XAUUSD", "NVDA", "AAPL", "TSLA", "XTIUSD", "BTCUSD", "ETHUSD", "SOLUSD"]
NOW = pd.Timestamp("2026-10-07 12:00", tz="UTC")


def test_asset_classes():
    assert [asset_class(s) for s in ["USTECH", "XAUUSD", "XTIUSD", "BTCUSD", "NVDA", "EURUSD"]] == \
        ["index", "gold", "oil", "crypto", "stock", "fx"]


def test_parse_calendar_keeps_high_impact_usd():
    rows = [
        {"title": "CPI m/m", "country": "USD", "date": "2026-10-07T08:30:00-04:00", "impact": "High"},
        {"title": "Crude Oil Inventories", "country": "USD", "date": "2026-10-07T10:30:00-04:00", "impact": "Medium"},
        {"title": "German PMI", "country": "EUR", "date": "2026-10-07T03:30:00-04:00", "impact": "High"},
        {"title": "Retail Sales", "country": "USD", "date": "2026-10-07T08:30:00-04:00", "impact": "Low"},
    ]
    ev = parse_calendar(rows, SYMS)
    assert [e.kind for e in ev] == ["cpi", "oil_inventory"]
    assert ev[0].time == pd.Timestamp("2026-10-07 12:30", tz="UTC")
    assert "USTECH" in ev[0].symbols and "BTCUSD" in ev[0].symbols
    assert ev[1].symbols == ["XTIUSD"]


def test_parse_rss_google_style():
    xml = """<rss><channel><item><title>Missile strike hits oil facility - Reuters</title>
      <link>https://x/1</link><pubDate>Wed, 07 Oct 2026 11:50:00 GMT</pubDate><source url="https://reuters.com">Reuters</source>
      </item><item><title>no date</title></item></channel></rss>"""
    h = parse_rss(xml, "https://news.google.com/rss")
    assert len(h) == 1 and h[0].title == "Missile strike hits oil facility" and h[0].source == "Reuters"
    assert h[0].time == pd.Timestamp("2026-10-07 11:50", tz="UTC")


def test_keyword_classifier_risk_off():
    a = KeywordClassifier().classify([Headline(NOW, "Russia launches missile strike on Kyiv", "x", "")], SYMS)[0]
    assert a.impact == 2 and a.effects["USTECH"] == -1 and a.effects["XAUUSD"] == 1 and a.effects["XTIUSD"] == 1


def test_keyword_classifier_ignores_non_market_headlines():
    noise = ["Gears of War: E-Day Launches October 6, 2026 on Xbox, Steam, and Game Pass",
             "Ohio man accused of stealing gun, shooting two people during home invasion",
             "Cold War memories stir Rubio in Iceland as he champions diplomacy",
             "Why has war returned to Ethiopia?",
             "Homeowner shoots intruder during Upstate NY home invasion"]
    res = KeywordClassifier().classify([Headline(NOW, t, "x", "") for t in noise], SYMS)
    assert all(a.impact == 0 and not a.effects for a in res)
    real = KeywordClassifier().classify([Headline(NOW, "Houthis claim missile attack on Abha airport in Saudi Arabia", "x", "")], SYMS)[0]
    assert real.impact == 2 and real.effects["XTIUSD"] == 1


def test_claude_classifier_parses_structured_output():
    payload = {"items": [{"index": 0, "impact": 3, "category": "central_bank", "summary": "Fed surprise cut",
                          "effects": [{"symbol": "USTECH", "direction": "up"}, {"symbol": "FAKE", "direction": "up"}]}]}
    fake = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(payload))])
    c = ClaudeClassifier.__new__(ClaudeClassifier)
    import anthropic

    c.anthropic, c.model, c.fallback = anthropic, "claude-opus-5-5", KeywordClassifier()
    c.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: fake)))
    a = c.classify([Headline(NOW, "Fed cuts rates in emergency move", "x", "")], SYMS)[0]
    assert a.impact == 3 and a.effects == {"USTECH": 1} and a.source == "claude"


def make_monitor(tmp_path, events=(), headlines=()):
    s = Settings()
    s.symbols = SYMS
    s.db_path = tmp_path / "n.db"
    j = Journal(s.db_path, mode="demo")
    reader = KeywordClassifier()
    reader.name = "claude"   # behave like the Claude reader so headlines can drive decisions
    m = NewsMonitor(s, j, classifier=reader, fetch_calendar_fn=lambda syms: list(events),
                    fetch_headlines_fn=lambda feeds, age, now: list(headlines), load_earnings_fn=lambda syms, d: [])
    return m, j


def test_monitor_states_blackout_and_flatten(tmp_path):
    cpi = ScheduledEvent(NOW + pd.Timedelta(minutes=10), "CPI m/m", "cpi", 3, [s for s in SYMS if s != "XTIUSD"])
    earn = ScheduledEvent(NOW + pd.Timedelta(minutes=30), "NVDA earnings", "earnings", 3, ["NVDA"], "earnings")
    war = Headline(NOW - pd.Timedelta(minutes=5), "Iran missile strike on tanker in Strait of Hormuz", "Reuters", "")
    m, j = make_monitor(tmp_path, [cpi, earn], [war])
    m.refresh(NOW)
    assert m.state("USTECH", NOW)["news_state"] == "breaking"       # headline outranks the upcoming release
    assert m.state("USTECH", NOW)["news_dir"] == -1
    assert m.state("XTIUSD", NOW)["news_dir"] == 1
    assert "news window" in m.blackout("US500", NOW)
    assert m.blackout("XTIUSD", NOW) is None
    assert m.must_flatten("NVDA", NOW) and not m.must_flatten("AAPL", NOW)
    later = NOW + pd.Timedelta(minutes=20)
    assert m.state("US500", later)["news_state"] in ("post_event", "breaking")
    assert "USTECH" in m.hot_symbols(NOW)
    assert j.news(10, kind="headline")[0]["impact"] == 2
    assert any(e["kind"] == "news" for e in j.events())


def _impulse_ctx(news, up=True):
    df = synthetic(120, 18000, seed=1)
    last = df.iloc[-1].copy()
    a = float((df["high"] - df["low"]).iloc[-20:].mean())
    o = float(df["close"].iloc[-2])
    c = o + (4 * a if up else -4 * a)
    df.iloc[-1] = [o, max(o, c) + a * 0.1, min(o, c) - a * 0.1, c, last["volume"]]
    return MarketContext("USTECH", df, news=news)


def test_news_momentum_rides_confirmed_move_only():
    news = {"news_state": "breaking", "news_kind": "central_bank", "news_dir": 1, "news_impact": 3, "news_age_min": 5,
            "news_title": "Fed cuts"}
    sig = NewsMomentum().generate(_impulse_ctx(news, up=True))
    assert sig and sig.side == "buy" and sig.stop < sig.entry
    assert NewsMomentum().generate(_impulse_ctx(news, up=False)) is None          # contradicts the headline
    assert NewsMomentum().generate(_impulse_ctx({"news_state": "clear"}, up=True)) is None


def test_news_fade_needs_spike_and_rejection():
    df = synthetic(120, 18000, seed=2)
    a = float((df["high"] - df["low"]).iloc[-30:].mean())
    base = float(df["close"].iloc[-5])
    df.iloc[-4] = [base, base + 4 * a, base - 0.1 * a, base + 3.8 * a, 1]
    df.iloc[-3] = [base + 3.8 * a, base + 4.5 * a, base + 3.5 * a, base + 4.3 * a, 1]
    df.iloc[-2] = [base + 4.3 * a, base + 4.4 * a, base + 3.9 * a, base + 4.0 * a, 1]
    df.iloc[-1] = [base + 4.0 * a, base + 4.1 * a, base + 2.0 * a, base + 2.2 * a, 1]
    news = {"news_state": "post_event", "news_kind": "cpi", "news_dir": 0, "news_impact": 3, "news_age_min": 20}
    sig = NewsFade().generate(MarketContext("USTECH", df, news=news))
    assert sig and sig.side == "sell" and sig.stop > base + 4.5 * a


def test_chart_view_is_json_ready():
    v = chart_view(MarketContext("USTECH", synthetic(400, 18000, seed=3)))
    json.dumps(v)
    assert len(v["t"]) == 160 and len(v["rsi"]) == 160 and v["read"]["trend"]


def test_manual_close_is_not_blamed_on_strategy(tmp_path):
    data = {"A": synthetic(400, 1.1, seed=5)}
    s = Settings()
    s.symbols = ["A"]
    s.db_path = tmp_path / "m.db"
    j = Journal(s.db_path, mode="demo")
    broker = SimBroker(data, warmup=300)
    agent = Agent(s, broker, j, rng=random.Random(1))
    entry = float(data["A"]["close"].iloc[300])
    pid = broker.place_market("A", "buy", 1.0, entry - 0.01, entry + 0.03, "structure_trend")
    tid = j.open_trade(shadow=0, symbol="A", strategy="structure_trend", side="buy", opened_at=broker.now().isoformat(),
                       entry=entry, stop=entry - 0.01, take_profit=entry + 0.03, qty=1.0, risk_amount=1000, risk_pct=0.05,
                       features={"regime": "x"}, decision={}, broker_id=pid)
    broker.step()
    before = learner_count = agent.learner.stats.get("structure_trend")
    broker.close_position(pid)                      # "you" close it mid-way
    agent.run_cycle()
    t = j.trade(tid)
    assert t["status"] == "closed" and "closed_manually" in t["postmortem"]["tags"]
    assert agent.learner.stats.get("structure_trend") is before is learner_count   # strategy stats untouched
    cont = j.trades("shadow=1 AND strategy='structure_trend'")
    assert cont and cont[0]["decision"]["continues"] == tid
    assert np.isclose(cont[0]["stop"], entry - 0.01)


def test_keyword_reader_never_puts_symbols_in_breaking_mode(tmp_path):
    s = Settings()
    s.symbols = SYMS
    s.db_path = tmp_path / "k.db"
    j = Journal(s.db_path, mode="demo")
    war = Headline(NOW - pd.Timedelta(minutes=5), "Houthis claim missile attack on Saudi oil terminal", "Reuters", "")
    m = NewsMonitor(s, j, classifier=KeywordClassifier(), fetch_calendar_fn=lambda syms: [],
                    fetch_headlines_fn=lambda f, a, n: [war], load_earnings_fn=lambda syms, d: [])
    m.refresh(NOW)
    assert m.state("XTIUSD", NOW)["news_state"] == "clear" and not m.hot_symbols(NOW)
    assert j.news(10, kind="headline")[0]["impact"] == 2          # still recorded for context
    assert m.reset_headlines() == 1 and not j.news(10, kind="headline")
