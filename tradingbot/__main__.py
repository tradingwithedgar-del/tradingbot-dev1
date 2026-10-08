"""Command line entry point.

    python -m tiim symbols             # log in and list the exact symbol names on your account
    python -m tiim run                 # trade on the account in .env (demo by default)
    python -m tiim news                # check news feeds, upcoming events and headline reading
    python -m tiim backtest --synthetic
    python -m tiim backtest --csv EURUSD=data/eurusd_15m.csv
    python -m tiim backtest --tradelocker --days 60
    python -m tiim dashboard           # http://localhost:8000
    python -m tiim why                 # what TIIM did in the last 24h and why it did/didn't trade
    python -m tiim status
    python -m tiim stop | resume
"""
from __future__ import annotations

import argparse
import logging
import time

from .config import Settings


def _require_symbols(broker, symbols: list[str]) -> None:
    missing = broker.check_symbols(symbols)
    if missing:
        lines = [f"These symbols don't exist on your TradeLocker account: {', '.join(missing)}"]
        for sym, similar in missing.items():
            lines.append(f"  {sym}: similar names -> {', '.join(similar) or '(none found)'}")
        lines.append("Fix BOT_SYMBOLS in .env (run `python -m tiim symbols` to see every name).")
        raise SystemExit("\n".join(lines))


def cmd_symbols(args) -> None:
    from .broker.tradelocker_broker import TradeLockerBroker

    s = Settings()
    s.validate()
    broker = TradeLockerBroker(s)
    print(f"Logged in to {s.tl_environment} (account currency {broker._account_ccy}).")
    names = broker.instrument_names()
    if args.search:
        names = [n for n in names if args.search.upper() in n.upper()]
    print(f"{len(names)} instruments" + (f" matching '{args.search}'" if args.search else "") + ":")
    for i in range(0, len(names), 6):
        print("  " + "  ".join(f"{n:<14}" for n in names[i:i + 6]))
    missing = broker.check_symbols(s.symbols)
    print(f"\nBOT_SYMBOLS = {','.join(s.symbols)}")
    if missing:
        for sym, similar in missing.items():
            print(f"  NOT FOUND: {sym}  (similar: {', '.join(similar) or 'none'})")
    else:
        print("  all symbols found - ready to run")


def cmd_news(args) -> None:
    import pandas as pd

    from .journal import Journal
    from .news import NewsMonitor

    s = Settings()
    j = Journal(s.db_path, mode=s.mode)
    mon = NewsMonitor(s, j)
    now = pd.Timestamp.now(tz="UTC")
    if args.reset:
        print(f"Cleared {mon.reset_headlines()} saved headlines - re-reading the latest ones now.")
    print(f"Reading headlines with: {mon.reader}" + ("" if mon.reader == "claude" else
          "\n  Keyword reader: headlines are shown for context only and don't drive trades."
          "\n  Scheduled events still do. Add ANTHROPIC_API_KEY to .env to let Claude judge headlines."))
    mon.refresh(now)
    mon.wait()
    print("\nUpcoming high-impact events (UTC):")
    for e in mon.upcoming(now, hours=24 * 7)[:25]:
        print(f"  {e.time:%a %d %b %H:%M}  {e.title:<40} -> {', '.join(e.symbols)}")
    rows = [r for r in j.news(200, kind="headline") if (r["impact"] or 0) >= s.news.min_impact]
    label = "Market-moving headlines" if mon.reader == "claude" else "Possibly relevant headlines (keyword reader)"
    print(f"\n{label} in the last {s.news.max_headline_age_minutes} min:")
    for r in rows[:20]:
        eff = ", ".join(f"{k} {'up' if v > 0 else 'down'}" for k, v in r["effects"].items()) or "no clear direction"
        print(f"  [{r['impact']}] {r['ts'][11:16]} {r['title'][:90]}\n        -> {eff}")
    if not rows:
        print("  none right now")
    print("\nNews state per symbol:")
    for sym in s.symbols:
        st = mon.state(sym, now)
        print(f"  {sym:<8} {st['news_state']:<15} {st['news_kind']:<14} {st['news_title'][:60]}")


def cmd_strategies(args) -> None:
    from .journal import Journal
    from .learning import EdgeStat
    from .strategies.registry import USER_DIR, disabled, discover

    s = Settings()
    classes, problems = discover()
    off = disabled(s.db_path.parent)
    stats = {}
    if s.db_path.exists():
        state = Journal(s.db_path, mode=s.mode).get(f"learner:{s.mode}") or {}
        stats = {k: EdgeStat(**v) for k, v in state.get("stats", {}).items()}
    print(f"TIIM strategy library ({len(classes)} strategies; yours go in {USER_DIR}):\n")
    for cls in classes:
        st = cls()
        e = stats.get(st.name)
        rec = f"{e.count} trades, {e.expected_r(e.p_mean(s.learning), s.risk.reward_multiple):+.2f}R/trade" if e else "no trades yet"
        flag = "off      " if st.name in off else ("live ok  " if st.name in s.live_approved_strategies else "demo only")
        print(f"  [{flag}] {st.name:<18} {st.title} - by {st.author} - {', '.join(st.markets)}")
        print(f"         {st.description}")
        print(f"         {rec}\n")
    for p in problems:
        print(f"  problem: {p}")
    print("Switch one off: add its name to data/strategies.json -> {\"disabled\": [\"name\"]}")
    print("Allow one on the live account: add it to LIVE_APPROVED_STRATEGIES in .env")


def cmd_run(args) -> None:
    from .agent import Agent
    from .broker.tradelocker_broker import TradeLockerBroker
    from .journal import Journal

    s = Settings()
    s.validate()
    if s.mode == "backtest":
        raise SystemExit("Use `python -m tiim backtest` for backtests.")
    broker = TradeLockerBroker(s)
    _require_symbols(broker, s.symbols)
    journal = Journal(s.db_path, mode=s.mode)
    news = earnings = None
    if s.news.enabled:
        from .news import NewsMonitor
        from .news.earnings_history import EarningsDesk

        news = NewsMonitor(s, journal)
        earnings = EarningsDesk(s, broker, journal, news)
        logging.info("News monitor on (headlines read by %s)", news.reader)
    agent = Agent(s, broker, journal, news=news, earnings=earnings)
    logging.info("Strategy library: %s", ", ".join(st.name for st in agent.core))
    banner = "LIVE ACCOUNT - REAL MONEY" if s.is_live else "demo account"
    logging.info("TIIM started on %s | symbols=%s tf=%s risk=%.1f%% target=%.0fR",
                 banner, s.symbols, s.timeframe, s.risk.risk_per_trade * 100, s.risk.reward_multiple)
    journal.event("start", f"TIIM started ({banner})")
    while True:
        try:
            agent.run_cycle()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.exception("cycle failed")
            journal.event("error", f"cycle failed: {e}")
        if agent.halted:
            logging.error(agent.halted)
        time.sleep(s.poll_seconds)


def cmd_backtest(args) -> None:
    from .backtest import load_csv, run_backtest, synthetic

    s = Settings()
    data, specs = {}, {}
    if args.synthetic:
        data = {"SYNTH_A": synthetic(args.bars, 1.10, seed=11), "SYNTH_B": synthetic(args.bars, 1.30, seed=23)}
    for item in args.csv or []:
        sym, path = item.split("=", 1)
        data[sym] = load_csv(path)
    if args.tradelocker:
        from .broker.tradelocker_broker import TradeLockerBroker

        s.mode = "demo"
        broker = TradeLockerBroker(s)
        from .broker.base import TIMEFRAME_SECONDS

        _require_symbols(broker, s.symbols)
        n = int(args.days * 86400 / TIMEFRAME_SECONDS[s.timeframe])
        for sym in s.symbols:
            data[sym] = broker.get_bars(sym, s.timeframe, n, max_days=args.days)
            specs[sym] = broker.spec(sym)
            print(f"  {sym}: {len(data[sym])} bars")
    if not data:
        raise SystemExit("Give --synthetic, --csv SYMBOL=path or --tradelocker")
    j = run_backtest(data, s, db_path=args.db or s.db_path.parent / "backtest.db", seed=args.seed, progress=True, specs=specs)
    closed = j.trades("status='closed' AND shadow=0 AND mode='backtest'")
    wins = sum(1 for t in closed if (t["r_multiple"] or 0) > 0)
    total_r = sum(t["r_multiple"] or 0 for t in closed)
    print(f"real trades: {len(closed)}  win rate: {wins / max(1, len(closed)):.1%}  total: {total_r:+.1f}R")
    print("open the dashboard with: python -m tiim dashboard --db data/backtest.db")


def cmd_dashboard(args) -> None:
    import uvicorn

    from .dashboard.app import create_app

    uvicorn.run(create_app(args.db), host=args.host, port=args.port)


def cmd_status(args) -> None:
    from .journal import Journal

    s = Settings()
    j = Journal(args.db or s.db_path, mode=s.mode)
    print(f"mode={s.mode} halted={j.get(f'halted:{s.mode}')} stop_file={s.stop_file.exists()}")
    for t in j.open_trades(shadow=False):
        print(f"  open #{t['id']} {t['symbol']} {t['side']} {t['strategy']} entry={t['entry']} sl={t['stop']} tp={t['take_profit']}")
    for e in j.events(10):
        print(f"  {e['ts']} [{e['kind']}] {e['message']}")


def cmd_why(args) -> None:
    from .diagnose import report
    from .journal import Journal

    s = Settings()
    print(report(Journal(args.db or s.db_path, mode=s.mode), s, hours=args.hours))


def cmd_stop(args) -> None:
    s = Settings()
    s.stop_file.parent.mkdir(parents=True, exist_ok=True)
    s.stop_file.write_text("no new entries\n")
    print(f"created {s.stop_file}: no new real trades will be opened (open trades keep their SL/TP).")


def cmd_resume(args) -> None:
    from .journal import Journal

    s = Settings()
    if s.stop_file.exists():
        s.stop_file.unlink()
    j = Journal(s.db_path, mode=s.mode)
    j.set(f"halted:{s.mode}", None)
    j.set(f"peak:{s.mode}", 0.0)  # drawdown is measured from the current equity again
    j.event("resume", "human resumed the agent")
    print("resumed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="tiim", description="TIIM - self-learning trading agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run").set_defaults(fn=cmd_run)
    sub.add_parser("strategies", help="list TIIM's strategy library").set_defaults(fn=cmd_strategies)
    nw = sub.add_parser("news", help="check the news feeds, calendar and how headlines are read")
    nw.add_argument("--reset", action="store_true", help="forget saved headlines and re-read them")
    nw.set_defaults(fn=cmd_news)
    sy = sub.add_parser("symbols", help="log in and list instrument names")
    sy.add_argument("--search", default="")
    sy.set_defaults(fn=cmd_symbols)
    b = sub.add_parser("backtest")
    b.add_argument("--synthetic", action="store_true")
    b.add_argument("--bars", type=int, default=3000)
    b.add_argument("--csv", action="append", help="SYMBOL=path.csv (repeatable)")
    b.add_argument("--tradelocker", action="store_true", help="download history from TradeLocker")
    b.add_argument("--days", type=int, default=60)
    b.add_argument("--db", default=None)
    b.add_argument("--seed", type=int, default=1)
    b.set_defaults(fn=cmd_backtest)
    d = sub.add_parser("dashboard")
    d.add_argument("--db", default=None)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.set_defaults(fn=cmd_dashboard)
    st = sub.add_parser("status")
    st.add_argument("--db", default=None)
    st.set_defaults(fn=cmd_status)
    w = sub.add_parser("why", help="what TIIM did recently and why it did or didn't trade")
    w.add_argument("--hours", type=int, default=24)
    w.add_argument("--db", default=None)
    w.set_defaults(fn=cmd_why)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    sub.add_parser("resume").set_defaults(fn=cmd_resume)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
