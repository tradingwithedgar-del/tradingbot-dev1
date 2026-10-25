"""Command line entry point.

    python -m tiim symbols             # log in and list the exact symbol names on your account
    python -m tiim run                 # trade on the account in .env (demo by default)
    python -m tiim news                # check news feeds, upcoming events and headline reading
    python -m tiim backtest --synthetic
    python -m tiim backtest --csv EURUSD=data/eurusd_15m.csv
    python -m tiim backtest --tradelocker --days 60
    python -m tiim dashboard           # http://localhost:8000
    python -m tiim why                 # what TIIM did in the last 24h and why it did/didn't trade
    python -m tiim desk                # the trading desk: latest brief, debates and report cards
    python -m tiim lab                 # run the Strategy Lab now (it also runs nightly)
    python -m tiim claude              # check that TIIM can reach Claude
    python -m tiim plan [--rebuild]    # this week's plan for every market
    python -m tiim status
    python -m tiim new-agent NAME      # retire this agent, start a new one with the same knowledge
    python -m tiim stop | resume
"""
from __future__ import annotations

import argparse
import logging
import time

from .config import Settings


def resolve_symbols(broker, s) -> None:
    """BOT_SYMBOLS=ALL means every instrument on the account (minus BOT_EXCLUDE)."""
    import os

    if [x.upper() for x in s.symbols] == ["ALL"]:
        skip = {x.strip() for x in os.getenv("BOT_EXCLUDE", "").split(",") if x.strip()}
        s.symbols = [n for n in broker.instrument_names() if n not in skip]
        logging.info("Trading every instrument on the account: %d markets", len(s.symbols))


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
    resolve_symbols(broker, s)
    _require_symbols(broker, s.symbols)
    journal = Journal(s.db_path, mode=s.mode)
    news = earnings = None
    if s.news.enabled:
        from .news import NewsMonitor
        from .news.earnings_history import EarningsDesk

        news = NewsMonitor(s, journal)
        earnings = EarningsDesk(s, broker, journal, news)
        logging.info("News monitor on (headlines read by %s)", news.reader)
    desk = None
    if s.desk.enabled:
        from .desk import Desk
        from .llm import make_llm

        llm = make_llm()
        if llm is None:
            logging.warning("Trading desk is off: no Claude access (set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY)")
        else:
            desk = Desk(s, journal, llm, news=news)
            logging.info("Trading desk on (%s): research=%s advocates=%s judge=%s, up to %d Claude calls a day",
                         llm.name, s.desk.model_research, s.desk.model_advocate, s.desk.model_judge, s.desk.daily_calls)
    agent = Agent(s, broker, journal, news=news, earnings=earnings, desk=desk)
    logging.info("Strategy library: %s", ", ".join(st.name for st in agent.core))
    banner = "LIVE ACCOUNT - REAL MONEY" if s.is_live else "demo account"
    logging.info("%s started on %s | symbols=%s tf=%s risk=%.1f%% target=%.0fR",
                 s.agent_name, banner, s.symbols, s.timeframe, s.risk.risk_per_trade * 100, s.risk.reward_multiple)
    journal.event("start", f"{s.agent_name} started ({banner})")
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


def cmd_desk(args) -> None:
    from .desk import scorecard, scorecard_text
    from .journal import Journal
    from .lab import summary
    from .llm import Budget

    s = Settings()
    j = Journal(args.db or s.db_path, mode=s.mode)
    b = j.get("desk:brief") or {}
    print(f"Trading desk ({'on' if s.desk.enabled else 'off'}), Claude calls today: "
          f"{Budget(j, s.desk.daily_calls).used}/{s.desk.daily_calls}")
    if b:
        print(f"\nResearch brief {b['at'][:16].replace('T', ' ')} UTC: {b.get('overview', '')}")
        for sym, m in (b.get("markets") or {}).items():
            print(f"  {sym:<8} {m['bias']:<7} {m['confidence']:>3}%  {m['drivers'][:90]}")
    rows = [t for t in j.trades("mode = ?", (s.mode,), order="id DESC", limit=300) if (t.get("decision") or {}).get("desk")][:8]
    if rows:
        print("\nRecent debates:")
    for t in rows:
        d = t["decision"]["desk"]
        res = f"{t['r_multiple']:+.2f}R" if t["status"] == "closed" else "open"
        print(f"  #{t['id']} {t['opened_at'][5:16].replace('T', ' ')} {t['symbol']} {t['side']} {t['strategy']}: "
              f"bull {d.get('bull', {}).get('conviction', '-')}/10 vs bear {d.get('bear', {}).get('conviction', '-')}/10 "
              f"-> {d['action'].upper()} ({'virtual' if t['shadow'] else 'real'}, {res})")
        if d.get("reason"):
            print(f"      {d['reason'][:140]}")
    print("\nReport cards:")
    print("\n".join(scorecard_text(scorecard(j, s))))
    print()
    print(summary(j.get("lab:report") or {}))


def cmd_plan(args) -> None:
    from .journal import Journal
    from .plan import text

    s = Settings()
    j = Journal(args.db or s.db_path, mode=s.mode)
    if args.rebuild:
        j.set(f"plan:{s.mode}", None)
        print("Weekly plan cleared: TIIM writes a new one on its next cycle (within a minute).")
        return
    print(text(j.get(f"plan:{s.mode}") or {}))


def cmd_claude(args) -> None:
    """Check TIIM's Claude access with one tiny call."""
    import os

    from .llm import make_llm
    from .news.classify import find_claude_cli

    Settings()
    key, tok = os.getenv("ANTHROPIC_API_KEY", ""), os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
    print(f"ANTHROPIC_API_KEY:        {'set' if key else 'empty'}")
    print(f"CLAUDE_CODE_OAUTH_TOKEN:  {'set (' + tok[:12] + '...)' if tok else 'empty'}")
    print(f"claude command:           {find_claude_cli() or 'NOT FOUND'}")
    llm = make_llm()
    if llm is None:
        print("\nNo Claude access. Put your `claude setup-token` token in .env as CLAUDE_CODE_OAUTH_TOKEN=... (see DEPLOY.md)")
        return
    print(f"\nUsing: {llm.name}. Asking Claude a test question...")
    r = llm.ask("Reply with JSON only.", "Say ok.", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                      "required": ["ok"], "additionalProperties": False}, "haiku")
    print("Claude answered: access works." if r.data is not None else f"Claude call FAILED: {r.error}")


def cmd_lab(args) -> None:
    from .broker.tradelocker_broker import TradeLockerBroker
    from .journal import Journal
    from .lab import run_nightly, summary
    from .llm import make_llm

    s = Settings()
    s.validate()
    j = Journal(s.db_path, mode=s.mode)
    if not s.experiments_allowed:
        raise SystemExit("The Strategy Lab only feeds the demo account.")
    broker = TradeLockerBroker(s)
    resolve_symbols(broker, s)
    report = run_nightly(s, j, broker, llm=make_llm() if s.lab.claude_ideas else None)
    print(summary(report))


def cmd_stop(args) -> None:
    s = Settings()
    s.stop_file.parent.mkdir(parents=True, exist_ok=True)
    s.stop_file.write_text("no new entries\n")
    print(f"created {s.stop_file}: no new real trades will be opened (open trades keep their SL/TP).")


def cmd_new_agent(args) -> None:
    """Hand the account to a new agent under a new name. The knowledge base (trades, learned odds, filters,
    experiments, weekly plan) stays in the journal, so the new agent starts with everything the old one learned.
    A backup of the journal is saved first. Stop the running agent before this, restart it after."""
    import re
    import shutil

    from .config import ROOT
    from .journal import Journal

    name = re.sub(r"[^A-Za-z0-9 _-]", "", args.name).strip()
    if not name:
        raise SystemExit("Give the new agent a name, e.g. `tiim new-agent NOVA`.")
    s = Settings()
    old = s.agent_name
    backups = s.db_path.parent / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    j = Journal(s.db_path, mode=s.mode)
    j.conn.execute("PRAGMA wal_checkpoint(FULL)")
    backup = backups / f"{old.lower()}-{stamp}.db"
    shutil.copy2(s.db_path, backup)
    env = ROOT / ".env"
    text = env.read_text() if env.exists() else ""
    line = f"AGENT_NAME={name}"
    text = re.sub(r"(?m)^AGENT_NAME=.*$", line, text) if re.search(r"(?m)^AGENT_NAME=", text) else text.rstrip("\n") + "\n" + line + "\n"
    env.write_text(text)
    j.event("handover", f"{old} retired; {name} takes over the account with all of {old}'s knowledge "
                        f"(journal backup: {backup.name})")
    print(f"{old} retired. {name} takes over with the same rules, strategy library and everything learned so far.")
    print(f"Backup of {old}'s journal: {backup}")
    print("Start it with: systemctl restart tiim   (and the dashboard: systemctl restart tiim-dashboard)")


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
    dk = sub.add_parser("desk", help="the trading desk: research brief, recent debates, report cards, Strategy Lab")
    dk.add_argument("--db", default=None)
    dk.set_defaults(fn=cmd_desk)
    pl = sub.add_parser("plan", help="this week's plan for every market")
    pl.add_argument("--rebuild", action="store_true", help="throw the plan away and make a new one now")
    pl.add_argument("--db", default=None)
    pl.set_defaults(fn=cmd_plan)
    sub.add_parser("claude", help="check that TIIM can reach Claude (one tiny call)").set_defaults(fn=cmd_claude)
    sub.add_parser("lab", help="run the Strategy Lab now (it also runs nightly)").set_defaults(fn=cmd_lab)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)
    na = sub.add_parser("new-agent", help="retire this agent and start a new one under a new name, same knowledge")
    na.add_argument("name")
    na.set_defaults(fn=cmd_new_agent)
    sub.add_parser("resume").set_defaults(fn=cmd_resume)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
