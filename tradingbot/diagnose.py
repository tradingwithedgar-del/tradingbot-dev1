"""`tiim why`: a plain-English summary of what TIIM did recently and why it did or didn't trade."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .config import Settings
from .journal import Journal
from .learning import CANDIDATE

# How a shadow decision's last reason maps to a human explanation.
REASON_GROUPS = [
    ("against its group", "against the market group it moves with (BTC for crypto, the index group for indices/stocks)"),
    ("waiting: ", "waiting for the market leader (BTC) to pick a direction"),
    ("against this week's plan", "against the weekly plan for that market"),
    ("chasing an exhausted move", "the move was already stretched (long 4H streak / far from average) - didn't chase"),
    ("against the higher timeframes", "against the higher-timeframe trend, no confirmed 4H/Daily reversal yet"),
    ("desk veto", "the trading desk vetoed it (still followed virtually to grade the veto)"),
    ("while the desk debated", "price moved while the desk debated"),
    ("desk is already debating", "the desk was busy with another setup on that market"),
    ("not positive", "TIIM's estimate for this setup was not profitable enough (still learning; it explores on demo)"),
    ("learned filter", "TIIM learned this setup loses in these conditions"),
    ("major news points the other way", "major news pointed against the trade"),
    ("news window", "high-impact news release window"),
    ("earnings", "earnings coming up"),
    ("spread", "spread too wide / off-market quote"),
    ("misquote", "quote looked like a misquote/spike"),
    ("stale quote", "price quote was stale"),
    ("stop too tight", "stop would have been too tight"),
    ("already in a real trade", "already had a trade on that symbol"),
    ("too close to the", "too close to a market close (gap protection)"),
    ("max open trades", "already at the maximum number of open trades"),
    ("correlated", "already enough trades in that market group (e.g. several US indices the same way)"),
    ("open risk", "already at the maximum total risk"),
    ("daily loss limit", "daily loss limit reached"),
    ("drawdown", "drawdown limit reached (halted)"),
    ("halted", "TIIM is halted"),
    ("STOP file", "paused with `tiim stop`"),
    ("position too small", "position size below the broker minimum"),
    ("instrument spec failed", "could not read the instrument's contract details"),
    ("order failed", "the broker rejected the order"),
    ("not approved for live", "strategy not approved for the live account"),
]


def _why(reasons: list[str]) -> str:
    last = " ".join(reasons[-2:]) if reasons else ""
    for key, text in REASON_GROUPS:
        if key.lower() in last.lower():
            return text
    return reasons[-1] if reasons else "no reason recorded"


def report(j: Journal, s: Settings, now: pd.Timestamp | None = None, hours: int = 24) -> str:
    now = now or pd.Timestamp.now(tz="UTC")
    since = (now - pd.Timedelta(hours=hours)).isoformat()
    out: list[str] = [f"{s.agent_name} over the last {hours} hours ({s.mode} account)", ""]

    halted = j.get(f"halted:{s.mode}")
    if halted:
        out += [f"!! {halted}" if halted.startswith("HALTED") else f"!! HALTED: {halted}", ""]
    if s.stop_file.exists():
        out.append("!! Paused: new trades are off (`tiim stop`). Run `tiim resume` to turn them back on.")

    last_bar = j.get(f"last_bar:{s.mode}", {}) or {}
    symbols = sorted(last_bar) if [x.upper() for x in s.symbols] == ["ALL"] else list(s.symbols)
    lines, live_n, closed_n, bad = [], 0, 0, []
    for sym in symbols:
        t = last_bar.get(sym)
        if not t:
            bad.append(f"  {sym:<8} never - no price data yet (market closed or symbol problem)")
            continue
        age = (now - pd.Timestamp(t)).total_seconds() / 3600
        flag = "" if age < 1 else ("  (market closed?)" if age < 72 else "  (NOT UPDATING - check `journalctl -u tiim`)")
        live_n += age < 1
        closed_n += 1 <= age < 72
        if len(symbols) <= 20 or age >= 72:
            (bad if age >= 72 else lines).append(f"  {sym:<8} {t[5:16].replace('T', ' ')} UTC, {age:.1f} h ago{flag}")
        elif age >= 72:
            bad.append(f"  {sym:<8} not updating")
    if len(symbols) > 20:
        out.append(f"Markets checked: {len(symbols)} - {live_n} updating now, {closed_n} idle (market closed?), "
                   f"{len(bad)} with no data or not updating")
    else:
        out.append("Markets checked (last candle analysed):")
    out += lines + bad

    rows = j.trades("mode = ? AND opened_at >= ?", (s.mode, since))
    real = [t for t in rows if not t["shadow"]]
    shadow = [t for t in rows if t["shadow"]]
    by_design = [t for t in shadow if t["experimental"] or CANDIDATE in t["strategy"]]
    blocked = [t for t in shadow if t not in by_design]
    out += ["", f"Setups found: {len(rows)}",
            f"  real trades opened:            {len(real)}",
            f"  blocked / passed on (virtual): {len(blocked)}",
            f"  experiments & tweak tests:     {len(by_design)}  (always virtual - that's by design)"]
    for t in real:
        if t["status"] == "closed":
            how = (t.get("decision") or {}).get("close_reason", "")
            res = f"{t['r_multiple']:+.2f}R {t['pnl'] or 0:+,.0f}" + (f" ({how})" if how else "")
        else:
            res = "still open"
        out.append(f"  REAL #{t['id']} {t['opened_at'][5:16].replace('T', ' ')} {t['symbol']} {t['side']} {t['strategy']} "
                   f"risk {(t['risk_pct'] or 0):.1%} -> {res}")
    closed = [t for t in real if t["status"] == "closed"]
    if closed:
        wins = sum(1 for t in closed if (t["r_multiple"] or 0) > 0.05)
        out.append(f"  closed: {len(closed)}, won {wins}, total {sum(t['r_multiple'] or 0 for t in closed):+.1f}R, "
                   f"{sum(t['pnl'] or 0 for t in closed):+,.0f} in account currency")
        by = Counter()
        for t in closed:
            by[t["strategy"]] += t["r_multiple"] or 0
        out.append("  R by strategy: " + ", ".join(f"{k} {v:+.1f}R" for k, v in by.most_common()))
    curve = [e for e in j.equity_curve(s.mode) if e["ts"] >= since]
    if curve:
        lo = min(curve, key=lambda e: e["equity"])
        out.append(f"  equity: {curve[0]['equity']:,.0f} -> {curve[-1]['equity']:,.0f} (lowest {lo['equity']:,.0f}, "
                   f"peak so far {curve[-1]['peak']:,.0f})")
    if blocked:
        out += ["", "Why setups were passed on:"]
        for text, n in Counter(_why((t.get("decision") or {}).get("why", [])) for t in blocked).most_common():
            out.append(f"  {n:>3} x {text}")
        out.append("  by strategy: " + ", ".join(f"{k} {v}" for k, v in Counter(t["strategy"] for t in blocked).most_common()))
    if not rows:
        out += ["", "No setup met any strategy's rules. That's normal on quiet days: TIIM's strategies are",
                "strict on purpose (fresh zone + rejection, confirmed divergence, clean pullback)."]

    errors = [e for e in j.events(500, s.mode) if e["kind"] == "error" and e["ts"] >= since]
    if errors:
        out += ["", f"Errors: {len(errors)} (latest first)"]
        out += [f"  {e['ts'][5:16].replace('T', ' ')} {e['message'][:110]}" for e in errors[:5]]
    desk_rows = [t for t in rows if (t.get("decision") or {}).get("desk")]
    if desk_rows or j.get("desk:brief") or j.get("lab:report"):
        from .desk import scorecard, scorecard_text
        from .lab import summary

        acts = Counter(t["decision"]["desk"]["action"] for t in desk_rows)
        out += ["", f"Trading desk: debated {len(desk_rows)} setups - take {acts.get('take', 0)}, half {acts.get('half', 0)}, "
                    f"skip {acts.get('skip', 0)}, unavailable {acts.get('none', 0)}  (details: `tiim desk`)"]
        out += scorecard_text(scorecard(j, s))
        lab = j.get("lab:report")
        if lab:
            out.append("  " + summary(lab).splitlines()[2])
    lessons = [e for e in j.events(500, s.mode) if e["kind"] in ("lesson", "tuning", "news", "manage", "evolution", "lab") and e["ts"] >= since]
    if lessons:
        out += ["", "Notable:"] + [f"  {e['ts'][5:16].replace('T', ' ')} [{e['kind']}] {e['message'][:100]}" for e in lessons[:8]]
    from .config import branded

    return branded("\n".join(out))


# --- performance autopsy -------------------------------------------------------------------
def _stats(rows: list[dict]) -> dict:
    rs = [float(t["r_multiple"]) for t in rows if t.get("r_multiple") is not None]
    pn = [float(t["pnl"] or 0) for t in rows]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    gp, gl = sum(x for x in pn if x > 0), -sum(x for x in pn if x < 0)
    return {"n": len(rows), "win": len(wins) / len(rs) if rs else 0.0, "avg_r": sum(rs) / len(rs) if rs else 0.0,
            "pnl": sum(pn), "pf": (gp / gl) if gl else float("inf") if gp else 0.0,
            "avg_win": sum(wins) / len(wins) if wins else 0.0, "avg_loss": sum(losses) / len(losses) if losses else 0.0}


def _line(label: str, st: dict) -> str:
    pf = "inf" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
    return (f"  {label[:24]:<24} n={st['n']:<4} win={st['win']:.0%} avgR={st['avg_r']:+.2f} "
            f"pnl={st['pnl']:+.0f} PF={pf}")


def _group(rows: list[dict], key, top: int = 12, worst: bool = True) -> list[str]:
    g: dict[str, list[dict]] = {}
    for t in rows:
        g.setdefault(str(key(t)), []).append(t)
    items = sorted(((k, _stats(v)) for k, v in g.items()), key=lambda kv: kv[1]["pnl"], reverse=not worst)
    return [_line(k, st) for k, st in items[:top]]


def autopsy(j: Journal, s: Settings, days: int = 45) -> str:
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).isoformat()
    out: list[str] = [f"AUTOPSY - {s.mode} account, last {days} days", "=" * 60]
    curve = j.equity_curve(s.mode)
    if curve:
        first, last = curve[0], curve[-1]
        peak = max(r["peak"] or 0 for r in curve)
        lo = min(r["equity"] for r in curve)
        out += [f"Equity: first recorded {first['equity']:.0f} ({first['ts'][:10]}), peak seen {peak:.0f}, "
                f"lowest {lo:.0f}, now {last['equity']:.0f} ({last['ts'][:10]})",
                f"  change since first record {(last['equity'] / first['equity'] - 1):+.1%}; "
                f"from peak {(last['equity'] / peak - 1) if peak else 0:+.1%}; "
                f"worst drawdown the agent itself recorded {max((r['drawdown'] or 0) for r in curve):.1%} "
                f"(halt limit {s.risk.max_drawdown_halt:.0%})"]
    else:
        out.append("No equity history recorded.")
    ev = j.events(2000, s.mode)
    halts = [e for e in ev if e["kind"] == "halt"]
    resumes = [e for e in ev if e["kind"] == "resume"]
    out.append(f"Halts: {len(halts)}   Resumes by you: {len(resumes)}"
               + (f"   (last resume {resumes[0]['ts'][:16]})" if resumes else ""))
    out.append(f"Settings now: risk/trade={s.risk.risk_per_trade:.1%} max open={s.risk.max_open_trades} "
               f"max open risk={s.risk.max_total_open_risk:.0%} same-group={s.risk.max_same_group} "
               f"markets={len(s.symbols)}")

    real = [t for t in j.trades("mode = ? AND shadow = 0 AND status = 'closed' AND closed_at >= ?", (s.mode, since), order="closed_at")]
    shadow = [t for t in j.trades("mode = ? AND shadow = 1 AND status = 'closed' AND closed_at >= ?", (s.mode, since), order="closed_at")]
    opened = j.open_trades(shadow=False)
    out += ["", f"REAL TRADES closed: {len(real)}   still open: {len(opened)}   shadow (paper-only): {len(shadow)}"]
    if not real:
        return "\n".join(out + ["No closed real trades in this window."])
    st = _stats(real)
    out.append(_line("ALL REAL", st))
    out.append(f"  average win {st['avg_win']:+.2f}R, average loss {st['avg_loss']:+.2f}R "
               f"(a 3R design needs only ~30% wins; losses worse than -1R mean slippage/gaps)")
    big = sorted((t for t in real if (t.get("r_multiple") or 0) < -1.5), key=lambda t: t["r_multiple"])[:5]
    if big:
        out.append("  losses worse than -1.5R (gaps/slippage/stop moved):")
        for t in big:
            out.append(f"    #{t['id']} {t['symbol']} {t['side']} {t['strategy']} {t['r_multiple']:+.1f}R "
                       f"{(t['decision'] or {}).get('close_reason', '')} {t['closed_at'][:16]}")
    streak = cur = 0
    for t in real:
        cur = cur + 1 if (t.get("r_multiple") or 0) <= 0 else 0
        streak = max(streak, cur)
    out.append(f"  longest losing streak: {streak}")
    cost = []
    for t in real:
        sp = (t.get("features") or {}).get("spread")
        d = abs((t["entry"] or 0) - (t["stop"] or 0))
        if sp and d:
            cost.append(float(sp) / d)
    if cost:
        out.append(f"  spread cost per trade ~{sum(cost) / len(cost):.2f}R on average (worst {max(cost):.2f}R) - "
                   f"paid on every entry before price moves")
    out += ["", "By strategy (worst first):"] + _group(real, lambda t: t["strategy"])
    out += ["", "Worst markets:"] + _group(real, lambda t: t["symbol"], 10)
    out += ["", "Best markets:"] + _group(real, lambda t: t["symbol"], 5, worst=False)
    out += ["", "By side:"] + _group(real, lambda t: t["side"])
    out += ["", "By session:"] + _group(real, lambda t: (t.get("features") or {}).get("session", "?"))
    out += ["", "By exit:"] + _group(real, lambda t: (t.get("decision") or {}).get("close_reason") or t.get("outcome") or "?")
    out += ["", "By risk % taken:"] + _group(real, lambda t: f"{(t.get('risk_pct') or 0):.1%}")
    out += ["", "By weekday (0=Mon):"] + _group(real, lambda t: (t.get("features") or {}).get("weekday", "?"))
    # clusters: days where many trades lost together (correlated exposure)
    days_: dict[str, list[dict]] = {}
    for t in real:
        days_.setdefault(t["closed_at"][:10], []).append(t)
    bad = sorted(((d, sum(float(x["pnl"] or 0) for x in v), len(v)) for d, v in days_.items()), key=lambda x: x[1])[:5]
    out += ["", "Worst days (money lost in one day, trades closed):"]
    out += [f"  {d}  {p:+.0f}  ({n} trades)" for d, p, n in bad]
    if shadow:
        out += ["", "SHADOW (paper) vs REAL by strategy - does the paper edge exist for real?"]
        names = sorted({t["strategy"] for t in real} | {t["strategy"] for t in shadow})
        for n in names[:15]:
            r_ = _stats([t for t in real if t["strategy"] == n])
            s_ = _stats([t for t in shadow if t["strategy"] == n])
            out.append(f"  {n[:22]:<22} real n={r_['n']:<3} avgR={r_['avg_r']:+.2f} | shadow n={s_['n']:<4} avgR={s_['avg_r']:+.2f}")
    return "\n".join(out)
