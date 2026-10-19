"""The trading desk: Claude agents that weigh each setup before TIIM risks money on it.

    Research Analyst  - every couple of hours (and on major news) writes a macro brief per market
    Bull advocate     - argues the strongest case to BUY the setup
    Bear advocate     - argues the strongest case to SELL / stay out
    (one rebuttal round each, when the daily budget allows)
    TIIM, the judge   - take, skip or half size, with a reason
    Scorekeeper       - grades the desk on real outcomes; if its vetoes cost money, it loses veto power

The judge can only take, skip or halve a trade. Stops, targets, sizing limits, daily/drawdown limits,
gap protection and the broker rules stay with the Risk Officer (risk.py, compliance.py, sessions.py),
which runs after the judge and can't be overruled.

Claude calls run on a single background worker so the trading loop never waits on them.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from .analysis import htf
from .config import Settings
from .journal import Journal
from .llm import Budget

log = logging.getLogger(__name__)

CALLS_FULL, CALLS_SHORT = 5, 3     # with / without the rebuttal round

PHILOSOPHY = ("TIIM trades price action first, top-down: the Monthly, Weekly, Daily and 4H structure and zones "
              "decide the direction, the 15-minute chart only times the entry. It reads supply/demand zones, divergences "
              "and market structure (HH/HL, LH/LL). Fundamentals and news only reinforce or warn; they never create a trade on their own. "
              "Every trade has its stop decided before entry: 'how much am I willing to lose to be wrong'.")

# --- schemas --------------------------------------------------------------------------------
BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "markets": {"type": "array", "items": {
            "type": "object",
            "properties": {"symbol": {"type": "string"},
                           "bias": {"type": "string", "enum": ["bull", "bear", "neutral"]},
                           "confidence": {"type": "integer"},
                           "drivers": {"type": "string"}, "risks": {"type": "string"}},
            "required": ["symbol", "bias", "confidence", "drivers", "risks"], "additionalProperties": False}},
        "overview": {"type": "string"},
    },
    "required": ["markets", "overview"], "additionalProperties": False,
}
CASE_SCHEMA = {
    "type": "object",
    "properties": {"case": {"type": "string"}, "conviction": {"type": "integer"}, "key_risk": {"type": "string"}},
    "required": ["case", "conviction", "key_risk"], "additionalProperties": False,
}
REBUTTAL_SCHEMA = {
    "type": "object",
    "properties": {"rebuttal": {"type": "string"}, "conviction": {"type": "integer"}},
    "required": ["rebuttal", "conviction"], "additionalProperties": False,
}
WEEKLY_SCHEMA = {
    "type": "object",
    "properties": {
        "overview": {"type": "string"},
        "markets": {"type": "array", "items": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}, "priority": {"type": "integer"}, "comment": {"type": "string"}},
            "required": ["symbol", "priority", "comment"], "additionalProperties": False}},
    },
    "required": ["overview", "markets"], "additionalProperties": False,
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["take", "half", "skip"]},
                   "confidence": {"type": "integer"}, "reason": {"type": "string"}},
    "required": ["action", "confidence", "reason"], "additionalProperties": False,
}


@dataclass
class Verdict:
    action: str                      # take | half | skip | none (desk unavailable: TIIM decides alone)
    reason: str = ""
    confidence: int = 0
    bull: dict = field(default_factory=dict)
    bear: dict = field(default_factory=dict)
    brief: dict = field(default_factory=dict)
    calls: int = 0
    at: str = ""

    def to_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason, "confidence": self.confidence, "bull": self.bull,
                "bear": self.bear, "brief": self.brief, "calls": self.calls, "at": self.at}


@dataclass
class Pending:
    sig: object
    ctx: object
    dec: dict
    future: Future


def _clip(n, lo=0, hi=10) -> int:
    try:
        return max(lo, min(hi, int(n)))
    except (TypeError, ValueError):
        return lo


def dossier(sig, ctx, dec: dict, brief: dict | None, events: list[str], plan: dict | None = None) -> str:
    """Everything the desk is allowed to know about one setup, in plain text."""
    f = sig.features
    dist = abs(sig.entry - sig.stop) or 1e-12
    rr = abs(sig.take_profit - sig.entry) / dist
    closes = [round(float(x), 5) for x in ctx.df["close"].iloc[-24:]]
    lines = [
        f"Market: {sig.symbol}, 15-minute chart, candle closing {ctx.time:%Y-%m-%d %H:%M} UTC.",
        f"Setup found by the Technical Analyst: {sig.side.upper()} via strategy '{sig.strategy}' - {sig.reason}.",
        f"Plan: entry {sig.entry:.5g}, stop {sig.stop:.5g}, target {sig.take_profit:.5g} ({rr:.1f}R), ATR {ctx.atr_now:.5g}"
        + (f", closes after {sig.max_bars} candles if neither is hit" if sig.max_bars else "") + ".",
        f"Chart context: structure {f.get('structure')} ({f.get('structure_seq', '')}), EMA50/200 bias {f.get('bias')}, "
        f"regime {f.get('regime')} / {f.get('trend_strength')} / volatility {f.get('volatility')}, RSI {f.get('rsi')}, "
        f"session {f.get('session')}, strategies agreeing on this side: {f.get('confluence', 0)}.",
        f"Last 24 closes (oldest first): {closes}",
        f"Top-down read: {htf.describe(ctx.htf)}. Higher-timeframe zone at price: {f.get('htf_zone', 'none')}; "
        f"nearest higher-timeframe zone in the trade's way: "
        + (f"{f['htf_room_r']}R away" if f.get("htf_room_r") is not None else "none") + ".",
        f"Reversal candlestick patterns: 15m {'with the trade: ' + f.get('candle', 'none') if f.get('candle_agree') == 'with' else 'against the trade' if f.get('candle_agree') == 'against' else 'none'}; "
        f"higher timeframe: {f.get('htf_candle', 'none')} ({f.get('htf_candle_agree', '-')} the trade).",
        f"Trend TIIM trades with: {f.get('htf_structure', 'unknown')}; this trade is {f.get('htf_align', 'unknown')} it"
        + (f" - confirmed reversal: {f['htf_reversal']}" if f.get("htf_reversal") else "") + ".",
        f"This week's plan for {sig.symbol}: {(plan or {}).get('summary', 'none')}",
        f"How stretched the move already is in the trade's direction: {f.get('extended', 'unknown')} "
        f"({htf.extension_text(f, sig.side)}).",
        f"TIIM's learned odds for this setup: estimated win rate {dec.get('p', 0):.0%}, expected {dec.get('expected_r', 0):+.2f}R "
        f"per trade (statistic '{dec.get('stat_key')}').",
        f"News state: {f.get('news_state', 'clear')}"
        + (f" - '{f.get('news_title')}' (impact {f.get('news_impact')}, pushes price {'up' if (f.get('news_dir') or 0) > 0 else 'down' if (f.get('news_dir') or 0) < 0 else 'nowhere clear'})"
           if f.get("news_title") else "") + ".",
    ]
    if brief:
        lines.append(f"Research Analyst brief ({brief.get('at', '')[:16]} UTC): bias {brief.get('bias')} "
                     f"(confidence {brief.get('confidence')}%). Drivers: {brief.get('drivers')}. Risks: {brief.get('risks')}.")
    else:
        lines.append("Research Analyst brief: none available right now.")
    if events:
        lines.append("Scheduled high-impact events in the next 6 hours: " + "; ".join(events))
    return "\n".join(lines)


class Desk:
    def __init__(self, settings: Settings, journal: Journal, llm, news=None, background: bool = True) -> None:
        self.s, self.j, self.llm, self.news = settings, journal, llm, news
        self.cfg = settings.desk
        self.budget = Budget(journal, self.cfg.daily_calls, key="desk")
        self.background = background
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="desk") if background else None
        self.pending: dict[tuple, Pending] = {}
        self._research: Future | None = None
        self._lock = threading.Lock()

    # --- availability ----------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.cfg.enabled and self.llm is not None and self.budget.remaining >= CALLS_SHORT

    def busy_with(self, symbol: str) -> bool:
        return any(k[0] == symbol for k in self.pending)

    def _run(self, fn, *args) -> Future:
        if self._pool is not None:
            return self._pool.submit(fn, *args)
        fut: Future = Future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:  # pragma: no cover
            fut.set_exception(e)
        return fut

    # --- research analyst ------------------------------------------------------------------
    def brief_for(self, symbol: str, now: pd.Timestamp | None = None, max_age_hours: float = 6) -> dict | None:
        b = self.j.get("desk:brief") or {}
        m = (b.get("markets") or {}).get(symbol)
        if not m:
            return None
        now = now or pd.Timestamp.now(tz="UTC")
        if (now - pd.Timestamp(b["at"])).total_seconds() > max_age_hours * 3600:
            return None
        return dict(m, at=b["at"])

    def research_due(self, now: pd.Timestamp) -> bool:
        if not (self.cfg.enabled and self.llm is not None) or (self._research and not self._research.done()):
            return False
        if self.budget.remaining < 10:          # keep the last calls for actual setups
            return False
        b = self.j.get("desk:brief") or {}
        if not b.get("at"):
            return True
        age_min = (now - pd.Timestamp(b["at"])).total_seconds() / 60
        if age_min >= self.cfg.research_every_minutes:
            return True
        major = self.j.news(limit=5, since=b["at"])
        return age_min >= 20 and any((n.get("impact") or 0) >= 3 for n in major)

    def research_tick(self, now: pd.Timestamp, snapshot: dict[str, dict]) -> bool:
        if not snapshot or not self.research_due(now):
            return False
        self._research = self._run(self._do_research, now, snapshot)
        return True

    def _events(self, now: pd.Timestamp, hours: int, symbol: str | None = None) -> list[str]:
        if not self.news:
            return []
        out = []
        for e in self.news.upcoming(now, hours):
            if e.impact >= 3 and e.time >= now and (symbol is None or not e.symbols or symbol in e.symbols):
                out.append(f"{e.time:%a %H:%M} UTC {e.title}")
        return out[:8]

    def _do_research(self, now: pd.Timestamp, snapshot: dict[str, dict]) -> dict | None:
        if not self.budget.take():
            return None
        heads = self.j.news(limit=80, kind="headline", since=(now - pd.Timedelta(hours=12)).isoformat())
        heads.sort(key=lambda n: (n.get("impact") or 0, n["ts"]), reverse=True)
        head_lines = [f"[{n['ts'][5:16]}] (impact {n.get('impact') or 0}) {n['title']}" for n in heads[:30]]
        mkt = [f"{s}: {v['price']:.5g}, 1 day {v['chg_1d']:+.2f}%, 5 days {v['chg_5d']:+.2f}%, structure {v['trend']}"
               for s, v in snapshot.items()]
        prompt = (
            f"Time: {now:%Y-%m-%d %H:%M} UTC.\nMarkets TIIM trades (USTECH = Nasdaq 100, US500 = S&P 500, US30 = Dow, "
            f"XAUUSD = gold, XTIUSD = WTI oil):\n" + "\n".join(mkt) +
            "\n\nHeadlines from the last 12 hours (most important first):\n" + ("\n".join(head_lines) or "none") +
            "\n\nScheduled high-impact events in the next 24 hours:\n" + ("\n".join(self._events(now, 24)) or "none") +
            "\n\nFor EACH market give the fundamental/macro bias for the next 1-2 days (bull, bear or neutral), "
            "confidence 0-100, the main drivers and the main risks, one short sentence each. Use neutral with low "
            "confidence when the news doesn't clearly point one way. Then a two-sentence overview.")
        reply = self.llm.ask("You are the research analyst on a trading desk. " + PHILOSOPHY +
                             " Your job is the fundamental backdrop only. Be specific and honest about uncertainty.",
                             prompt, BRIEF_SCHEMA, self.cfg.model_research)
        if reply.data is None:
            log.warning("research brief failed: %s", reply.error)
            return None
        markets = {}
        for m in reply.data.get("markets", []):
            if m.get("symbol") in snapshot:
                markets[m["symbol"]] = {"bias": m["bias"], "confidence": _clip(m.get("confidence"), 0, 100),
                                        "drivers": str(m.get("drivers", ""))[:300], "risks": str(m.get("risks", ""))[:300]}
        brief = {"at": now.isoformat(), "overview": str(reply.data.get("overview", ""))[:600], "markets": markets}
        self.j.set("desk:brief", brief)
        self.j.event("desk", f"Research brief: {brief['overview'][:160]}")
        return brief

    # --- weekly briefing ------------------------------------------------------------------
    def weekly_briefing(self, week: str, now: pd.Timestamp) -> None:
        self._run(self._do_briefing, week, now)

    def _do_briefing(self, week: str, now: pd.Timestamp) -> None:
        book = self.j.get(f"plan:{self.s.mode}") or {}
        if book.get("week") != week or not self.budget.take():
            return
        lines = [f"{s}: {p['summary']} Frames {p['frames']}. Scenarios: " + " | ".join(x["text"] for x in p["scenarios"])
                 for s, p in sorted(book["markets"].items())]
        b = self.j.get("desk:brief") or {}
        prompt = (f"Week {week}. TIIM's mechanical top-down plan for each market:\n" + "\n".join(lines) +
                  f"\n\nLatest fundamental brief: {b.get('overview', 'none')}\n"
                  "Scheduled high-impact events this week:\n" + ("\n".join(self._events(now, 24 * 6)) or "none") +
                  "\n\nWrite the weekly briefing: a 3-4 sentence overview of the week, then for each market one "
                  "sentence: what to focus on, and any event or risk that should make TIIM careful. Priority 1 = best "
                  "opportunity this week, 3 = least. Do not change the plan's direction; comment on it.")
        r = self.llm.ask("You are the head of a trading desk writing the weekly plan briefing. " + PHILOSOPHY,
                         prompt, WEEKLY_SCHEMA, self.cfg.model_research)
        if r.data is None:
            log.warning("weekly briefing failed: %s", r.error)
            return
        book = self.j.get(f"plan:{self.s.mode}") or {}
        if book.get("week") != week:
            return
        book["briefing"] = str(r.data.get("overview", ""))[:900]
        book["notes"] = {m["symbol"]: f"(priority {m.get('priority', 2)}) {str(m.get('comment', ''))[:300]}"
                         for m in r.data.get("markets", []) if m.get("symbol") in book["markets"]}
        self.j.set(f"plan:{self.s.mode}", book)
        self.j.event("plan", f"Weekly briefing {week}: {book['briefing'][:160]}")

    # --- debate + judge --------------------------------------------------------------------
    def submit(self, sig, ctx, dec: dict) -> bool:
        """Queue a setup for the desk. False when the desk can't take it (TIIM then decides alone)."""
        key = (sig.symbol, sig.strategy)
        if not self.available or key in self.pending:
            return False
        self.pending[key] = Pending(sig, ctx, dec, self._run(self.review, sig, ctx, dec))
        return True

    def poll(self) -> list[tuple[Pending, Verdict]]:
        done = []
        for key, p in list(self.pending.items()):
            if not p.future.done():
                continue
            del self.pending[key]
            try:
                v = p.future.result()
            except Exception as e:  # pragma: no cover - defensive
                log.exception("desk review crashed")
                v = Verdict("none", f"desk error: {e}")
            done.append((p, v))
        return done

    def _ask(self, system: str, prompt: str, schema: dict, model: str) -> dict | None:
        if not self.budget.take():
            return None
        r = self.llm.ask(system, prompt, schema, model)
        if r.data is None:
            log.warning("desk call failed: %s", r.error)
        return r.data

    def review(self, sig, ctx, dec: dict) -> Verdict:
        now = pd.Timestamp(ctx.time)
        brief = self.brief_for(sig.symbol, pd.Timestamp.now(tz="UTC")) or {}
        plan = ((self.j.get(f"plan:{self.s.mode}") or {}).get("markets") or {}).get(sig.symbol)
        text = dossier(sig, ctx, dec, brief, self._events(now, 6, sig.symbol), plan)
        calls0 = self.budget.used
        adv = "You are an advocate on a trading desk. " + PHILOSOPHY + " Argue your side as strongly as the evidence " \
              "honestly allows, in at most 80 words. Conviction 0-10: 0 = your side has no real case, 10 = overwhelming."
        bull = self._ask(adv, text + "\n\nYou are the BULL: make the case that price goes UP from here "
                         "(for a sell setup, that means the trade fails).", CASE_SCHEMA, self.cfg.model_advocate)
        bear = self._ask(adv, text + "\n\nYou are the BEAR: make the case that price goes DOWN from here "
                         "(for a buy setup, that means the trade fails).", CASE_SCHEMA, self.cfg.model_advocate)
        if bull is None or bear is None:
            return Verdict("none", "desk unavailable (Claude call failed or daily budget used up) - TIIM decided alone",
                           calls=self.budget.used - calls0, at=pd.Timestamp.now(tz="UTC").isoformat())
        bull = {"case": str(bull["case"])[:600], "conviction": _clip(bull["conviction"]), "key_risk": str(bull["key_risk"])[:200]}
        bear = {"case": str(bear["case"])[:600], "conviction": _clip(bear["conviction"]), "key_risk": str(bear["key_risk"])[:200]}
        if self.cfg.rebuttal and self.budget.remaining >= 3:
            for me, other, side in ((bull, bear, "BULL"), (bear, bull, "BEAR")):
                r = self._ask(adv, text + f"\n\nYou are the {side}. Your case: {me['case']}\nThe other side argued: "
                              f"{other['case']}\nReply to their strongest point in at most 50 words, then restate your "
                              "conviction 0-10.", REBUTTAL_SCHEMA, self.cfg.model_advocate)
                if r:
                    me["rebuttal"] = str(r["rebuttal"])[:400]
                    me["conviction"] = _clip(r["conviction"])
        record = self.scorecard()
        judge_prompt = (
            text + "\n\nBULL case (conviction " + str(bull["conviction"]) + "/10): " + bull["case"]
            + (" Rebuttal: " + bull["rebuttal"] if bull.get("rebuttal") else "")
            + "\nBEAR case (conviction " + str(bear["conviction"]) + "/10): " + bear["case"]
            + (" Rebuttal: " + bear["rebuttal"] if bear.get("rebuttal") else "")
            + f"\n\nThis is a {sig.side.upper()} setup. Your own track record as judge: {record['judge']['text']}\n"
            "Decide: 'take' (full planned risk), 'half' (half the risk: decent setup with a real, specific concern), or "
            "'skip' (the evidence clearly says this trade is likely to fail now). You cannot change the stop, the target "
            "or raise the risk. Skipping costs TIIM the trade, so skip only for a concrete reason, not general caution. "
            "Give confidence 0-100 and a reason in at most 40 words.")
        v = self._ask("You are TIIM, the judge and trader of this desk. " + PHILOSOPHY, judge_prompt, VERDICT_SCHEMA,
                      self.cfg.model_judge)
        at = pd.Timestamp.now(tz="UTC").isoformat()
        if v is None or v.get("action") not in ("take", "half", "skip"):
            return Verdict("none", "judge unavailable - TIIM decided alone", bull=bull, bear=bear, brief=brief,
                           calls=self.budget.used - calls0, at=at)
        return Verdict(v["action"], str(v["reason"])[:300], _clip(v["confidence"], 0, 100), bull, bear, brief,
                       self.budget.used - calls0, at)

    # --- scorekeeper -----------------------------------------------------------------------
    def scorecard(self) -> dict:
        return scorecard(self.j, self.s)

    def veto_allowed(self) -> bool:
        return self.scorecard()["judge"]["veto_power"]


def scorecard(j: Journal, s: Settings) -> dict:
    """Report cards from real outcomes. Vetoed setups are followed virtually, so a veto can be graded."""
    rows = [t for t in j.trades("mode = ? AND status = 'closed'", (s.mode,), order="id DESC", limit=600)
            if (t.get("decision") or {}).get("desk")]
    vetoed = [t for t in rows if t["decision"]["desk"]["action"] == "skip" and t["shadow"]]
    taken = [t for t in rows if t["decision"]["desk"]["action"] in ("take", "half") and not t["shadow"]]
    half = [t for t in taken if t["decision"]["desk"]["action"] == "half"]
    full = [t for t in taken if t["decision"]["desk"]["action"] == "take"]
    avg = lambda ts: sum(t["r_multiple"] or 0 for t in ts) / len(ts) if ts else 0.0
    saved = sum(1 for t in vetoed if (t["r_multiple"] or 0) < 0)
    veto_avg = avg(vetoed)
    veto_power = not (len(vetoed) >= s.desk.veto_review_after and veto_avg > s.desk.veto_max_avg_r)
    if vetoed:
        jt = (f"{len(vetoed)} vetoes so far: {saved} would have lost, {len(vetoed) - saved} would have won "
              f"(vetoed setups averaged {veto_avg:+.2f}R). Trades you took averaged {avg(taken):+.2f}R over {len(taken)}.")
    else:
        jt = "no graded decisions yet."
    debated = [t for t in rows if t["decision"]["desk"].get("bull") and t["decision"]["desk"].get("bear")]
    right = 0
    for t in debated:
        d = t["decision"]["desk"]
        up = (t["side"] == "buy") == ((t["r_multiple"] or 0) > 0)
        b, br = d["bull"].get("conviction", 0), d["bear"].get("conviction", 0)
        right += int((b > br and up) or (br > b and not up))
    briefed = [t for t in rows if (t.get("features") or {}).get("brief_agree") in ("with", "against")]
    w = [t for t in briefed if t["features"]["brief_agree"] == "with"]
    a = [t for t in briefed if t["features"]["brief_agree"] == "against"]
    wr = lambda ts: sum(1 for t in ts if (t["r_multiple"] or 0) > 0) / len(ts) if ts else 0.0
    return {
        "judge": {"vetoes": len(vetoed), "vetoes_saved": saved, "veto_avg_r": round(veto_avg, 3), "taken": len(taken),
                  "taken_avg_r": round(avg(taken), 3), "half": len(half), "half_avg_r": round(avg(half), 3),
                  "full": len(full), "full_avg_r": round(avg(full), 3), "veto_power": veto_power, "text": jt},
        "debate": {"graded": len(debated), "stronger_side_right": right,
                   "accuracy": round(right / len(debated), 3) if debated else None},
        "research": {"with": len(w), "with_win_rate": round(wr(w), 3), "against": len(a), "against_win_rate": round(wr(a), 3)},
    }


def scorecard_text(card: dict) -> list[str]:
    jd, db, rs = card["judge"], card["debate"], card["research"]
    out = [f"  Judge: {jd['text']}" + ("" if jd["veto_power"] else "  VETO POWER SUSPENDED (vetoes were costing money)")]
    if jd["half"]:
        out.append(f"         half-size calls averaged {jd['half_avg_r']:+.2f}R ({jd['half']}), full-size {jd['full_avg_r']:+.2f}R ({jd['full']})")
    if db["graded"]:
        out.append(f"  Debate: the side arguing harder was right {db['stronger_side_right']} of {db['graded']} times")
    if rs["with"] or rs["against"]:
        out.append(f"  Research: trades with the brief won {rs['with_win_rate']:.0%} ({rs['with']}), "
                   f"against it {rs['against_win_rate']:.0%} ({rs['against']})")
    return out
