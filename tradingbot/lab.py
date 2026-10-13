"""Strategy Lab: invents strategies, backtests them on real price history, and only hands TIIM the
ones that pass the gate. Runs every night as its own low-priority process (`tiim lab`).

The gate (settings in LabConfig / .env):
  * win rate of at least LAB_MIN_WIN_RATE (70%) on the history used to pick the strategy, AND again on
    the most recent 30% of history it was never tuned on (a strategy that only fits the past fails)
  * at least LAB_MIN_TRADES trades, so a lucky handful can't pass
  * after trading costs it must not lose money overall (LAB_REQUIRE_PROFIT=1, default): a strategy that
    wins 7 of 10 trades by +0.2R but loses 3 by -1R has a 70% win rate and still loses
Approved strategies trade on demo only. Live still needs you to approve them by name.

Ideas come from random combinations of the indicator conditions, mutations of near-misses, and
(when Claude is available) a few ideas proposed by Claude in the same condition language. Nothing
Claude writes is ever run as code: it can only pick conditions and numbers from a fixed menu.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import indicators as ind
from .config import LabConfig
from .strategies.experimental import CONDITIONS, Genome

log = logging.getLogger(__name__)

# Conditions that can be computed for the whole history at once. (in_zone / structure_with need the
# bar-by-bar zone and swing logic, so lab strategies don't use them.)
LAB_CONDITIONS = [c for c in CONDITIONS if c not in ("in_zone", "structure_with")]
STOP_ATR = (0.6, 3.0)
TARGET_R = (0.3, 3.0)
MAX_BARS = (0, 48)             # 0 = no time stop
HOLD_CAP = 200                 # bars; a backtest trade still open after this is closed at market


@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    total_r: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_r(self) -> float:
        return self.total_r / self.trades if self.trades else 0.0

    def add(self, other: "Result") -> "Result":
        return Result(self.trades + other.trades, self.wins + other.wins, self.total_r + other.total_r)

    def to_dict(self) -> dict:
        return {"trades": self.trades, "win_rate": round(self.win_rate, 3), "avg_r": round(self.avg_r, 3),
                "total_r": round(self.total_r, 1)}


class Frame:
    """Indicator cache for one symbol's history."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.atr = ind.atr(df, 14).to_numpy(float)
        self._cache: dict = {}

    def get(self, name: str, *args):
        key = (name, args)
        if key not in self._cache:
            src = self.df["close"] if name in {"ema", "rsi", "macd", "bollinger"} else self.df
            self._cache[key] = getattr(ind, name)(src, *args)
        return self._cache[key]

    def cond(self, name: str, p: dict, long: bool) -> np.ndarray:
        """The same rules as experimental._eval, for every bar at once (bar i uses data up to i only)."""
        c = self.df["close"]
        if name == "rsi_extreme":
            r = self.get("rsi", int(p["period"]))
            return (r < p["level"] if long else r > 100 - p["level"]).to_numpy()
        if name == "rsi_cross_mid":
            r = self.get("rsi", int(p["period"]))
            return ((r.shift() < 50) & (r >= 50) if long else (r.shift() > 50) & (r <= 50)).to_numpy()
        if name == "price_vs_ema":
            e = self.get("ema", int(p["period"]))
            return (c > e if long else c < e).to_numpy()
        if name == "ema_stack":
            f, s = self.get("ema", int(p["fast"])), self.get("ema", int(p["slow"]))
            return (f > s if long else f < s).to_numpy()
        if name == "macd_hist_turn":
            h = self.get("macd")["hist"]
            h1, h2 = h.shift(1), h.shift(2)
            return ((h > h1) & (h1 <= h2) if long else (h < h1) & (h1 >= h2)).to_numpy()
        if name == "bb_touch":
            bb = self.get("bollinger", int(p["period"]), float(p["k"]))
            return (self.df["low"] <= bb["lower"] if long else self.df["high"] >= bb["upper"]).to_numpy()
        if name == "stoch_cross":
            st = self.get("stochastic")
            k, d = st["k"], st["d"]
            if long:
                m = (k.shift() <= d.shift()) & (k > d) & (k < p["level"] + 10)
            else:
                m = (k.shift() >= d.shift()) & (k < d) & (k > 90 - p["level"])
            return m.to_numpy()
        if name == "adx_strong":
            return (self.get("adx", 14) >= p["level"]).to_numpy()
        if name == "momentum_candle":
            body = self.df["close"] - self.df["open"]
            atr = pd.Series(self.atr, index=self.df.index)
            return (body > p["atr_mult"] * atr if long else -body > p["atr_mult"] * atr).to_numpy()
        raise KeyError(name)

    def signals(self, g: Genome) -> tuple[np.ndarray, np.ndarray]:
        n = len(self.c)
        buy, sell = np.ones(n, bool), np.ones(n, bool)
        for name, p in g.conditions:
            buy &= np.nan_to_num(self.cond(name, p, True), nan=0).astype(bool)
            sell &= np.nan_to_num(self.cond(name, p, False), nan=0).astype(bool)
        buy[:210] = sell[:210] = False          # same warm-up TIIM needs live
        return buy, sell & ~buy                 # live checks buy first


def simulate(fr: Frame, g: Genome, start: int, end: int, cost_atr: float, default_rr: float) -> Result:
    """Trade the genome over bars [start, end): one position at a time, stop checked before target."""
    buy, sell = fr.signals(g)
    rr = g.target_r or default_rr
    res = Result()
    i = start
    while i < end - 1:
        if not (buy[i] or sell[i]) or not fr.atr[i] > 0:
            i += 1
            continue
        long = bool(buy[i])
        entry, dist = fr.c[i], g.stop_atr * fr.atr[i]
        stop = entry - dist if long else entry + dist
        target = entry + rr * dist if long else entry - rr * dist
        stop_n = g.max_bars if g.max_bars else HOLD_CAP
        j0, j1 = i + 1, min(end, i + 1 + stop_n)
        hi, lo = fr.h[j0:j1], fr.l[j0:j1]
        hit_stop = lo <= stop if long else hi >= stop
        hit_tgt = hi >= target if long else lo <= target
        s_at = int(np.argmax(hit_stop)) if hit_stop.any() else 10**9
        t_at = int(np.argmax(hit_tgt)) if hit_tgt.any() else 10**9
        if s_at == 10**9 and t_at == 10**9:
            exit_at = j1 - 1
            r = ((fr.c[exit_at] - entry) if long else (entry - fr.c[exit_at])) / dist
        elif s_at <= t_at:                       # same candle: assume the stop came first
            exit_at, r = j0 + s_at, -1.0
        else:
            exit_at, r = j0 + t_at, rr
        r -= cost_atr * fr.atr[i] / dist
        res = res.add(Result(1, int(r > 0), r))
        i = exit_at + 1
    return res


def evaluate(frames: dict[str, Frame], g: Genome, cfg: LabConfig, default_rr: float = 3.0) -> tuple[Result, Result]:
    """(in-sample, out-of-sample) results pooled over all symbols."""
    ins, oos = Result(), Result()
    for fr in frames.values():
        n = len(fr.c)
        cut = int(n * (1 - cfg.out_of_sample))
        ins = ins.add(simulate(fr, g, 210, cut, cfg.spread_atr, default_rr))
        oos = oos.add(simulate(fr, g, cut, n, cfg.spread_atr, default_rr))
    return ins, oos


def passes(ins: Result, oos: Result, cfg: LabConfig) -> tuple[bool, str]:
    if ins.trades < cfg.min_trades:
        return False, f"only {ins.trades} trades (needs {cfg.min_trades})"
    if oos.trades < max(15, int(cfg.min_trades * 0.25)):
        return False, f"only {oos.trades} trades in the unseen period"
    if ins.win_rate < cfg.min_win_rate:
        return False, f"win rate {ins.win_rate:.1%} (needs {cfg.min_win_rate:.0%})"
    if oos.win_rate < cfg.min_win_rate:
        return False, f"win rate fell to {oos.win_rate:.1%} on unseen data"
    if cfg.require_profit and (ins.total_r <= 0 or oos.total_r <= 0):
        return False, f"wins often but loses money overall ({ins.total_r:+.0f}R / {oos.total_r:+.0f}R)"
    return True, "passed"


# --- idea generation ------------------------------------------------------------------------
def _params(rng: random.Random, name: str) -> dict:
    out = {}
    for k, (lo, hi, is_int) in CONDITIONS[name].items():
        out[k] = rng.randint(int(lo), int(hi)) if is_int else round(rng.uniform(lo, hi), 2)
    return out


def random_idea(rng: random.Random) -> Genome:
    names = rng.sample(LAB_CONDITIONS, rng.randint(2, 4))
    return Genome([(n, _params(rng, n)) for n in names], "atr", round(rng.uniform(*STOP_ATR), 2), style="lab",
                  target_r=round(rng.uniform(*TARGET_R), 1), max_bars=rng.choice([0, 0, rng.randint(4, MAX_BARS[1])]))


def mutate_idea(g: Genome, rng: random.Random) -> Genome:
    conds = [(n, dict(p)) for n, p in g.conditions]
    roll = rng.random()
    if roll < 0.35 and conds:
        n, p = conds[rng.randrange(len(conds))]
        if p:
            k = rng.choice(sorted(p))
            lo, hi, is_int = CONDITIONS[n][k]
            p[k] = rng.randint(int(lo), int(hi)) if is_int else round(rng.uniform(lo, hi), 2)
    elif roll < 0.5 and len(conds) < 4:
        unused = [n for n in LAB_CONDITIONS if n not in {c[0] for c in conds}]
        if unused:
            n = rng.choice(unused)
            conds.append((n, _params(rng, n)))
    elif roll < 0.6 and len(conds) > 2:
        conds.pop(rng.randrange(len(conds)))
    stop = round(min(STOP_ATR[1], max(STOP_ATR[0], g.stop_atr * rng.uniform(0.8, 1.25))), 2)
    target = round(min(TARGET_R[1], max(TARGET_R[0], (g.target_r or 3.0) * rng.uniform(0.75, 1.25))), 1)
    bars = g.max_bars if rng.random() < 0.7 else rng.choice([0, rng.randint(4, MAX_BARS[1])])
    return Genome(conds, "atr", stop, g.generation + 1, g.gid, "lab", target, bars)


def clean_idea(d: dict) -> Genome | None:
    """Turn a proposal (e.g. from Claude) into a valid genome, or None. Only menu items survive."""
    try:
        conds = []
        for c in d.get("conditions", [])[:4]:
            name = c["name"]
            if name not in LAB_CONDITIONS or name in {x[0] for x in conds}:
                continue
            p = {}
            for k, (lo, hi, is_int) in CONDITIONS[name].items():
                v = float(c.get("params", {}).get(k, (lo + hi) / 2))
                v = min(hi, max(lo, v))
                p[k] = int(round(v)) if is_int else round(v, 2)
            conds.append((name, p))
        if len(conds) < 2:
            return None
        return Genome(conds, "atr", round(min(STOP_ATR[1], max(STOP_ATR[0], float(d["stop_atr"]))), 2), style="lab",
                      target_r=round(min(TARGET_R[1], max(TARGET_R[0], float(d["target_r"]))), 1),
                      max_bars=int(min(MAX_BARS[1], max(0, int(d.get("max_bars", 0))))))
    except (KeyError, TypeError, ValueError):
        return None


IDEA_SCHEMA = {
    "type": "object",
    "properties": {"ideas": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "conditions": {"type": "array", "items": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": LAB_CONDITIONS},
                               "params": {"type": "object", "additionalProperties": {"type": "number"}}},
                "required": ["name", "params"], "additionalProperties": False}},
            "stop_atr": {"type": "number"}, "target_r": {"type": "number"}, "max_bars": {"type": "integer"},
            "why": {"type": "string"}},
        "required": ["conditions", "stop_atr", "target_r", "max_bars", "why"], "additionalProperties": False}}},
    "required": ["ideas"], "additionalProperties": False,
}


def claude_ideas(llm, budget, n: int, near_misses: list[dict], cfg: LabConfig, model: str) -> list[Genome]:
    if llm is None or n <= 0 or (budget is not None and not budget.take()):
        return []
    menu = {k: {p: [lo, hi] for p, (lo, hi, _) in v.items()} for k, v in CONDITIONS.items() if k in LAB_CONDITIONS}
    prompt = (
        f"Design {n} entry rules for 15-minute charts of US indices, gold, oil, US mega-cap stocks and crypto. "
        f"Goal: a win rate of at least {cfg.min_win_rate:.0%} on {cfg.min_trades}+ trades without losing money overall "
        "after costs. Each rule is 2-4 conditions that must all be true on the same candle (written for a long; "
        "shorts are mirrored automatically), plus a stop distance in ATR (0.6-3), a target in R (0.3-3, R = stop "
        "distance) and an optional time stop in candles (0 = none, up to 48).\n"
        f"Available conditions and parameter ranges: {json.dumps(menu)}\n"
        f"Closest attempts so far (did not pass): {json.dumps(near_misses[:6])}\n"
        "Prefer ideas that differ from these attempts.")
    reply = llm.ask("You are a quantitative strategy researcher. Be concrete and stay inside the menu.",
                    prompt, IDEA_SCHEMA, model)
    if reply.data is None:
        log.warning("Claude strategy ideas failed: %s", reply.error)
        return []
    return [g for g in (clean_idea(d) for d in reply.data.get("ideas", [])) if g is not None]


# --- the nightly run ------------------------------------------------------------------------
def run_lab(data: dict[str, pd.DataFrame], cfg: LabConfig, existing: dict, seed: int | None = None,
            llm=None, budget=None, model: str = "sonnet", default_rr: float = 3.0) -> dict:
    """Search, test and gate. `existing` is the learner population (to re-check lab strategies).
    Returns the report the agent merges: approved genomes, withdrawn ids, and what was tried."""
    rng = random.Random(seed)
    frames = {s: Frame(df) for s, df in data.items() if len(df) > 400}
    tested: list[dict] = []
    approved: list[dict] = []
    seen: set[str] = set()

    def test(g: Genome, origin: str) -> None:
        if g.gid in seen or g.gid in existing:
            return
        seen.add(g.gid)
        ins, oos = evaluate(frames, g, cfg, default_rr)
        ok, why = passes(ins, oos, cfg)
        row = {"gid": g.gid, "description": g.describe(), "origin": origin, "passed": ok, "why": why,
               "in_sample": ins.to_dict(), "unseen": oos.to_dict(), "genome": g.to_json()}
        tested.append(row)
        if ok:
            approved.append({"gid": g.gid, "genome": g.to_json(), "description": g.describe(),
                             "stats": {"in_sample": ins.to_dict(), "unseen": oos.to_dict(), "origin": origin}})

    def score(r: dict) -> tuple:
        enough = r["in_sample"]["trades"] >= cfg.min_trades
        return (enough, r["in_sample"]["win_rate"] if enough else r["in_sample"]["trades"], r["in_sample"]["total_r"])

    budget_n = max(10, cfg.candidates_per_night)
    for _ in range(budget_n * 2 // 3):
        test(random_idea(rng), "random")
    misses = sorted((r for r in tested if not r["passed"]), key=score, reverse=True)
    ideas = claude_ideas(llm, budget, cfg.claude_ideas, [
        {"rule": r["description"], "win_rate": r["in_sample"]["win_rate"], "trades": r["in_sample"]["trades"],
         "total_r": r["in_sample"]["total_r"]} for r in misses[:6]], cfg, model)
    for g in ideas:
        test(g, "claude")
    for _ in range(budget_n - budget_n * 2 // 3):
        pool = sorted((r for r in tested if not r["passed"]), key=score, reverse=True)[:8]
        if not pool:
            break
        parent = Genome.from_json(rng.choice(pool)["genome"])
        test(mutate_idea(parent, rng), "mutation")

    # Re-check strategies the lab added earlier on today's (fresher) history.
    withdrawn = []
    for gid, info in existing.items():
        if not info.get("lab") or info.get("status") == "retired":
            continue
        g = Genome.from_json(info["genome"])
        ins, oos = evaluate(frames, g, cfg, default_rr)
        ok, why = passes(ins, oos, cfg)
        grace = ins.win_rate >= cfg.min_win_rate - 0.05 and oos.win_rate >= cfg.min_win_rate - 0.05
        if not ok and not grace:
            withdrawn.append(gid)

    approved.sort(key=lambda a: (a["stats"]["unseen"]["win_rate"], a["stats"]["unseen"]["total_r"]), reverse=True)
    active_now = sum(1 for i in existing.values() if i.get("lab") and i.get("status") != "retired") - len(withdrawn)
    approved = approved[:max(0, cfg.max_active - active_now)]
    best = sorted(tested, key=score, reverse=True)[:10]
    return {
        "ran_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbols": sorted(frames), "bars": {s: len(f.c) for s, f in frames.items()},
        "gate": {"min_win_rate": cfg.min_win_rate, "min_trades": cfg.min_trades, "require_profit": cfg.require_profit},
        "tested": len(tested), "passed": sum(1 for r in tested if r["passed"]), "claude_ideas": len(ideas),
        "approved": approved, "withdrawn": withdrawn,
        "best": [{k: r[k] for k in ("gid", "description", "origin", "passed", "why", "in_sample", "unseen")} for r in best],
    }


def run_nightly(settings, journal, broker, llm=None) -> dict:
    """Fetch history from the broker, run the lab, hand results to the agent through the journal."""
    from .llm import Budget

    data = {}
    for sym in settings.symbols:
        try:
            df = broker.get_bars(sym, settings.timeframe, settings.lab.history_bars)
            if len(df) > 400:
                data[sym] = df
        except Exception as e:  # pragma: no cover - network
            log.warning("lab: no history for %s: %s", sym, e)
    population = (journal.get(f"learner:{journal.mode}") or {}).get("population", {})
    budget = Budget(journal, 2 if settings.lab.claude_ideas else 0, key="lab") if llm else None
    report = run_lab(data, settings.lab, population, llm=llm, budget=budget, model=settings.desk.model_research,
                     default_rr=settings.risk.reward_multiple)
    journal.set("lab:report", report)
    journal.event("lab", f"Strategy Lab tested {report['tested']} ideas on {len(data)} markets: {report['passed']} passed "
                         f"the gate, {len(report['approved'])} added, {len(report['withdrawn'])} withdrawn")
    return report   # the running agent merges it (Learner.merge_lab)


def summary(report: dict) -> str:
    if not report:
        return "The Strategy Lab hasn't run yet (it runs nightly; `tiim lab` runs it now)."
    g = report["gate"]
    lines = [f"Strategy Lab, {report['ran_at'][:16].replace('T', ' ')} UTC: tested {report['tested']} ideas "
             f"({report['claude_ideas']} from Claude) on {len(report['symbols'])} markets",
             f"Gate: win rate >= {g['min_win_rate']:.0%} on past and unseen data, {g['min_trades']}+ trades"
             + (", not losing money overall" if g["require_profit"] else ""),
             f"Passed: {report['passed']}, added to TIIM: {len(report['approved'])}, withdrawn: {len(report['withdrawn'])}",
             "Closest / best:"]
    for r in report["best"][:6]:
        i, u = r["in_sample"], r["unseen"]
        lines.append(f"  {'PASS' if r['passed'] else 'fail'} {r['gid']} {i['win_rate']:.0%}/{u['win_rate']:.0%} win, "
                     f"{i['trades']}+{u['trades']} trades, {i['total_r']:+.0f}R/{u['total_r']:+.0f}R - {r['why']}")
        lines.append(f"       {r['description'][:110]}")
    return "\n".join(lines)

