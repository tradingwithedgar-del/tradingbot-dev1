"""The learning brain.

1. Bayesian edge tracking - for every strategy, and every strategy x market condition
   (regime, session, structure, bias, symbol, ...), keep a recency-weighted Beta
   posterior of the win rate plus average win/loss in R. Old trades decay so the
   agent follows a changing market.
2. Decision - a signal is traded only if its expected R is positive. On demo the
   agent explores (Thompson sampling); on live it uses a conservative estimate.
3. Learned filters - (strategy, condition) pairs with enough evidence of negative
   expectancy are blocked. Blocked signals are still traded *virtually* (shadow), so
   a filter is lifted automatically if that condition starts working again.
4. Self-tuning - post-mortem tags (stop too tight, immediately wrong, gave back
   profit...) nudge each strategy's parameters within safe bounds.
5. Evolution (demo only) - experimental indicator strategies are bred, promoted and
   retired by their results. Promotion to live always needs human approval.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .config import LearningConfig, RiskConfig
from .journal import Journal
from .strategies import ExperimentalStrategy, Genome, Strategy, mutate, random_genome

CONDITION_KEYS = ["regime", "trend_strength", "volatility", "structure", "bias", "session", "rsi_zone",
                  "symbol", "with_structure", "zone_fresh", "divergence", "confluence"]

# Which parameter makes a strategy more selective, and in which direction.
STRICTER = {
    "sd_reversal": ("min_wick_ratio", +0.05),
    "divergence": ("min_osc_gap", +1.0),
    "structure_trend": ("max_pullback_atr", -0.1),
}


@dataclass
class EdgeStat:
    wins: float = 0.0
    losses: float = 0.0
    win_r: float = 0.0     # decayed sum of R on wins
    loss_r: float = 0.0    # decayed sum of |R| on losses
    count: int = 0         # raw, undecayed number of trades
    weight: float = 0.0    # decayed effective sample size

    def update(self, r: float, w: float, decay: float) -> None:
        self.wins *= decay
        self.losses *= decay
        self.win_r *= decay
        self.loss_r *= decay
        self.weight = self.weight * decay + w
        self.count += 1
        if r > 0:
            self.wins += w
            self.win_r += w * r
        else:
            self.losses += w
            self.loss_r += w * abs(r)

    def p_params(self, cfg: LearningConfig) -> tuple[float, float]:
        return cfg.prior_wins + self.wins, cfg.prior_losses + self.losses

    def p_mean(self, cfg: LearningConfig) -> float:
        a, b = self.p_params(cfg)
        return a / (a + b)

    def p_quantile(self, cfg: LearningConfig, z: float) -> float:
        """Normal approximation of a Beta quantile (z<0 lower bound, z>0 upper bound)."""
        a, b = self.p_params(cfg)
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return min(1.0, max(0.0, m + z * sd))

    def avg_win(self, rr: float) -> float:
        return (self.win_r + rr * 1.0) / (self.wins + 1.0)   # shrink toward the planned target

    def avg_loss(self) -> float:
        return (self.loss_r + 1.0) / (self.losses + 1.0)     # shrink toward -1R

    def expected_r(self, p: float, rr: float) -> float:
        return p * self.avg_win(rr) - (1 - p) * self.avg_loss()


@dataclass
class Decision:
    action: str            # "trade" | "shadow" | "skip"
    p: float
    expected_r: float
    stat_key: str
    reasons: list[str]


class Learner:
    def __init__(self, journal: Journal, cfg: LearningConfig, risk_cfg: RiskConfig, live: bool,
                 rng: random.Random | None = None) -> None:
        self.j = journal
        self.cfg = cfg
        self.rr = risk_cfg.reward_multiple
        self.shadow_weight = risk_cfg.shadow_weight
        self.live = live
        self.rng = rng or random.Random()
        self.stats: dict[str, EdgeStat] = {}
        self.filters: dict[str, list[str]] = {}       # strategy -> blocked "key=value" conditions
        self.params: dict[str, dict[str, float]] = {}  # strategy -> tuned parameter values
        self.breakeven: dict[str, float] = {}          # strategy -> move stop to BE at this R
        self.population: dict[str, dict] = {}          # experimental genomes
        self.tune_counter: dict[str, int] = {}
        self.load()

    # --- persistence -----------------------------------------------------------
    @property
    def _key(self) -> str:
        return f"learner:{self.j.mode}"

    def load(self) -> None:
        state = self.j.get(self._key)
        if state is None and self.j.mode == "live":
            state = self.j.get("learner:demo")  # live starts from what demo learned
            if state:
                state["population"] = {}  # experiments never carry into live
        if not state:
            return
        self.stats = {k: EdgeStat(**v) for k, v in state.get("stats", {}).items()}
        self.filters = state.get("filters", {})
        self.params = state.get("params", {})
        self.breakeven = state.get("breakeven", {})
        self.population = state.get("population", {})
        self.tune_counter = state.get("tune_counter", {})

    def save(self) -> None:
        self.j.set(self._key, {
            "stats": {k: asdict(v) for k, v in self.stats.items()},
            "filters": self.filters, "params": self.params, "breakeven": self.breakeven,
            "population": self.population, "tune_counter": self.tune_counter,
        })

    # --- edge statistics ---------------------------------------------------------
    def stat(self, key: str) -> EdgeStat:
        return self.stats.setdefault(key, EdgeStat())

    @staticmethod
    def condition_items(features: dict, symbol: str) -> list[str]:
        out = []
        for k in CONDITION_KEYS:
            v = symbol if k == "symbol" else features.get(k)
            if v is not None and v != "":
                out.append(f"{k}={v}")
        return out

    def record(self, trade: dict) -> None:
        r = float(trade.get("r_multiple") or 0.0)
        w = self.shadow_weight if trade.get("shadow") else 1.0
        s = trade["strategy"]
        self.stat(s).update(r, w, self.cfg.decay)
        if not trade.get("shadow"):
            self.stat(f"{s}|real").update(r, 1.0, self.cfg.decay)
        for cond in self.condition_items(trade.get("features") or {}, trade["symbol"]):
            self.stat(f"{s}|{cond}").update(r, w, self.cfg.decay)
        self._refresh_filters(s)
        self.save()

    # --- decisions -------------------------------------------------------------------
    def decide(self, strategy: Strategy, features: dict, symbol: str, live_approved: list[str]) -> Decision:
        s = strategy.name
        reasons: list[str] = []
        if strategy.experimental:
            status = self.population.get(s, {}).get("status", "shadow")
            if self.live:
                return Decision("skip", 0, 0, s, ["experimental strategies never trade live"])
            if status != "demo" and status != "live_candidate":
                return Decision("shadow", 0, 0, s, [f"experimental genome in {status} stage"])
        elif self.live and s not in live_approved:
            return Decision("shadow", 0, 0, s, ["strategy not approved for live"])

        conds = self.condition_items(features, symbol)
        blocked = [c for c in conds if c in self.filters.get(s, [])]
        if blocked:
            return Decision("shadow", 0, 0, s, [f"learned filter: {', '.join(blocked)}"])

        # Most specific statistic with enough evidence: regime-level, else strategy-level.
        key = s
        regime_key = f"{s}|regime={features.get('regime')}"
        if regime_key in self.stats and self.stats[regime_key].count >= 8:
            key = regime_key
        st = self.stat(key)
        if self.live:
            p = st.p_quantile(self.cfg, -0.5)   # cautious with real money
            reasons.append("live: conservative win-rate estimate")
        else:
            a, b = st.p_params(self.cfg)
            p = self.rng.betavariate(a, b)      # explore on demo
            reasons.append("demo: Thompson-sampled win rate")
        er = st.expected_r(p, self.rr)
        if er <= self.cfg.min_expected_r:
            return Decision("shadow", p, er, key, reasons + [f"expected {er:+.2f}R not positive"])
        return Decision("trade", p, er, key, reasons + [f"expected {er:+.2f}R"])

    # --- learned filters ---------------------------------------------------------------
    def _refresh_filters(self, strategy: str) -> None:
        breakeven_p = 1.0 / (1.0 + self.rr)
        old = set(self.filters.get(strategy, []))
        new = set()
        prefix = f"{strategy}|"
        for key, st in self.stats.items():
            if not key.startswith(prefix) or key == f"{strategy}|real" or st.count < self.cfg.filter_min_samples:
                continue
            exp = st.expected_r(st.p_mean(self.cfg), self.rr)
            if exp < self.cfg.filter_max_expectancy and st.p_quantile(self.cfg, 1.0) < breakeven_p:
                new.add(key[len(prefix):])
        for c in sorted(new - old):
            self.j.event("lesson", f"{strategy}: stop trading when {c} (negative edge, now shadow-only)",
                         {"strategy": strategy, "condition": c, "stat": asdict(self.stats[prefix + c])})
        for c in sorted(old - new):
            self.j.event("lesson", f"{strategy}: {c} recovered, filter lifted", {"strategy": strategy, "condition": c})
        self.filters[strategy] = sorted(new)

    # --- self-tuning from post-mortems -------------------------------------------------
    def tune(self, strategy: Strategy, recent: list[dict]) -> list[str]:
        """Adjust parameters using the last N finalized post-mortems for this strategy."""
        s = strategy.name
        self.tune_counter[s] = self.tune_counter.get(s, 0) + 1
        if self.tune_counter[s] < max(5, self.cfg.tune_window // 2) or len(recent) < self.cfg.tune_window:
            return []
        self.tune_counter[s] = 0
        losses = [t for t in recent if (t.get("r_multiple") or 0) < 0]
        wins = [t for t in recent if (t.get("r_multiple") or 0) > 0]
        if not losses:
            return []

        def share(tag: str, pool: list[dict]) -> float:
            return sum(tag in (t.get("postmortem") or {}).get("tags", []) for t in pool) / max(1, len(pool))

        changes: list[str] = []
        if "stop_atr_buffer" in strategy.params and share("stop_too_tight", losses) > 0.35:
            old = strategy.p("stop_atr_buffer")
            new = strategy.set_param("stop_atr_buffer", old * 1.2)
            if new != old:
                changes.append(f"widened stop buffer {old:.2f} -> {new:.2f} ATR ({share('stop_too_tight', losses):.0%} of losses were stop-hunts)")
        if s in STRICTER and share("immediately_wrong", losses) > 0.5:
            prm, step = STRICTER[s]
            old = strategy.p(prm)
            new = strategy.set_param(prm, old + step)
            if new != old:
                changes.append(f"stricter entry: {prm} {old:.2f} -> {new:.2f} ({share('immediately_wrong', losses):.0%} of losses never worked)")
        gb = share("gave_back_profit", losses)
        if gb > 0.3 and s not in self.breakeven:
            self.breakeven[s] = 1.5
            changes.append(f"enable break-even stop at +1.5R ({gb:.0%} of losses were in profit first)")
        elif gb < 0.1 and s in self.breakeven and share("survived_drawdown_win", wins) < 0.1:
            del self.breakeven[s]
            changes.append("disable break-even stop (rarely needed)")
        if changes:
            self.params[s] = strategy.param_values()
            for c in changes:
                self.j.event("tuning", f"{s}: {c}", {"strategy": s, "params": self.params[s]})
            self.save()
        return changes

    def apply_params(self, strategies: list[Strategy]) -> None:
        for st in strategies:
            if st.name in self.params:
                st.load_param_values(self.params[st.name])

    # --- evolution of experimental strategies --------------------------------------------
    def experimental_strategies(self) -> list[ExperimentalStrategy]:
        out = []
        for gid, info in self.population.items():
            if info["status"] != "retired":
                out.append(ExperimentalStrategy(Genome.from_json(info["genome"])))
        return out

    def evolve(self, now: str) -> None:
        if self.live:
            return
        for gid, info in self.population.items():
            if info["status"] == "retired":
                continue
            st = self.stats.get(gid)
            if st is None or st.count < self.cfg.promote_min_trades:
                continue
            exp = st.expected_r(st.p_mean(self.cfg), self.rr)
            info["expectancy"] = round(exp, 3)
            if exp < self.cfg.retire_max_expectancy:
                info["status"] = "retired"
                self.j.event("evolution", f"retired {gid} ({exp:+.2f}R over {st.count} trades)", info)
            elif info["status"] == "shadow" and exp >= self.cfg.promote_min_expectancy:
                info["status"] = "demo"
                self.j.event("evolution", f"promoted {gid} to real demo trades ({exp:+.2f}R over {st.count} shadow trades)", info)
            elif info["status"] == "demo":
                real = self.stats.get(f"{gid}|real")
                if real and real.count >= self.cfg.live_candidate_min_trades and real.expected_r(real.p_mean(self.cfg), self.rr) > 0.2:
                    info["status"] = "live_candidate"
                    self.j.event("approval_needed",
                                 f"{gid} is a LIVE CANDIDATE: {Genome.from_json(info['genome']).describe()} - add it to LIVE_APPROVED_STRATEGIES to allow it on live",
                                 info)
        active = [g for g, i in self.population.items() if i["status"] != "retired"]
        ranked = sorted(active, key=lambda g: self.population[g].get("expectancy", 0.0), reverse=True)
        attempts = 0
        while len(active) < self.cfg.population_size and attempts < 100:
            attempts += 1
            if ranked and self.rng.random() < 0.6:
                parent = Genome.from_json(self.population[ranked[0]]["genome"])
                child = mutate(parent, self.rng)
            else:
                child = random_genome(self.rng)
            if child.gid in self.population:
                continue
            self.population[child.gid] = {"genome": child.to_json(), "status": "shadow", "born": now,
                                          "description": child.describe(), "expectancy": 0.0}
            active.append(child.gid)
            self.j.event("evolution", f"new experiment {child.gid}: {child.describe()}", {"parent": child.parent})
        self.save()
