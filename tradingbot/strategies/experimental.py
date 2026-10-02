"""Self-built indicator strategies (demo account only).

Each experimental strategy is a "genome": 2-4 entry conditions picked from an
indicator library plus a stop rule. The learner breeds new genomes, keeps the ones
with positive expectancy and retires the losers. They start as shadow (virtual)
trades, can be promoted to real demo trades, and can only reach the live account
after a human approves them by name.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

from ..analysis.context import MarketContext
from ..analysis.structure import last_of
from ..analysis.zones import zone_at_price
from .base import Signal, Strategy

# Each condition: (name, {param: (lo, hi, is_int)}). Evaluated for a long; shorts are mirrored.
CONDITIONS: dict[str, dict[str, tuple[float, float, bool]]] = {
    "rsi_extreme": {"period": (7, 21, True), "level": (20, 40, True)},
    "rsi_cross_mid": {"period": (7, 21, True)},
    "price_vs_ema": {"period": (20, 200, True)},
    "ema_stack": {"fast": (5, 30, True), "slow": (40, 120, True)},
    "macd_hist_turn": {},
    "bb_touch": {"period": (14, 30, True), "k": (1.5, 2.5, False)},
    "stoch_cross": {"level": (15, 35, True)},
    "adx_strong": {"level": (18, 35, True)},
    "in_zone": {},
    "structure_with": {},
    "momentum_candle": {"atr_mult": (0.6, 1.5, False)},
}


def _eval(ctx: MarketContext, cond: str, prm: dict, side: str) -> bool:
    long = side == "buy"
    c = ctx.df["close"]
    if cond == "rsi_extreme":
        r = float(ctx.series("rsi", int(prm["period"])).iloc[-1])
        return r < prm["level"] if long else r > 100 - prm["level"]
    if cond == "rsi_cross_mid":
        r = ctx.series("rsi", int(prm["period"]))
        return (r.iloc[-2] < 50 <= r.iloc[-1]) if long else (r.iloc[-2] > 50 >= r.iloc[-1])
    if cond == "price_vs_ema":
        e = float(ctx.series("ema", int(prm["period"])).iloc[-1])
        return c.iloc[-1] > e if long else c.iloc[-1] < e
    if cond == "ema_stack":
        f = float(ctx.series("ema", int(prm["fast"])).iloc[-1])
        s = float(ctx.series("ema", int(prm["slow"])).iloc[-1])
        return f > s if long else f < s
    if cond == "macd_hist_turn":
        h = ctx.series("macd")["hist"]
        return (h.iloc[-1] > h.iloc[-2] and h.iloc[-2] <= h.iloc[-3]) if long else (h.iloc[-1] < h.iloc[-2] and h.iloc[-2] >= h.iloc[-3])
    if cond == "bb_touch":
        bb = ctx.series("bollinger", int(prm["period"]), float(prm["k"]))
        return ctx.bar["low"] <= bb["lower"].iloc[-1] if long else ctx.bar["high"] >= bb["upper"].iloc[-1]
    if cond == "stoch_cross":
        st = ctx.series("stochastic")
        k, d = st["k"], st["d"]
        if long:
            return k.iloc[-2] <= d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < prm["level"] + 10
        return k.iloc[-2] >= d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 90 - prm["level"]
    if cond == "adx_strong":
        return float(ctx.series("adx", 14).iloc[-1]) >= prm["level"]
    if cond == "in_zone":
        return zone_at_price(ctx.zones, "demand" if long else "supply", float(ctx.bar["low"]), float(ctx.bar["high"])) is not None
    if cond == "structure_with":
        return ctx.trend.startswith("up") if long else ctx.trend.startswith("down")
    if cond == "momentum_candle":
        body = float(ctx.bar["close"] - ctx.bar["open"])
        return body > prm["atr_mult"] * ctx.atr_now if long else -body > prm["atr_mult"] * ctx.atr_now
    raise KeyError(cond)


@dataclass
class Genome:
    conditions: list[tuple[str, dict]]
    stop_mode: str = "atr"      # "atr" | "swing"
    stop_atr: float = 1.5
    generation: int = 0
    parent: str = ""

    def to_json(self) -> str:
        return json.dumps({"conditions": self.conditions, "stop_mode": self.stop_mode, "stop_atr": self.stop_atr,
                           "generation": self.generation, "parent": self.parent}, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        d = json.loads(s)
        return cls([(c[0], c[1]) for c in d["conditions"]], d["stop_mode"], d["stop_atr"], d.get("generation", 0), d.get("parent", ""))

    @property
    def gid(self) -> str:
        body = json.dumps({"c": self.conditions, "m": self.stop_mode, "s": round(self.stop_atr, 2)}, sort_keys=True)
        return "exp_" + hashlib.sha1(body.encode()).hexdigest()[:8]

    def describe(self) -> str:
        parts = [f"{n}({', '.join(f'{k}={v}' for k, v in p.items())})" if p else n for n, p in self.conditions]
        return " AND ".join(parts) + f" | stop={self.stop_mode}:{self.stop_atr:.2f}ATR"


def _rand_params(rng: random.Random, spec: dict) -> dict:
    out = {}
    for k, (lo, hi, is_int) in spec.items():
        out[k] = rng.randint(int(lo), int(hi)) if is_int else round(rng.uniform(lo, hi), 2)
    return out


def random_genome(rng: random.Random) -> Genome:
    names = rng.sample(sorted(CONDITIONS), rng.randint(2, 4))
    return Genome([(n, _rand_params(rng, CONDITIONS[n])) for n in names], rng.choice(["atr", "swing"]), round(rng.uniform(1.0, 2.5), 2))


def mutate(g: Genome, rng: random.Random) -> Genome:
    conds = [(n, dict(p)) for n, p in g.conditions]
    roll = rng.random()
    if roll < 0.4 and conds:
        i = rng.randrange(len(conds))
        n, p = conds[i]
        if p:
            k = rng.choice(sorted(p))
            lo, hi, is_int = CONDITIONS[n][k]
            p[k] = rng.randint(int(lo), int(hi)) if is_int else round(rng.uniform(lo, hi), 2)
    elif roll < 0.6 and len(conds) < 4:
        unused = [n for n in CONDITIONS if n not in {c[0] for c in conds}]
        if unused:
            n = rng.choice(unused)
            conds.append((n, _rand_params(rng, CONDITIONS[n])))
    elif roll < 0.75 and len(conds) > 2:
        conds.pop(rng.randrange(len(conds)))
    stop_atr = round(min(3.0, max(0.8, g.stop_atr * rng.uniform(0.85, 1.15))), 2)
    stop_mode = g.stop_mode if rng.random() > 0.15 else ("swing" if g.stop_mode == "atr" else "atr")
    return Genome(conds, stop_mode, stop_atr, g.generation + 1, g.gid)


class ExperimentalStrategy(Strategy):
    experimental = True

    def __init__(self, genome: Genome) -> None:
        super().__init__()
        self.genome = genome
        self.name = genome.gid

    def generate(self, ctx: MarketContext) -> Signal | None:
        if len(ctx.df) < 210:
            return None
        for side in ("buy", "sell"):
            if all(_eval(ctx, n, p, side) for n, p in self.genome.conditions):
                a = ctx.atr_now
                if self.genome.stop_mode == "swing":
                    sw = last_of(ctx.swings, "L" if side == "buy" else "H")
                    if sw is None:
                        continue
                    stop = sw.price - 0.2 * a if side == "buy" else sw.price + 0.2 * a
                    # keep stop sane: between 0.8 and 3 ATR
                    dist = min(max(abs(ctx.close - stop), 0.8 * a), 3 * a)
                else:
                    dist = self.genome.stop_atr * a
                stop = ctx.close - dist if side == "buy" else ctx.close + dist
                return self._signal(ctx, side, stop, self.genome.describe())
        return None
