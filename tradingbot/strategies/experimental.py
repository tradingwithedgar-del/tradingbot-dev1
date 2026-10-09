"""Self-built indicator strategies (demo account only).

Each experimental strategy is a "genome": 2-4 entry conditions picked from an
indicator library plus a stop rule. A genome is either a "swing" trade (normal stop, the
account's 3R target) or a "scalp" on the same 15m chart: tighter stop (never under the
compliance floor of 0.5 ATR), a 1-2R target and a time stop that closes it after a few candles. The learner breeds new genomes, keeps the ones
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
    style: str = "swing"        # "swing" | "scalp"
    target_r: float = 0.0       # scalp only: target in R (swing uses the account's reward multiple)
    max_bars: int = 0           # scalp only: close after this many candles if neither stop nor target hit

    def to_json(self) -> str:
        return json.dumps({"conditions": self.conditions, "stop_mode": self.stop_mode, "stop_atr": self.stop_atr,
                           "generation": self.generation, "parent": self.parent, "style": self.style,
                           "target_r": self.target_r, "max_bars": self.max_bars}, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        d = json.loads(s)
        return cls([(c[0], c[1]) for c in d["conditions"]], d["stop_mode"], d["stop_atr"], d.get("generation", 0),
                    d.get("parent", ""), d.get("style", "swing"), d.get("target_r", 0.0), d.get("max_bars", 0))

    @property
    def scalp(self) -> bool:
        return self.style == "scalp"

    @property
    def gid(self) -> str:
        body = {"c": self.conditions, "m": self.stop_mode, "s": round(self.stop_atr, 2)}
        if self.scalp:   # swing genomes keep the ids they had before scalps existed
            body.update(t=round(self.target_r, 2), b=self.max_bars)
        return "exp_" + hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:8]

    def describe(self) -> str:
        parts = [f"{n}({', '.join(f'{k}={v}' for k, v in p.items())})" if p else n for n, p in self.conditions]
        text = " AND ".join(parts) + f" | stop={self.stop_mode}:{self.stop_atr:.2f}ATR"
        if self.scalp:
            text = "SCALP " + text + f" | target {self.target_r:.1f}R | out after {self.max_bars} candles"
        return text


def _rand_params(rng: random.Random, spec: dict) -> dict:
    out = {}
    for k, (lo, hi, is_int) in spec.items():
        out[k] = rng.randint(int(lo), int(hi)) if is_int else round(rng.uniform(lo, hi), 2)
    return out


# Scalp ranges. The stop floor stays above the compliance minimum (0.5 ATR) so a scalp is a real
# trade with a real stop, never latency/tick scalping (PlexyTrade forbids that kind of trading).
SCALP_STOP_ATR = (0.6, 1.0)
SCALP_TARGET_R = (1.0, 2.0)
SCALP_MAX_BARS = (2, 8)          # 30 min - 2 h on the 15m chart
SCALP_SHARE = 0.35               # share of brand-new genomes that are scalps


def random_genome(rng: random.Random) -> Genome:
    names = rng.sample(sorted(CONDITIONS), rng.randint(2, 4))
    conds = [(n, _rand_params(rng, CONDITIONS[n])) for n in names]
    if rng.random() < SCALP_SHARE:
        return Genome(conds, "atr", round(rng.uniform(*SCALP_STOP_ATR), 2), style="scalp",
                      target_r=round(rng.uniform(*SCALP_TARGET_R), 1), max_bars=rng.randint(*SCALP_MAX_BARS))
    return Genome(conds, rng.choice(["atr", "swing"]), round(rng.uniform(1.0, 2.5), 2))


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
    if g.scalp:
        lo, hi = SCALP_STOP_ATR
        stop_atr = round(min(hi, max(lo, g.stop_atr * rng.uniform(0.85, 1.15))), 2)
        target = round(min(SCALP_TARGET_R[1], max(SCALP_TARGET_R[0], g.target_r + rng.choice([-0.2, 0, 0.2]))), 1)
        bars = min(SCALP_MAX_BARS[1], max(SCALP_MAX_BARS[0], g.max_bars + rng.choice([-1, 0, 1])))
        return Genome(conds, "atr", stop_atr, g.generation + 1, g.gid, "scalp", target, bars)
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
                sig = self._signal(ctx, side, stop, self.genome.describe())
                if sig is not None and self.genome.scalp:
                    sig.reward_multiple, sig.max_bars = self.genome.target_r, self.genome.max_bars
                return sig
        return None
