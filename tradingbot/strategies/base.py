from __future__ import annotations

from dataclasses import dataclass, field

from ..analysis.context import MarketContext


@dataclass
class Signal:
    symbol: str
    strategy: str
    side: str            # "buy" | "sell"
    entry: float
    stop: float
    take_profit: float = 0.0
    reason: str = ""
    features: dict = field(default_factory=dict)
    experimental: bool = False

    @property
    def risk_distance(self) -> float:
        return abs(self.entry - self.stop)

    def set_target(self, reward_multiple: float) -> None:
        d = self.risk_distance
        self.take_profit = self.entry + reward_multiple * d if self.side == "buy" else self.entry - reward_multiple * d


@dataclass
class Param:
    value: float
    lo: float
    hi: float


class Strategy:
    """A strategy turns a MarketContext into at most one Signal.

    `params` are the knobs the learner is allowed to tune within [lo, hi].
    """

    name = "base"
    experimental = False

    def __init__(self) -> None:
        self.params: dict[str, Param] = {}

    def p(self, key: str) -> float:
        return self.params[key].value

    def set_param(self, key: str, value: float) -> float:
        prm = self.params[key]
        prm.value = min(prm.hi, max(prm.lo, value))
        return prm.value

    def param_values(self) -> dict[str, float]:
        return {k: v.value for k, v in self.params.items()}

    def load_param_values(self, values: dict[str, float]) -> None:
        for k, v in values.items():
            if k in self.params:
                self.set_param(k, float(v))

    def generate(self, ctx: MarketContext) -> Signal | None:  # pragma: no cover - interface
        raise NotImplementedError

    def _signal(self, ctx: MarketContext, side: str, stop: float, reason: str, **extra) -> Signal | None:
        entry = ctx.close
        if (side == "buy" and stop >= entry) or (side == "sell" and stop <= entry):
            return None
        feats = ctx.features()
        feats.update(extra)
        return Signal(ctx.symbol, self.name, side, entry, stop, reason=reason, features=feats, experimental=self.experimental)
