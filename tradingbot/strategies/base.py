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


ALL_MARKETS = ("index", "stock", "gold", "metal", "oil", "gas", "crypto", "fx")


class Strategy:
    """A strategy turns a MarketContext into at most one Signal.

    Every strategy in the library describes itself, so TIIM can pull the right ones for the
    situation and you can read what each one does on the dashboard:
        name         unique id (lowercase, no spaces)
        title        short human name
        description  what it trades and why, in plain words
        author       "TIIM" for built-ins, your name for strategies you add
        markets      asset classes it may trade (see ALL_MARKETS)
        needs        extra data it requires: "news", "earnings" (skipped when that data isn't available)
        stricter     (param, step) - how TIIM makes it more selective when its trades keep failing early
    `params` are the knobs TIIM is allowed to tune, each within [lo, hi].
    """

    name = "base"
    title = ""
    description = ""
    author = "TIIM"
    markets: tuple[str, ...] = ALL_MARKETS
    needs: tuple[str, ...] = ()
    stricter: tuple[str, float] | None = None
    experimental = False

    def __init__(self) -> None:
        self.params: dict[str, Param] = {}

    def applies(self, ctx: MarketContext) -> bool:
        """Whether this strategy should even look at this situation."""
        from ..news.assets import asset_class

        if asset_class(ctx.symbol) not in self.markets:
            return False
        if "news" in self.needs and not ctx.news:
            return False
        if "earnings" in self.needs and not ctx.earnings:
            return False
        return True

    def clone(self, name: str | None = None) -> "Strategy":
        """Same strategy with the same parameters, optionally under another name (used for testing variants)."""
        twin = type(self)()
        twin.load_param_values(self.param_values())
        if name:
            twin.name = name
        return twin

    def info(self) -> dict:
        return {"name": self.name, "title": self.title or self.name, "description": self.description,
                "author": self.author, "markets": list(self.markets), "needs": list(self.needs),
                "params": self.param_values()}

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
