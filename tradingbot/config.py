"""Central configuration.

Every number that controls money lives here so it can be reviewed in one place.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BOT_DATA_DIR", ROOT / "data"))

# US indices, gold, NVIDIA, Apple, Tesla, US oil. Exact names differ per broker:
# run `python -m tradingbot symbols` to see what your TradeLocker account calls them.
DEFAULT_SYMBOLS = "US30,US500,NAS100,XAUUSD,NVDA,AAPL,TSLA,USOIL"


@dataclass
class RiskConfig:
    # "How much am I willing to lose to be wrong?" -> decided BEFORE entry.
    risk_per_trade: float = 0.05        # 5% of equity at risk per trade
    reward_multiple: float = 3.0        # target = 3R -> 15% gain on a 5% risk
    # Survival rules. A sequence of 5% losses compounds fast; these stop the bleeding.
    max_open_trades: int = 2
    max_total_open_risk: float = 0.10   # never more than 10% of equity at risk at once
    daily_loss_limit: float = 0.10      # stop opening trades for the day after -10%
    max_drawdown_halt: float = 0.30     # full halt at -30% from peak; needs a human `resume`
    # As drawdown grows, risk per trade is scaled down linearly towards min_risk_per_trade.
    # Risk is NEVER scaled up after losses (no martingale).
    drawdown_risk_scaling: bool = True
    min_risk_per_trade: float = 0.01
    # Shadow (virtual) trades for experiments never risk real money.
    shadow_weight: float = 0.5          # how much a shadow result counts in learning vs a real one


@dataclass
class ComplianceConfig:
    """Guardrails derived from the PlexyTrade account opening agreement.

    The agreement forbids arbitrage, price/execution/platform manipulation, trading on
    errors/omissions/misquotes, Negative Balance Protection abuse and "any other strategy
    deemed abusive". These checks keep the bot clearly on the legitimate side.
    """
    max_spread_atr_fraction: float = 0.35   # refuse to trade if spread > 35% of ATR (illiquid/off-market quote)
    max_quote_jump_atr: float = 3.0         # refuse if live quote is > 3 ATR away from last close (misquote / spike)
    min_stop_atr_fraction: float = 0.5      # stops tighter than 0.5 ATR look like latency scalping -> refuse
    max_orders_per_minute: int = 4
    max_quote_age_seconds: float = 120.0    # stale quote -> do not trade on it
    allow_opposite_positions: bool = False  # no self-hedging on the same symbol


@dataclass
class LearningConfig:
    decay: float = 0.97                 # older trades fade so the agent tracks a changing market
    prior_wins: float = 1.0             # weak Beta prior centred near breakeven for 3R (25%)
    prior_losses: float = 3.0
    min_expected_r: float = 0.0         # only take signals with positive expected R
    filter_min_samples: int = 12        # evidence needed before a condition is blocked
    filter_max_expectancy: float = -0.25  # block a (strategy, condition) when expectancy is below this
    tune_window: int = 20               # trades considered when tuning strategy parameters
    postmortem_watch_bars: int = 40     # bars to keep watching price after a loss
    # Experimental indicator strategies (demo only).
    population_size: int = 8
    promote_min_trades: int = 20
    promote_min_expectancy: float = 0.3
    retire_max_expectancy: float = 0.0
    live_candidate_min_trades: int = 30


@dataclass
class Settings:
    mode: str = field(default_factory=lambda: os.getenv("BOT_MODE", "demo").lower())
    symbols: list[str] = field(
        default_factory=lambda: [s.strip() for s in os.getenv("BOT_SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]
    )
    timeframe: str = field(default_factory=lambda: os.getenv("BOT_TIMEFRAME", "15m"))
    history_bars: int = 400
    poll_seconds: int = 20
    db_path: Path = field(default_factory=lambda: DATA_DIR / "bot.db")
    risk: RiskConfig = field(default_factory=RiskConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    # Strategies the human has explicitly approved for the live account.
    live_approved_strategies: list[str] = field(
        default_factory=lambda: [s.strip() for s in os.getenv("LIVE_APPROVED_STRATEGIES", "sd_reversal,divergence,structure_trend").split(",") if s.strip()]
    )

    # TradeLocker credentials
    tl_environment: str = field(default_factory=lambda: os.getenv("TL_ENVIRONMENT", "https://demo.tradelocker.com"))
    tl_email: str = field(default_factory=lambda: os.getenv("TL_EMAIL", ""))
    tl_password: str = field(default_factory=lambda: os.getenv("TL_PASSWORD", ""))
    tl_server: str = field(default_factory=lambda: os.getenv("TL_SERVER", ""))
    tl_acc_num: int = field(default_factory=lambda: int(os.getenv("TL_ACC_NUM") or 0))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def experiments_allowed(self) -> bool:
        # Indicator experiments are only ever allowed off the live account.
        return not self.is_live

    @property
    def stop_file(self) -> Path:
        return self.db_path.parent / "STOP"

    def validate(self) -> None:
        if self.mode not in {"demo", "live", "backtest"}:
            raise ValueError(f"BOT_MODE must be demo, live or backtest, got {self.mode!r}")
        if self.is_live:
            if os.getenv("ALLOW_LIVE_TRADING", "NO") != "YES":
                raise RuntimeError("Live mode requires ALLOW_LIVE_TRADING=YES in the environment.")
            if "demo" in self.tl_environment:
                raise RuntimeError("BOT_MODE=live but TL_ENVIRONMENT points at the demo server.")
        if self.mode == "demo" and "live" in self.tl_environment:
            raise RuntimeError("BOT_MODE=demo but TL_ENVIRONMENT points at the live server.")
        if not 0 < self.risk.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be in (0, 5%].")
