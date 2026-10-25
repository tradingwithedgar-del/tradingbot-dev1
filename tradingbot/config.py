"""Central configuration.

Every number that controls money lives here so it can be reviewed in one place.
Values can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# A subscription token pasted with extra text around it (or pasted twice): keep just the token.
_tok = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")
if _tok and not re.fullmatch(r"sk-ant-oat\w*-[A-Za-z0-9_-]+", _tok):
    _m = re.search(r"sk-ant-oat\w*-[A-Za-z0-9_-]+", _tok)
    if _m:
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = _m.group(0)

# A Claude subscription token (from `claude setup-token`, starts with sk-ant-oat) pasted into
# ANTHROPIC_API_KEY by mistake: use it as the subscription token it is.
if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant-oat"):
    if not os.getenv("CLAUDE_CODE_OAUTH_TOKEN"):
        os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = os.environ["ANTHROPIC_API_KEY"]
    del os.environ["ANTHROPIC_API_KEY"]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BOT_DATA_DIR", ROOT / "data"))

# BOT_SYMBOLS=ALL trades every instrument on the account (BOT_EXCLUDE removes names from that).
# US indices, gold, NVIDIA, Apple, Tesla, WTI oil and crypto, named as on PlexyTrade's TradeLocker.
# Full list of PlexyTrade names: SYMBOLS.md. `python -m tiim symbols` checks them.
DEFAULT_SYMBOLS = "US30,US500,USTECH,XAUUSD,NVDA,AAPL,TSLA,XTIUSD,BTCUSD,ETHUSD,SOLUSD"


@dataclass
class RiskConfig:
    # "How much am I willing to lose to be wrong?" -> decided BEFORE entry.
    risk_per_trade: float = 0.05        # 5% of equity at risk per trade
    reward_multiple: float = 3.0        # target = 3R -> 15% gain on a 5% risk
    # Survival rules. A sequence of 5% losses compounds fast; these stop the bleeding.
    max_open_trades: int = 2
    max_total_open_risk: float = 0.10   # never more than 10% of equity at risk at once
    # Correlated markets (e.g. US30/US500/USTECH) move together: at most this many open trades
    # in the same asset class AND direction, so "5 trades" is never really one big bet.
    max_same_group: int = 2
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
    # Strategy improvement (demo only): each strategy is shadowed by a tweaked "challenger" variant.
    challenger_min_trades: int = 25        # trades the challenger needs before it is judged
    challenger_max_trades: int = 45        # give up on it after this many without a clear win
    challenger_margin_r: float = 0.15      # must beat the current version by this much expectancy


@dataclass
class NewsConfig:
    """News awareness: scheduled events (economic calendar, earnings) and breaking headlines."""
    enabled: bool = field(default_factory=lambda: os.getenv("NEWS_ENABLED", "1") not in {"0", "false", "no", "NO"})
    # Claude reads headlines and decides which symbols they move. Without ANTHROPIC_API_KEY a
    # simple keyword reader is used instead.
    model: str = field(default_factory=lambda: os.getenv("NEWS_MODEL", "claude-opus-5-5"))
    headline_poll_seconds: int = 90
    calendar_refresh_minutes: int = 180   # the free calendar rate-limits; it changes rarely anyway
    max_headline_age_minutes: int = 120
    max_headlines_per_call: int = 25
    min_impact: int = 2                    # 0-3; headlines below this are ignored for trading
    pre_event_block_minutes: int = 15      # no new trades this long before a high-impact release
    post_event_block_minutes: int = 5      # ...nor right after it (spreads explode)
    post_event_window_minutes: int = 60    # how long a release/headline counts as "news is driving price"
    earnings_block_hours: int = 20         # no new stock trades this long before earnings
    flatten_before_earnings_minutes: int = 45  # close open stock trades before earnings
    fast_timeframe: str = "5m"             # timeframe used to react while news is hot
    # Protecting open trades when major news breaks against them (price action still opened the trade).
    guard_min_impact: int = 3
    guard_breakeven_r: float = 0.3         # in profit by this much -> stop moved to break-even
    guard_close_r: float = 1.5             # in profit by this much -> take the profit now
    feeds: list[str] = field(default_factory=lambda: [f for f in os.getenv("NEWS_FEEDS", "").split(",") if f.strip()])


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


@dataclass
class DeskConfig:
    """The trading desk: Claude agents that research, argue and judge each setup (see DESK.md)."""
    enabled: bool = field(default_factory=lambda: os.getenv("DESK_ENABLED", "1") not in {"0", "false", "no", "NO"})
    # Claude Code model aliases (opus / sonnet / haiku) or full model ids when an API key is used.
    model_research: str = field(default_factory=lambda: os.getenv("DESK_MODEL_RESEARCH", "sonnet"))
    model_advocate: str = field(default_factory=lambda: os.getenv("DESK_MODEL_ADVOCATE", "sonnet"))
    model_judge: str = field(default_factory=lambda: os.getenv("DESK_MODEL_JUDGE", "opus"))
    # Claude Pro has usage limits: the desk never makes more than this many calls a day.
    daily_calls: int = field(default_factory=lambda: _env_int("DESK_DAILY_CALLS", 150))
    research_markets: int = field(default_factory=lambda: _env_int("DESK_RESEARCH_MARKETS", 40))  # biggest movers briefed
    rebuttal: bool = field(default_factory=lambda: os.getenv("DESK_REBUTTAL", "1") not in {"0", "false", "no", "NO"})
    research_every_minutes: int = field(default_factory=lambda: _env_int("DESK_RESEARCH_MINUTES", 120))
    max_price_drift_atr: float = 0.3     # setup is dropped if price ran away while the desk debated
    # Scorekeeper: an agent whose advice doesn't beat plain TIIM loses its say.
    veto_review_after: int = 20           # vetoes needed before the judge's veto record is judged
    veto_max_avg_r: float = 0.15          # vetoed setups averaging better than this = vetoes are costing money


@dataclass
class LabConfig:
    """Strategy Lab: invents strategies and backtests them on real history every night."""
    min_win_rate: float = field(default_factory=lambda: float(os.getenv("LAB_MIN_WIN_RATE") or 70) / 100)
    min_trades: int = field(default_factory=lambda: _env_int("LAB_MIN_TRADES", 100))
    # Safety floor on top of the win rate: a strategy that wins often but still loses money overall
    # (tiny wins, big losses) is rejected. LAB_REQUIRE_PROFIT=0 turns this off.
    require_profit: bool = field(default_factory=lambda: os.getenv("LAB_REQUIRE_PROFIT", "1") not in {"0", "false", "no", "NO"})
    history_bars: int = field(default_factory=lambda: _env_int("LAB_HISTORY_BARS", 12000))   # ~6 months of 15m
    out_of_sample: float = 0.3            # last 30% of history is never used to pick, only to confirm
    candidates_per_night: int = field(default_factory=lambda: _env_int("LAB_CANDIDATES", 150))
    claude_ideas: int = field(default_factory=lambda: _env_int("LAB_CLAUDE_IDEAS", 6))
    spread_atr: float = 0.05              # trading cost charged on every backtest trade, in ATR
    max_active: int = 6                   # lab strategies trading at once (best first)
    max_markets: int = field(default_factory=lambda: _env_int("LAB_MAX_MARKETS", 30))   # a rotating sample per night


@dataclass
class Settings:
    # The agent's name (shown in logs, the dashboard and the desk's prompts). Its knowledge lives in the journal,
    # so renaming never loses anything: see `tiim new-agent`.
    agent_name: str = field(default_factory=lambda: (os.getenv("AGENT_NAME") or "TIIM").strip() or "TIIM")
    mode: str = field(default_factory=lambda: os.getenv("BOT_MODE", "demo").lower())
    symbols: list[str] = field(
        default_factory=lambda: [s.strip() for s in os.getenv("BOT_SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]
    )
    timeframe: str = field(default_factory=lambda: os.getenv("BOT_TIMEFRAME", "15m"))
    # Top-down filter: no real 15m trade against the Monthly/Weekly/Daily/4H bias unless price is at a
    # higher-timeframe zone (HTF_FILTER=0 turns it off; blocked setups are still followed virtually).
    htf_filter: bool = field(default_factory=lambda: os.getenv("HTF_FILTER", "1") not in {"0", "false", "no", "NO"})
    # Don't chase: no real trade in the direction of a move that is already exhausted (long 4H streak, far
    # from the 4H average, big day, extreme RSI). EXTENSION_FILTER=0 turns it off.
    # Weekly plan: real trades only in the direction planned for the week (PLAN_FILTER=0 turns it off).
    plan_filter: bool = field(default_factory=lambda: os.getenv("PLAN_FILTER", "1") not in {"0", "false", "no", "NO"})
    extension_filter: bool = field(default_factory=lambda: os.getenv("EXTENSION_FILTER", "1") not in {"0", "false", "no", "NO"})
    history_bars: int = 400
    poll_seconds: int = 20
    db_path: Path = field(default_factory=lambda: DATA_DIR / "bot.db")
    risk: RiskConfig = field(default_factory=RiskConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    desk: DeskConfig = field(default_factory=DeskConfig)
    lab: LabConfig = field(default_factory=LabConfig)
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

    # The demo account trades more at once so TIIM gets more samples to learn from.
    DEMO_LIMITS = {"max_open_trades": 20, "max_total_open_risk": 0.40, "max_same_group": 4}

    def __post_init__(self) -> None:
        if self.mode == "demo":
            for k, v in self.DEMO_LIMITS.items():
                setattr(self.risk, k, v)
        # Explicit overrides from .env win over both defaults (risk given in %, e.g. MAX_OPEN_RISK=20).
        if os.getenv("RISK_PER_TRADE"):
            self.risk.risk_per_trade = float(os.getenv("RISK_PER_TRADE")) / 100
            self.risk.min_risk_per_trade = min(self.risk.min_risk_per_trade, self.risk.risk_per_trade)
        if os.getenv("MAX_OPEN_TRADES"):
            self.risk.max_open_trades = int(os.getenv("MAX_OPEN_TRADES"))
        if os.getenv("MAX_OPEN_RISK"):
            self.risk.max_total_open_risk = float(os.getenv("MAX_OPEN_RISK")) / 100
        if os.getenv("MAX_SAME_GROUP"):
            self.risk.max_same_group = int(os.getenv("MAX_SAME_GROUP"))

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
        max_trades, max_risk = (30, 0.60) if self.mode != "live" else (10, 0.30)
        if not 1 <= self.risk.max_open_trades <= max_trades or not 0 < self.risk.max_total_open_risk <= max_risk:
            raise ValueError(f"MAX_OPEN_TRADES must be 1-{max_trades} and MAX_OPEN_RISK at most {max_risk:.0%} "
                             f"on the {self.mode} account.")
        if not 0 < self.risk.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be in (0, 5%].")
