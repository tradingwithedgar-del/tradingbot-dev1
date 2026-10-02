"""Risk manager. The stop is decided first (where am I wrong?), then size is solved so
that being wrong costs exactly the allowed risk. Nothing here ever increases risk
after a loss."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .broker.base import InstrumentSpec
from .config import RiskConfig


@dataclass
class SizeDecision:
    ok: bool
    qty: float = 0.0
    risk_pct: float = 0.0
    risk_amount: float = 0.0
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def current_risk_pct(self, equity: float, peak: float) -> float:
        base = self.cfg.risk_per_trade
        if not self.cfg.drawdown_risk_scaling or peak <= 0:
            return base
        dd = max(0.0, (peak - equity) / peak)
        # Linear scale-down: full risk at 0 DD, min risk at the halt threshold.
        frac = min(1.0, dd / self.cfg.max_drawdown_halt)
        return max(self.cfg.min_risk_per_trade, base - frac * (base - self.cfg.min_risk_per_trade))

    def check_limits(self, equity: float, peak: float, day_start_equity: float, open_risk_pct: float, open_count: int) -> str | None:
        """Return a reason string if a new trade is not allowed right now."""
        if peak > 0 and (peak - equity) / peak >= self.cfg.max_drawdown_halt:
            return f"max drawdown {self.cfg.max_drawdown_halt:.0%} reached - halted for human review"
        if day_start_equity > 0 and (day_start_equity - equity) / day_start_equity >= self.cfg.daily_loss_limit:
            return f"daily loss limit {self.cfg.daily_loss_limit:.0%} reached"
        if open_count >= self.cfg.max_open_trades:
            return f"max open trades ({self.cfg.max_open_trades})"
        if open_risk_pct + self.cfg.min_risk_per_trade > self.cfg.max_total_open_risk + 1e-9:
            return f"total open risk {open_risk_pct:.1%} at cap"
        return None

    def size(self, equity: float, peak: float, entry: float, stop: float, spec: InstrumentSpec, open_risk_pct: float) -> SizeDecision:
        dist = abs(entry - stop)
        if dist <= 0 or equity <= 0:
            return SizeDecision(False, reason="invalid stop distance or equity")
        risk_pct = min(self.current_risk_pct(equity, peak), self.cfg.max_total_open_risk - open_risk_pct)
        if risk_pct <= 0:
            return SizeDecision(False, reason="no risk budget left")
        risk_amount = equity * risk_pct
        raw_qty = risk_amount / (dist * spec.value_per_point)
        qty = math.floor(raw_qty / spec.qty_step + 1e-9) * spec.qty_step
        qty = min(round(qty, 8), spec.max_qty)
        if qty < spec.min_qty:
            return SizeDecision(False, reason=f"position too small (needs {raw_qty:.4f}, min {spec.min_qty})")
        actual = qty * dist * spec.value_per_point
        return SizeDecision(True, qty, actual / equity, actual)
