"""PlexyTrade terms-of-use guardrails.

From the PlexyTrade Account Opening Agreement ("Market Abuse and Manipulation" and
related clauses): the company does not permit arbitrage, forbids manipulation of its
prices, execution or platform, forbids transactions based on errors, omissions or
misquotes, and lists Negative Balance Protection abuse and "any other strategy deemed
abusive" (patterns designed to extract artificial or asymmetric gain at the company's
expense rather than legitimate market participation). EAs, scalping, hedging and news
trading are permitted.

What the agent IS allowed to exploit: genuine market inefficiencies visible in its own
price data (structure, liquidity zones, divergences, session/volatility behaviour).
What it must never do, and what this module blocks:
  * latency/feed arbitrage (it only ever uses the broker's own feed; no external price comparison)
  * trading off-market / erroneous / stale quotes or abnormal spreads
  * micro-stop latency-style scalping
  * order spam
  * opposite positions on the same symbol (self-hedging) or across accounts
  * martingale / grid / size-up-after-loss behaviour (NBP abuse) - enforced in risk.py
"""
from __future__ import annotations

import time
from collections import deque

import pandas as pd

from .broker.base import Position, Quote
from .config import ComplianceConfig


class Compliance:
    def __init__(self, cfg: ComplianceConfig) -> None:
        self.cfg = cfg
        self._orders: deque[float] = deque()

    def check(self, side: str, symbol: str, entry: float, stop: float, last_close: float, atr: float,
              quote: Quote, positions: list[Position], now: pd.Timestamp | None = None) -> str | None:
        if atr <= 0:
            return "ATR unavailable"
        if quote.spread < 0 or quote.bid <= 0:
            return "invalid quote"
        if quote.spread > self.cfg.max_spread_atr_fraction * atr:
            return f"spread {quote.spread:.5f} abnormal vs ATR {atr:.5f} (possible off-market quote)"
        if abs(quote.mid - last_close) > self.cfg.max_quote_jump_atr * atr:
            return "quote far from last close - possible misquote/spike, not trading on it"
        if now is not None and (now - quote.time).total_seconds() > self.cfg.max_quote_age_seconds:
            return "stale quote"
        if abs(entry - stop) < self.cfg.min_stop_atr_fraction * atr:
            return "stop too tight (latency-scalping pattern not allowed)"
        if not self.cfg.allow_opposite_positions:
            for p in positions:
                if p.symbol == symbol and p.side != side:
                    return "opposite position open on same symbol (no self-hedging)"
        t = time.monotonic()
        while self._orders and t - self._orders[0] > 60:
            self._orders.popleft()
        if len(self._orders) >= self.cfg.max_orders_per_minute:
            return "order rate limit"
        return None

    def record_order(self) -> None:
        self._orders.append(time.monotonic())
