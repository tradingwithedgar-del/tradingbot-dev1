"""Bar-by-bar simulated broker used for backtests and tests.

Orders placed on bar i fill at the OPEN of bar i+1 (plus half the spread), and
stop/target are checked on every following bar's high/low. If a single bar
touches both stop and target, the stop is assumed to have been hit first
(pessimistic, so backtests don't flatter the agent).
"""
from __future__ import annotations

import itertools

import pandas as pd

from .base import Broker, ClosedInfo, InstrumentSpec, Position, Quote


FX_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK", "SGD", "HKD", "ZAR", "MXN",
            "PLN", "TRY", "CNH"}


def is_fx_pair(symbol: str) -> bool:
    s = symbol.upper()
    return len(s) >= 6 and s[:3] in FX_CODES and s[3:6] in FX_CODES


def default_contract(symbol: str) -> float:
    """Typical CFD contract sizes, used only when the broker's real spec isn't available."""
    s = symbol.upper()
    if s.startswith("XAU"):
        return 100.0
    if s.startswith("XAG"):
        return 5000.0
    if any(x in s for x in ("OIL", "WTI", "BRENT", "XTI", "XBR")):
        return 1000.0
    if is_fx_pair(s):
        return 100_000.0
    return 1.0  # indices, single stocks, crypto: 1 unit = 1 index point / 1 share


class SimBroker(Broker):
    """Steps through the union of all symbols' bar times, so instruments with different
    trading hours (stocks vs. indices vs. gold) stay aligned in time."""

    name = "sim"

    def __init__(self, data: dict[str, pd.DataFrame], start_balance: float = 10_000.0, spread_frac: float = 0.0001,
                 warmup: int = 250, specs: dict[str, InstrumentSpec] | None = None) -> None:
        self.data = data
        self.specs = specs or {}
        idx = data[next(iter(data))].index
        for df in list(data.values())[1:]:
            idx = idx.union(df.index)
        self.timeline = idx.sort_values()
        self.cursor = min(warmup, len(self.timeline) - 1)   # timeline index of the latest CLOSED bar
        self._balance = start_balance
        self.spread_frac = spread_frac
        self._ids = itertools.count(1)
        self.pending: list[dict] = []
        self.open: dict[str, dict] = {}
        self.closed: dict[str, ClosedInfo] = {}

    # --- time control -------------------------------------------------------
    @property
    def length(self) -> int:
        return len(self.timeline)

    def now(self) -> pd.Timestamp:
        return self.timeline[self.cursor]

    def _pos(self, symbol: str) -> int:
        """Position of the symbol's latest bar at or before now (-1 if none yet)."""
        return int(self.data[symbol].index.searchsorted(self.now(), side="right")) - 1

    def _has_bar_now(self, symbol: str) -> bool:
        p = self._pos(symbol)
        return p >= 0 and self.data[symbol].index[p] == self.now()

    def step(self) -> bool:
        """Advance the clock one bar. Symbols with a bar at the new time fill their pending
        orders at its open, then have stops/targets checked against its high/low."""
        if self.cursor + 1 >= self.length:
            return False
        self.cursor += 1
        still_pending = []
        for o in self.pending:
            if not self._has_bar_now(o["symbol"]):
                still_pending.append(o)   # market closed for this symbol: fill at its next open
                continue
            bar = self.data[o["symbol"]].iloc[self._pos(o["symbol"])]
            half = self._spread(o["symbol"]) / 2
            o["entry"] = float(bar["open"]) + (half if o["side"] == "buy" else -half)
            self.open[o["id"]] = o
        self.pending = still_pending
        for pid, o in list(self.open.items()):
            if not self._has_bar_now(o["symbol"]):
                continue
            bar = self.data[o["symbol"]].iloc[self._pos(o["symbol"])]
            hi, lo = float(bar["high"]), float(bar["low"])
            exit_price = None
            if o["side"] == "buy":
                if lo <= o["stop"]:
                    exit_price = min(o["stop"], float(bar["open"]))  # gap through the stop fills worse
                elif hi >= o["tp"]:
                    exit_price = o["tp"]
            else:
                if hi >= o["stop"]:
                    exit_price = max(o["stop"], float(bar["open"]))
                elif lo <= o["tp"]:
                    exit_price = o["tp"]
            if exit_price is not None:
                self._close(pid, exit_price)
        return True

    def _close(self, pid: str, price: float) -> None:
        o = self.open.pop(pid)
        direction = 1 if o["side"] == "buy" else -1
        pnl = (price - o["entry"]) * direction * o["qty"] * self.spec(o["symbol"]).value_per_point
        self._balance += pnl
        self.closed[pid] = ClosedInfo(price, self.now(), pnl, entry_price=o["entry"])

    def _last_close(self, symbol: str) -> float:
        return float(self.data[symbol]["close"].iloc[max(0, self._pos(symbol))])

    def _spread(self, symbol: str) -> float:
        return self._last_close(symbol) * self.spread_frac

    # --- Broker API -------------------------------------------------------------
    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        p = self._pos(symbol)
        return self.data[symbol].iloc[max(0, p + 1 - count): p + 1]

    def get_quote(self, symbol: str) -> Quote:
        c = self._last_close(symbol)
        half = self._spread(symbol) / 2
        return Quote(c - half, c + half, self.now())

    def balance(self) -> float:
        return self._balance

    def equity(self) -> float:
        eq = self._balance
        for o in self.open.values():
            d = 1 if o["side"] == "buy" else -1
            eq += (self._last_close(o["symbol"]) - o["entry"]) * d * o["qty"] * self.spec(o["symbol"]).value_per_point
        return eq

    def open_positions(self) -> list[Position]:
        out = [Position(o["id"], o["symbol"], o["side"], o["qty"], o["entry"]) for o in self.open.values()]
        out += [Position(o["id"], o["symbol"], o["side"], o["qty"], o["est_entry"]) for o in self.pending]
        return out

    def spec(self, symbol: str) -> InstrumentSpec:
        return self.specs.get(symbol) or InstrumentSpec(value_per_point=default_contract(symbol), max_qty=1_000_000)

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float, tag: str) -> str:
        pid = str(next(self._ids))
        self.pending.append({"id": pid, "symbol": symbol, "side": side, "qty": qty, "stop": stop, "tp": take_profit,
                             "tag": tag, "est_entry": self._last_close(symbol)})
        return pid

    def close_position(self, position_id: str) -> None:
        if position_id in self.open:
            o = self.open[position_id]
            self._close(position_id, self._last_close(o["symbol"]))

    def modify_stop(self, position_id: str, stop: float) -> bool:
        if position_id in self.open:
            self.open[position_id]["stop"] = stop
            return True
        return False

    def closed_info(self, position_id, symbol, side, stop, take_profit) -> ClosedInfo | None:
        return self.closed.get(position_id)

    def fill_price(self, position_id: str) -> float | None:
        o = self.open.get(position_id)
        return o["entry"] if o else None
