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


def default_contract(symbol: str) -> float:
    s = symbol.upper()
    if s.startswith("XAU"):
        return 100.0
    if s.startswith("XAG"):
        return 5000.0
    if any(s.startswith(x) for x in ("US30", "NAS", "SPX", "GER", "UK100", "BTC", "ETH")):
        return 1.0
    return 100_000.0


class SimBroker(Broker):
    name = "sim"

    def __init__(self, data: dict[str, pd.DataFrame], start_balance: float = 10_000.0, spread_frac: float = 0.0001,
                 warmup: int = 250) -> None:
        self.data = data
        self.cursor = warmup                 # index of the latest CLOSED bar
        self._balance = start_balance
        self.spread_frac = spread_frac
        self._ids = itertools.count(1)
        self.pending: list[dict] = []
        self.open: dict[str, dict] = {}
        self.closed: dict[str, ClosedInfo] = {}

    # --- time control -------------------------------------------------------
    @property
    def length(self) -> int:
        return min(len(df) for df in self.data.values())

    def now(self) -> pd.Timestamp:
        return next(iter(self.data.values())).index[self.cursor]

    def step(self) -> bool:
        """Advance one bar: fill pending orders at its open, then process stops/targets."""
        if self.cursor + 1 >= self.length:
            return False
        self.cursor += 1
        for o in self.pending:
            bar = self.data[o["symbol"]].iloc[self.cursor]
            half = self._spread(o["symbol"]) / 2
            o["entry"] = float(bar["open"]) + (half if o["side"] == "buy" else -half)
            o["open_idx"] = self.cursor
            self.open[o["id"]] = o
        self.pending = []
        for pid, o in list(self.open.items()):
            bar = self.data[o["symbol"]].iloc[self.cursor]
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

    def _spread(self, symbol: str) -> float:
        return float(self.data[symbol]["close"].iloc[self.cursor]) * self.spread_frac

    # --- Broker API -------------------------------------------------------------
    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        df = self.data[symbol]
        return df.iloc[max(0, self.cursor + 1 - count): self.cursor + 1]

    def get_quote(self, symbol: str) -> Quote:
        c = float(self.data[symbol]["close"].iloc[self.cursor])
        half = self._spread(symbol) / 2
        return Quote(c - half, c + half, self.now())

    def balance(self) -> float:
        return self._balance

    def equity(self) -> float:
        eq = self._balance
        for o in self.open.values():
            c = float(self.data[o["symbol"]]["close"].iloc[self.cursor])
            d = 1 if o["side"] == "buy" else -1
            eq += (c - o["entry"]) * d * o["qty"] * self.spec(o["symbol"]).value_per_point
        return eq

    def open_positions(self) -> list[Position]:
        out = [Position(o["id"], o["symbol"], o["side"], o["qty"], o["entry"]) for o in self.open.values()]
        out += [Position(o["id"], o["symbol"], o["side"], o["qty"], o["est_entry"]) for o in self.pending]
        return out

    def spec(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(value_per_point=default_contract(symbol))

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float, tag: str) -> str:
        pid = str(next(self._ids))
        self.pending.append({"id": pid, "symbol": symbol, "side": side, "qty": qty, "stop": stop, "tp": take_profit,
                             "tag": tag, "est_entry": float(self.data[symbol]["close"].iloc[self.cursor])})
        return pid

    def close_position(self, position_id: str) -> None:
        if position_id in self.open:
            o = self.open[position_id]
            self._close(position_id, float(self.data[o["symbol"]]["close"].iloc[self.cursor]))

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
