from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1H": 3600, "4H": 14400, "1D": 86400}


@dataclass
class Quote:
    bid: float
    ask: float
    time: pd.Timestamp

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        return (self.ask + self.bid) / 2


@dataclass
class Position:
    id: str
    symbol: str
    side: str
    qty: float
    entry: float


@dataclass
class InstrumentSpec:
    # Account-currency P&L for 1 unit of qty when price moves by 1.0
    value_per_point: float
    qty_step: float = 0.01
    min_qty: float = 0.01
    max_qty: float = 100.0


@dataclass
class ClosedInfo:
    exit_price: float
    exit_time: pd.Timestamp
    pnl: float | None = None   # account currency, None if the broker didn't report it
    estimated: bool = False
    entry_price: float | None = None


class Broker:
    is_live = False
    name = "base"

    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """Closed bars only, columns open/high/low/close/volume, UTC DatetimeIndex."""
        raise NotImplementedError

    def get_quote(self, symbol: str) -> Quote:
        raise NotImplementedError

    def equity(self) -> float:
        raise NotImplementedError

    def balance(self) -> float:
        raise NotImplementedError

    def open_positions(self) -> list[Position]:
        raise NotImplementedError

    def spec(self, symbol: str) -> InstrumentSpec:
        raise NotImplementedError

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float, tag: str) -> str:
        raise NotImplementedError

    def close_position(self, position_id: str) -> None:
        raise NotImplementedError

    def modify_stop(self, position_id: str, stop: float) -> bool:
        raise NotImplementedError

    def fill_price(self, position_id: str) -> float | None:
        return None

    def closed_info(self, position_id: str, symbol: str, side: str, stop: float, take_profit: float) -> ClosedInfo | None:
        raise NotImplementedError
