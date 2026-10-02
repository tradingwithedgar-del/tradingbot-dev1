"""TradeLocker connection (demo or live) via the official `tradelocker` Python package.

Position sizing depends on InstrumentSpec.value_per_point. It is derived from the
instrument's contract size (lotSize) and the quote->account currency rate. Verify the
first few demo trades: the risk shown in the TradeLocker UI should match the bot's
risk_amount. Overrides can be put in data/instruments.json:
    {"XAUUSD": {"value_per_point": 100, "qty_step": 0.01, "min_qty": 0.01}}
"""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from ..config import Settings
from .base import TIMEFRAME_SECONDS, Broker, ClosedInfo, InstrumentSpec, Position, Quote

log = logging.getLogger(__name__)


class TradeLockerBroker(Broker):
    name = "tradelocker"

    def __init__(self, settings: Settings) -> None:
        from tradelocker import TLAPI  # imported lazily so backtests don't need credentials

        self.settings = settings
        self.is_live = settings.is_live
        if not (settings.tl_email and settings.tl_password and settings.tl_server):
            raise RuntimeError("TL_EMAIL, TL_PASSWORD and TL_SERVER must be set (see .env.example).")
        self.api = TLAPI(
            environment=settings.tl_environment,
            username=settings.tl_email,
            password=settings.tl_password,
            server=settings.tl_server,
            acc_num=settings.tl_acc_num,
            log_level="warning",
        )
        self._ids: dict[str, int] = {}
        self._specs: dict[str, InstrumentSpec] = {}
        overrides_path = settings.db_path.parent / "instruments.json"
        self._overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
        self._account_ccy = self._detect_account_currency()

    # --- helpers ---------------------------------------------------------------
    def _iid(self, symbol: str) -> int:
        if symbol not in self._ids:
            self._ids[symbol] = int(self.api.get_instrument_id_from_symbol_name(symbol))
        return self._ids[symbol]

    def _symbol_from_iid(self, iid: int) -> str:
        for s, i in self._ids.items():
            if i == iid:
                return s
        return str(self.api.get_symbol_name_from_instrument_id(iid))

    def _detect_account_currency(self) -> str:
        try:
            accounts = self.api.get_trade_accounts()
            wanted = int(getattr(self.api, "acc_num", 0) or 0)
            match = [a for a in accounts if int(a.get("accNum", -1)) == wanted] or accounts[:1]
            if match:
                return str(match[0].get("currency", "USD"))
        except Exception:  # pragma: no cover - network
            log.exception("could not detect account currency, assuming USD")
        return "USD"

    def _quote_to_account_rate(self, symbol: str) -> float:
        quote_ccy = symbol[3:6].upper() if len(symbol) >= 6 else self._account_ccy
        acct = self._account_ccy.upper()
        if quote_ccy == acct or not quote_ccy.isalpha():
            return 1.0
        for pair, invert in ((quote_ccy + acct, False), (acct + quote_ccy, True)):
            try:
                q = self.get_quote(pair)
                return 1.0 / q.mid if invert else q.mid
            except Exception:
                continue
        raise RuntimeError(f"Cannot convert {quote_ccy} to {acct} for {symbol}; add it to data/instruments.json")

    # --- Broker API -------------------------------------------------------------
    def get_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        secs = TIMEFRAME_SECONDS[timeframe]
        lookback_days = max(2, int(count * secs / 86400 * 1.6) + 2)  # weekends / gaps
        raw = self.api.get_price_history(self._iid(symbol), resolution=timeframe, lookback_period=f"{lookback_days}D")
        if raw.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = raw.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
        # drop the bar that is still forming
        now = pd.Timestamp.now(tz="UTC")
        df = df[df.index + pd.Timedelta(seconds=secs) <= now]
        return df.iloc[-count:]

    def get_quote(self, symbol: str) -> Quote:
        q = self.api.get_quotes(self._iid(symbol))
        return Quote(float(q["bp"]), float(q["ap"]), pd.Timestamp.now(tz="UTC"))

    def _state(self) -> dict:
        return self.api.get_account_state()

    def balance(self) -> float:
        return float(self._state().get("balance", 0.0))

    def equity(self) -> float:
        st = self._state()
        return float(st.get("projectedBalance", st.get("balance", 0.0)))

    def open_positions(self) -> list[Position]:
        df = self.api.get_all_positions()
        out = []
        for _, r in df.iterrows():
            out.append(Position(str(int(r["id"])), self._symbol_from_iid(int(r["tradableInstrumentId"])),
                                str(r["side"]).lower(), float(r["qty"]), float(r["avgPrice"])))
        return out

    def spec(self, symbol: str) -> InstrumentSpec:
        if symbol in self._overrides:
            return InstrumentSpec(**self._overrides[symbol])
        if symbol not in self._specs:
            d = self.api.get_instrument_details(self._iid(symbol))
            lot_size = float(d.get("lotSize") or d.get("contractSize") or 100_000)
            step = float(d.get("lotStep") or 0.01)
            min_q = float(d.get("minLot") or d.get("minOrderSize") or step)
            max_q = float(d.get("maxLot") or d.get("maxOrderSize") or 100)
            self._specs[symbol] = InstrumentSpec(lot_size, step, min_q, max_q)
        spec = self._specs[symbol]
        # conversion rate changes, so apply it fresh each time
        return InstrumentSpec(spec.value_per_point * self._quote_to_account_rate(symbol), spec.qty_step, spec.min_qty, spec.max_qty)

    def place_market(self, symbol: str, side: str, qty: float, stop: float, take_profit: float, tag: str) -> str:
        order_id = self.api.create_order(
            self._iid(symbol), quantity=qty, side=side, type_="market",
            stop_loss=stop, stop_loss_type="absolute", take_profit=take_profit, take_profit_type="absolute",
            strategy_id=tag[:31],
        )
        if not order_id:
            raise RuntimeError(f"order rejected: {symbol} {side} {qty}")
        for _ in range(10):
            pid = self.api.get_position_id_from_order_id(int(order_id))
            if pid:
                return str(pid)
            time.sleep(0.5)
        raise RuntimeError(f"order {order_id} placed but no position id found; check the platform")

    def close_position(self, position_id: str) -> None:
        self.api.close_position(position_id=int(position_id))

    def modify_stop(self, position_id: str, stop: float) -> bool:
        return bool(self.api.modify_position(int(position_id), {"stopLoss": float(stop)}))

    def closed_info(self, position_id: str, symbol: str, side: str, stop: float, take_profit: float) -> ClosedInfo | None:
        """Find the fill that closed the position in order history; fall back to the nearer of SL/TP."""
        try:
            hist = self.api.get_all_orders(history=True, lookback_period="7D")
            m = hist[(hist["positionId"].astype("Int64") == int(position_id)) & (hist["status"] == "Filled")]
            m = m[m["side"].str.lower() != side]
            if len(m):
                row = m.iloc[-1]
                t = pd.to_datetime(int(row.get("lastModified", 0) or 0), unit="ms", utc=True)
                return ClosedInfo(float(row["avgPrice"]), t)
        except Exception:
            log.exception("could not read order history for position %s", position_id)
        q = self.get_quote(symbol)
        px = q.bid if side == "buy" else q.ask
        guess = stop if abs(px - stop) < abs(px - take_profit) else take_profit
        return ClosedInfo(guess, pd.Timestamp.now(tz="UTC"), estimated=True)

    def fill_price(self, position_id: str) -> float | None:
        for p in self.open_positions():
            if p.id == position_id:
                return p.entry
        return None
