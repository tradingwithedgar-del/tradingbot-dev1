"""SQLite trade journal: every real and shadow trade, the equity curve, the agent's
lessons/events and its learned state. The dashboard reads only from here."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT, shadow INTEGER DEFAULT 0, symbol TEXT, strategy TEXT, experimental INTEGER DEFAULT 0,
    side TEXT, status TEXT, opened_at TEXT, closed_at TEXT,
    entry REAL, stop REAL, take_profit REAL, qty REAL, risk_amount REAL, risk_pct REAL,
    exit_price REAL, pnl REAL, r_multiple REAL, outcome TEXT,
    mfe_r REAL DEFAULT 0, mae_r REAL DEFAULT 0, bars_held INTEGER DEFAULT 0,
    reason TEXT, features TEXT, decision TEXT, postmortem TEXT, chart TEXT, broker_id TEXT,
    exit_estimated INTEGER DEFAULT 0, watch TEXT
);
CREATE INDEX IF NOT EXISTS trades_status ON trades(status);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, mode TEXT, balance REAL, equity REAL, peak REAL, drawdown REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, mode TEXT, kind TEXT, message TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_COLS = {"features", "decision", "postmortem", "chart", "watch"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path | str, mode: str = "demo") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.clock = now_iso  # the agent swaps in bar time during backtests
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()

    # --- trades -----------------------------------------------------------------
    def open_trade(self, **fields: Any) -> int:
        fields.setdefault("mode", self.mode)
        fields.setdefault("status", "open")
        for k in JSON_COLS & fields.keys():
            fields[k] = None if fields[k] is None else json.dumps(fields[k], default=str)
        cols = ", ".join(fields)
        qs = ", ".join("?" for _ in fields)
        with self._lock:
            cur = self.conn.execute(f"INSERT INTO trades ({cols}) VALUES ({qs})", list(fields.values()))
            self.conn.commit()
            return int(cur.lastrowid)

    def update_trade(self, trade_id: int, **fields: Any) -> None:
        for k in JSON_COLS & fields.keys():
            fields[k] = None if fields[k] is None else json.dumps(fields[k], default=str)
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self.conn.execute(f"UPDATE trades SET {sets} WHERE id = ?", [*fields.values(), trade_id])
            self.conn.commit()

    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        for k in JSON_COLS:
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    def trades(self, where: str = "1=1", params: tuple = (), order: str = "id", limit: int | None = None) -> list[dict]:
        sql = f"SELECT * FROM trades WHERE {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [self._row(r) for r in rows]

    def trade(self, trade_id: int) -> dict | None:
        rows = self.trades("id = ?", (trade_id,))
        return rows[0] if rows else None

    def open_trades(self, shadow: bool | None = None) -> list[dict]:
        if shadow is None:
            return self.trades("status = 'open' AND mode = ?", (self.mode,))
        return self.trades("status = 'open' AND mode = ? AND shadow = ?", (self.mode, int(shadow)))

    def watching_trades(self) -> list[dict]:
        return self.trades("status = 'closed' AND watch IS NOT NULL AND mode = ?", (self.mode,))

    # --- equity / events / kv ------------------------------------------------------
    def record_equity(self, balance: float, equity: float, peak: float, ts: str | None = None) -> None:
        dd = 0.0 if peak <= 0 else (peak - equity) / peak
        with self._lock:
            self.conn.execute("INSERT INTO equity VALUES (?,?,?,?,?,?)", (ts or now_iso(), self.mode, balance, equity, peak, dd))
            self.conn.commit()

    def event(self, kind: str, message: str, data: Any = None, ts: str | None = None) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO events (ts, mode, kind, message, data) VALUES (?,?,?,?,?)",
                              (ts or self.clock(), self.mode, kind, message, json.dumps(data, default=str) if data is not None else None))
            self.conn.commit()

    def events(self, limit: int = 200, mode: str | None = None) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM events WHERE mode = ? ORDER BY id DESC LIMIT ?", (mode or self.mode, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if d["data"] else None
            out.append(d)
        return out

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.conn.execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                              (key, json.dumps(value, default=str)))
            self.conn.commit()

    def equity_curve(self, mode: str | None = None) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM equity WHERE mode = ? ORDER BY ts", (mode or self.mode,)).fetchall()
        return [dict(r) for r in rows]
