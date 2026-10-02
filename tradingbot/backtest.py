"""Backtest the full agent (strategies + learning + risk + compliance) on historical bars.

Data sources:
  * CSV files with columns time,open,high,low,close[,volume]
  * TradeLocker history (demo credentials)
  * synthetic random-walk data - ONLY for smoke-testing the plumbing; it has no real edge.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import Agent
from .broker.sim import SimBroker
from .config import Settings
from .journal import Journal


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in df.columns if c in {"time", "timestamp", "date", "datetime", "t"})
    t = df[tcol]
    df.index = pd.to_datetime(t, unit="ms", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()


def synthetic(n: int = 3000, start: float = 1.10, seed: int = 7, freq: str = "15min") -> pd.DataFrame:
    """Regime-switching random walk with some mean reversion. Plumbing tests only."""
    rng = np.random.default_rng(seed)
    vol = 0.0006
    drift = 0.0
    price = start
    rows = []
    for _ in range(n):
        if rng.random() < 0.01:
            drift = rng.choice([-1, 0, 1]) * vol * 0.15
            vol = float(np.clip(vol * rng.uniform(0.6, 1.6), 0.0002, 0.002))
        ret = drift + rng.normal(0, vol)
        o = price
        c = max(1e-6, o * (1 + ret))
        h = max(o, c) * (1 + abs(rng.normal(0, vol * 0.6)))
        lo = min(o, c) * (1 - abs(rng.normal(0, vol * 0.6)))
        rows.append((o, h, lo, c, float(rng.integers(50, 500))))
        price = c
    idx = pd.date_range("2025-01-06", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close", "volume"])


def run_backtest(data: dict[str, pd.DataFrame], settings: Settings | None = None, db_path: str | Path | None = None,
                 start_balance: float = 10_000.0, seed: int = 1, progress: bool = False) -> Journal:
    settings = settings or Settings()
    settings.mode = "backtest"
    settings.symbols = list(data)
    if db_path is not None:
        settings.db_path = Path(db_path)
    journal = Journal(settings.db_path, mode="backtest")
    broker = SimBroker(data, start_balance=start_balance, warmup=max(220, min(settings.history_bars, 400)))
    agent = Agent(settings, broker, journal, rng=random.Random(seed))
    total = broker.length - broker.cursor
    k = 0
    while True:
        agent.run_cycle()
        k += 1
        if progress and k % 250 == 0:
            print(f"  {k}/{total} bars, equity {broker.equity():,.2f}")
        if not broker.step():
            break
    journal.event("backtest", f"backtest finished: equity {broker.equity():,.2f} from {start_balance:,.2f}", ts=agent.now())
    return journal
