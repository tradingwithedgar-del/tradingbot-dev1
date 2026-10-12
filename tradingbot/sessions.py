"""Gap protection. A stop only works while the market is open: a stock held overnight, or an
index/gold/oil trade held over the weekend, can open far beyond its stop (TSLA #340 lost 2.85R
that way). So TIIM is flat before those closes and doesn't open trades it would have to dump
minutes later. Crypto trades 24/7 and is exempt. Times are New York time (handles DST)."""
from __future__ import annotations

from datetime import time

import pandas as pd

from .news.assets import asset_class

NY = "America/New_York"
STOCK_NO_ENTRY, STOCK_FLATTEN, STOCK_CLOSE = time(15, 0), time(15, 45), time(16, 0)
WEEKEND_NO_ENTRY, WEEKEND_FLATTEN, WEEKEND_CLOSE = time(15, 0), time(16, 0), time(17, 0)
FRIDAY = 4


def _ny(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return (ts if ts.tzinfo else ts.tz_localize("UTC")).tz_convert(NY)


def flatten_reason(symbol: str, ts: pd.Timestamp) -> str | None:
    """Why an open trade on `symbol` must be closed now, or None."""
    cls = asset_class(symbol)
    if cls == "crypto":
        return None
    t = _ny(ts)
    if cls == "stock" and t.weekday() < 5 and STOCK_FLATTEN <= t.time() < STOCK_CLOSE:
        return "closed before the stock market close (overnight gaps jump over stops)"
    if t.weekday() == FRIDAY and WEEKEND_FLATTEN <= t.time() < WEEKEND_CLOSE:
        return "closed before the weekend (weekend gaps jump over stops)"
    return None


def entry_block(symbol: str, ts: pd.Timestamp) -> str | None:
    """Why no new trade should be opened now (too close to a close it would have to be dumped at)."""
    cls = asset_class(symbol)
    if cls == "crypto":
        return None
    t = _ny(ts)
    if cls == "stock" and t.weekday() < 5 and t.time() >= STOCK_NO_ENTRY:
        return "too close to the stock market close (would be held into an overnight gap)"
    if t.weekday() == FRIDAY and t.time() >= WEEKEND_NO_ENTRY:
        return "too close to the weekend close (would be held into a weekend gap)"
    return None
