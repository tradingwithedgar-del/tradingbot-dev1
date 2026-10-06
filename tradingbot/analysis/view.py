"""Snapshot of what the agent sees on a symbol, for the dashboard chart.

Saved every time a bar closes: candles, EMAs, RSI, MACD, active supply/demand zones, labelled
swing points (HH/HL/LH/LL), recent divergences, and the agent's read of trend, bias, regime and news.
"""
from __future__ import annotations

import math

from .context import MarketContext

VIEW_BARS = 160


def _num(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, 6)


def chart_view(ctx: MarketContext, bars: int = VIEW_BARS) -> dict:
    df = ctx.df
    start = max(0, len(df) - bars)
    idx = df.index
    t = [ts.isoformat() for ts in idx[start:]]

    def series(s) -> list:
        return [_num(v) for v in s.iloc[start:]]

    macd = ctx.series("macd")
    zones = []
    for z in ctx.zones:
        if z.created_idx < start - 200:   # long-forgotten zones aren't drawn
            continue
        zones.append({
            "kind": z.kind, "top": z.top, "bottom": z.bottom,
            "from": idx[max(z.base_start, start)].isoformat(), "fresh": z.fresh, "touches": z.touches,
            "strength": round(z.impulse_atr, 2),
        })
    swings = [{"t": idx[s.idx].isoformat(), "price": s.price, "kind": s.kind, "label": s.label}
              for s in ctx.swings if s.idx >= start]
    divs = []
    for d in ctx.divergences:
        if d.first.idx >= start and ctx.i - d.second.confirmed_at <= 60:
            divs.append({"kind": d.kind, "osc": d.oscillator,
                         "t1": idx[d.first.idx].isoformat(), "p1": d.first.price,
                         "t2": idx[d.second.idx].isoformat(), "p2": d.second.price})
    return {
        "symbol": ctx.symbol,
        "updated": idx[-1].isoformat(),
        "t": t,
        "open": series(df["open"]), "high": series(df["high"]), "low": series(df["low"]), "close": series(df["close"]),
        "ema50": series(ctx.series("ema", 50)), "ema200": series(ctx.series("ema", 200)),
        "rsi": series(ctx.series("rsi", 14)),
        "macd_hist": series(macd["hist"]),
        "zones": zones, "swings": swings, "divergences": divs,
        "read": {
            "trend": ctx.trend, "bias": ctx.bias, "regime": ctx.regime["regime"],
            "atr": _num(ctx.atr_now), "rsi": _num(ctx.series("rsi", 14).iloc[-1]),
            "news": ctx.news or {},
        },
    }
