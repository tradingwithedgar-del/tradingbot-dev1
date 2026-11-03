"""Markets that move together trade together.

  * Crypto: BTCUSD is the market mover. Its direction (its open trade, otherwise its higher-timeframe trend) is
    the direction for every other crypto. Until BTC has a direction, the rest wait.
  * Indices: each region moves as a group (US: US30, US500, USTECH, RUSS2000; Europe; Asia-Pacific). When the
    others in the group are going one way, trading the opposite way is not allowed.
  * Stocks follow the indices of their industry (tech follows USTECH, financials/industrials/staples follow the
    Dow and S&P, energy follows oil and the S&P, ...).

"Going a way" = the higher-timeframe structure trend TIIM already trades with (htf.structure_dir), and any
real open trade counts double. A group has a consensus when at least two thirds of the votes agree; with no
consensus the normal rules apply. Blocked setups are still followed virtually.
"""
from __future__ import annotations

CRYPTO_LEADER = "BTCUSD"
CRYPTO_FOLLOWERS = {"BTCEUR", "ETHUSD", "ETHEUR", "SOLUSD", "LTCUSD", "LTCEUR", "XRPUSD"}

US_INDICES = ("US30", "US500", "USTECH", "RUSS2000")
EUROPE = ("DE40", "F40", "UK100", "ES35", "NETH25", "STOXX50", "SWI20")
ASIA = ("JP225", "HK50", "CHINA50", "AUS200")

# industry -> the markets that set the direction for its stocks
SECTORS = {
    "technology": (("USTECH", "US500"), "AAPL ADBE AMD AVGO CRM CSCO IBM INTC META MSFT NFLX NVDA ORCL QCOM TSLA"),
    "china tech": (("CHINA50", "USTECH"), "BABA"),
    "financials": (("US30", "US500"), "BAC BRKB JPM MA V WFC"),
    "healthcare": (("US500", "US30"), "ABBV ABT JNJ LLY MRK PFE TMO UNH"),
    "consumer": (("US500", "US30"), "COST DIS GM HD KO MCD NKE PEP PG WMT CMCSA TMUS VZ"),
    "industrials": (("US30", "US500"), "BA GE UPS"),
    "energy": (("XTIUSD", "US500"), "CVX XOM"),
}
STOCK_ANCHORS = {sym: (name, anchors) for name, (anchors, syms) in SECTORS.items() for sym in syms.split()}
DEFAULT_STOCK = ("US stocks", ("US500", "US30", "USTECH"))
CONSENSUS = 2 / 3


def _vote(sym: str, dirs: dict[str, int], open_sides: dict[str, list[str]]) -> int:
    """+1/-1 for a market going up/down, counting a real open trade twice; 0 = no opinion."""
    v = dirs.get(sym, 0)
    for side in open_sides.get(sym, []):
        v += 2 * (1 if side == "buy" else -1)
    return (v > 0) - (v < 0)


def _consensus(members, dirs, open_sides) -> tuple[int, int, int]:
    """(direction, up_votes, down_votes); direction 0 when there is no two-thirds agreement."""
    up = sum(1 for m in members if _vote(m, dirs, open_sides) > 0)
    down = sum(1 for m in members if _vote(m, dirs, open_sides) < 0)
    n = up + down
    if n and up / n >= CONSENSUS:
        return 1, up, down
    if n and down / n >= CONSENSUS:
        return -1, up, down
    return 0, up, down


def is_us_stock(symbol: str) -> bool:
    from .news.assets import asset_class

    return asset_class(symbol) == "stock"


def requirement(symbol: str, dirs: dict[str, int], open_sides: dict[str, list[str]]) -> tuple[int, str] | None:
    """The direction `symbol` has to trade, as (+1/-1, who sets it), (0, why it must wait), or None (no rule)."""
    if symbol in CRYPTO_FOLLOWERS:
        d = _vote(CRYPTO_LEADER, dirs, open_sides)
        return (d, f"{CRYPTO_LEADER} is going {'up' if d > 0 else 'down'}") if d else (
            0, f"{CRYPTO_LEADER}, the crypto market mover, has no clear direction yet")
    for group, name in ((US_INDICES, "US indices"), (EUROPE, "European indices"), (ASIA, "Asia-Pacific indices")):
        if symbol in group:
            d, up, down = _consensus([m for m in group if m != symbol], dirs, open_sides)
            return (d, f"the other {name} are going {'up' if d > 0 else 'down'} ({up} up, {down} down)") if d else None
    if is_us_stock(symbol):
        name, anchors = STOCK_ANCHORS.get(symbol, DEFAULT_STOCK)
        d, up, down = _consensus(anchors, dirs, open_sides)
        return (d, f"{name}: {'/'.join(anchors)} are going {'up' if d > 0 else 'down'} ({up} up, {down} down)") if d else None
    return None


def check(symbol: str, side: str, dirs: dict[str, int], open_sides: dict[str, list[str]]) -> str | None:
    """Why this trade breaks the group rule, or None."""
    req = requirement(symbol, dirs, open_sides)
    if req is None:
        return None
    d, why = req
    if d == 0:
        return f"waiting: {why}"
    if (d > 0) != (side == "buy"):
        return f"against its group: {why} - a {side} would go the other way"
    return None
