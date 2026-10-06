"""Which kind of market a symbol is, so news can be mapped to the symbols it moves."""
from __future__ import annotations

from ..broker.sim import is_fx_pair

INDICES = {"US30", "US500", "USTECH", "RUSS2000", "NAS100", "US100", "SPX500", "UK100", "DE40", "F40", "JP225",
           "HK50", "AUS200", "CHINA50", "ES35", "NETH25", "STOXX50", "SWI20"}
CRYPTO = ("BTC", "ETH", "SOL", "LTC", "XRP")


def asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s in INDICES:
        return "index"
    if s.startswith("XAU"):
        return "gold"
    if s.startswith(("XAG", "XPT", "XPD")):
        return "metal"
    if s.startswith(("XTI", "XBR")) or "OIL" in s or "WTI" in s:
        return "oil"
    if s.startswith("XNG"):
        return "gas"
    if s.startswith(CRYPTO):
        return "crypto"
    if is_fx_pair(s):
        return "fx"
    return "stock"


US_INDICES = {"US30", "US500", "USTECH", "RUSS2000", "NAS100", "US100", "SPX500"}


def usd_driven(symbol: str) -> bool:
    """Moved by US macro data (CPI, Fed, jobs): US indices and stocks, and anything priced in USD."""
    s = symbol.upper()
    return s in US_INDICES or asset_class(s) == "stock" or s.endswith("USD")
