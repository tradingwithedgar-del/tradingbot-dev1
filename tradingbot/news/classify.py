"""Turn headlines into per-symbol market impact.

ClaudeClassifier (used when ANTHROPIC_API_KEY is set) reads a batch of headlines and returns,
for each, how important it is and which traded symbols it should push up or down.
KeywordClassifier is a free, cruder fallback.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from .assets import asset_class
from .headlines import Headline

log = logging.getLogger(__name__)
CATEGORIES = ["geopolitics", "central_bank", "macro_data", "earnings", "energy", "crypto", "company", "other"]


@dataclass
class Assessment:
    impact: int = 0                       # 0 noise, 1 minor, 2 notable, 3 major market mover
    category: str = "other"
    summary: str = ""
    effects: dict[str, int] = field(default_factory=dict)   # symbol -> +1 up / -1 down
    source: str = "keywords"


# --- keyword fallback ----------------------------------------------------------------
# (pattern, impact, category, {asset_class: direction})
RULES = [
    (r"\b(war|invasion|invades|missile|airstrike|air strike|bombing|military strike|attack on)\b", 3, "geopolitics",
     {"index": -1, "stock": -1, "gold": +1, "oil": +1, "crypto": -1}),
    (r"\b(nuclear)\b", 3, "geopolitics", {"index": -1, "stock": -1, "gold": +1, "crypto": -1}),
    (r"\b(ceasefire|cease-fire|peace deal|truce|peace talks)\b", 2, "geopolitics",
     {"index": +1, "stock": +1, "gold": -1, "oil": -1, "crypto": +1}),
    (r"\b(new tariffs?|raises? tariffs?|tariff hike|sanctions on)\b", 2, "geopolitics",
     {"index": -1, "stock": -1, "gold": +1, "crypto": -1}),
    (r"\b(trade deal|tariff pause|cuts? tariffs?|lifts? sanctions)\b", 2, "geopolitics",
     {"index": +1, "stock": +1, "gold": -1, "crypto": +1}),
    (r"\b(cuts? (interest )?rates?|rate cut|dovish)\b", 2, "central_bank",
     {"index": +1, "stock": +1, "gold": +1, "crypto": +1}),
    (r"\b(raises? (interest )?rates?|rate hike|hawkish)\b", 2, "central_bank",
     {"index": -1, "stock": -1, "gold": -1, "crypto": -1}),
    (r"\b(hotter[- ]than[- ]expected|inflation (surges|jumps|accelerates))\b", 2, "macro_data",
     {"index": -1, "stock": -1, "gold": -1, "crypto": -1}),
    (r"\b(cooler[- ]than[- ]expected|inflation (eases|slows|cools))\b", 2, "macro_data",
     {"index": +1, "stock": +1, "gold": +1, "crypto": +1}),
    (r"\b(opec\+? (cuts?|reduces?)|output cut|supply disruption|strait of hormuz)\b", 2, "energy", {"oil": +1}),
    (r"\b(opec\+? (raises?|boosts?|increases?)|output (hike|increase))\b", 2, "energy", {"oil": -1}),
    (r"\b(bank (collapse|failure|run)|default|recession)\b", 2, "macro_data",
     {"index": -1, "stock": -1, "gold": +1, "crypto": -1}),
    (r"\b(sec approves|etf approval|etf inflows)\b", 2, "crypto", {"crypto": +1}),
    (r"\b(exchange hack|crypto ban|sec sues)\b", 2, "crypto", {"crypto": -1}),
]


# Headlines that use market words in a non-market sense (games, crime, history, opinion...).
NOT_MARKET = re.compile(
    r"\b(game|gaming|gears of war|xbox|playstation|nintendo|steam|walkthrough|trailer|movie|film|tv series|episode|"
    r"season \d|review|home invasion|intruder|homeowner|burglar|police|sheriff|suspect|murder|stabbing|shooting|"
    r"shots fired|cold war|memories|remember|recalls|anniversary|history of|why has|explainer|opinion|podcast|"
    r"quiz|recipe|celebrity|nfl|nba|mlb|soccer|football)\b", re.I)
# Geopolitical headlines only count when they involve a market-relevant actor or place.
MARKET_ACTORS = re.compile(
    r"\b(iran|israel|gaza|hamas|hezbollah|houthi|saudi|opec|hormuz|red sea|russia|ukraine|moscow|kyiv|china|beijing|"
    r"taiwan|north korea|nato|pentagon|white house|trump|biden|u\.?s\.? (strikes?|troops|military)|federal reserve|fed)\b",
    re.I)
KEYWORD_MAX_IMPACT = 2   # keyword reading can't judge context, so it never triggers the impact-3 protections


class KeywordClassifier:
    """Free fallback. Crude: it only gives the learner context; it never rates a headline as a
    major (impact 3) market mover, so it can't trigger trade protection or block trades on its own.
    Set ANTHROPIC_API_KEY to let Claude judge headlines properly."""

    name = "keywords"

    def classify(self, headlines: list[Headline], symbols: list[str]) -> list[Assessment]:
        out = []
        for h in headlines:
            a = Assessment(summary=h.title)
            if NOT_MARKET.search(h.title):
                out.append(a)
                continue
            for pattern, impact, cat, dirs in RULES:
                if cat == "geopolitics" and not MARKET_ACTORS.search(h.title):
                    continue
                if re.search(pattern, h.title, re.I):
                    impact = min(impact, KEYWORD_MAX_IMPACT)
                    if impact > a.impact:
                        a.impact, a.category = impact, cat
                    for s in symbols:
                        d = dirs.get(asset_class(s))
                        if d:
                            a.effects[s] = d
            out.append(a)
        return out


# --- Claude ------------------------------------------------------------------------------
SYSTEM = """You are the news desk for an automated trading agent. You read breaking headlines and judge,
for each one, whether it is likely to move any of the instruments being traded in the next hour, and in
which direction. Be conservative: most headlines are noise (impact 0-1). Use impact 3 only for genuine
market movers (war escalation, surprise central-bank action, major sanctions/tariffs, big inflation or
jobs surprises, OPEC decisions, earnings shocks of mega-caps, exchange/ETF news for crypto).
Only list an instrument in effects when you expect a clear directional move; leave it out when unsure.
Old news, opinion pieces, previews and recaps of moves that already happened are impact 0-1."""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "impact": {"type": "integer", "enum": [0, 1, 2, 3]},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "summary": {"type": "string"},
                    "effects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "direction": {"type": "string", "enum": ["up", "down"]},
                            },
                            "required": ["symbol", "direction"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["index", "impact", "category", "summary", "effects"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


class ClaudeClassifier:
    name = "claude"

    def __init__(self, model: str) -> None:
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.fallback = KeywordClassifier()

    def classify(self, headlines: list[Headline], symbols: list[str]) -> list[Assessment]:
        if not headlines:
            return []
        lines = "\n".join(f"{i}. [{h.time:%Y-%m-%d %H:%M} UTC, {h.source}] {h.title}" for i, h in enumerate(headlines))
        prompt = (f"Instruments traded: {', '.join(symbols)} (USTECH = Nasdaq 100, US500 = S&P 500, US30 = Dow, "
                  f"XAUUSD = gold, XTIUSD = WTI crude oil).\n\nHeadlines:\n{lines}\n\n"
                  "Return one item per headline, using its index.")
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self.anthropic.AuthenticationError:
            log.error("ANTHROPIC_API_KEY rejected - falling back to keyword news reading")
            return self.fallback.classify(headlines, symbols)
        except self.anthropic.RateLimitError:
            log.warning("Claude rate limited - keyword news reading for this batch")
            return self.fallback.classify(headlines, symbols)
        except self.anthropic.APIStatusError as e:
            log.warning("Claude news call failed (%s) - keyword fallback", e.status_code)
            return self.fallback.classify(headlines, symbols)
        except self.anthropic.APIConnectionError:
            log.warning("Claude unreachable - keyword fallback")
            return self.fallback.classify(headlines, symbols)
        if response.stop_reason in ("refusal", "max_tokens"):
            log.warning("Claude news call ended with %s - keyword fallback", response.stop_reason)
            return self.fallback.classify(headlines, symbols)
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            items = json.loads(text)["items"]
        except (ValueError, KeyError):
            return self.fallback.classify(headlines, symbols)
        out = [Assessment(summary=h.title, source="claude") for h in headlines]
        wanted = set(symbols)
        for it in items:
            i = it.get("index")
            if not isinstance(i, int) or not 0 <= i < len(out):
                continue
            a = out[i]
            a.impact, a.category, a.summary = int(it["impact"]), it["category"], it["summary"]
            a.effects = {e["symbol"]: (1 if e["direction"] == "up" else -1) for e in it["effects"] if e["symbol"] in wanted}
        return out


def make_classifier(model: str):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ClaudeClassifier(model)
        except ImportError:
            log.warning("anthropic package not installed - keyword news reading")
    return KeywordClassifier()
