"""Breaking headlines from public RSS feeds (no keys needed)."""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd
import requests

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0 (tradingbot news monitor)"}


def google_news(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}+when:1h&hl=en-US&gl=US&ceid=US:en"


DEFAULT_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",          # CNBC top news
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",      # MarketWatch top stories
    google_news("stock market OR Wall Street OR Nasdaq"),
    google_news("Federal Reserve OR inflation OR interest rates"),
    google_news("war OR missile OR invasion OR sanctions OR ceasefire OR tariffs"),
    google_news("oil prices OR OPEC OR crude"),
    google_news("gold price"),
    google_news("bitcoin OR ethereum OR crypto"),
    google_news("Nvidia OR Apple OR Tesla"),
]


@dataclass
class Headline:
    time: pd.Timestamp
    title: str
    source: str
    link: str

    @property
    def key(self) -> str:
        norm = re.sub(r"\W+", " ", self.title.lower()).strip()
        return hashlib.sha1(norm.encode()).hexdigest()[:16]


def parse_rss(xml_text: str, feed: str = "") -> list[Headline]:
    out = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        src_el = item.find("source")
        source = (src_el.text if src_el is not None and src_el.text else feed.split("/")[2] if "//" in feed else feed)
        # Google News appends " - Source" to titles
        if src_el is not None and src_el.text and title.endswith(" - " + src_el.text):
            title = title[: -len(" - " + src_el.text)]
        try:
            t = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate") or "")).tz_convert("UTC")
        except Exception:
            continue
        out.append(Headline(t, title, source, (item.findtext("link") or "").strip()))
    return out


def fetch_headlines(feeds: list[str], max_age_minutes: int, now: pd.Timestamp) -> list[Headline]:
    seen: dict[str, Headline] = {}
    for f in feeds or DEFAULT_FEEDS:
        try:
            r = requests.get(f, headers=HEADERS, timeout=15)
            r.raise_for_status()
        except Exception as e:
            log.warning("news feed failed (%s): %s", f[:60], e)
            continue
        for h in parse_rss(r.text, f):
            if (now - h.time).total_seconds() <= max_age_minutes * 60 and h.key not in seen:
                seen[h.key] = h
    return sorted(seen.values(), key=lambda h: h.time)
