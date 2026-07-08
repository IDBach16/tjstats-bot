"""Wire service (long-form): pull recent FanGraphs & Baseball Savant / MLB.com
Statcast articles as candidate leads the columnist can react to.

Unlike the leaderboard feeds (which hand the writer exact numbers), these leads
carry a short, ATTRIBUTED summary of a real published article. The columnist
writes an ORIGINAL reaction thread that credits the outlet + author and links
back — it never reproduces the article's text (summaries are truncated and the
writer is told to react, not copy).
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests

from .feeds import Lead

log = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")
_DC = "{http://purl.org/dc/elements/1.1/}"

MAX_AGE_DAYS = 4     # only react to fresh articles
PER_SOURCE = 4       # candidate articles kept per source

# Sources must be plain-`requests`-reachable RSS (the bot runs headless on CI).
# Baseball Savant itself publishes no article feed, so its Statcast analysis is
# pulled from the MLB.com news feed and filtered to Statcast-flavored pieces via
# ``statcast_only`` below. Add/remove feeds here — nothing else changes.
ARTICLE_SOURCES = [
    {"outlet": "FanGraphs", "url": "https://blogs.fangraphs.com/feed/",
     "statcast_only": False},
    {"outlet": "Baseball Savant / MLB.com",
     "url": "https://www.mlb.com/feeds/news/rss.xml", "statcast_only": True},
]

# Keep a general feed (MLB.com) down to Statcast-driven analysis. Terms are
# specific on purpose — a bare "expected" also matches "expected to start", so
# only the xStat abbreviations / distinctive tracking terms are listed.
_STATCAST_HINTS = (
    "statcast", "savant", "exit velo", "exit velocity", "launch angle",
    "bat speed", "bat-tracking", "bat tracking", "xwoba", "xera", "xba",
    "expected stats", "expected batting", "barrel rate", "barreled",
    "hard-hit", "hardest-hit", "hardest hit", "sprint speed", "arm strength",
    "spin rate", "pitch shape", "stuff+", "swing path", "run value",
    "whiff rate", "chase rate", "attack angle",
)
# Drop audio / chat / non-article items (mostly FanGraphs).
_SKIP_HINTS = ("episode", "podcast", "chat:", " chat ", "prospects chat",
               "power hour", "mailbag", "livestream", "sunday notes")


def _norm_ws(s: str) -> str:
    """Collapse whitespace and drop non-breaking spaces from feed text."""
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _clean(text: str, limit: int = 600) -> str:
    """Strip HTML tags/entities and collapse whitespace into a short summary."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = _norm_ws(text)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _fresh(pubdate: str) -> bool:
    try:
        dt = parsedate_to_datetime(pubdate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt >= datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
    except Exception:
        return False


def _parse_feed(xml_text: str) -> list[dict]:
    """Parse an RSS 2.0 feed into item dicts (stdlib only, no feedparser)."""
    import xml.etree.ElementTree as ET

    items: list[dict] = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        def g(tag: str) -> str:
            el = it.find(tag)
            return el.text.strip() if el is not None and el.text else ""

        title, link = g("title"), g("link")
        if not title or not link:
            continue
        items.append({
            "title": title,
            "link": link,
            "summary": g("description"),
            "pubDate": g("pubDate"),
            "author": g(f"{_DC}creator"),
        })
    return items


def build_article_leads() -> list[Lead]:
    """Return recent FanGraphs / Savant article leads (kind='article')."""
    leads: list[Lead] = []
    for src in ARTICLE_SOURCES:
        try:
            r = requests.get(src["url"], timeout=20, headers={"User-Agent": _UA})
            r.raise_for_status()
            items = _parse_feed(r.text)
        except Exception:
            log.warning("article feed failed: %s", src["outlet"], exc_info=True)
            continue

        kept = 0
        for it in items:
            if kept >= PER_SOURCE:
                break
            title = _norm_ws(it["title"])
            tl = title.lower()
            if any(h in tl for h in _SKIP_HINTS):
                continue
            if not _fresh(it["pubDate"]):
                continue
            blob = (title + " " + it["summary"]).lower()
            if src["statcast_only"] and not any(h in blob for h in _STATCAST_HINTS):
                continue
            summary = _clean(it["summary"])
            if len(summary) < 40:      # need something real to react to
                continue
            leads.append(Lead(
                kind="article",
                subject=title[:120],
                player_id=0,
                is_pitcher=False,
                angle=f"React to this {src['outlet']} piece and credit the source.",
                rank=kept + 1, total=0, is_red=False,
                facts={
                    "outlet": src["outlet"],
                    "author": _norm_ws(it["author"]),
                    "url": it["link"],
                    "title": title,
                    "summary": summary,
                    "published": it["pubDate"],
                },
            ))
            kept += 1
        log.info("article feed %s: %d leads", src["outlet"], kept)
    return leads
