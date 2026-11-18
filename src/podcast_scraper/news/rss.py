"""The RSS/Atom transport for news: lists a feed's new articles as URLs to fetch (E05).

The feed is read through the shared downloader (conditional GET when the run enables it). The
cursor is the newest ``published_at`` listed; the next run lists only articles published after it.
An article without a date is listed every run, and its id makes it the same item each time.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Tuple
from urllib.parse import urlsplit

from ..content_types import SourceItem, Transport
from ..rss import downloader
from .fetch import TIMEOUT_S, user_agent

logger = logging.getLogger(__name__)

NAME = "rss"


def _iso(parsed: Any) -> Optional[str]:
    """feedparser's UTC ``time.struct_time`` as ISO 8601 UTC."""
    if not parsed:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed)
    except (TypeError, ValueError):
        return None


def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[SourceItem], Optional[str]]:
    """The feed's articles published after ``cursor``, newest cursor last."""
    import feedparser

    url = str(getattr(source, "url", "") or "")
    resp = downloader.fetch_rss_feed_url(url, user_agent(source), TIMEOUT_S)
    if resp is None:
        logger.warning("news rss: feed %s could not be fetched; cursor kept", url)
        return [], cursor
    parsed = feedparser.parse(resp.content)
    feed_language = (parsed.feed.get("language") or "").split("-")[0].lower() or None
    items: List[SourceItem] = []
    newest = cursor
    for entry in parsed.entries:
        link = entry.get("link") or ""
        # feedparser copies a permalink guid into ``link``; only an absolute web URL is an article.
        if urlsplit(link).scheme not in ("http", "https") or not urlsplit(link).netloc:
            continue
        published = _iso(entry.get("published_parsed") or entry.get("updated_parsed"))
        if cursor and published and published <= cursor:
            continue
        guid = entry.get("id") or link
        metadata = {
            "title": entry.get("title"),
            "published_at": published,
            "description": entry.get("summary"),
            "byline_raw": entry.get("author") or None,
            "transport_ids": {NAME: guid},
            "feed_language": feed_language,
        }
        items.append(SourceItem(item_id=guid, metadata=metadata, url=link))
        if published and (newest is None or published > newest):
            newest = published
    return items, newest


TRANSPORT = Transport(name=NAME, list_items=list_items)
