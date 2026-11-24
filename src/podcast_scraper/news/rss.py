"""The RSS/Atom transport for news: lists a feed's new articles as URLs to fetch (E05).

The feed is read through the shared downloader (conditional GET when the run enables it). The
cursor is the newest ``published_at`` listed; the next run lists only articles published after it.
After a cursor the oldest are taken first, so a feed that publishes more than its cap between runs
is caught up over the next runs instead of losing the ones in between; the first run (no cursor)
takes the newest. An article without a date is listed every run, and its id makes it the same item
each time.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, List, Optional, Tuple
from urllib.parse import urlsplit

from ..content_types import SourceItem, Transport
from ..rss import downloader
from . import listing
from .fetch import source_spec, TIMEOUT_S, user_agent

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


def _entry_iso(entry: Any) -> Optional[str]:
    return _iso(entry.get("published_parsed") or entry.get("updated_parsed"))


def _in_cap_order(entries: List[Any], cursor: Optional[str]) -> List[Any]:
    """Dated entries oldest first after a cursor, newest first without one; then undated ones in
    feed order. The cap takes from the front."""

    def key(e: Any) -> str:
        return _entry_iso(e) or ""

    dated = sorted((e for e in entries if key(e)), key=key, reverse=not cursor)
    return dated + [e for e in entries if not key(e)]


def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[SourceItem], Optional[str]]:
    """The feed's articles published after ``cursor`` (oldest first; newest first without a
    cursor), at most the source's ``max_items``, filtered by its URL patterns."""
    import feedparser

    url = str(getattr(source, "url", "") or "")
    resp = downloader.fetch_rss_feed_url(url, user_agent(source), TIMEOUT_S)
    if resp is None:
        logger.warning("news rss: feed %s could not be fetched; cursor kept", url)
        return [], cursor
    parsed = feedparser.parse(resp.content)
    feed_language = (parsed.feed.get("language") or "").split("-")[0].lower() or None
    spec = source_spec(source)
    include = re.compile(spec.include_url_pattern) if spec.include_url_pattern else None
    exclude = re.compile(spec.exclude_url_pattern) if spec.exclude_url_pattern else None
    items: List[SourceItem] = []
    newest = cursor
    waiting = 0
    for entry in _in_cap_order(parsed.entries, cursor):
        link = entry.get("link") or ""
        # feedparser copies a permalink guid into ``link``; only an absolute web URL is an article.
        if urlsplit(link).scheme not in ("http", "https") or not urlsplit(link).netloc:
            continue
        if (include and not include.search(link)) or (exclude and exclude.search(link)):
            continue
        published = _iso(entry.get("published_parsed") or entry.get("updated_parsed"))
        if cursor and published and published <= cursor:
            continue
        waiting += 1
        if len(items) >= spec.max_items:
            continue
        guid = entry.get("id") or link
        metadata = {
            "title": entry.get("title"),
            "published_at": published,
            "description": entry.get("summary"),
            "byline_raw": entry.get("author") or None,
            "transport_ids": {NAME: guid},
            "feed_language": feed_language,
            # The feed's own text for the item: its full content, else its description.
            "feed_html": (entry.get("content") or [{}])[0].get("value") or entry.get("summary"),
        }
        items.append(SourceItem(item_id=guid, metadata=metadata, url=link))
        if published and (newest is None or published > newest):
            newest = published
    dated = [d for d in map(_entry_iso, parsed.entries) if d]
    listing.report(
        NAME,
        source,
        cursor_before=cursor,
        cursor_after=newest,
        waiting=waiting,
        taken=len(items),
        oldest_listed=min(dated) if dated else None,
    )
    return items, newest


TRANSPORT = Transport(name=NAME, list_items=list_items)
