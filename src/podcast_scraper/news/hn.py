"""The Hacker News transport (Algolia's API): front-page stories as URLs to fetch (E05).

A story links to an article elsewhere; the news type fetches and extracts that page as for an RSS
item, so one article reached through Hacker News and through its publisher's feed is one item
(its id is the canonical URL's). A story with no link (Ask HN, a poll) is not an article and is
skipped. The story's author is who submitted it, not who wrote the article, so it is never a
byline. A story's cursor value is its ``created_at_i``, compared as a number; the run moves the
cursor only past stories that wrote (``workflow.text_in``).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple
from urllib.parse import urlsplit

from ..content_types import SourceItem, Transport
from ..rss import downloader
from .fetch import source_spec, TIMEOUT_S, user_agent

logger = logging.getLogger(__name__)

NAME = "hn"


def _iso(epoch: Any) -> Optional[str]:
    try:
        moment = datetime.fromtimestamp(int(epoch), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None
    return moment.isoformat().replace("+00:00", "Z")


def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[SourceItem], Optional[str]]:
    """The front page's linked stories created after ``cursor`` (an epoch second), newest first,
    filtered by its URL patterns. The run caps what it fetches at the source's ``max_items``."""
    url = str(getattr(source, "url", "") or "")
    resp = downloader.fetch_url(url, user_agent(source), TIMEOUT_S)
    if resp is None:
        # As a podcast feed's (stages/scraping.py): a soft feed incident, the cursor kept.
        raise ValueError(f"Failed to fetch RSS feed. (news hn: {url})")
    try:
        hits = resp.json().get("hits") or []
    except ValueError as exc:
        raise ValueError(f"Failed to parse RSS XML: news hn: {url} did not answer JSON") from exc
    spec = source_spec(source)
    include = re.compile(spec.include_url_pattern) if spec.include_url_pattern else None
    exclude = re.compile(spec.exclude_url_pattern) if spec.exclude_url_pattern else None
    after = int(cursor) if cursor and cursor.isdigit() else None
    items: List[SourceItem] = []
    newest = after
    for hit in sorted(hits, key=lambda h: int(h.get("created_at_i") or 0), reverse=True):
        link = str(hit.get("url") or "")
        created = int(hit.get("created_at_i") or 0)
        if urlsplit(link).scheme not in ("http", "https") or not urlsplit(link).netloc:
            continue
        if (include and not include.search(link)) or (exclude and exclude.search(link)):
            continue
        if after is not None and created <= after:
            continue
        story = str(hit.get("objectID") or "")
        metadata = {
            "title": hit.get("title"),
            "published_at": _iso(created),
            "transport_ids": {NAME: story},
        }
        items.append(
            SourceItem(item_id=story or link, metadata=metadata, url=link, cursor=str(created))
        )
        if newest is None or created > newest:
            newest = created
    return items, (str(newest) if newest is not None else cursor)


TRANSPORT = Transport(
    name=NAME,
    list_items=list_items,
    cursor_key=int,
    max_items=lambda source: source_spec(source).max_items,
)
