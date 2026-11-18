"""The news type's fetch: an item its transport listed as a URL becomes an item with its text.

robots.txt first (RFC 9309), then the page through the shared downloader (its retries, SSRF guard,
circuit breaker and the run's per-host throttle), then the main text (``extract``). The page is
kept as the item's raw cache (RFC-134 §5).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .. import config
from ..content_types import SourceItem
from ..rss import downloader
from . import extract

logger = logging.getLogger(__name__)

TIMEOUT_S = 30

_ROBOTS: Dict[str, Optional[RobotFileParser]] = {}
_ROBOTS_LOCK = threading.Lock()


def user_agent(source: Any) -> str:
    """The source's own user agent when its definition names one, else the run's default."""
    return str(getattr(source, "user_agent", None) or config.DEFAULT_USER_AGENT)


def _robots(origin: str, agent: str) -> Optional[RobotFileParser]:
    """The origin's parsed robots.txt; an empty parser (allow all) for a 4xx, ``None`` (disallow
    all) for a 5xx or an unreachable host (RFC 9309 §2.3.1)."""
    with _ROBOTS_LOCK:
        if origin in _ROBOTS:
            return _ROBOTS[origin]
    status, body = downloader.http_get_any_status(origin + "/robots.txt", agent, TIMEOUT_S)
    parser: Optional[RobotFileParser] = RobotFileParser()
    if 200 <= status < 300:
        assert parser is not None
        parser.parse(body.decode("utf-8", errors="replace").splitlines())
    elif 400 <= status < 500:
        assert parser is not None
        parser.parse([])
    else:
        parser = None
    with _ROBOTS_LOCK:
        _ROBOTS[origin] = parser
    return parser


def reset_robots_cache() -> None:
    """Forget every origin's robots.txt (tests, and a long-running service between runs)."""
    with _ROBOTS_LOCK:
        _ROBOTS.clear()


def robots_allow(url: str, agent: str) -> bool:
    """Whether robots.txt lets ``agent`` fetch ``url``."""
    parts = urlsplit(url)
    parser = _robots(f"{parts.scheme}://{parts.netloc}", agent)
    return parser is not None and parser.can_fetch(agent, url)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch_body(item: SourceItem, source: Any) -> Optional[SourceItem]:
    """``item`` with its page's main text, or ``None`` when robots.txt refuses it, the fetch fails
    or no text is found."""
    if not item.url:
        return None
    agent = user_agent(source)
    if not robots_allow(item.url, agent):
        logger.info("news: robots.txt disallows %s; skipped", item.url)
        return None
    resp = downloader.fetch_url(item.url, agent, TIMEOUT_S)
    if resp is None:
        return None
    html = resp.text
    found = extract.main_text(html, item.url)
    if found is None:
        logger.warning("news: no text found on %s; skipped", item.url)
        return None
    fields = {
        **item.metadata,
        "fetched_at": _now(),
        "text_status": extract.text_status(found.text, item.url),
        "language": found.language or item.metadata.get("feed_language"),
        "extraction": found.method,
    }
    return SourceItem(
        item_id=item.item_id,
        metadata=fields,
        text=found.text,
        url=item.url,
        raw=html,
        raw_suffix=".html",
    )
