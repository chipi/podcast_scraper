"""The news type's fetch: an item its transport listed as a URL becomes an item with its text.

robots.txt first (RFC 9309), then the page through the shared downloader (its retries, SSRF guard,
circuit breaker and the run's per-host throttle), then the main text (``extract``). The page is
kept as the item's raw cache (RFC-134 §5).
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from .. import config
from ..content_types import SourceItem
from ..rss import downloader
from ..rss.feeds_spec import NewsSourceSpec
from . import extract

logger = logging.getLogger(__name__)

TIMEOUT_S = 30

_ROBOTS: Dict[str, Optional[RobotFileParser]] = {}
_ROBOTS_LOCK = threading.Lock()


def source_spec(source: Any) -> NewsSourceSpec:
    """The source's ``news:`` settings from its feeds.spec entry; the defaults when it has none."""
    spec = getattr(source, "news", None)
    return spec if isinstance(spec, NewsSourceSpec) else NewsSourceSpec()


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


#: Where an item's language came from (recorded as ``news_article.language_source``).
LANGUAGE_OVERRIDE = "override"
LANGUAGE_PAGE = "page"
LANGUAGE_FEED = "rss"
LANGUAGE_DETECTED = "detected"
LANGUAGE_NONE = "none"
#: From this many words an article's text overrules a declared language it contradicts (an
#: episode's transcript needs 200). Measured on all8 (257 items, 45 under 200 words): any
#: threshold from 20 to 75 changes exactly one item, a 75-word English post under a German
#: server's lang=de, rightly; 100 and above change none.
NEWS_LANGUAGE_MIN_WORDS = 50


def resolve_language(
    text: str, *, override: Any, page: Optional[str], feed: Optional[str]
) -> Tuple[Optional[str], str]:
    """``(language, source)`` for one item, as an episode's is resolved (``languages``): the
    source's override, else the page's ``<html lang>``, else the feed's ``<language>``; then the
    text itself, which wins when it clearly reads as another language than the declared one (the
    podcast's ``transcript_language_contradiction``), and decides when nothing was declared."""
    from ..languages import normalize_language_tag, primary_language
    from ..languages_guard import transcript_language_contradiction
    from ..text_language import detect_language, words

    declared, source = None, LANGUAGE_NONE
    for value, origin in (
        (override, LANGUAGE_OVERRIDE),
        (page, LANGUAGE_PAGE),
        (feed, LANGUAGE_FEED),
    ):
        tag = normalize_language_tag(value) if isinstance(value, str) and value.strip() else None
        if tag:
            declared, source = primary_language(tag), origin
            break
    detected = detect_language(words(text))
    if declared is None:
        return (detected, LANGUAGE_DETECTED) if detected else (None, LANGUAGE_NONE)
    if source != LANGUAGE_OVERRIDE and detected and detected != declared:
        if transcript_language_contradiction(text, declared, min_words=NEWS_LANGUAGE_MIN_WORDS):
            return detected, LANGUAGE_DETECTED
    return declared, source


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
    found = extract.main_text(
        html,
        item.url,
        feed_html=item.metadata.get("feed_html"),
        text_from=source_spec(source).text_from,
    )
    if found is None:
        logger.warning("news: no text found on %s; skipped", item.url)
        return None
    return complete(
        item,
        source,
        found.text,
        method=found.method,
        declared_language=found.language,
        byline_raw=item.metadata.get("byline_raw") or extract.page_author(html, item.url),
        raw=html,
        raw_suffix=".html",
    )


def complete(
    item: SourceItem,
    source: Any,
    text: str,
    *,
    method: str,
    declared_language: Optional[str],
    byline_raw: Optional[str],
    raw: Optional[str],
    raw_suffix: str,
) -> Optional[SourceItem]:
    """``item`` with its text and every field the contract asks of the writer: its language
    (resolved and gated as an episode's), its source's rights, its text status, its byline.
    ``None`` when its language is not one we ingest. Shared by the page fetch and by a
    transport whose API gives the text itself (the Guardian)."""
    from ..languages import is_language_enabled

    url = item.url or ""
    language, language_source = resolve_language(
        text,
        override=getattr(source, "language_override", None),
        page=declared_language,
        feed=item.metadata.get("feed_language"),
    )
    if not is_language_enabled(language):
        # As an episode in a language we do not ingest: skipped with the reason, never written
        # as English (config/languages.yaml, D-29).
        logger.info(
            "news: %s is in %r (from %s), not an enabled language; skipped",
            url,
            language,
            language_source,
        )
        return None
    spec = source_spec(source)
    fields = {
        **item.metadata,
        "fetched_at": _now(),
        "text_status": spec.text_status or extract.text_status(text, url),
        "language": language,
        "language_source": language_source,
        "extraction": method,
        "serveable": spec.serveable,
        "mineable": spec.mineable,
        "licence": spec.licence,
        "paywalled": spec.paywalled,
        "editorial_type": spec.editorial_type,
        "byline_raw": byline_raw,
        "personal_publisher": spec.personal,
        "translate": spec.translate,
    }
    if spec.live_blog_url_pattern and re.search(spec.live_blog_url_pattern, url):
        fields["live_blog"] = True
    return SourceItem(
        item_id=item.item_id, metadata=fields, text=text, url=url, raw=raw, raw_suffix=raw_suffix
    )
