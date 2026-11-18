"""A news item's metadata document, exactly as the News 1.0 contract has it
(``docs/architecture/corpus/news-item-1.0.md``): shared fields at the top level, the type's own
under ``news_article``.

Built from the document the shared metadata stage made (for its summary) and the item's context:
its ids, times, and the fields its transport and fetch gave.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

from ..content_types import ItemContext
from ..news_items import (
    EDITORIAL_TYPES,
    F_BYLINE,
    F_CONTENT_TYPE,
    F_EDITORIAL_TYPE,
    F_ITEM_ID,
    F_LANGUAGE,
    F_PAYWALLED,
    F_PUBLISHER_ID,
    F_SERVEABLE,
    F_SOURCE_ID,
    F_SUMMARY,
    F_TEXT_STATUS,
    F_TITLE,
    F_URL,
    NEWS_ARTICLE,
    SERVEABLE_METADATA,
)
from ..publishers import load_registry, Publisher, Registry
from ..utils.canonical_url import canonical_url
from . import byline

#: What a source with no stated rights may show: the narrowest value that still shows the item.
DEFAULT_SERVEABLE = SERVEABLE_METADATA


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _summary_text(doc: Mapping[str, Any]) -> Optional[str]:
    """The summary the platform wrote, as one string (the summary stage's ``short_summary``)."""
    summary = doc.get("summary")
    if not isinstance(summary, Mapping):
        return None
    bullets = [b for b in summary.get("bullets") or [] if isinstance(b, str) and b.strip()]
    text = " ".join(bullets[:2]) if bullets else str(summary.get("raw_text") or "")
    return text.strip() or None


def publisher(registry: Registry, source_id: Optional[str], url: str) -> Optional[Publisher]:
    """The source's publisher, else the publisher whose domain the URL's host matches."""
    pub = registry.for_source(source_id) if source_id else None
    if pub is None:
        host = urlsplit(url).hostname or ""
        pub = registry.for_domain(host) if host else None
    return pub


def shape(
    doc: Dict[str, Any], item: ItemContext, *, registry: Optional[Registry] = None
) -> Dict[str, Any]:
    """The contract's document for ``item``; ``doc`` is the shared stage's document."""
    f = item.fields
    url = canonical_url(str(f.get("url") or ""))
    reg = registry if registry is not None else load_registry()
    pub = publisher(reg, item.source_id, url)
    raw_episode = doc.get("episode")
    episode: Mapping[str, Any] = raw_episode if isinstance(raw_episode, Mapping) else {}
    out: Dict[str, Any] = {
        F_CONTENT_TYPE: NEWS_ARTICLE,
        F_ITEM_ID: item.item_id,
        F_URL: url,
        F_SOURCE_ID: item.source_id,
        F_TITLE: f.get("title") or episode.get("title"),
        **dict(item.times),
        "ingested_at": _now(),
        F_LANGUAGE: f.get("language"),
        "licence": f.get("licence"),
        F_PAYWALLED: f.get("paywalled"),
        F_SERVEABLE: f.get("serveable") or DEFAULT_SERVEABLE,
        "mineable": f.get("mineable", True),
    }
    if pub is not None:
        out[F_PUBLISHER_ID] = pub.id
    summary = _summary_text(doc)
    if summary:
        out[F_SUMMARY] = summary
    out["fetched_at"] = f.get("fetched_at")
    out["body_sha256"] = f.get("body_sha256")
    raw = f.get("byline_raw")
    editorial = f.get("editorial_type")
    own: Dict[str, Any] = {
        "transport_ids": dict(f.get("transport_ids") or {}),
        F_BYLINE: byline.parse(raw, organisations=[pub.name] if pub else []),
        "byline_raw": raw,
        F_EDITORIAL_TYPE: editorial if editorial in EDITORIAL_TYPES else None,
        F_TEXT_STATUS: f.get("text_status"),
    }
    if f.get("live_blog") is True:
        own["live_blog"] = True
    out[NEWS_ARTICLE] = own
    return out
