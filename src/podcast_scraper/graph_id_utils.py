"""Stable graph node ids for GI and KG artifacts.

Episode nodes share ``episode:{episode_id}`` across layers. Topic, Entity, and Person
use global slug-style ids. Insight and Quote
use opaque hashes keyed by episode + content so they stay unique without embedding
episode id in the string (``properties.episode_id`` remains the anchor).

Canonical label→slug rules live in ``podcast_scraper.identity.slugify``;
this module applies graph-specific fallbacks (``topic``, ``unknown``) and length caps.
"""

from __future__ import annotations

import hashlib
from typing import Any

from podcast_scraper.identity.slugify import slugify as canonical_slugify


def slugify_label(label: str, max_len: int = 80) -> str:
    """Lowercase filesystem-safe slug from a human label (CIL slugify + KG max length).

    Empty or unslugifiable labels return ``topic`` so topic dedupe in GI/KG stays safe.
    """
    base = (label or "").strip()
    if not base:
        return "topic"
    try:
        s = canonical_slugify(base)
    except ValueError:
        return "topic"
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s


def episode_node_id(episode_id: str) -> str:
    """Episode anchor node id (shared by GI and KG)."""
    return f"episode:{episode_id}"


def topic_node_id_from_slug(slug: str) -> str:
    """Topic node id from an already-normalized slug."""
    return f"topic:{slug}"


def entity_node_id(entity_kind: str, name: str) -> str:
    """KG entity node id: ``person:{slug}`` or ``org:{slug}``."""
    ek = entity_kind if entity_kind in ("person", "organization") else "person"
    base = (name or "").strip()
    slug = slugify_label(base) if base else "unknown"
    if ek == "organization":
        return f"org:{slug}"
    return f"person:{slug}"


def person_node_id(display_name: str) -> str:
    """GIL person node id from diarization / display name (global by canonical slug)."""
    base = (display_name or "").strip()
    if not base:
        return "person:unknown"
    try:
        slug = canonical_slugify(base)
    except ValueError:
        return "person:unknown"
    return f"person:{slug}"


def gil_insight_node_id(episode_id: str, index: int, insight_text: str) -> str:
    """Opaque insight id (stable for fixed episode, index, and text prefix)."""
    text_key = (insight_text or "").strip()[:2000]
    payload = f"{episode_id}\0{index}\0{text_key}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:16]
    return f"insight:{h}"


def gil_quote_node_id(
    episode_id: str,
    quote_index: int,
    quote_text: str,
    char_start: Any,
    char_end: Any,
) -> str:
    """Opaque quote id (stable for fixed episode, index, span, and text prefix)."""
    text_key = (quote_text or "").strip()[:2000]
    payload = f"{episode_id}\0{quote_index}\0{text_key}\0{char_start}\0{char_end}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()[:16]
    return f"quote:{h}"
