"""Canonical slugifier for CIL identifiers.

See ``docs/guides/GIL_KG_CIL_CROSS_LAYER.md`` for canonical identity rules (shared
namespace).
"""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    """Canonical slugifier for CIL identifiers.

    Behaviour:
    - Unicode normalise (NFKD), strip non-ASCII (diacritics dropped for stable IDs)
    - Lowercase
    - Replace whitespace and punctuation with hyphens
    - Collapse consecutive hyphens
    - Strip leading/trailing hyphens
    - Raise ValueError if result is empty (preserves original input in error message)

    Args:
        text: Raw human-readable label or name.

    Returns:
        Non-empty ASCII slug.

    Raises:
        ValueError: If the slug is empty after normalisation.
    """
    original = text
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    if not text:
        raise ValueError(f"Slug is empty after normalisation of: {original!r}")
    return text


def person_id(name: str) -> str:
    """Return canonical ``person:{slug}`` (Phase 2+); slugifier is shared from Phase 1."""
    return f"person:{slugify(name)}"


def org_id(name: str) -> str:
    """Return canonical ``org:{slug}`` (Phase 2+)."""
    return f"org:{slugify(name)}"


def topic_id(label: str) -> str:
    """Return canonical ``topic:{slug}`` (aligned with existing ``topic:`` nodes)."""
    return f"topic:{slugify(label)}"
