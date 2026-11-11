"""The content type of a corpus item (RFC-134, ADR-177).

Podcast episodes are the only type written today, and their metadata carries no ``content_type``
field until R3b (ADR-178): an absent field reads as a podcast episode. Shared reads (the catalog
builders, the search indexer's walk) take only podcast episodes unless a caller asks for other
types, so a surface sees a new type only when it opts in (ADR-177 decision 8).
"""

from __future__ import annotations

from typing import Any, Collection

PODCAST_EPISODE = "podcast_episode"

#: The default for every shared read: podcast episodes only.
PODCAST_ONLY: frozenset[str] = frozenset({PODCAST_EPISODE})


def item_content_type(doc: Any) -> str:
    """The type a metadata document declares; none declared means a podcast episode."""
    if isinstance(doc, dict):
        declared = doc.get("content_type")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    return PODCAST_EPISODE


def is_wanted(doc: Any, content_types: Collection[str]) -> bool:
    """Whether a reader that takes ``content_types`` reads this document."""
    return item_content_type(doc) in content_types
