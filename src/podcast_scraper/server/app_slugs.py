"""Episode slug contract for the consumer Learning Platform API (#1067, RFC-098 §4).

Derives a stable, URL-safe ``slug`` per episode from *stable* inputs — the feed id
and episode id — so the consumer API (``/api/app/*``) can address episodes by slug
rather than the opaque internal ``episode_id``. Keying on ``(feed_id, episode_id)``
(both stable across re-scrapes) keeps a slug valid when an episode is re-processed.

No new persistence: slugs are computed on the fly from the catalog (the same
per-request scan the catalog/relational layers already do), consistent with the
"plain files, no DB" decision (PRD-035 D2). A persisted ``episode_slugs`` index can
be added later if the scan cost ever warrants it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from podcast_scraper.identity.slugify import slugify
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)

_HASH_LEN = 10  # hex chars of the stable discriminator suffix


def _feed_slug(feed_id: str) -> str:
    """Readable, URL-safe prefix from a feed id; ``"feed"`` when empty/unslug-able."""
    fid = (feed_id or "").strip()
    if not fid:
        return "feed"
    try:
        return slugify(fid)
    except ValueError:
        return "feed"


def episode_slug(feed_id: str, episode_id: str | None, metadata_relpath: str) -> str:
    """Deterministic, URL-safe slug ``{feed-slug}-{hash}`` for one episode.

    The hash is taken over the stable identity ``(feed_id, episode_id)``; when an
    episode has no ``episode_id`` we fall back to the metadata relpath so every
    episode still gets a deterministic slug. Title/run-suffix never affect the slug,
    so it survives re-scrapes.
    """
    if episode_id:
        key = f"{feed_id}\x00{episode_id}"
    else:
        key = f"{feed_id}\x00path:{metadata_relpath}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LEN]
    return f"{_feed_slug(feed_id)}-{digest}"


def slug_for_row(row: CatalogEpisodeRow) -> str:
    """Slug for a catalog row (thin wrapper over :func:`episode_slug`)."""
    return episode_slug(row.feed_id, row.episode_id, row.metadata_relative_path)


def resolve_slug(corpus_root: Path, slug: str) -> CatalogEpisodeRow | None:
    """Return the (deduplicated) catalog row whose slug matches ``slug``, else ``None``.

    O(episodes) per call — acceptable at corpus scale; swap for a persisted index
    later if needed.
    """
    want = (slug or "").strip()
    if not want:
        return None
    for row in build_catalog_rows_cumulative(corpus_root):
        if slug_for_row(row) == want:
            return row
    return None
