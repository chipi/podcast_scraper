"""The catalog from corpus rows: ``CatalogEpisodeRow`` built from ``intents.catalog_items``.

Every field is a value the writers stored, taken by the catalog's own reader for it (public's
type_attrs writer and ``corpus_rows.item_row``), so a row equals the one built from files:

- the item's own fields are its columns and its ``type_attrs`` keys, named after the
  ``CatalogEpisodeRow`` field each one fills;
- the feed fields are the ones this item states (``source_attrs``), as the file catalog reads
  them per episode, not the source's merged row;
- the artwork paths are stored raw and checked on disk here (``_verified_artwork_relpath``), as
  the file catalog checks them when it reads, so artwork downloaded later shows;
- the GI, KG and bridge paths are derived from ``metadata_relative_path`` as the file catalog
  derives them; ``has_gi`` and ``has_kg`` are the item's columns.

An item with no ``metadata_relative_path`` has no catalog row: every surface keys on it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

from sqlalchemy.engine import Connection

from podcast_scraper.builders.bridge_artifact_paths import bridge_json_path_adjacent_to_metadata
from podcast_scraper.content_types import PODCAST_ONLY
from podcast_scraper.db import intents
from podcast_scraper.server.corpus_catalog import (
    _gi_kg_relpaths_from_metadata,
    _verified_artwork_relpath,
    CatalogEpisodeRow,
)


def _text(attrs: Mapping[str, Any], key: str) -> Optional[str]:
    value = attrs.get(key)
    return value if isinstance(value, str) else None


def _texts(attrs: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = attrs.get(key)
    if not isinstance(value, list):
        return ()
    return tuple(v for v in value if isinstance(v, str))


def _int(attrs: Mapping[str, Any], key: str) -> Optional[int]:
    value = attrs.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _publish_date(published_at: Optional[datetime]) -> Optional[str]:
    if published_at is None:
        return None
    return published_at.astimezone(timezone.utc).date().isoformat()


def catalog_row(corpus_root: Path, item: Mapping[str, Any]) -> Optional[CatalogEpisodeRow]:
    """The catalog row of one ``intents.catalog_items`` row, or ``None`` when it names no
    metadata path."""
    attrs: Mapping[str, Any] = item.get("type_attrs") or {}
    stated: Mapping[str, Any] = item.get("source_attrs") or {}
    feed: Mapping[str, Any] = stated.get("type_attrs") or {}
    relpath = _text(attrs, "metadata_relative_path")
    if not relpath:
        return None
    gi_rel, kg_rel = _gi_kg_relpaths_from_metadata(relpath)
    return CatalogEpisodeRow(
        metadata_relative_path=relpath,
        feed_id=item.get("raw_feed_id") or "",
        feed_title=_text(stated, "title"),
        episode_id=item.get("external_item_id"),
        episode_title=item["title"],
        publish_date=_publish_date(item.get("published_at")),
        summary_title=_text(attrs, "summary_title"),
        summary_bullets=_texts(attrs, "summary_bullets"),
        summary_text=_text(attrs, "summary_text"),
        gi_relative_path=gi_rel,
        kg_relative_path=kg_rel,
        bridge_relative_path=bridge_json_path_adjacent_to_metadata(relpath),
        has_gi=bool(item.get("has_gi")),
        has_kg=bool(item.get("has_kg")),
        has_bridge=attrs.get("has_bridge") is True,
        feed_image_url=_text(feed, "feed_image_url"),
        episode_image_url=_text(attrs, "episode_image_url"),
        duration_seconds=item.get("duration_s"),
        episode_number=_int(attrs, "episode_number"),
        feed_image_local_relpath=_verified_artwork_relpath(
            corpus_root, _text(attrs, "feed_image_local_relpath")
        ),
        episode_image_local_relpath=_verified_artwork_relpath(
            corpus_root, _text(attrs, "episode_image_local_relpath")
        ),
        feed_rss_url=_text(feed, "feed_rss_url"),
        feed_description=_text(feed, "feed_description"),
        episode_description=_text(attrs, "episode_description"),
        feed_category=_text(stated, "category"),
        feed_authors=_texts(feed, "feed_authors"),
        feed_language=_text(stated, "language"),
        episode_language=item.get("language"),
        feed_last_updated=_text(feed, "feed_last_updated"),
        feed_licence_json=_text(stated, "licence"),
        episode_licence_json=_text(attrs, "episode_licence_json"),
    )


def catalog_from_rows(
    conn: Connection, corpus_root: Path, *, content_types: Collection[str] = PODCAST_ONLY
) -> list[CatalogEpisodeRow]:
    """The catalog of the served items of ``content_types``, in the file catalog's order."""
    rows = [catalog_row(corpus_root, item) for item in intents.catalog_items(conn, content_types)]
    out = [row for row in rows if row is not None]
    out.sort(key=lambda r: r.sort_key())
    return out
