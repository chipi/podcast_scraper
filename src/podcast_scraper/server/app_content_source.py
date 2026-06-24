"""Pluggable content source for the consumer catalog (#1078, RFC-099 §3).

The catalog list endpoints read episodes through a :class:`ContentSource` so the backend
can evolve without reshaping the API. The MVP implementation, :class:`LocalCorpusSource`,
enumerates the **already-processed local corpus** (PRD-038 local-content MVP) — every
episode it lists is effectively *ready*. When scrape-on-demand (#1069) lands, a
``DiscoverySource`` can implement the same Protocol to surface not-yet-processed content,
with **no change** to the routes or the ``/api/app/episodes`` response shape.

No new persistence (PRD-035 D2): everything is derived from the per-request corpus scan
the catalog/slug layers already perform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Optional, Protocol

from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import (
    _load_metadata_doc,
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
    episode_list_summary_preview,
    episode_list_topics,
    filter_rows,
)
from podcast_scraper.server.schemas import AppEpisodeSummary


@dataclass(frozen=True)
class EpisodeListResult:
    """A page of catalog episodes plus the total matching the filter."""

    items: list[AppEpisodeSummary]
    total: int


class ContentSource(Protocol):
    """Catalog backend the consumer list routes depend on (the #1069 swap seam)."""

    def list_episodes(
        self,
        *,
        feed_id: Optional[str] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> EpisodeListResult:
        """Return a newest-first page of episodes and the total matching the filter."""
        ...


def _has_transcript(corpus_root: Path, metadata_relpath: str) -> bool:
    """Whether the episode's metadata references a transcript file (playable signal)."""
    doc = _load_metadata_doc(corpus_root / metadata_relpath)
    content = doc.get("content") if isinstance(doc, dict) else None
    if not isinstance(content, dict):
        return False
    tr = content.get("transcript_file")
    return isinstance(tr, str) and bool(tr.strip())


def _row_to_summary(corpus_root: Path, row: CatalogEpisodeRow) -> AppEpisodeSummary:
    """Map a catalog row to the consumer card shape (one metadata read for transcript)."""
    has_transcript = _has_transcript(corpus_root, row.metadata_relative_path)
    has_summary = bool(row.summary_title or row.summary_bullets or row.summary_text)
    return AppEpisodeSummary(
        slug=slug_for_row(row),
        title=row.episode_title,
        feed_id=row.feed_id,
        podcast_title=row.feed_title,
        publish_date=row.publish_date,
        duration_seconds=row.duration_seconds,
        episode_image_url=row.episode_image_url,
        feed_image_url=row.feed_image_url,
        status="ready" if has_transcript else "pending",
        summary_preview=episode_list_summary_preview(row),
        topics=episode_list_topics(row.summary_bullets),
        has_transcript=has_transcript,
        has_summary=has_summary,
        has_gi=row.has_gi,
        has_kg=row.has_kg,
        has_bridge=row.has_bridge,
    )


class LocalCorpusSource:
    """ContentSource over the already-processed local corpus (MVP backend)."""

    def __init__(self, corpus_root: Path) -> None:
        self._root = Path(corpus_root)

    def list_episodes(
        self,
        *,
        feed_id: Optional[str] = None,
        status: Optional[str] = None,
        offset: int = 0,
        limit: int = 20,
    ) -> EpisodeListResult:
        """Newest-first page from the corpus scan, optionally scoped by feed/status.

        Common path (no status filter) maps only the page slice — bounded metadata reads.
        A status filter maps the full filtered set first (O(episodes) reads, acceptable at
        corpus scale; status filtering is an uncommon query in the local-content MVP).
        """
        off = max(0, offset)
        lim = max(1, min(200, limit))
        rows = filter_rows(build_catalog_rows_cumulative(self._root), feed_id=feed_id)

        if status in ("ready", "pending"):
            mapped = [_row_to_summary(self._root, r) for r in rows]
            mapped = [m for m in mapped if m.status == status]
            return EpisodeListResult(items=mapped[off : off + lim], total=len(mapped))

        page_rows = rows[off : off + lim]
        return EpisodeListResult(
            items=[_row_to_summary(self._root, r) for r in page_rows], total=len(rows)
        )


def get_content_source(app_state: Any, corpus_root: Path) -> ContentSource:
    """Return the configured content source, defaulting to :class:`LocalCorpusSource`.

    A future ``DiscoverySource`` (#1069) can be set on ``app.state.content_source`` to
    override the default without touching the routes.
    """
    configured = getattr(app_state, "content_source", None)
    if configured is not None:
        return cast(ContentSource, configured)
    return LocalCorpusSource(corpus_root)
