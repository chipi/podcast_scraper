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

import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Optional, Protocol

from podcast_scraper.server.app_artwork import artwork_url
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import (
    _load_metadata_doc,
    CatalogEpisodeRow,
    episode_list_topics,
    filter_rows,
)
from podcast_scraper.server.schemas import AppEpisodeSummary

#: Max bullets surfaced on a card's expand-on-demand insights view.
_MAX_CARD_BULLETS = 8


def _lede_key(s: str) -> str:
    """Comparison key for "is this just the title again?".

    Ignores case, spacing and edge punctuation.
    """
    return " ".join(s.lower().split()).strip(" .,:;!?-–—\"'")


def _card_lede(row: CatalogEpisodeRow, *, max_len: int = 150) -> str | None:
    """The episode's ``summary_title`` — the ONE thing that goes in a small space.

    This used to be a fallback chain: summary title, else the first bullet, else the prose body's
    first sentence, truncated to 150. Every branch is a different SHAPE, so the same slot rendered a
    headline on one card, a bullet on the next and half a sentence with an ellipsis on a third.
    Marko's report was that the line "feels like it has 2 parts sometimes" — that is the chain, seen
    from the outside.

    So the slot has one source. ``summary.title`` is generated for every episode (36/36 on the
    validation corpus), is short by construction (median 56 chars, max 77 — it never reaches the
    cap), and is the field written to be read at a glance. The bullets and the prose body are the
    BIG summary and belong on the episode detail surface, not squeezed into a card line.

    A title that merely restates the episode title is still skipped rather than returned. The card
    renders the episode title directly above this line, so echoing it prints the same sentence twice
    and spends the card's only descriptive line saying nothing new. Comparison ignores case, spacing
    and edge punctuation, so "Risk Is a Systems Property." still counts as a restatement. It fires
    rarely — 1 of 36 on the validation corpus — but the one case is exactly the one that looks
    broken.
    """
    ep_key = _lede_key(row.episode_title or "")
    candidate = (row.summary_title or "").strip()
    if not candidate or (ep_key and _lede_key(candidate) == ep_key):
        return None
    return candidate if len(candidate) <= max_len else candidate[: max_len - 1].rstrip() + "…"


def _card_bullets(row: CatalogEpisodeRow) -> list[str]:
    """The full summary bullets for the card's expand-on-demand insights view."""
    return [str(b).strip() for b in row.summary_bullets if str(b).strip()][:_MAX_CARD_BULLETS]


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


def transcript_relpath(content: dict) -> str | None:
    """Episode transcript relpath from a metadata ``content`` block, or ``None``.

    Canonical key is ``transcript_file_path`` (written by the pipeline, read by the search
    indexer); ``transcript_file`` is accepted as a defensive fallback.
    """
    tr = content.get("transcript_file_path") or content.get("transcript_file")
    return tr.strip() if isinstance(tr, str) and tr.strip() else None


def transcript_corpus_relpath(metadata_relpath: str, transcript_rel: str) -> str:
    """Resolve a run-relative ``transcript_file_path`` to a corpus-root-relative path.

    ``transcript_file_path`` is stored relative to the **run directory** (the parent of the
    metadata dir): e.g. metadata ``feeds/F/run_R/metadata/ep.metadata.json`` → transcript
    ``feeds/F/run_R/transcripts/ep.txt``. For a flat corpus (``metadata/ep.metadata.json``)
    the run dir is ``""`` and the result is just ``transcript_rel``.
    """
    run_dir = posixpath.dirname(posixpath.dirname(metadata_relpath))
    return posixpath.normpath(posixpath.join(run_dir, transcript_rel.lstrip("/")))


def _has_transcript(corpus_root: Path, metadata_relpath: str) -> bool:
    """Whether the episode's metadata references a transcript file (playable signal)."""
    doc = _load_metadata_doc(corpus_root / metadata_relpath)
    content = doc.get("content") if isinstance(doc, dict) else None
    if not isinstance(content, dict):
        return False
    return transcript_relpath(content) is not None


def row_to_summary(corpus_root: Path, row: CatalogEpisodeRow) -> AppEpisodeSummary:
    """Map a catalog row to the consumer card shape (one metadata read for transcript)."""
    has_transcript = _has_transcript(corpus_root, row.metadata_relative_path)
    has_summary = bool(row.summary_title or row.summary_bullets or row.summary_text)
    local_art = row.episode_image_local_relpath or row.feed_image_local_relpath
    return AppEpisodeSummary(
        slug=slug_for_row(row),
        title=row.episode_title,
        feed_id=row.feed_id,
        podcast_title=row.feed_title,
        publish_date=row.publish_date,
        duration_seconds=row.duration_seconds,
        episode_image_url=row.episode_image_url,
        feed_image_url=row.feed_image_url,
        artwork_url=artwork_url(local_art, "thumb"),
        status="ready" if has_transcript else "pending",
        summary_preview=_card_lede(row),
        summary_text=(row.summary_text or "").strip() or None,
        summary_bullets=_card_bullets(row),
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
        rows = filter_rows(cached_catalog(self._root), feed_id=feed_id)

        if status in ("ready", "pending"):
            mapped = [row_to_summary(self._root, r) for r in rows]
            mapped = [m for m in mapped if m.status == status]
            return EpisodeListResult(items=mapped[off : off + lim], total=len(mapped))

        page_rows = rows[off : off + lim]
        return EpisodeListResult(
            items=[row_to_summary(self._root, r) for r in page_rows], total=len(rows)
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
