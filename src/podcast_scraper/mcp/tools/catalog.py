"""Catalog / navigation MCP tools (RFC-095 slice 3).

Wrap the clean :mod:`podcast_scraper.server.corpus_catalog` scan functions plus the
extracted ``top_persons`` capability. Read-only browsing of the corpus: feeds, episodes,
one episode's detail, and the top voices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context import CorpusContext


def _row_summary(row: Any) -> Dict[str, Any]:
    """Compact episode dict for list views (provenance + identity, not full body)."""
    return {
        "metadata_path": row.metadata_relative_path,
        "episode_id": row.episode_id,
        "episode_title": row.episode_title,
        "feed_id": row.feed_id,
        "feed_title": row.feed_title,
        "publish_date": row.publish_date,
        "summary_title": row.summary_title,
        "has_gi": row.has_gi,
        "has_kg": row.has_kg,
    }


def list_feeds(ctx: CorpusContext) -> Dict[str, Any]:
    """List the shows (feeds) in the corpus, with display title and episode counts."""
    from ...server.corpus_catalog import aggregate_feeds, build_catalog_rows_cumulative

    rows = build_catalog_rows_cumulative(ctx.corpus_dir)
    return {"feeds": aggregate_feeds(rows)}


def list_episodes(
    ctx: CorpusContext,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List episodes (newest-first), optionally filtered by feed substring and ``since`` date.

    ``since`` is a ``YYYY-MM-DD`` lower bound on publish date. Returns compact rows; use
    ``episode_detail`` for the full summary of one episode.
    """
    from ...server.corpus_catalog import build_catalog_rows_cumulative

    cap = max(1, min(500, int(limit)))
    feed_q = (feed or "").strip()
    since_q = (since or "").strip()
    out: List[Dict[str, Any]] = []
    for row in build_catalog_rows_cumulative(ctx.corpus_dir):
        if feed_q and feed_q not in (row.feed_id or ""):
            continue
        if since_q and (row.publish_date or "") < since_q:
            continue
        out.append(_row_summary(row))
        if len(out) >= cap:
            break
    return {"episodes": out, "count": len(out)}


def episode_detail(ctx: CorpusContext, metadata_path: str) -> Dict[str, Any]:
    """Full detail for one episode by its ``metadata_path`` (from a list/search result)."""
    from ...server.corpus_catalog import catalog_row_for_metadata_path

    row = catalog_row_for_metadata_path(ctx.corpus_dir, metadata_path)
    if row is None:
        return {"metadata_path": metadata_path, "episode": None, "error": "not_found"}
    detail = _row_summary(row)
    detail.update(
        {
            "summary_text": row.summary_text,
            "gi_relative_path": row.gi_relative_path if row.has_gi else None,
            "kg_relative_path": row.kg_relative_path if row.has_kg else None,
            "duration_seconds": row.duration_seconds,
            "episode_number": row.episode_number,
            "feed_rss_url": row.feed_rss_url,
        }
    )
    return {"metadata_path": metadata_path, "episode": detail}


def top_people(ctx: CorpusContext, limit: int = 10) -> Dict[str, Any]:
    """The corpus's top voices — people ranked by grounded (quote-backed) insight count."""
    from ...server.routes.corpus_persons import top_persons

    return top_persons(ctx.corpus_dir, max(1, min(50, int(limit))))
