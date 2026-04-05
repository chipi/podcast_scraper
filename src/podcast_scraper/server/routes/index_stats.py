"""GET /api/index/stats — FAISS vector index metrics (RFC-062 M3)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server.schemas import IndexStatsBody, IndexStatsEnvelope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["index"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {root}")
        return root
    return fallback


@router.get("/index/stats", response_model=IndexStatsEnvelope)
async def index_stats(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (contains search/). Omit to use server default output_dir.",
    ),
) -> IndexStatsEnvelope:
    """Return FAISS index statistics when ``<corpus>/search`` is loadable."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return IndexStatsEnvelope(available=False, reason="no_corpus_path")

    index_dir = (root / "search").resolve()
    try:
        from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE
    except ImportError:
        return IndexStatsEnvelope(
            available=False,
            reason="faiss_unavailable",
            index_path=str(index_dir),
        )

    vec = index_dir / VECTORS_FILE
    if not vec.is_file():
        return IndexStatsEnvelope(
            available=False,
            reason="no_index",
            index_path=str(index_dir),
        )

    try:
        store = FaissVectorStore.load(index_dir)
        raw = store.stats()
    except Exception:
        logger.exception("Failed to load vector index from %s", index_dir)
        return IndexStatsEnvelope(
            available=False,
            reason="load_failed",
            index_path=str(index_dir),
        )

    body = IndexStatsBody(
        total_vectors=raw.total_vectors,
        doc_type_counts=dict(raw.doc_type_counts),
        feeds_indexed=list(raw.feeds_indexed),
        embedding_model=raw.embedding_model,
        embedding_dim=raw.embedding_dim,
        last_updated=raw.last_updated,
        index_size_bytes=raw.index_size_bytes,
    )
    return IndexStatsEnvelope(
        available=True,
        index_path=str(index_dir),
        stats=body,
    )
