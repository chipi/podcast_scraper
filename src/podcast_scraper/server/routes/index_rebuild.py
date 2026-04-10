"""POST /api/index/rebuild — background FAISS index update (GitHub #507 follow-up)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.search.cli_handlers import _minimal_vector_config
from podcast_scraper.search.index_source_mtime import invalidate_newest_index_source_mtime_cache
from podcast_scraper.search.indexer import index_corpus
from podcast_scraper.server.index_rebuild import CorpusRebuildGate, gate_for_corpus
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import IndexRebuildAccepted

logger = logging.getLogger(__name__)

router = APIRouter(tags=["index"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


def _parse_csv_types(raw: Optional[str]) -> Optional[List[str]]:
    if not raw or not str(raw).strip():
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _spawn_rebuild_thread(
    corpus_key: str,
    output_dir: str,
    *,
    rebuild: bool,
    vector_index_path: Optional[str],
    vector_embedding_model: Optional[str],
    vector_faiss_index_mode: Optional[str],
    vector_index_types: Optional[List[str]],
    gate: CorpusRebuildGate,
) -> None:
    """Run ``index_corpus`` off the request thread; clear gate + mtime cache in ``finally``."""
    err: Optional[str] = None
    try:
        cfg = _minimal_vector_config(
            output_dir,
            vector_index_path=vector_index_path,
            vector_embedding_model=vector_embedding_model,
            vector_faiss_index_mode=vector_faiss_index_mode,
            vector_index_types=vector_index_types,
        )
        index_corpus(output_dir, cfg, rebuild=rebuild)
    except Exception as exc:
        logger.exception("Background index rebuild failed for %s", corpus_key)
        err = str(exc)
    finally:
        gate.end(err)
        invalidate_newest_index_source_mtime_cache(corpus_key)


@router.post(
    "/index/rebuild",
    response_model=IndexRebuildAccepted,
    status_code=202,
    responses={409: {"description": "Rebuild already running for this corpus"}},
)
async def trigger_index_rebuild(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (same as index/stats). Omit for server default.",
    ),
    rebuild: bool = Query(
        default=False,
        description="If true, delete existing FAISS dir and rebuild from scratch.",
    ),
    embedding_model: str | None = Query(
        default=None,
        description="Override ``vector_embedding_model`` for this run (optional).",
    ),
    vector_index_path: str | None = Query(
        default=None,
        description="Optional relative or absolute vector index directory override.",
    ),
    vector_faiss_index_mode: str | None = Query(
        default=None,
        description="Optional FAISS mode: auto, flat, ivf_flat, ivfpq.",
    ),
    vector_index_types: str | None = Query(
        default=None,
        description="Comma-separated doc types to embed (optional).",
    ),
) -> IndexRebuildAccepted:
    """Queue an incremental (or full) index build; poll ``GET /api/index/stats`` for progress."""
    try:
        import podcast_scraper.search.faiss_store  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="FAISS is not available in this Python environment.",
        ) from None

    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        raise HTTPException(
            status_code=400,
            detail="Corpus path is required (query or server default).",
        )

    corpus_key = str(root.resolve())
    gate = gate_for_corpus(request.app, root)
    if not gate.try_begin():
        raise HTTPException(
            status_code=409,
            detail="Index rebuild is already running for this corpus.",
        )

    vit = _parse_csv_types(vector_index_types)
    thread = threading.Thread(
        target=_spawn_rebuild_thread,
        name=f"index-rebuild-{corpus_key}",
        kwargs={
            "corpus_key": corpus_key,
            "output_dir": corpus_key,
            "rebuild": rebuild,
            "vector_index_path": vector_index_path,
            "vector_embedding_model": embedding_model,
            "vector_faiss_index_mode": vector_faiss_index_mode,
            "vector_index_types": vit,
            "gate": gate,
        },
        daemon=True,
    )
    thread.start()
    return IndexRebuildAccepted(
        accepted=True,
        corpus_path=corpus_key,
        rebuild=rebuild,
    )
