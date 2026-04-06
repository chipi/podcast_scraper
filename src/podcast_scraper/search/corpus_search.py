"""Shared semantic corpus search (CLI + HTTP viewer)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Sequence

from podcast_scraper.providers.ml import embedding_loader
from podcast_scraper.providers.ml.model_registry import ModelRegistry
from podcast_scraper.search.cli_handlers import (
    _enrich_hit,
    _episode_to_gi_path,
    _hit_passes_cli_filters,
    _parse_since,
    _resolve_index_dir,
)
from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


@dataclass
class CorpusSearchOutcome:
    """Result of ``run_corpus_search`` (HTTP maps this to JSON; CLI maps to exit codes)."""

    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    """Machine-readable: ``empty_query``, ``no_index``, ``load_failed``, ``embed_failed``."""
    detail: Optional[str] = None
    """Optional human/debug message (logged server-side; may be omitted in API)."""


def run_corpus_search(
    output_dir: Path,
    query: str,
    *,
    doc_types: Optional[Sequence[str]] = None,
    feed: Optional[str] = None,
    since: Optional[str] = None,
    speaker: Optional[str] = None,
    grounded_only: bool = False,
    top_k: int = 10,
    index_path: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> CorpusSearchOutcome:
    """Embed ``query``, search FAISS, apply metadata filters, return enriched rows."""
    q = query.strip()
    if not q:
        return CorpusSearchOutcome(error="empty_query")

    index_dir = _resolve_index_dir(output_dir, index_path)
    if not (index_dir / VECTORS_FILE).is_file():
        return CorpusSearchOutcome(error="no_index", detail=str(index_dir))

    try:
        store = FaissVectorStore.load(index_dir)
    except Exception as exc:
        logger.warning("corpus_search load failed: %s", format_exception_for_log(exc))
        return CorpusSearchOutcome(error="load_failed", detail=str(exc))

    idx_model = store.stats().embedding_model
    if isinstance(embedding_model, str) and embedding_model.strip():
        if ModelRegistry.resolve_evidence_model_id(
            embedding_model.strip()
        ) != ModelRegistry.resolve_evidence_model_id(idx_model):
            logger.warning(
                "search: embedding_model %s differs from index model %s",
                embedding_model,
                idx_model,
            )
        model_id = embedding_model.strip()
    else:
        model_id = idx_model

    t_embed0 = time.perf_counter()
    try:
        qvec = embedding_loader.encode(
            q,
            model_id,
            return_numpy=False,
            allow_download=False,
        )
        if isinstance(qvec, list) and qvec and isinstance(qvec[0], float):
            qemb = cast(List[float], qvec)
        else:
            return CorpusSearchOutcome(error="embed_failed", detail="expected single embedding")
    except Exception as exc:
        logger.warning("corpus_search embed failed: %s", format_exception_for_log(exc))
        return CorpusSearchOutcome(error="embed_failed", detail=str(exc))
    embed_sec = time.perf_counter() - t_embed0

    types_norm: Optional[List[str]] = None
    if doc_types:
        types_norm = [x.strip().lower() for x in doc_types if isinstance(x, str) and x.strip()]
        if not types_norm:
            types_norm = None

    faiss_filters: Optional[Dict[str, Any]] = None
    if types_norm and len(types_norm) == 1:
        faiss_filters = {"doc_type": types_norm[0]}

    top_k = max(1, min(int(top_k), 100))
    fetch_k = min(max(top_k * 25, top_k), max(store.ntotal, 1))

    t_search0 = time.perf_counter()
    hits = store.search(
        qemb,
        top_k=fetch_k,
        filters=faiss_filters,
        overfetch_factor=1,
    )
    search_sec = time.perf_counter() - t_search0
    logger.info(
        "corpus_search: embed_sec=%.4f search_sec=%.4f ntotal=%s fetch_k=%s",
        embed_sec,
        search_sec,
        store.ntotal,
        fetch_k,
    )

    since_dt = _parse_since(since) if isinstance(since, str) and since.strip() else None
    gi_cache = _episode_to_gi_path(output_dir)
    filtered: List[SearchResult] = []
    for h in hits:
        dt = h.metadata.get("doc_type")
        if types_norm and len(types_norm) > 1:
            if dt not in types_norm:
                continue
        elif types_norm and len(types_norm) == 1:
            if dt != types_norm[0]:
                continue

        if _hit_passes_cli_filters(
            h,
            feed_substr=feed,
            since_dt=since_dt,
            speaker_substr=speaker,
            grounded_only=grounded_only,
            gi_by_episode=gi_cache,
        ):
            filtered.append(h)
        if len(filtered) >= top_k:
            break

    enriched = [_enrich_hit(h, gi_cache) for h in filtered]
    return CorpusSearchOutcome(results=enriched)
