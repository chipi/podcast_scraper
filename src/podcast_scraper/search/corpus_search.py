"""Shared semantic corpus search (CLI + HTTP viewer)."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Sequence, Set

from podcast_scraper.providers.ml import embedding_loader
from podcast_scraper.providers.ml.model_registry import ModelRegistry
from podcast_scraper.search.cli_handlers import (
    _enrich_hit,
    _episode_to_gi_path,
    _hit_passes_cli_filters,
    _metadata_relpath_by_scope_from_corpus,
    _parse_since,
    _resolve_index_dir,
)
from podcast_scraper.search.faiss_store import FaissVectorStore, VECTORS_FILE
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)

_DEDUPE_KG_DOC_TYPES = frozenset({"kg_entity", "kg_topic"})
_KG_SURFACE_MAX_EPISODE_IDS = 48


def _normalize_kg_surface_text(text: str) -> str:
    """Lowercase, trim, collapse whitespace (aligns with graph Entity/Topic name dedupe idea)."""
    raw = (text or "").lower().strip()
    return re.sub(r"\s+", " ", raw)


def dedupe_kg_surface_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate ``kg_entity`` / ``kg_topic`` hits that share the same embedded surface text.

    Keeps the highest-scoring row first in list order; adds ``kg_surface_match_count`` and
    ``kg_surface_episode_ids`` when more than one episode contributed.
    """
    out: List[Dict[str, Any]] = []
    kg_winners: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        dt = meta.get("doc_type")
        if dt not in _DEDUPE_KG_DOC_TYPES:
            out.append(row)
            continue
        text = str(row.get("text") or "")
        sk = f"{dt}\0{_normalize_kg_surface_text(text)}"
        if sk not in kg_winners:
            kg_winners[sk] = row
            out.append(row)
            continue
        winner = kg_winners[sk]
        wmeta = winner.get("metadata")
        if not isinstance(wmeta, dict):
            wmeta = {}
            winner["metadata"] = wmeta
        ep = meta.get("episode_id")
        new_id = ep.strip() if isinstance(ep, str) and ep.strip() else None

        collected: List[str] = []
        seen: Set[str] = set()
        raw_prev = wmeta.get("kg_surface_episode_ids")
        if isinstance(raw_prev, list):
            for x in raw_prev:
                if isinstance(x, str) and x.strip() and x.strip() not in seen:
                    seen.add(x.strip())
                    collected.append(x.strip())
        else:
            w_ep = wmeta.get("episode_id")
            if isinstance(w_ep, str) and w_ep.strip() and w_ep.strip() not in seen:
                seen.add(w_ep.strip())
                collected.append(w_ep.strip())
        if new_id and new_id not in seen:
            seen.add(new_id)
            collected.append(new_id)
        wmeta["kg_surface_episode_ids"] = collected[:_KG_SURFACE_MAX_EPISODE_IDS]
        wmeta["kg_surface_match_count"] = len(collected)
    return out


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
    dedupe_kg_surfaces: bool = True,
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
    rel_by_scope = _metadata_relpath_by_scope_from_corpus(output_dir)
    filtered: List[SearchResult] = []
    collect_cap = fetch_k if dedupe_kg_surfaces else top_k
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
        if len(filtered) >= collect_cap:
            break

    title_cache: Dict[str, tuple[str, str]] = {}
    enriched = [
        _enrich_hit(
            h,
            gi_cache,
            metadata_relpath_by_scope=rel_by_scope,
            corpus_root=output_dir,
            title_cache=title_cache,
        )
        for h in filtered
    ]
    if dedupe_kg_surfaces:
        enriched = dedupe_kg_surface_rows(enriched)
    enriched = enriched[:top_k]
    return CorpusSearchOutcome(results=enriched)
