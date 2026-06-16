"""Shared semantic corpus search (CLI + HTTP viewer)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from podcast_scraper.search.cil_lift_overrides import load_cil_lift_overrides
from podcast_scraper.search.cli_handlers import (
    _enrich_hit,
    _hit_passes_cli_filters,
    _metadata_relpath_by_scope_from_corpus,
    _parse_since,
    merged_episode_gi_paths,
)
from podcast_scraper.search.hybrid_search import hybrid_candidates
from podcast_scraper.search.protocol import SearchResult
from podcast_scraper.search.topic_clusters import load_topic_cluster_enrichment_map
from podcast_scraper.search.transcript_chunk_lift import (
    lift_row_if_transcript,
    TranscriptLiftGiCache,
)

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
    lift_stats: Optional[Dict[str, int]] = None
    """Per-response lift counters; set on success after ``top_k`` slice."""


def _attach_topic_cluster_metadata(rows: List[Dict[str, Any]], corpus_root: Path) -> None:
    """Join ``topic_clusters.json`` into ``kg_topic`` metadata (query-time join)."""
    m = load_topic_cluster_enrichment_map(corpus_root)
    if not m:
        return
    for row in rows:
        meta = row.get("metadata")
        if not isinstance(meta, dict):
            continue
        if meta.get("doc_type") != "kg_topic":
            continue
        sid = meta.get("source_id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        info = m.get(sid.strip())
        if info:
            meta["topic_cluster"] = dict(info)


def _lift_stats_for_page(enriched: List[Dict[str, Any]]) -> Dict[str, int]:
    transcript_returned = 0
    lift_applied = 0
    for r in enriched:
        meta_r = r.get("metadata")
        if isinstance(meta_r, dict) and meta_r.get("doc_type") == "transcript":
            transcript_returned += 1
        if isinstance(r.get("lifted"), dict):
            lift_applied += 1
    return {
        "transcript_hits_returned": transcript_returned,
        "lift_applied": lift_applied,
    }


def _enrich_lift_and_slice(
    filtered: List[SearchResult],
    output_dir: Path,
    gi_cache: Dict[str, Path],
    rel_by_scope: Dict[str, str],
    *,
    top_k: int,
    dedupe_kg_surfaces: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
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
    lift_overrides = load_cil_lift_overrides(output_dir)
    lift_cache = TranscriptLiftGiCache()
    for row in enriched:
        meta = row.get("metadata")
        if not isinstance(meta, dict) or meta.get("doc_type") != "transcript":
            continue
        ep = meta.get("episode_id")
        if not isinstance(ep, str) or not ep.strip():
            continue
        gpath = gi_cache.get(ep.strip())
        if gpath is not None and gpath.is_file():
            lift_row_if_transcript(
                row,
                output_dir,
                gpath,
                lift_cache,
                lift_overrides,
            )
    if dedupe_kg_surfaces:
        enriched = dedupe_kg_surface_rows(enriched)
    _attach_topic_cluster_metadata(enriched, output_dir)
    page = enriched[:top_k]
    return page, _lift_stats_for_page(page)


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
    """Embed ``query``, search the LanceDB index, apply metadata filters, return enriched rows."""
    q = query.strip()
    if not q:
        return CorpusSearchOutcome(error="empty_query")

    top_k = max(1, min(int(top_k), 100))
    types_norm: Optional[List[str]] = None
    if doc_types:
        types_norm = [x.strip().lower() for x in doc_types if isinstance(x, str) and x.strip()]
        if not types_norm:
            types_norm = None

    # ADR-099 / #995: the LanceDB two-tier index is the single search path — no FAISS
    # fallback. ``hybrid_candidates`` returns None when there is no usable index (or a
    # query-embedding failure); surface that as ``no_index`` rather than a second path.
    candidates = hybrid_candidates(
        output_dir,
        q,
        top_k=top_k,
        doc_types=doc_types,
        embedding_model=embedding_model,
    )
    if candidates is None:
        return CorpusSearchOutcome(error="no_index", detail="no LanceDB index (run `cli index`)")
    return _filter_and_enrich(
        candidates,
        output_dir,
        types_norm=types_norm,
        feed=feed,
        since=since,
        speaker=speaker,
        grounded_only=grounded_only,
        top_k=top_k,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
        collect_cap=len(candidates),
    )


def _filter_and_enrich(
    hits: Sequence[SearchResult],
    output_dir: Path,
    *,
    types_norm: Optional[List[str]],
    feed: Optional[str],
    since: Optional[str],
    speaker: Optional[str],
    grounded_only: bool,
    top_k: int,
    dedupe_kg_surfaces: bool,
    collect_cap: int,
) -> CorpusSearchOutcome:
    """Apply metadata filters + enrich/lift/dedupe a candidate list (backend-agnostic).

    Drives the LanceDB two-tier hybrid retrieval path (RFC-090 Phase 2 / ADR-099) — the
    single search path since FAISS was retired (#995) — producing the enriched response shape.
    """
    since_dt = _parse_since(since) if isinstance(since, str) and since.strip() else None
    gi_cache = merged_episode_gi_paths(output_dir)
    rel_by_scope = _metadata_relpath_by_scope_from_corpus(output_dir)
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
        if len(filtered) >= collect_cap:
            break

    enriched, lift_stats = _enrich_lift_and_slice(
        filtered,
        output_dir,
        gi_cache,
        rel_by_scope,
        top_k=top_k,
        dedupe_kg_surfaces=dedupe_kg_surfaces,
    )
    return CorpusSearchOutcome(results=enriched, lift_stats=lift_stats)
