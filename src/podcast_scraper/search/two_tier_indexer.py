"""From-corpus two-tier indexer (RFC-090 / wire-live follow-up B).

Builds the two-tier LanceDB index (#855) directly from corpus artifacts, so a
**fresh** corpus becomes hybrid-searchable without first having a FAISS index to
migrate (#858). It reuses the proven FAISS-indexer extraction —
``discover_metadata_files`` → ``_collect_docs_for_episode`` — which already yields
insight rows and timestamped transcript chunks, then re-embeds the text and upserts
into the ``insight`` (Tier 2) and ``segment`` (Tier 1) tables.

Relationship to the migration (#858): the migration is the fast path for a corpus
that already has FAISS (it reuses those embeddings verbatim); this indexer is the
native path for corpora that don't. Both produce the same two-tier layout the live
hybrid search (RFC-090 Phase 2) reads. Unlike the migration, this path **populates
insight↔segment links** (``linked_insight_ids`` / ``source_segment_id``) from the
gi.json ``SUPPORTED_BY`` edges + quote timestamps, so the compound-result path
(``dedup``) actually fires on a natively-built index. The migration leaves them
empty (FAISS gives it no edges), so compounds need a native (re)index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Dict, List, Optional, Tuple

from ..providers.ml import embedding_loader
from .backend import AuxDocument, InsightDocument, SegmentDocument
from .backends.lancedb_backend import LanceDBBackend
from .corpus_scope import discover_metadata_files, episode_root_from_metadata_path
from .indexer import _collect_docs_for_episode, _gi_path, _load_metadata_file
from .segments import link_insights_to_segments

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TARGET_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32

# Non-tiered FAISS surfaces indexed into the aux tier for full coverage.
_AUX_DOC_TYPES = frozenset({"kg_entity", "kg_topic", "quote", "summary"})


@dataclass
class TwoTierIndexStats:
    """Counts from a from-corpus two-tier index build."""

    episodes: int = 0
    segments: int = 0
    insights: int = 0
    aux: int = 0
    linked: int = 0


def _embed(text: str, model_id: str, *, allow_download: bool) -> List[float]:
    vec = embedding_loader.encode(text, model_id, return_numpy=False, allow_download=allow_download)
    return [float(x) for x in cast(List[float], vec)]


def _insight_grounding_quotes(gi_path: Path) -> Dict[str, Tuple[float, Optional[float]]]:
    """Map each insight node id → its first grounding quote's (start_s, end_s).

    Reads the episode's ``*.gi.json`` (Insight ``SUPPORTED_BY`` Quote; quotes carry
    ``timestamp_*_ms``). This is what lets ``link_insights_to_segments`` connect an
    insight to the transcript segment it was spoken in — the input the compound-result
    path (``dedup``) needs but the FAISS-derived migration can't supply.
    """
    if not gi_path.is_file():
        return {}
    try:
        art = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quote_ts: Dict[str, Tuple[float, Optional[float]]] = {}
    for node in art.get("nodes") or []:
        if node.get("type") == "Quote":
            props = node.get("properties") or {}
            start = props.get("timestamp_start_ms")
            if start is not None:
                end = props.get("timestamp_end_ms")
                quote_ts[node.get("id")] = (
                    float(start) / 1000.0,
                    float(end) / 1000.0 if end is not None else None,
                )
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for edge in art.get("edges") or []:
        if edge.get("type") == "SUPPORTED_BY":
            insight_id, quote_id = edge.get("from"), edge.get("to")
            if insight_id not in out and quote_id in quote_ts:
                out[insight_id] = quote_ts[quote_id]
    return out


def build_two_tier_index(
    corpus_dir: str | Path,
    lance_path: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embed_dim: Optional[int] = None,
    limit_episodes: Optional[int] = None,
    allow_download: bool = False,
) -> TwoTierIndexStats:
    """Build the two-tier LanceDB index at *lance_path* from the corpus at *corpus_dir*.

    Walks episode metadata, extracts insight + transcript-chunk rows, embeds each with
    *model_id*, and upserts into the segment/insight tables (idempotent merge on id).
    ``embed_dim`` is derived from *model_id* when None (so a non-MiniLM model can't
    silently mismatch the schema). ``limit_episodes`` caps the walk. Returns counts.
    """
    out = Path(corpus_dir)
    stats = TwoTierIndexStats()
    backend: Optional[LanceDBBackend] = None

    def _ensure_backend(vec_len: int) -> LanceDBBackend:
        # Lazy: size the schema from the model's real dim (or an explicit override) on
        # the first embedded doc, so a non-MiniLM model can't silently mismatch — and an
        # empty corpus creates no index at all.
        nonlocal backend
        if backend is None:
            backend = LanceDBBackend(str(lance_path), embed_dim=embed_dim or vec_len)
        return backend

    for meta_path in discover_metadata_files(out):
        if limit_episodes is not None and stats.episodes >= limit_episodes:
            break
        doc = _load_metadata_file(meta_path)
        if not doc:
            continue
        stats.episodes += 1
        episode_root = episode_root_from_metadata_path(meta_path)
        meta_rel = meta_path.resolve().relative_to(out.resolve()).as_posix()
        rows = _collect_docs_for_episode(
            episode_root,
            meta_path,
            doc,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            metadata_relative_path=meta_rel,
        )
        # Collect this episode's docs first, then link insights to the segment that
        # contains their grounding quote, then upsert — so linked_insight_ids /
        # source_segment_id are populated (the compound-result path needs them).
        seg_docs: List[SegmentDocument] = []
        ins_docs: List[InsightDocument] = []
        aux_docs: List[AuxDocument] = []
        node_id_by_insight: Dict[str, Optional[str]] = {}
        for doc_id, text, meta in rows:
            doc_type = meta.get("doc_type")
            if doc_type == "insight":
                ins_docs.append(
                    InsightDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        entity_type="insight",
                        confidence=0.0,
                        derived=bool(meta.get("grounded")),
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )
                node_id_by_insight[doc_id] = meta.get("source_id")
            elif doc_type == "transcript":
                seg_docs.append(
                    SegmentDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        start_time=float(meta.get("timestamp_start_ms") or 0.0) / 1000.0,
                        end_time=float(meta.get("timestamp_end_ms") or 0.0) / 1000.0,
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )
            elif doc_type in _AUX_DOC_TYPES:
                aux_docs.append(
                    AuxDocument(
                        id=doc_id,
                        text=text,
                        show_id=meta.get("feed_id") or "",
                        episode_id=meta.get("episode_id") or "",
                        doc_type=doc_type,
                        embedding=_embed(text, model_id, allow_download=allow_download),
                    )
                )

        grounding = _insight_grounding_quotes(_gi_path(episode_root, meta_path, doc))
        insight_quotes = [
            (ins.id, *grounding[node_id])
            for ins in ins_docs
            if (node_id := node_id_by_insight.get(ins.id)) in grounding
        ]
        mapping = link_insights_to_segments(seg_docs, insight_quotes)
        for ins in ins_docs:
            if ins.id in mapping:
                ins.source_segment_id = mapping[ins.id]
        stats.linked += len(mapping)

        for seg in seg_docs:
            _ensure_backend(len(seg.embedding)).upsert_segment(seg)
            stats.segments += 1
        for ins in ins_docs:
            _ensure_backend(len(ins.embedding)).upsert_insight(ins)
            stats.insights += 1
        for aux in aux_docs:
            _ensure_backend(len(aux.embedding)).upsert_aux(aux)
            stats.aux += 1

    if backend is not None:
        backend.write_index_meta(model_id)  # query path must embed in the same space
        backend.create_indices()
    return stats
