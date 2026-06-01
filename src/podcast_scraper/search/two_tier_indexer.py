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
hybrid search (RFC-090 Phase 2) reads. Insight↔segment linking
(``linked_insight_ids`` for compound results) is not populated here yet — like the
migration — so compounds degrade to separate insight+segment rows; wiring the
SUPPORTED_BY edges through is a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast, List, Optional

from ..providers.ml import embedding_loader
from .backend import InsightDocument, SegmentDocument
from .backends.lancedb_backend import DEFAULT_EMBED_DIM, LanceDBBackend
from .corpus_scope import discover_metadata_files, episode_root_from_metadata_path
from .indexer import _collect_docs_for_episode, _load_metadata_file

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TARGET_TOKENS = 256
DEFAULT_OVERLAP_TOKENS = 32


@dataclass
class TwoTierIndexStats:
    """Counts from a from-corpus two-tier index build."""

    episodes: int = 0
    segments: int = 0
    insights: int = 0


def _embed(text: str, model_id: str, *, allow_download: bool) -> List[float]:
    vec = embedding_loader.encode(text, model_id, return_numpy=False, allow_download=allow_download)
    return [float(x) for x in cast(List[float], vec)]


def build_two_tier_index(
    corpus_dir: str | Path,
    lance_path: str | Path,
    *,
    model_id: str = DEFAULT_MODEL,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    limit_episodes: Optional[int] = None,
    allow_download: bool = False,
) -> TwoTierIndexStats:
    """Build the two-tier LanceDB index at *lance_path* from the corpus at *corpus_dir*.

    Walks episode metadata, extracts insight + transcript-chunk rows, embeds each with
    *model_id*, and upserts into the segment/insight tables (idempotent merge on id).
    ``limit_episodes`` caps the walk for smoke runs. Returns per-tier counts.
    """
    out = Path(corpus_dir)
    backend = LanceDBBackend(str(lance_path), embed_dim=embed_dim)
    stats = TwoTierIndexStats()

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
        for doc_id, text, meta in rows:
            doc_type = meta.get("doc_type")
            if doc_type == "insight":
                backend.upsert_insight(
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
                stats.insights += 1
            elif doc_type == "transcript":
                backend.upsert_segment(
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
                stats.segments += 1

    if stats.segments or stats.insights:
        backend.create_indices()
    return stats
