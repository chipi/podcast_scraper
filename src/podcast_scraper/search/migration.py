"""FAISS → two-tier LanceDB migration (RFC-090 Stage 4 / #858).

Re-projects an existing single-index FAISS store into the two-tier LanceDB layout
(#855): ``insight`` rows become Tier-2 ``InsightDocument``s, ``transcript`` chunks
become Tier-1 ``SegmentDocument``s. **Embeddings are reused verbatim** — the FAISS
vectors are the same MiniLM projections the new index would compute, so the
migration neither re-embeds nor changes the vector space. That is deliberate: it
makes the Stage-4 A/B (``scripts/eval_two_tier_retrieval.py``) a controlled
experiment — vector signal held constant, the only new variable is BM25 + RRF
fusion, which is exactly the RFC-090 hybrid hypothesis.

Other FAISS doc types (``quote``/``summary``/``kg_entity``/``kg_topic``) are not
part of the two-tier retrieval surface and are skipped; they remain reachable via
the KG-proximity signal (#859), which reads the corpus graph, not this index.

A from-corpus build path (chunk via #857 ``build_segment_documents`` + embed via
``indexer`` + link insights) is the eventual production indexer; it is out of
scope for the migration, whose job is to stand up a LanceDB index comparable to
the live FAISS one for the gating eval.

This is the first step an operator runs when upgrading a deployed 2.6 corpus to
2.7. Ordering, version-stamping, dry-run and rollback around it — the managed
upgrade-path runner that will register this step — are tracked in #862. The
migration itself is idempotent (merge-insert on ``id``), so re-runs are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .backend import AuxDocument, InsightDocument, SegmentDocument
from .backends.lancedb_backend import LanceDBBackend
from .faiss_store import FaissVectorStore

# Non-tiered FAISS surfaces carried into the aux tier for full hybrid coverage.
_AUX_DOC_TYPES = frozenset({"kg_entity", "kg_topic", "quote", "summary"})


@dataclass
class MigrationStats:
    """Counts from a FAISS → LanceDB migration run."""

    segments: int = 0
    insights: int = 0
    aux: int = 0
    skipped: int = 0
    embed_dim: int = 0


def _aux_from_faiss(doc_id: str, meta: Dict, embedding: list) -> AuxDocument:
    return AuxDocument(
        id=doc_id,
        text=meta.get("text") or "",
        show_id=meta.get("feed_id") or "",
        episode_id=meta.get("episode_id") or "",
        doc_type=str(meta.get("doc_type") or ""),
        embedding=embedding,
    )


def _segment_from_faiss(doc_id: str, meta: Dict, embedding: list) -> SegmentDocument:
    return SegmentDocument(
        id=doc_id,
        text=meta.get("text") or "",
        show_id=meta.get("feed_id") or "",
        episode_id=meta.get("episode_id") or "",
        start_time=float(meta.get("timestamp_start_ms") or 0.0) / 1000.0,
        end_time=float(meta.get("timestamp_end_ms") or 0.0) / 1000.0,
        embedding=embedding,
        speaker_id=meta.get("speaker_id"),
    )


def _insight_from_faiss(doc_id: str, meta: Dict, embedding: list) -> InsightDocument:
    return InsightDocument(
        id=doc_id,
        text=meta.get("text") or "",
        show_id=meta.get("feed_id") or "",
        episode_id=meta.get("episode_id") or "",
        entity_type="insight",
        confidence=float(meta.get("confidence") or 0.0),
        # FAISS marks quote-grounded insights with ``grounded``; reuse it as the
        # derived flag so the two-tier payload keeps the provenance bit.
        derived=bool(meta.get("grounded")),
        embedding=embedding,
    )


def migrate_faiss_to_lance(
    faiss_dir: str | Path,
    lance_path: str | Path,
    *,
    limit_per_tier: Optional[int] = None,
) -> MigrationStats:
    """Re-project the FAISS store at *faiss_dir* into a two-tier LanceDB index.

    Reuses the stored embeddings; only ``insight`` and ``transcript`` doc types
    cross over (Tier 2 and Tier 1 respectively). ``limit_per_tier`` caps each tier
    for smoke runs. Returns per-tier counts. Idempotent (merge-insert on ``id``).
    """
    store = FaissVectorStore.load(Path(faiss_dir))
    vectors = store.export_vectors_by_doc_id()
    metadata = store.metadata_by_doc_id
    backend = LanceDBBackend(str(lance_path), embed_dim=store.embedding_dim)

    stats = MigrationStats(embed_dim=store.embedding_dim)
    for doc_id, meta in metadata.items():
        doc_type = meta.get("doc_type")
        vec = vectors.get(doc_id)
        if vec is None:
            stats.skipped += 1
            continue
        emb = vec.tolist()
        if doc_type == "transcript":
            if limit_per_tier is not None and stats.segments >= limit_per_tier:
                continue
            backend.upsert_segment(_segment_from_faiss(doc_id, meta, emb))
            stats.segments += 1
        elif doc_type == "insight":
            if limit_per_tier is not None and stats.insights >= limit_per_tier:
                continue
            backend.upsert_insight(_insight_from_faiss(doc_id, meta, emb))
            stats.insights += 1
        elif doc_type in _AUX_DOC_TYPES:
            if limit_per_tier is not None and stats.aux >= limit_per_tier:
                continue
            backend.upsert_aux(_aux_from_faiss(doc_id, meta, emb))
            stats.aux += 1
        else:
            stats.skipped += 1

    # Record the model so the query path embeds in the same space (else silent mismatch).
    backend.write_index_meta(store.embedding_model)
    if stats.segments or stats.insights or stats.aux:
        backend.create_indices()
    return stats
