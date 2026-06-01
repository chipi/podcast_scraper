"""Hybrid retrieval bridge for the live search path (RFC-090 Phase 2 / wire-live).

Connects the dormant two-tier stack (``RetrievalLayer`` over ``LanceDBBackend``,
#855-861) to the serving path (``corpus_search.run_corpus_search``) without changing
the response contract: hybrid candidates are mapped to the same ``SearchResult``
shape the FAISS path produces, then run through the *identical* filter/enrich/lift
pipeline.

**Opt-in and non-regressing.** It activates only when ``serving.hybrid_enabled`` is
true in ``config/search.yaml`` (default false) *and* a LanceDB index exists for the
corpus. Any miss — flag off, no index, embed failure — returns ``None`` so the caller
falls back to FAISS. This matters because the two-tier index covers only the
*insight* and *segment* (transcript) tiers; kg_entity / kg_topic / quote / summary
still live only in FAISS, so flipping hybrid on narrows coverage to those two tiers
by design. Full-coverage hybrid (kg/quote tiers in LanceDB) is a follow-up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast, List, Optional, Sequence

import yaml

from ..providers.ml import embedding_loader
from .backend import CompoundResult, ScoredResult, Tier
from .protocol import SearchResult

logger = logging.getLogger(__name__)

_SEARCH_CONFIG = Path("config/search.yaml")
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def hybrid_search_enabled() -> bool:
    """True when ``serving.hybrid_enabled`` is set in ``config/search.yaml``."""
    try:
        doc = yaml.safe_load(_SEARCH_CONFIG.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    serving = doc.get("serving") if isinstance(doc, dict) else None
    return bool(isinstance(serving, dict) and serving.get("hybrid_enabled"))


def lance_index_dir(output_dir: Path) -> Path:
    """Per-corpus LanceDB index location (co-located with the corpus, #858/#862)."""
    return Path(output_dir) / "search" / "lance_index"


def _tier_for(doc_types: Optional[Sequence[str]]) -> Tier:
    """Map requested doc types to a two-tier scope (insight | segment | all)."""
    if not doc_types:
        return "all"
    wanted = {t.strip().lower() for t in doc_types if isinstance(t, str) and t.strip()}
    if wanted <= {"insight"}:
        return "insight"
    if wanted <= {"transcript", "segment"}:
        return "segment"
    return "all"


def _doc_type_for_tier(source_tier: str) -> str:
    # Segments are transcript chunks in the FAISS vocabulary the enrich/lift expects.
    return "insight" if source_tier == "insight" else "transcript"


def _to_search_result(result: ScoredResult) -> SearchResult:
    payload = dict(result.payload or {})
    metadata = {
        "doc_type": _doc_type_for_tier(result.source_tier),
        "text": payload.get("text", ""),
        "episode_id": payload.get("episode_id", ""),
        "feed_id": payload.get("show_id", ""),
    }
    # Carry segment timestamps (seconds → ms) so transcript lift can locate the span.
    if "start_time" in payload:
        metadata["timestamp_start_ms"] = int(float(payload.get("start_time") or 0.0) * 1000)
        metadata["timestamp_end_ms"] = int(float(payload.get("end_time") or 0.0) * 1000)
    if payload.get("speaker_id"):
        metadata["speaker_id"] = payload["speaker_id"]
    return SearchResult(doc_id=str(result.doc_id), score=float(result.score), metadata=metadata)


def _flatten(result: object) -> List[ScoredResult]:
    # A compound contributes both its insight and segment as standalone rows.
    if isinstance(result, CompoundResult):
        return [result.insight, result.segment]
    if isinstance(result, ScoredResult):
        return [result]
    return []


def hybrid_candidates(
    output_dir: Path,
    query: str,
    *,
    top_k: int,
    doc_types: Optional[Sequence[str]] = None,
    embedding_model: Optional[str] = None,
    fetch_multiplier: int = 25,
) -> Optional[List[SearchResult]]:
    """Hybrid candidates as ``SearchResult`` rows, or ``None`` to fall back to FAISS.

    Returns ``None`` (not an empty list) on any condition that should defer to FAISS:
    missing LanceDB index or a query-embedding failure. An empty list means the index
    was searched and genuinely had no hits.
    """
    index_dir = lance_index_dir(output_dir)
    if not index_dir.exists():
        return None

    model_id = (
        embedding_model.strip()
        if isinstance(embedding_model, str) and embedding_model.strip()
        else _DEFAULT_MODEL
    )
    try:
        qvec = embedding_loader.encode(query, model_id, return_numpy=False, allow_download=False)
    except Exception as exc:  # noqa: BLE001 - any embed failure → FAISS fallback
        logger.warning("hybrid_search embed failed (%s); falling back to FAISS", exc)
        return None
    if not (isinstance(qvec, list) and qvec and isinstance(qvec[0], float)):
        return None
    qemb = cast(List[float], qvec)

    try:
        from .backends.lancedb_backend import LanceDBBackend
        from .retrieval import RetrievalLayer

        layer = RetrievalLayer(LanceDBBackend(str(index_dir)))
        fetch_k = max(top_k * fetch_multiplier, top_k)
        results = layer.retrieve(query, qemb, k=fetch_k, tier=_tier_for(doc_types))
    except Exception as exc:  # noqa: BLE001 - backend/index error → FAISS fallback
        logger.warning("hybrid_search retrieve failed (%s); falling back to FAISS", exc)
        return None

    rows: List[SearchResult] = []
    for result in results:
        rows.extend(_to_search_result(r) for r in _flatten(result))
    return rows
