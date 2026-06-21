"""Hybrid retrieval bridge for the live search path (RFC-090 Phase 2 / ADR-099).

Connects the two-tier stack (``RetrievalLayer`` over ``LanceDBBackend``, #855-861) to
the serving path (``corpus_search.run_corpus_search``): hybrid candidates are mapped to
the ``SearchResult`` shape, then run through the filter/enrich/lift pipeline.

**Always-on, single index.** Since FAISS was retired (#995, ADR-099) the LanceDB
two-tier index is the only search path; there is no toggle. The segment + insight
tiers cover transcript and insight docs, and the *aux* tier covers kg_entity /
kg_topic / quote / summary, so the index spans every doc type. ``hybrid_candidates``
returns ``None`` only when there is no usable index (no index, embed failure, dim
mismatch); the caller then reports ``no_index`` (there is no FAISS fallback).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Sequence, TYPE_CHECKING

import yaml

from .. import config as _config
from ..providers.ml import embedding_loader
from ..utils.path_validation import (
    normpath_if_under_root,
    safe_relpath_under_corpus_root,
    safe_resolve_directory,
)
from .backend import CompoundResult, ScoredResult, Tier
from .protocol import SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .query_router import QueryRouter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _search_config_path() -> Optional[Path]:
    """Locate ``search.yaml`` robustly (env override > CWD > repo-root anchor).

    A bare ``Path("config/search.yaml")`` is CWD-relative and silently missing in a
    deployed container (audit H1). Candidates cover dev (CWD / editable source tree)
    and prod (the file shipped next to the working dir).
    """
    env = os.getenv("PODCAST_SEARCH_CONFIG")
    candidates = [Path(env)] if env else []
    candidates.append(Path("config/search.yaml"))
    candidates.append(Path(__file__).resolve().parents[3] / "config" / "search.yaml")
    return next((p for p in candidates if p.is_file()), None)


def _load_search_config() -> Dict[str, Any]:
    """Parse ``search.yaml`` if found, else ``{}`` (never raises)."""
    path = _search_config_path()
    if path is None:
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _serving_router() -> "Optional[QueryRouter]":
    """Build the configured query router (RFC-092 #860), or None to use rules default."""
    cfg = _load_search_config().get("router")
    if not isinstance(cfg, dict):
        return None
    from .query_router import get_query_router

    return get_query_router(str(cfg.get("mode") or "rules"), model_path=cfg.get("model_path"))


def lance_index_dir(output_dir: Path) -> Path:
    """Per-corpus LanceDB index location (co-located with the corpus, #858/#862)."""
    return Path(output_dir) / "search" / "lance_index"


_AUX_DOC_TYPES = frozenset({"kg_entity", "kg_topic", "quote", "summary"})


def _tier_for(doc_types: Optional[Sequence[str]]) -> Tier:
    """Map requested doc types to a tier scope (insight | segment | aux | all)."""
    if not doc_types:
        return "all"
    wanted = {t.strip().lower() for t in doc_types if isinstance(t, str) and t.strip()}
    if wanted <= {"insight"}:
        return "insight"
    if wanted <= {"transcript", "segment"}:
        return "segment"
    if wanted <= _AUX_DOC_TYPES:
        return "aux"
    return "all"


def _doc_type_for_result(result: ScoredResult, payload: Dict) -> str:
    # Segment = transcript; aux rows carry their own doc_type.
    if result.source_tier == "insight":
        return "insight"
    if result.source_tier == "aux":
        return str(payload.get("doc_type") or "aux")
    return "transcript"


def _to_search_result(result: ScoredResult) -> SearchResult:
    payload = dict(result.payload or {})
    metadata = {
        "doc_type": _doc_type_for_result(result, payload),
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
    # Carry the episode publish date so the shared date/``since`` filter — and the
    # digest topic-band window join — work on the hybrid path. Without it, every
    # hybrid hit is dropped whenever ``since`` is set.
    if payload.get("publish_date"):
        metadata["publish_date"] = payload["publish_date"]
    # Canonical graph node id so a result's "Show on graph"
    # affordance resolves — the viewer reads metadata.source_id for focusable tiers
    # (insight / quote / kg_topic / kg_entity). Absent it, no graph handoff renders.
    if payload.get("source_id"):
        metadata["source_id"] = payload["source_id"]
    return SearchResult(doc_id=str(result.doc_id), score=float(result.score), metadata=metadata)


def _flatten(result: object) -> List[ScoredResult]:
    # A compound contributes both its insight and segment as standalone rows.
    if isinstance(result, CompoundResult):
        return [result.insight, result.segment]
    if isinstance(result, ScoredResult):
        return [result]
    return []


class QueryEmbeddingError(Exception):
    """The query could not be embedded (model missing/offline or empty vector).

    Distinct from "no index": re-indexing won't help — the index is fine, the
    *query*-side embedding failed. Callers map this to ``embed_failed`` so the UI
    can say "model missing or offline" instead of the misleading "run indexing".
    """


def hybrid_candidates(
    output_dir: Path,
    query: str,
    *,
    top_k: int,
    doc_types: Optional[Sequence[str]] = None,
    embedding_model: Optional[str] = None,
    # ADR-099 Stage 2 (#995): the ×25 over-fetch was an in-memory-index habit — cheap to pull 600
    # rows from an in-memory array, then filter in Python. On the database it dominates cost
    # (limit 600 ≈ 0.37s vs limit 100 ≈ 0.09s per query). Native in-engine RRF fusion needs
    # far fewer candidates; ×6 keeps enough headroom for the downstream type/feed/since
    # filters in ``_filter_and_enrich`` while cutting the per-query cost ~4×.
    fetch_multiplier: int = 6,
) -> Optional[List[SearchResult]]:
    """Hybrid candidates as ``SearchResult`` rows, or ``None`` when there is no usable index.

    Returns ``None`` (not an empty list) when the index cannot serve the query: missing
    LanceDB index, a stale schema, or a query-embedding failure — the caller then reports
    ``no_index`` (FAISS was retired, #995; there is no fallback). An empty list means the
    index was searched and genuinely had no hits.
    """
    # py/path-injection sanitizer chain (CodeQL Type 1, docs/ci/CODEQL_DISMISSALS.md;
    # mirrors jobs_log_path): resolve the corpus dir, confine the CONSTANT index subpath
    # under it, then pragma the sink. The corpus path is sanitized cross-function at the
    # route, which CodeQL cannot model.
    root_res = safe_resolve_directory(Path(output_dir))
    if root_res is None:
        return None
    root_s = os.path.normpath(str(root_res))
    verified = safe_relpath_under_corpus_root(root_res, "search/lance_index")
    if not verified:
        return None
    index_dir_str = normpath_if_under_root(os.path.normpath(verified), root_s)
    if not index_dir_str:
        return None
    # codeql[py/path-injection] -- index_dir_str via normpath_if_under_root (Type 1).
    if not os.path.isdir(index_dir_str):
        return None
    index_dir = Path(index_dir_str)

    # Self-healing schema guard: an index built before a schema bump lacks columns the
    # read path expects (e.g. publish_date → date filters silently drop every hit), so
    # skip it (report no_index) until a (re)index moment rebuilds it.
    from .backends.lancedb_backend import (
        lance_index_is_stale,
        LANCE_SCHEMA_VERSION,
        stored_schema_version,
    )

    if lance_index_is_stale(index_dir):
        logger.warning(
            "hybrid_search: lance index at %s has a stale schema (v%s < v%s); reporting "
            "no_index — rebuild via `cli index-two-tier` or `cli upgrade`",
            index_dir,
            stored_schema_version(index_dir),
            LANCE_SCHEMA_VERSION,
        )
        return None

    try:
        from .backends.lancedb_backend import LanceDBBackend
        from .index_pool import get_lance_backend
        from .retrieval import RetrievalLayer

        # ADR-099 / #995: reuse one warm backend per index_dir instead of opening the
        # LanceDB connection + index readers on every query (~0.8 s) — the query on a
        # warm table is ~7 ms. Pooling also removes the concurrent-cold-init segfault.
        backend = get_lance_backend(index_dir, lambda: LanceDBBackend(str(index_dir)))
    except Exception as exc:  # noqa: BLE001 - cannot open index → report no_index
        logger.warning("hybrid_search open failed (%s); reporting no_index", exc)
        return None

    # Embed the query in the SAME space the index was built in: explicit override >
    # the model recorded at build time > MiniLM default. Mismatched models silently
    # return wrong results, so the index's own model is the source of truth.
    meta = backend.read_index_meta() or {}
    model_id = (
        embedding_model.strip()
        if isinstance(embedding_model, str) and embedding_model.strip()
        else str(meta.get("embedding_model") or _DEFAULT_MODEL)
    )
    try:
        _cfg = _config.Config()
        qvec = embedding_loader.encode(
            query,
            model_id,
            return_numpy=False,
            allow_download=False,
            remote_endpoint=_cfg.vector_embedding_endpoint,
            provider=_cfg.vector_embedding_provider,
        )
    except Exception as exc:  # noqa: BLE001 - any embed failure → embed_failed (not no_index)
        logger.warning("hybrid_search embed failed (%s); reporting embed_failed", exc)
        raise QueryEmbeddingError(str(exc)) from exc
    if not (isinstance(qvec, list) and qvec and isinstance(qvec[0], float)):
        logger.warning("hybrid_search produced an empty/invalid query vector; embed_failed")
        raise QueryEmbeddingError("query embedding returned an empty or invalid vector")
    qemb = cast(List[float], qvec)

    expected_dim = meta.get("embed_dim")
    if isinstance(expected_dim, int) and len(qemb) != expected_dim:
        logger.warning(
            "hybrid_search dim mismatch (query %d != index %d for model %s); no_index",
            len(qemb),
            expected_dim,
            model_id,
        )
        return None

    try:
        layer = RetrievalLayer(backend, router=_serving_router())
        fetch_k = max(top_k * fetch_multiplier, top_k)
        results = layer.retrieve(query, qemb, k=fetch_k, tier=_tier_for(doc_types))
    except Exception as exc:  # noqa: BLE001 - backend/index error → report no_index
        logger.warning("hybrid_search retrieve failed (%s); reporting no_index", exc)
        return None

    rows: List[SearchResult] = []
    for result in results:
        rows.extend(_to_search_result(r) for r in _flatten(result))
    return rows
