"""Vector index staleness heuristics for viewer API (GitHub #507)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from podcast_scraper import config
from podcast_scraper.providers.ml.model_registry import ModelRegistry
from podcast_scraper.search.corpus_scope import discover_metadata_files
from podcast_scraper.search.index_source_mtime import newest_index_source_mtime_epoch
from podcast_scraper.utils.corpus_episode_paths import corpus_search_parent_hint

logger = logging.getLogger(__name__)

# Slack so clock / filesystem granularity does not false-positive staleness.
_STALE_AFTER_INDEX_SEC = 1.0

REASON_ARTIFACTS_NEWER = "artifacts_newer_than_index"
REASON_NO_INDEX_BUT_METADATA = "no_index_but_metadata"
REASON_EMBEDDING_MODEL_MISMATCH = "embedding_model_mismatch"
REASON_CORPUS_SEARCH_PARENT_HINT = "corpus_search_parent_hint"
REASON_MULTI_FEED_BATCH_INCOMPLETE = "multi_feed_batch_incomplete"


def _parse_index_last_updated_epoch(iso_str: str) -> Optional[float]:
    s = (iso_str or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _epoch_to_utc_iso(epoch: float) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _read_multi_feed_overall_ok(parent: Path) -> Optional[bool]:
    summary = parent / "corpus_run_summary.json"
    if not summary.is_file():
        return None
    try:
        doc = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.debug("Unreadable corpus_run_summary.json at %s", parent)
        return None
    ok = doc.get("overall_ok")
    return ok if isinstance(ok, bool) else None


@dataclass(frozen=True)
class IndexStalenessFields:
    """Extra fields merged into ``GET /api/index/stats`` responses."""

    reindex_recommended: bool
    reindex_reasons: List[str]
    artifact_newest_mtime: Optional[str]
    search_root_hints: List[str]


def compute_index_staleness(
    corpus_root: Path,
    *,
    index_available: bool,
    index_reason: Optional[str],
    index_last_updated: Optional[str],
    index_embedding_model: Optional[str],
    embedding_model_query: Optional[str],
) -> IndexStalenessFields:
    """Derive staleness flags for a resolved corpus root."""
    hints = corpus_search_parent_hint(corpus_root)
    reasons: List[str] = []
    if hints:
        reasons.append(REASON_CORPUS_SEARCH_PARENT_HINT)

    artifact_epoch = newest_index_source_mtime_epoch(corpus_root)
    artifact_iso = _epoch_to_utc_iso(artifact_epoch) if artifact_epoch is not None else None

    has_metadata = bool(discover_metadata_files(corpus_root))
    recommend = False

    batch_ok = _read_multi_feed_overall_ok(corpus_root)
    if batch_ok is False:
        reasons.append(REASON_MULTI_FEED_BATCH_INCOMPLETE)

    if not index_available:
        if has_metadata and index_reason in ("no_index", "load_failed"):
            reasons.append(REASON_NO_INDEX_BUT_METADATA)
            recommend = True
        return IndexStalenessFields(
            reindex_recommended=recommend,
            reindex_reasons=sorted(set(reasons)),
            artifact_newest_mtime=artifact_iso,
            search_root_hints=hints,
        )

    index_epoch = _parse_index_last_updated_epoch(index_last_updated or "")
    if (
        artifact_epoch is not None
        and index_epoch is not None
        and artifact_epoch > index_epoch + _STALE_AFTER_INDEX_SEC
    ):
        reasons.append(REASON_ARTIFACTS_NEWER)
        recommend = True
    elif artifact_epoch is not None and index_epoch is None:
        reasons.append(REASON_ARTIFACTS_NEWER)
        recommend = True

    expected = (
        embedding_model_query.strip()
        if embedding_model_query and embedding_model_query.strip()
        else config.Config().vector_embedding_model
    )
    idx_model = (index_embedding_model or "").strip()
    if idx_model and expected:
        try:
            resolved_idx = ModelRegistry.resolve_evidence_model_id(idx_model)
            resolved_exp = ModelRegistry.resolve_evidence_model_id(expected)
            models_differ = resolved_idx != resolved_exp
        except ValueError:
            models_differ = idx_model != expected.strip()
        if models_differ:
            reasons.append(REASON_EMBEDDING_MODEL_MISMATCH)
            recommend = True

    return IndexStalenessFields(
        reindex_recommended=recommend,
        reindex_reasons=sorted(set(reasons)),
        artifact_newest_mtime=artifact_iso,
        search_root_hints=hints,
    )
