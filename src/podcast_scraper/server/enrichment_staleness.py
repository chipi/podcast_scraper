"""Corpus-level enrichment freshness — the enrichment mirror of ``index_staleness`` (RFC-118 §5).

``compute_enrichment_staleness`` answers, from on-disk facts alone (no enrichment run,
no models): *is the corpus enrichment current, and if not, why?* Per-enricher rows plus
a rolled-up ``reenrich_recommended`` flag, with typed reasons:

* ``never_ran`` — no output envelope / no run-summary entry for the enricher.
* ``enricher_version_changed`` — the output was produced by an older enricher version.
* ``last_run_failed_or_timed_out`` — the last recorded outcome was not ``ok``
  (this is exactly what the efdca585 topic_consensus timeout looked like).
* ``corpus_artifacts_newer`` — a gi/kg artifact is newer than the output.

Surfaced on ``GET /api/enrichment/stats`` (operator UI widget) and the MCP
``corpus_status`` tool. The explicit-full lever this recommends is
``POST /api/jobs/enrichment`` with ``force=true`` / MCP ``reenrich``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REASON_NEVER_RAN = "never_ran"
REASON_VERSION_CHANGED = "enricher_version_changed"
REASON_LAST_RUN_FAILED = "last_run_failed_or_timed_out"
REASON_ARTIFACTS_NEWER = "corpus_artifacts_newer"


@dataclass(frozen=True)
class EnricherFreshnessRow:
    """One enricher's freshness verdict."""

    enricher_id: str
    scope: str  # "corpus" | "episode"
    stale: bool
    reasons: List[str]
    last_status: Optional[str]
    last_computed_at: Optional[str]
    current_version: str
    output_version: Optional[str]


@dataclass(frozen=True)
class EnrichmentStalenessFields:
    """Rolled-up corpus enrichment freshness (RFC-118 §5)."""

    reenrich_recommended: bool
    reenrich_reasons: List[str]
    enrichers: List[EnricherFreshnessRow] = field(default_factory=list)
    artifact_newest_mtime: Optional[str] = None
    last_run_status: Optional[str] = None
    last_run_finished_at: Optional[str] = None


def _iso_to_epoch(iso_str: Optional[str]) -> Optional[float]:
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _epoch_to_iso(epoch: Optional[float]) -> Optional[str]:
    if epoch is None:
        return None
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_run_summary(corpus_root: Path) -> Dict[str, Any]:
    import json

    from podcast_scraper.enrichment.paths import enrichment_run_summary_path

    try:
        raw = json.loads(enrichment_run_summary_path(corpus_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def compute_enrichment_staleness(corpus_root: Path) -> EnrichmentStalenessFields:
    """Per-enricher freshness rows + a rolled-up ``reenrich_recommended`` flag.

    Corpus-scope enrichers are judged from their output ENVELOPE (version, status,
    computed_at vs newest gi/kg artifact). Episode-scope enrichers are judged more
    coarsely from the last run-summary entry — their per-episode staleness is the
    executor's own input-fingerprint gate, which needs no operator surface.
    """
    from podcast_scraper.enrichment.eval.admission import known_enricher_manifests
    from podcast_scraper.enrichment.paths import corpus_enrichment_path
    from podcast_scraper.enrichment.staleness import load_envelope
    from podcast_scraper.search.index_source_mtime import newest_index_source_mtime_epoch

    corpus_root = Path(corpus_root)
    newest_epoch = newest_index_source_mtime_epoch(corpus_root)
    run_summary = _load_run_summary(corpus_root)
    per_raw = run_summary.get("per_enricher")
    per_enricher_summary: Dict[str, Any] = per_raw if isinstance(per_raw, dict) else {}

    rows: List[EnricherFreshnessRow] = []
    for enricher_id, manifest in sorted(known_enricher_manifests().items()):
        scope = manifest.scope.value
        reasons: List[str] = []
        last_status: Optional[str] = None
        computed_at: Optional[str] = None
        output_version: Optional[str] = None

        if scope == "corpus":
            envelope = load_envelope(corpus_enrichment_path(corpus_root, manifest.writes))
            if not isinstance(envelope, dict):
                reasons.append(REASON_NEVER_RAN)
            else:
                last_status = str(envelope.get("status") or "") or None
                computed_at = str(envelope.get("computed_at") or "") or None
                output_version = str(envelope.get("enricher_version") or "") or None
                if output_version != manifest.version:
                    reasons.append(REASON_VERSION_CHANGED)
                if last_status != "ok":
                    reasons.append(REASON_LAST_RUN_FAILED)
                computed_epoch = _iso_to_epoch(computed_at)
                if (
                    newest_epoch is not None
                    and computed_epoch is not None
                    and newest_epoch > computed_epoch
                ):
                    reasons.append(REASON_ARTIFACTS_NEWER)
        else:
            entry = per_enricher_summary.get(enricher_id)
            if not isinstance(entry, dict):
                reasons.append(REASON_NEVER_RAN)
            else:
                last_status = str(entry.get("status") or "") or None
                computed_at = str(run_summary.get("finished_at") or "") or None
                if last_status not in ("ok", "skipped"):
                    reasons.append(REASON_LAST_RUN_FAILED)
                computed_epoch = _iso_to_epoch(computed_at)
                if (
                    newest_epoch is not None
                    and computed_epoch is not None
                    and newest_epoch > computed_epoch
                ):
                    reasons.append(REASON_ARTIFACTS_NEWER)

        rows.append(
            EnricherFreshnessRow(
                enricher_id=enricher_id,
                scope=scope,
                stale=bool(reasons),
                reasons=reasons,
                last_status=last_status,
                last_computed_at=computed_at,
                current_version=manifest.version,
                output_version=output_version,
            )
        )

    # The rolled-up flag reads CORPUS-scope rows plus the overall last run outcome.
    # Episode-scope rows inform the table but do not drive the recommendation — the
    # executor's per-episode fingerprint gate self-heals those on the next run.
    rollup_reasons: List[str] = []
    for row in rows:
        if row.scope == "corpus" and row.stale:
            rollup_reasons.extend(row.reasons)
    last_run_status = str(run_summary.get("status") or "") or None
    if last_run_status not in (None, "ok") and REASON_LAST_RUN_FAILED not in rollup_reasons:
        rollup_reasons.append(REASON_LAST_RUN_FAILED)

    return EnrichmentStalenessFields(
        reenrich_recommended=bool(rollup_reasons),
        reenrich_reasons=sorted(set(rollup_reasons)),
        enrichers=rows,
        artifact_newest_mtime=_epoch_to_iso(newest_epoch),
        last_run_status=last_run_status,
        last_run_finished_at=str(run_summary.get("finished_at") or "") or None,
    )
