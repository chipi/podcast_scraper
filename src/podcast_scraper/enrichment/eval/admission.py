"""Admission — the cascade point where eval data decides registry membership.

``data/eval`` accuracy metrics + each enricher's manifest-declared
``accuracy_gate`` → the set of *admitted* enricher ids. Registration,
``profile_sets``, and the UI config route all consult this set, so a failing
eval removes an enricher everywhere at once (mirrors how a failing provider
regression gate pulls a provider). An enricher with no declared gate is always
admitted, so the shipping set is unchanged until a gate is authored.

Data flow::

    known_enricher_manifests() ─► gate_specs_from_manifests() ─┐
                                                               ├─► admitted_enricher_ids()
    load_latest_eval_metrics(data/eval) ───────────────────────┘        │
                                                                        ▼
                                              AdmissionResult(admitted, decisions)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from podcast_scraper.enrichment.eval.gate import evaluate_gate, GateDecision
from podcast_scraper.enrichment.protocol import AccuracyGateSpec, EnricherManifest

logger = logging.getLogger(__name__)

# Conventional location of per-enricher gate metrics, relative to the eval root
# (``data/eval`` by default). One file per enricher:
#   data/eval/enrichment/<enricher_id>/gate_metrics.json
# shape: {"enricher_id": "...", "metrics": {"precision": 0.8, ...}, "run_id": ...}
_ENRICHMENT_EVAL_SUBDIR = "enrichment"
_GATE_METRICS_FILENAME = "gate_metrics.json"


@dataclass(frozen=True)
class AdmissionResult:
    """Admitted enricher ids + the per-enricher decision that produced them."""

    admitted: list[str] = field(default_factory=list)
    decisions: dict[str, GateDecision] = field(default_factory=dict)

    def is_admitted(self, enricher_id: str) -> bool:
        """True when this enricher cleared its gate (or declares none)."""
        d = self.decisions.get(enricher_id)
        return bool(d and d.promoted)


def admitted_enricher_ids(
    candidate_ids: Sequence[str],
    gate_specs: Mapping[str, AccuracyGateSpec | None],
    eval_metrics: Mapping[str, dict[str, float]] | None = None,
) -> AdmissionResult:
    """Filter ``candidate_ids`` through their gates given the latest metrics.

    Pure + fully injectable (no disk / no import side effects) so it is trivial
    to unit-test. ``gate_specs`` maps id → its manifest gate (``None`` = no
    gate). ``eval_metrics`` maps id → the flat ``{metric: value}`` map from the
    latest ``data/eval`` record (missing → the gate's ``on_missing_data``
    policy governs). Order of ``candidate_ids`` is preserved in ``admitted``.
    """
    metrics = eval_metrics or {}
    admitted: list[str] = []
    decisions: dict[str, GateDecision] = {}
    for eid in candidate_ids:
        decision = evaluate_gate(eid, gate_specs.get(eid), metrics.get(eid))
        decisions[eid] = decision
        if decision.promoted:
            admitted.append(eid)
        else:
            logger.info("enrichment admission: %s not admitted — %s", eid, decision.reason)
    return AdmissionResult(admitted=admitted, decisions=decisions)


def known_enricher_manifests() -> dict[str, EnricherManifest]:
    """Map ``enricher_id → EnricherManifest`` for all built-in enrichers.

    Reads each enricher's *class-level* ``manifest`` attribute — no
    instantiation, so provider-injected ML enrichers (topic_similarity,
    topic_consensus, stance_timeline) are included without wiring a
    provider. This is the
    manifest source the gate + UI config read for ``accuracy_gate`` /
    ``config_schema`` / ``provider_requirement`` without a live registry.
    """
    from podcast_scraper.enrichment.enrichers import (
        GroundingRateEnricher,
        GuestCoappearanceEnricher,
        InsightDensityEnricher,
        InsightSentimentEnricher,
        TemporalVelocityEnricher,
        TopicConsensusEnricher,
        TopicCooccurrenceCorpusEnricher,
        TopicSimilarityEnricher,
        TopicThemeClustersEnricher,
    )

    classes = (
        TopicCooccurrenceCorpusEnricher,
        TopicThemeClustersEnricher,
        TemporalVelocityEnricher,
        GroundingRateEnricher,
        GuestCoappearanceEnricher,
        InsightDensityEnricher,
        InsightSentimentEnricher,
        TopicSimilarityEnricher,
        TopicConsensusEnricher,
    )
    return {cls.manifest.id: cls.manifest for cls in classes}


def gate_specs_from_manifests(
    manifests: Mapping[str, EnricherManifest],
) -> dict[str, AccuracyGateSpec | None]:
    """Project ``id → manifest`` to ``id → accuracy_gate`` (the gate's input)."""
    return {eid: m.accuracy_gate for eid, m in manifests.items()}


def _default_eval_root() -> Path:
    """Repo-root ``data/eval`` — enrichment/eval/admission.py → parents[4] = repo root."""
    return Path(__file__).resolve().parents[4] / "data" / "eval"


def load_latest_eval_metrics(
    eval_root: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Load per-enricher gate metrics from ``<eval_root>/enrichment/*/gate_metrics.json``.

    Returns ``{enricher_id: {metric: value}}``, empty when nothing has been
    recorded yet (the current state — no gold/eval authored). Robust to
    missing / malformed files: a bad file is logged + skipped, never raised,
    so a single corrupt record can't gate every enricher off.
    """
    root = (eval_root or _default_eval_root()) / _ENRICHMENT_EVAL_SUBDIR
    out: dict[str, dict[str, float]] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob(f"*/{_GATE_METRICS_FILENAME}")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("enrichment admission: skipping unreadable %s (%s)", path, exc)
            continue
        if not isinstance(doc, dict):
            continue
        eid = str(doc.get("enricher_id") or path.parent.name)
        raw = doc.get("metrics")
        if not isinstance(raw, dict):
            continue
        metrics: dict[str, float] = {}
        for k, v in raw.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[str(k)] = float(v)
        out[eid] = metrics
    return out


def write_gate_metrics(
    metrics: Mapping[str, dict[str, float]],
    *,
    eval_root: Path | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """Persist per-enricher metrics as the ``gate_metrics.json`` the gate reads.

    The persistence counterpart to :func:`load_latest_eval_metrics`, closing the
    loop end to end::

        run_scorers → metrics_by_enricher → write_gate_metrics
                    → load_latest_eval_metrics → gate → admission

    Writes one file per enricher to
    ``<eval_root>/enrichment/<id>/gate_metrics.json`` (parent dirs created).
    Returns the written paths, sorted.
    """
    root = (eval_root or _default_eval_root()) / _ENRICHMENT_EVAL_SUBDIR
    written: list[Path] = []
    for eid, values in metrics.items():
        target_dir = root / eid
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / _GATE_METRICS_FILENAME
        doc: dict[str, Any] = {"enricher_id": eid, "metrics": dict(values)}
        if run_id is not None:
            doc["run_id"] = run_id
        path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return sorted(written)


def admit_enrichers(
    candidate_ids: Sequence[str],
    *,
    eval_root: Path | None = None,
) -> AdmissionResult:
    """Convenience: gate ``candidate_ids`` using built-in manifests + on-disk eval data.

    The one call registration / ``profile_sets`` make. Reads manifest gates +
    the latest ``data/eval`` metrics, then delegates to the pure
    :func:`admitted_enricher_ids`.
    """
    specs = gate_specs_from_manifests(known_enricher_manifests())
    metrics = load_latest_eval_metrics(eval_root)
    return admitted_enricher_ids(candidate_ids, specs, metrics)


__all__ = [
    "AdmissionResult",
    "admit_enrichers",
    "admitted_enricher_ids",
    "gate_specs_from_manifests",
    "known_enricher_manifests",
    "load_latest_eval_metrics",
    "write_gate_metrics",
]
