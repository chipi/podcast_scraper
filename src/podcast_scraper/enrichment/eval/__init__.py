"""Enricher accuracy-eval + gating framework (RFC-088 amendment, 2026-07).

The grading + gating counterpart to the enrichment runtime. Where the runtime
(``enrichment.protocol`` / ``registry`` / ``profile_sets``) declares, registers,
and runs enrichers, this package declares, registers, and runs *scorers* that
grade enricher output against gold — and turns the resulting metrics into an
admission decision that cascades to the registry → profiles → UI config, exactly
as provider quality gates (``evaluation.regression``) gate providers.

Public surface:

* Scorer side — ``AccuracyScorer`` / ``ScorerManifest`` / ``ScoreResult`` /
  ``ScorerRegistry`` / ``register_builtin_scorers`` / ``run_scorers``.
* Gold side — ``EXPECTED_ENRICHMENT_KEY`` + ``gold_for`` (one generic block per
  enricher id — no per-enricher field names).
* Gate side — ``evaluate_gate`` / ``GateDecision``.
* Cascade — ``admit_enrichers`` / ``admitted_enricher_ids`` / ``AdmissionResult``.

See ``docs/wip/RFC-088-ENRICHER-ACCURACY-GATE-2026-07.md``.
"""

from __future__ import annotations

from podcast_scraper.enrichment.eval.admission import (
    AdmissionResult,
    admit_enrichers,
    admitted_enricher_ids,
    known_enricher_manifests,
    load_latest_eval_metrics,
    write_gate_metrics,
)
from podcast_scraper.enrichment.eval.gate import evaluate_gate, GateDecision, GateViolation
from podcast_scraper.enrichment.eval.gold import EXPECTED_ENRICHMENT_KEY, gold_for
from podcast_scraper.enrichment.eval.protocol import (
    AccuracyScorer,
    ScoreResult,
    ScorerManifest,
)
from podcast_scraper.enrichment.eval.registry import ScorerRegistry
from podcast_scraper.enrichment.eval.runner import metrics_by_enricher, run_scorers
from podcast_scraper.enrichment.eval.scorers import register_builtin_scorers

__all__ = [
    "EXPECTED_ENRICHMENT_KEY",
    "AccuracyScorer",
    "AdmissionResult",
    "GateDecision",
    "GateViolation",
    "ScoreResult",
    "ScorerManifest",
    "ScorerRegistry",
    "admit_enrichers",
    "admitted_enricher_ids",
    "evaluate_gate",
    "gold_for",
    "known_enricher_manifests",
    "load_latest_eval_metrics",
    "metrics_by_enricher",
    "register_builtin_scorers",
    "run_scorers",
    "write_gate_metrics",
]
