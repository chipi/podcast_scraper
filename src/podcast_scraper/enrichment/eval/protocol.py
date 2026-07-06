"""Enricher accuracy-scorer protocol — the grading counterpart to the runtime.

Mirrors :class:`podcast_scraper.enrichment.protocol.Enricher`: where an
``Enricher`` declares a ``manifest`` and *produces* an output envelope, an
:class:`AccuracyScorer` declares a ``manifest`` and *grades* that envelope
against gold, emitting metrics.

The metrics feed the accuracy gate (``eval.gate``), which — mirroring the
provider quality gate (``evaluation.regression.RegressionRule``) — decides
whether the enricher is admitted to the registry, and therefore into profiles
and UI config.

Scoring is synchronous: grading is a pure comparison of two dicts (output vs
gold) with no ML providers in the loop, so authors don't pay the async tax the
runtime protocol carries for its embedding / NLI backends.

See ``docs/wip/RFC-088-ENRICHER-ACCURACY-GATE-2026-07.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ScorerManifest:
    """Declares which enricher a scorer grades and which metrics it emits.

    ``enricher_id`` MUST equal the graded enricher's
    :attr:`EnricherManifest.id` — that key ties output, gold, gate, and
    admission together (there is exactly one scorer per enricher id).
    """

    enricher_id: str  # == EnricherManifest.id of the enricher this grades
    version: str  # semver, e.g. "1.0.0"
    metrics: tuple[str, ...]  # metric names this scorer emits (precision, mae, …)
    description: str
    # Optional JSON-Schema fragment describing the gold block this scorer
    # expects under ``expected_enrichment[enricher_id]``. Used by the gold
    # validator + (later) fixture-authoring tooling. ``None`` = unconstrained.
    gold_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Canonical grading output — one enricher's metrics against gold.

    ``skipped`` distinguishes "measured and scored 0" from "not measurable"
    (no output produced, or no gold authored). A skipped result carries no
    metrics and must NOT be read as a failing score — the gate treats a
    skipped enricher via its ``on_missing_data`` policy, not as ``0.0``.
    """

    enricher_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    sample_count: int = 0  # gold items graded
    skipped: bool = False  # True when output or gold was absent
    notes: str | None = None
    # Optional per-item breakdown (e.g. per-topic precision) for drill-down;
    # never read by the gate, which consumes ``metrics`` only.
    details: dict[str, Any] | None = None


@runtime_checkable
class AccuracyScorer(Protocol):
    """The canonical accuracy-scorer protocol.

    PEP 544 structural typing + ``@runtime_checkable``, consistent with the
    runtime :class:`Enricher` protocol. Single-method, stateless units.
    """

    @property
    def manifest(self) -> ScorerManifest:
        """Declare the graded enricher id + emitted metric names."""
        ...

    def score(
        self,
        *,
        output: dict[str, Any],
        gold: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> ScoreResult:
        """Grade ``output`` (the enricher's emitted envelope ``data``) against
        ``gold`` (the ``expected_enrichment[enricher_id]`` block).

        ``config`` carries the enricher's per-run knobs (e.g. ``top_k``) so a
        ranking scorer can measure precision at the same K the enricher ran
        with. Return a :class:`ScoreResult`; never raise on ordinary
        empty/malformed input — return ``ScoreResult(skipped=True, …)`` so the
        gate's ``on_missing_data`` policy governs, not an exception.
        """
        ...


__all__ = ["AccuracyScorer", "ScoreResult", "ScorerManifest"]
