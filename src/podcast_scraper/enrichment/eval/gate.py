"""Accuracy gate — turns eval metrics into a promote/reject decision.

The enricher-side analogue of the provider quality gate
(``podcast_scraper.evaluation.regression.RegressionRule`` /
``RegressionChecker``). A provider ``RegressionRule`` fires on a *delta vs
baseline*; an :class:`AccuracyGateRule` (declared on the enricher's manifest)
fires on an *absolute floor*. :func:`evaluate_gate` reads the latest
``data/eval`` metrics for one enricher and returns a :class:`GateDecision`;
``eval.admission`` consumes those decisions to build the admitted set that
cascades to the registry → profiles → UI config.

Semantics:

* No gate declared → promoted (no accuracy bar).
* Gate declared, no eval metrics → the spec's ``on_missing_data`` policy decides.
* Gate declared, metrics present → promoted iff every ``severity == "error"``
  rule is cleared. ``warning`` / ``info`` rules record a violation but do not
  block (advisory, surfaced to the UI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from podcast_scraper.enrichment.protocol import AccuracyGateSpec

_ON_MISSING_ADMIT = "admit"
_ON_MISSING_REJECT = "reject"


@dataclass(frozen=True)
class GateViolation:
    """One unmet rule within a gate decision."""

    metric_name: str
    min_value: float
    actual: float | None  # None → metric absent from the eval metrics
    severity: str


@dataclass(frozen=True)
class GateDecision:
    """The outcome of evaluating one enricher's accuracy gate.

    ``reason`` is a short human-facing string (surfaced in logs + the UI so an
    operator sees *why* an enricher is off, not just that it is absent).
    """

    enricher_id: str
    promoted: bool
    reason: str
    violations: tuple[GateViolation, ...] = field(default_factory=tuple)
    had_metrics: bool = False


def _fmt_metric(actual: float | None) -> str:
    return "n/a" if actual is None else f"{actual:.2f}"


def evaluate_gate(
    enricher_id: str,
    spec: AccuracyGateSpec | None,
    metrics: dict[str, float] | None,
) -> GateDecision:
    """Evaluate one enricher's accuracy gate against its latest eval metrics.

    ``spec`` is the manifest's :attr:`EnricherManifest.accuracy_gate` (``None``
    when the enricher declares no bar). ``metrics`` is the flat
    ``{metric_name: value}`` map from the latest ``data/eval`` record for this
    enricher (``None`` / empty when nothing has measured it).
    """
    if spec is None:
        return GateDecision(enricher_id, promoted=True, reason="no accuracy gate declared")

    if not metrics:
        if spec.on_missing_data == _ON_MISSING_ADMIT:
            return GateDecision(
                enricher_id,
                promoted=True,
                reason="no eval data yet (on_missing_data=admit)",
            )
        # Default + explicit "reject": excluded until a passing eval is recorded.
        return GateDecision(
            enricher_id,
            promoted=False,
            reason="no eval data yet (on_missing_data=reject)",
        )

    violations: list[GateViolation] = []
    for rule in spec.rules:
        actual = metrics.get(rule.metric_name)
        if actual is None or actual < rule.min_value:
            violations.append(
                GateViolation(
                    metric_name=rule.metric_name,
                    min_value=rule.min_value,
                    actual=actual,
                    severity=rule.severity,
                )
            )

    blocking = [v for v in violations if v.severity == "error"]
    promoted = not blocking
    if promoted:
        reason = (
            "cleared all error-severity rules"
            if not violations
            else "cleared all error-severity rules (advisory warnings present)"
        )
    else:
        parts = [f"{v.metric_name} {_fmt_metric(v.actual)} < {v.min_value:.2f}" for v in blocking]
        reason = "gated: " + ", ".join(parts)
    return GateDecision(
        enricher_id=enricher_id,
        promoted=promoted,
        reason=reason,
        violations=tuple(violations),
        had_metrics=True,
    )


__all__ = ["GateDecision", "GateViolation", "evaluate_gate"]
