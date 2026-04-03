"""PRD-017 success metrics over per-episode ``gi.json`` artifacts (file-based, no DB).

Aggregates grounding rate, quote validity, and density for operator/CI gates.
Thresholds default to PRD-017 targets; use :func:`enforce_prd017_thresholds` with
``--enforce`` in ``scripts/tools/gil_quality_metrics.py``.

**Quote validity (file-based view):** :func:`compute_gil_quality_metrics` checks
schema/evidence fields (spans, ``transcript_ref``, timestamps). A **pipeline run**
can additionally compute **verbatim** match vs transcript slices via
``Metrics.record_gi_success_counts`` when the transcript is available — the two
rates measure related but not identical things; compare like with like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io import collect_gi_paths_from_inputs, read_artifact
from .load import build_inspect_output
from .schema import validate_artifact


def _quote_evidence_valid(q: Any) -> bool:
    """True if span and timestamps look usable (PRD-017 quote validity)."""
    ev = q.evidence
    if ev.char_start < 0 or ev.char_end < 0 or ev.char_end <= ev.char_start:
        return False
    if not (ev.transcript_ref or "").strip():
        return False
    ts0 = q.timestamp_start_ms
    ts1 = q.timestamp_end_ms
    if ts0 is None or ts1 is None:
        return False
    if ts0 < 0 or ts1 < 0 or ts1 < ts0:
        return False
    return True


@dataclass
class GilQualityMetrics:
    """Aggregated metrics over one or more GIL artifacts."""

    artifact_paths: int = 0
    artifacts_with_insight_and_quote: int = 0
    total_insights: int = 0
    grounded_insights: int = 0
    total_quotes: int = 0
    valid_quotes: int = 0
    total_insights_per_artifact: List[int] = field(default_factory=list)
    total_quotes_per_artifact: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def extraction_coverage(self) -> float:
        """Share of artifacts that have at least one insight and one quote."""
        if self.artifact_paths <= 0:
            return 0.0
        return self.artifacts_with_insight_and_quote / self.artifact_paths

    def grounded_insight_rate(self) -> float:
        """Share of insights with ``grounded=True``."""
        if self.total_insights <= 0:
            return 0.0
        return self.grounded_insights / self.total_insights

    def quote_validity_rate(self) -> float:
        """Share of supporting quotes passing span + timestamp checks."""
        if self.total_quotes <= 0:
            return 0.0
        return self.valid_quotes / self.total_quotes

    def avg_insights_per_artifact(self) -> float:
        """Mean insight count per artifact over collected per-artifact totals."""
        if not self.total_insights_per_artifact:
            return 0.0
        return sum(self.total_insights_per_artifact) / len(self.total_insights_per_artifact)

    def avg_quotes_per_artifact(self) -> float:
        """Mean quote count per artifact over collected per-artifact totals."""
        if not self.total_quotes_per_artifact:
            return 0.0
        return sum(self.total_quotes_per_artifact) / len(self.total_quotes_per_artifact)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "artifact_paths": self.artifact_paths,
            "artifacts_with_insight_and_quote": self.artifacts_with_insight_and_quote,
            "extraction_coverage": round(self.extraction_coverage(), 4),
            "total_insights": self.total_insights,
            "grounded_insights": self.grounded_insights,
            "grounded_insight_rate": round(self.grounded_insight_rate(), 4),
            "total_quotes": self.total_quotes,
            "valid_quotes": self.valid_quotes,
            "quote_validity_rate": round(self.quote_validity_rate(), 4),
            "avg_insights_per_artifact": round(self.avg_insights_per_artifact(), 4),
            "avg_quotes_per_artifact": round(self.avg_quotes_per_artifact(), 4),
            "errors": list(self.errors),
        }


def compute_gil_quality_metrics(
    paths: List[Any],
    *,
    strict_schema: bool = False,
) -> GilQualityMetrics:
    """Load artifacts from paths (files or dirs), validate, and compute PRD-017 metrics."""
    raw_paths = [Path(p) for p in paths]
    try:
        gi_paths = collect_gi_paths_from_inputs(raw_paths)
    except (FileNotFoundError, ValueError) as e:
        m = GilQualityMetrics()
        m.errors.append(str(e))
        return m

    out = GilQualityMetrics(artifact_paths=len(gi_paths))
    for gpath in gi_paths:
        try:
            data = read_artifact(gpath, validate=False)
            validate_artifact(data, strict=strict_schema)
            inspect = build_inspect_output(data, transcript_text=None)
        except Exception as e:
            out.errors.append(f"{gpath}: {e}")
            continue

        n_ins = len(inspect.insights)
        n_quotes = sum(len(i.supporting_quotes) for i in inspect.insights)
        out.total_insights_per_artifact.append(n_ins)
        out.total_quotes_per_artifact.append(n_quotes)
        if n_ins >= 1 and n_quotes >= 1:
            out.artifacts_with_insight_and_quote += 1

        for ins in inspect.insights:
            out.total_insights += 1
            if ins.grounded:
                out.grounded_insights += 1
            for q in ins.supporting_quotes:
                out.total_quotes += 1
                if _quote_evidence_valid(q):
                    out.valid_quotes += 1

    return out


def enforce_prd017_thresholds(
    m: GilQualityMetrics,
    *,
    min_extraction_coverage: float = 0.80,
    min_grounded_insight_rate: float = 0.90,
    min_quote_validity_rate: float = 0.95,
    min_avg_insights: float = 5.0,
    min_avg_quotes: float = 10.0,
) -> Tuple[bool, List[str]]:
    """Return (all_passed, list of human-readable failures)."""
    failures: List[str] = []
    if m.artifact_paths <= 0:
        failures.append("No .gi.json artifacts found to score.")
        return False, failures
    if m.errors:
        failures.append(f"{len(m.errors)} artifact(s) failed to load or validate.")
    if m.total_insights == 0 and m.artifact_paths > 0 and len(m.errors) < m.artifact_paths:
        failures.append("No insights found in any artifact (check extraction or paths).")
    if m.extraction_coverage() + 1e-9 < min_extraction_coverage:
        failures.append(
            f"extraction_coverage {m.extraction_coverage():.3f} < {min_extraction_coverage} "
            "(artifacts with ≥1 insight and ≥1 quote)"
        )
    if m.total_insights > 0 and m.grounded_insight_rate() + 1e-9 < min_grounded_insight_rate:
        failures.append(
            f"grounded_insight_rate {m.grounded_insight_rate():.3f} < {min_grounded_insight_rate}"
        )
    if m.total_quotes > 0 and m.quote_validity_rate() + 1e-9 < min_quote_validity_rate:
        failures.append(
            f"quote_validity_rate {m.quote_validity_rate():.3f} < {min_quote_validity_rate}"
        )
    if m.avg_insights_per_artifact() + 1e-9 < min_avg_insights:
        failures.append(
            f"avg_insights_per_artifact {m.avg_insights_per_artifact():.3f} < {min_avg_insights}"
        )
    if m.avg_quotes_per_artifact() + 1e-9 < min_avg_quotes:
        failures.append(
            f"avg_quotes_per_artifact {m.avg_quotes_per_artifact():.3f} < {min_avg_quotes}"
        )
    return len(failures) == 0, failures
