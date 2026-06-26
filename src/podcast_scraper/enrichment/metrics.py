"""``EnrichmentMetrics`` — per-enricher counters + record helpers.

The framework owns the metric surface; enrichers just write through.
At end of run, the executor flushes all per-enricher records into
``Metrics.enrichment`` (workflow/metrics.py field) so the existing
``.to_json() / .to_csv() / log_metrics()`` paths surface enrichment
data automatically.

Per chunk-1 lock audit §I2: ``error_samples`` is capped at 5 via
``__post_init__`` — older samples popped on push to avoid unbounded
growth on chronically failing enrichers.

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§"Metrics + observability + analytics" and §"Per-enricher metric
record".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from podcast_scraper.enrichment.protocol import (
    EnricherResult,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_QUARANTINED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
)

# Hard cap for error_samples — newest 5 only (per chunk-1 lock audit §I2).
ERROR_SAMPLES_CAP = 5


@dataclass
class EnrichmentMetrics:
    """Per-enricher counters + diagnostic samples.

    The framework instantiates one of these per enricher per run and
    calls ``record_result(...)`` after every enrich() invocation. The
    aggregation across runs flows through workflow/metrics.py.
    """

    enricher_id: str
    enricher_version: str
    scope: str  # "episode" | "corpus"
    tier: str  # serialized EnricherTier

    runs_total: int = 0
    runs_ok: int = 0
    runs_failed: int = 0
    runs_timeout: int = 0
    runs_quarantined: int = 0
    runs_cancelled: int = 0
    runs_skipped: int = 0

    duration_seconds: float = 0.0
    retries_total: int = 0

    # Circuit transitions by transition type (e.g. "closed→open": 2).
    circuit_transitions: dict[str, int] = field(default_factory=dict)

    output_records_total: int = 0
    scorer_calls_total: int = 0
    # Scorer failure breakdown by error_class.
    scorer_failures_total: dict[str, int] = field(default_factory=dict)

    # ml / llm tier: token + cost accounting; deterministic stays 0.
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    last_run_status: str = ""
    last_run_started_at: str = ""
    last_run_finished_at: str = ""
    model_id: str = ""
    model_version: str = ""

    # Most recent ERROR_SAMPLES_CAP failure samples (popped FIFO).
    error_samples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Enforce the cap on construction (in case a loaded record exceeds it).
        if len(self.error_samples) > ERROR_SAMPLES_CAP:
            self.error_samples = self.error_samples[-ERROR_SAMPLES_CAP:]

    # ------------------------------------------------------------------ record

    def record_result(
        self,
        result: EnricherResult,
        *,
        started_at: str,
        finished_at: str,
        duration_s: float | None = None,
    ) -> None:
        """Update counters + samples from one EnricherResult.

        ``duration_s`` overrides the result's ``duration_ms`` when set
        (e.g. when the executor measures wall-clock more precisely
        than the enricher does).
        """
        self.runs_total += 1
        self.last_run_status = result.status
        self.last_run_started_at = started_at
        self.last_run_finished_at = finished_at
        if duration_s is None:
            duration_s = result.duration_ms / 1000.0
        self.duration_seconds += float(duration_s)
        self.retries_total += int(result.retry_count)

        # Status counter.
        if result.status == STATUS_OK:
            self.runs_ok += 1
        elif result.status == STATUS_FAILED:
            self.runs_failed += 1
        elif result.status == STATUS_TIMEOUT:
            self.runs_timeout += 1
        elif result.status == STATUS_QUARANTINED:
            self.runs_quarantined += 1
        elif result.status == STATUS_CANCELLED:
            self.runs_cancelled += 1
        elif result.status == STATUS_SKIPPED:
            self.runs_skipped += 1

        # Records written (meaningful when status == "ok").
        if result.status == STATUS_OK:
            self.output_records_total += int(result.records_written)

        # Error sample (capped FIFO).
        if result.status in (STATUS_FAILED, STATUS_TIMEOUT, STATUS_QUARANTINED):
            self._push_error_sample(
                {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": result.status,
                    "error": result.error,
                    "error_class": result.error_class,
                    "retry_count": result.retry_count,
                }
            )

    def record_scorer_call(self, *, error_class: str | None = None) -> None:
        """One scorer invocation. ``error_class != None`` increments failure breakdown."""
        self.scorer_calls_total += 1
        if error_class:
            self.scorer_failures_total[error_class] = (
                self.scorer_failures_total.get(error_class, 0) + 1
            )

    def record_tokens(self, *, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Accumulate token counts from a scorer/provider call."""
        if tokens_in:
            self.tokens_in += int(tokens_in)
        if tokens_out:
            self.tokens_out += int(tokens_out)

    def record_cost(self, *, cost_usd: float) -> None:
        """Accumulate USD cost from a scorer/provider call."""
        if cost_usd and cost_usd > 0.0:
            self.cost_usd += float(cost_usd)

    def record_circuit_transition(self, *, from_state: str, to_state: str) -> None:
        """Record a circuit-breaker state transition (e.g. closed → open)."""
        key = f"{from_state}->{to_state}"
        self.circuit_transitions[key] = self.circuit_transitions.get(key, 0) + 1

    def set_model(self, *, model_id: str, model_version: str = "") -> None:
        """Stamp the backing model id + version on the record."""
        self.model_id = model_id
        self.model_version = model_version

    # ------------------------------------------------------------------ helpers

    def _push_error_sample(self, sample: dict[str, Any]) -> None:
        """Append a sample, evicting oldest when over cap."""
        self.error_samples.append(sample)
        while len(self.error_samples) > ERROR_SAMPLES_CAP:
            self.error_samples.pop(0)

    # ------------------------------------------------------------------ serialize

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for envelope / dashboard / run-summary use."""
        return asdict(self)


def new_metrics_for(
    *,
    enricher_id: str,
    enricher_version: str,
    scope: str,
    tier: str,
) -> EnrichmentMetrics:
    """Construct a fresh ``EnrichmentMetrics`` keyed on the enricher manifest."""
    return EnrichmentMetrics(
        enricher_id=enricher_id,
        enricher_version=enricher_version,
        scope=scope,
        tier=tier,
    )
