"""Resilience model for the enrichment layer — state machine, retry,
circuit-breaker, heartbeat watchdog, cost-cap enforcement.

Reuses the existing primitives:

* ``utils/retry.py`` — backoff calculation idiom.
* ``utils/llm_circuit_breaker.py`` — circuit-breaker state pattern.
* ``utils/retryable_errors.py`` — failure taxonomy for network/HTTP errors.

Adds enrichment-specific machinery the existing utils don't cover:

* Per-tier policy table (deterministic / embedding / ml / llm).
* Per-enricher state machine: READY → RUNNING → OK | RETRY | QUARANTINED →
  cooldown → READY → AUTO-DISABLED.
* Enrichment-specific failure taxonomy: ``BadInputError``,
  ``DependencyAccessError``, ``ScorerTimeoutError``, ``ModelLoadError``,
  ``RunTimeoutError`` (in addition to the existing taxonomy).
* Heartbeat watchdog for stall detection.
* Cost-cap enforcement (per-enricher + run-wide).

State outcomes flow to ``EnricherResult.status`` — the executor never sees
raised exceptions out of an enricher's ``enrich()``.

See ``docs/wip/RFC-088-ENRICHMENT-LAYER-IMPLEMENTATION-PLAN.md``
§Resilience model.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum

from podcast_scraper.enrichment.envelope import EnvelopeShapeError
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherTier,
    STATUS_FAILED,
    STATUS_QUARANTINED,
    STATUS_TIMEOUT,
)
from podcast_scraper.utils.retryable_errors import is_retryable_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure taxonomy — enrichment-specific exceptions
# ---------------------------------------------------------------------------


class BadInputError(ValueError):
    """Required input artifact is missing or malformed.

    Non-retryable — the enricher cannot do its job without the input.
    The executor records ``status: "failed"`` with
    ``error_class: "BadInputError"``.
    """


class DependencyAccessError(Exception):
    """Transient backend access failure (LanceDB lock, file I/O, network).

    Retryable — backoff and try again. Reused across all tiers.
    """


class ScorerTimeoutError(Exception):
    """Per-call timeout from a scorer (NLI, embedding, LLM).

    Retryable — retry the single failed call without resetting the
    rest of the enrichment progress.
    """


class ModelLoadError(Exception):
    """Local ML model failed to load (corrupt cache, OOM during init, etc).

    Retryable once — frees memory + retries the load. If the second
    load also fails, the circuit opens immediately (skip subsequent
    runs of this enricher until cooldown).
    """


class RunTimeoutError(Exception):
    """Whole-enricher hard timeout (``asyncio.wait_for`` fired).

    Non-retryable — the executor cancels the run; partial output
    preserved with ``status: "timeout"``.
    """


# ---------------------------------------------------------------------------
# Classification of failures
# ---------------------------------------------------------------------------


class RetryClass(Enum):
    NON_RETRYABLE = "non_retryable"
    RETRYABLE = "retryable"
    RETRYABLE_ONCE = "retryable_once"


def classify_failure(exc: BaseException) -> RetryClass:
    """Classify an exception into the enrichment failure taxonomy.

    Order matters — most specific first. The HTTP/network branch
    defers to ``utils/retryable_errors.is_retryable_error`` only when
    the exception is structurally HTTP-shaped (has ``.response.status_code``
    or ``.status_code``); arbitrary unknown exceptions are NON_RETRYABLE
    (safety net for enricher bugs — a bare ``RuntimeError`` from an
    enricher body is a code error, not a transient backend issue).
    """
    # Non-retryable first.
    if isinstance(exc, EnvelopeShapeError):
        return RetryClass.NON_RETRYABLE
    if isinstance(exc, BadInputError):
        return RetryClass.NON_RETRYABLE
    if isinstance(exc, RunTimeoutError):
        return RetryClass.NON_RETRYABLE

    # Retryable-once: heavy resource init paths.
    if isinstance(exc, MemoryError):  # OOM
        return RetryClass.RETRYABLE_ONCE
    if isinstance(exc, ModelLoadError):
        return RetryClass.RETRYABLE_ONCE

    # Retryable: transient.
    if isinstance(exc, DependencyAccessError):
        return RetryClass.RETRYABLE
    if isinstance(exc, ScorerTimeoutError):
        return RetryClass.RETRYABLE
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError subclass on 3.11+
        return RetryClass.RETRYABLE

    # Network / HTTP failures: defer to the existing taxonomy ONLY when
    # the exception is structurally HTTP-shaped. Arbitrary exceptions
    # (e.g. a bare RuntimeError raised from an enricher body) are NOT
    # HTTP-shaped and fall through to the non-retryable safety net.
    if _looks_http_shaped(exc) and isinstance(exc, Exception):
        return RetryClass.RETRYABLE if is_retryable_error(exc) else RetryClass.NON_RETRYABLE

    # Default safety-net: treat unknown exceptions as non-retryable.
    # The executor wraps these in EnricherResult(status="failed"),
    # logs them, and fires a Sentry breadcrumb.
    return RetryClass.NON_RETRYABLE


def _looks_http_shaped(exc: BaseException) -> bool:
    """True when *exc* carries the HTTP-error shape ``utils/retryable_errors`` consumes.

    Specifically: has a ``.response.status_code`` (httpx / requests) or
    a top-level ``.status_code`` (the OpenAI SDK pattern). Bare
    ``RuntimeError`` / ``ValueError`` etc. raised by an enricher body
    are NOT HTTP-shaped and route to the safety net.
    """
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        return True
    return hasattr(exc, "status_code")


# ---------------------------------------------------------------------------
# Per-tier policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierPolicy:
    """Per-tier retry + circuit + auto-disable + concurrency policy."""

    max_retries: int
    initial_backoff_s: float
    backoff_factor: float
    max_backoff_s: float
    # Circuit threshold: consecutive failures within a single run.
    # ``None`` disables (deterministic tier).
    circuit_threshold: int | None
    # Cross-run auto-disable threshold: consecutive failed runs.
    auto_disable_threshold: int
    # Per-tier max concurrent enrichers within a phase.
    concurrency: int
    # Default hard timeout for the enricher's whole run.
    # ``None`` defers to the executor's scope-based default
    # (60s episode / 600s corpus).
    default_timeout_s: int | None


# Per-tier defaults per the chunk-1 lock-audit / plan body table.
DEFAULT_POLICIES: dict[EnricherTier, TierPolicy] = {
    EnricherTier.DETERMINISTIC: TierPolicy(
        max_retries=0,
        initial_backoff_s=0.0,
        backoff_factor=1.0,
        max_backoff_s=0.0,
        circuit_threshold=None,  # n/a — no retries means no circuit
        auto_disable_threshold=5,
        concurrency=4,
        default_timeout_s=None,
    ),
    EnricherTier.EMBEDDING: TierPolicy(
        max_retries=3,
        initial_backoff_s=1.0,
        backoff_factor=2.0,
        max_backoff_s=30.0,
        circuit_threshold=5,
        auto_disable_threshold=3,
        concurrency=2,
        default_timeout_s=None,
    ),
    EnricherTier.ML: TierPolicy(
        max_retries=2,
        initial_backoff_s=5.0,
        backoff_factor=2.0,
        max_backoff_s=60.0,
        circuit_threshold=3,
        auto_disable_threshold=2,
        concurrency=1,
        default_timeout_s=None,
    ),
    EnricherTier.LLM: TierPolicy(
        max_retries=5,
        initial_backoff_s=2.0,
        backoff_factor=2.0,
        max_backoff_s=120.0,
        circuit_threshold=3,
        auto_disable_threshold=2,
        concurrency=4,  # rate-limit decided at runtime by provider config
        default_timeout_s=None,
    ),
}


def policy_for(tier: EnricherTier) -> TierPolicy:
    """Return the default policy for a tier."""
    return DEFAULT_POLICIES[tier]


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


def compute_backoff(attempt: int, policy: TierPolicy) -> float:
    """Exponential backoff with jitter, capped at ``policy.max_backoff_s``.

    ``attempt`` is 1-indexed (first retry == attempt 1). Returns 0.0
    when the policy disables retries (deterministic tier).
    """
    if policy.max_retries == 0 or attempt < 1:
        return 0.0
    base = policy.initial_backoff_s * (policy.backoff_factor ** (attempt - 1))
    capped = min(base, policy.max_backoff_s)
    # 10% jitter, deterministic enough for tests via seeded ``random``.
    jitter = random.uniform(0, 0.1 * capped)
    return capped + jitter


# ---------------------------------------------------------------------------
# Circuit breaker — per-enricher, per-run
# ---------------------------------------------------------------------------


class CircuitStatus(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class EnricherCircuitState:
    """Per-enricher circuit-breaker state within a single run.

    ``consecutive_failures`` resets to 0 on success. When it reaches
    ``policy.circuit_threshold``, the circuit transitions to
    ``OPEN`` — the executor stops calling this enricher for the rest
    of the run and writes ``status: "quarantined"`` for any later
    invocations scheduled.
    """

    status: CircuitStatus = CircuitStatus.CLOSED
    consecutive_failures: int = 0
    opened_at: float | None = None

    def record_success(self) -> None:
        self.status = CircuitStatus.CLOSED
        self.consecutive_failures = 0
        self.opened_at = None

    def record_failure(self, policy: TierPolicy) -> None:
        self.consecutive_failures += 1
        threshold = policy.circuit_threshold
        if (
            threshold is not None
            and self.consecutive_failures >= threshold
            and self.status is not CircuitStatus.OPEN
        ):
            self.status = CircuitStatus.OPEN
            self.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        return self.status is CircuitStatus.OPEN


# ---------------------------------------------------------------------------
# Heartbeat watchdog — per-enricher stall detection
# ---------------------------------------------------------------------------


@dataclass
class HeartbeatWatchdog:
    """Track per-enricher heartbeat timestamps to detect stalls.

    Long-running enricher bodies emit a heartbeat via
    ``record_heartbeat()`` every batch. The executor periodically
    checks ``is_stalled()``; when True, it logs a WARNING and
    optionally escalates to cancel.
    """

    enricher_id: str
    expected_interval_s: float
    last_heartbeat_at: float = field(default_factory=time.monotonic)

    def record_heartbeat(self, *, now: float | None = None) -> None:
        self.last_heartbeat_at = time.monotonic() if now is None else now

    def is_stalled(self, *, now: float | None = None, factor: float = 2.0) -> bool:
        """True when no heartbeat has been recorded for > factor × expected.

        Default factor is 2× — picks up stalls without flagging on
        normal jitter. Pass a smaller factor in tests to make stall
        detection deterministic.
        """
        moment = time.monotonic() if now is None else now
        elapsed = moment - self.last_heartbeat_at
        return elapsed > (self.expected_interval_s * factor)


# ---------------------------------------------------------------------------
# Cost-cap enforcement (REPLAN-O7: plumbing in chunk 1)
# ---------------------------------------------------------------------------


@dataclass
class CostCapState:
    """Cost accounting across an enrichment run.

    Per-enricher quarantine fires when an enricher's accumulated
    cost in this run exceeds its ``manifest.max_cost_usd_per_run``.
    Run-wide abort fires when the total cost across enrichers in
    this run exceeds ``max_total_cost_usd_per_run``.
    """

    per_enricher_cost: dict[str, float] = field(default_factory=dict)
    run_total_cost: float = 0.0

    def record_cost(self, enricher_id: str, cost_usd: float) -> None:
        """Add ``cost_usd`` to both the per-enricher and the run-wide totals."""
        if cost_usd <= 0.0:
            return
        self.per_enricher_cost[enricher_id] = (
            self.per_enricher_cost.get(enricher_id, 0.0) + cost_usd
        )
        self.run_total_cost += cost_usd

    def per_enricher_cap_exceeded(self, enricher_id: str, manifest: EnricherManifest) -> bool:
        """True when the enricher's recorded cost exceeds its per-enricher cap.

        ``manifest.max_cost_usd_per_run is None`` → unbounded, never fires.
        """
        cap = manifest.max_cost_usd_per_run
        if cap is None:
            return False
        return self.per_enricher_cost.get(enricher_id, 0.0) >= cap

    def run_wide_cap_exceeded(self, max_total: float | None) -> bool:
        """True when the run-wide accumulated cost exceeds the run cap.

        ``max_total is None`` → unbounded run-wide cap (default).
        """
        if max_total is None:
            return False
        return self.run_total_cost >= max_total


# ---------------------------------------------------------------------------
# Convenience: map status to a final EnricherResult.status
# ---------------------------------------------------------------------------


def status_for_exception(exc: BaseException) -> str:
    """Map a (final) exception to the appropriate ``EnricherResult.status``.

    Used by the executor when a retry loop exhausts and only the
    terminal exception remains.
    """
    if isinstance(exc, RunTimeoutError):
        return STATUS_TIMEOUT
    if isinstance(exc, EnvelopeShapeError):
        return STATUS_FAILED
    if isinstance(exc, (BadInputError, DependencyAccessError, ScorerTimeoutError)):
        return STATUS_FAILED
    return STATUS_FAILED


__all__ = [
    "BadInputError",
    "CircuitStatus",
    "CostCapState",
    "DEFAULT_POLICIES",
    "DependencyAccessError",
    "EnricherCircuitState",
    "HeartbeatWatchdog",
    "ModelLoadError",
    "RetryClass",
    "RunTimeoutError",
    "ScorerTimeoutError",
    "STATUS_QUARANTINED",
    "TierPolicy",
    "classify_failure",
    "compute_backoff",
    "policy_for",
    "status_for_exception",
]
