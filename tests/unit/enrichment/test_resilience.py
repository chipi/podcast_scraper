"""Unit tests for ``enrichment.resilience``."""

from __future__ import annotations

import random

import pytest

from podcast_scraper.enrichment.envelope import EnvelopeShapeError
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherScope,
    EnricherTier,
    STATUS_FAILED,
    STATUS_TIMEOUT,
)
from podcast_scraper.enrichment.resilience import (
    BadInputError,
    CircuitStatus,
    classify_failure,
    compute_backoff,
    CostCapState,
    DEFAULT_POLICIES,
    DependencyAccessError,
    EnricherCircuitState,
    HeartbeatWatchdog,
    ModelLoadError,
    policy_for,
    RetryClass,
    RunTimeoutError,
    ScorerTimeoutError,
    status_for_exception,
    TierPolicy,
)

# ---------------------------------------------------------------------------
# Per-tier defaults (the policy matrix from the plan)
# ---------------------------------------------------------------------------


def test_default_policies_cover_all_four_tiers() -> None:
    assert set(DEFAULT_POLICIES.keys()) == set(EnricherTier)


def test_deterministic_tier_has_no_retries_no_circuit() -> None:
    p = policy_for(EnricherTier.DETERMINISTIC)
    assert p.max_retries == 0
    assert p.circuit_threshold is None
    assert p.auto_disable_threshold == 5


def test_embedding_tier_policy_matches_plan() -> None:
    p = policy_for(EnricherTier.EMBEDDING)
    assert p.max_retries == 3
    assert p.initial_backoff_s == pytest.approx(1.0)
    assert p.max_backoff_s == pytest.approx(30.0)
    assert p.circuit_threshold == 5
    assert p.auto_disable_threshold == 3
    assert p.concurrency == 2


def test_ml_tier_policy_matches_plan() -> None:
    p = policy_for(EnricherTier.ML)
    assert p.max_retries == 2
    assert p.initial_backoff_s == pytest.approx(5.0)
    assert p.max_backoff_s == pytest.approx(60.0)
    assert p.circuit_threshold == 3
    assert p.auto_disable_threshold == 2
    assert p.concurrency == 1


def test_llm_tier_policy_matches_plan() -> None:
    p = policy_for(EnricherTier.LLM)
    assert p.max_retries == 5
    assert p.max_backoff_s == pytest.approx(120.0)
    assert p.circuit_threshold == 3
    assert p.auto_disable_threshold == 2


# ---------------------------------------------------------------------------
# Failure taxonomy + classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        EnvelopeShapeError("bad shape"),
        BadInputError("missing bridge"),
        RunTimeoutError("hard timeout"),
    ],
)
def test_classify_failure_non_retryable(exc: BaseException) -> None:
    assert classify_failure(exc) == RetryClass.NON_RETRYABLE


@pytest.mark.parametrize(
    "exc",
    [
        DependencyAccessError("lance lock"),
        ScorerTimeoutError("per-pair timeout"),
        TimeoutError("asyncio cancelled"),
    ],
)
def test_classify_failure_retryable(exc: BaseException) -> None:
    assert classify_failure(exc) == RetryClass.RETRYABLE


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("OOM during inference"),
        ModelLoadError("corrupt cache"),
    ],
)
def test_classify_failure_retryable_once(exc: BaseException) -> None:
    assert classify_failure(exc) == RetryClass.RETRYABLE_ONCE


def test_classify_failure_unknown_exception_is_non_retryable() -> None:
    """Safety-net default: unknown exceptions are non-retryable.

    A bare ``RuntimeError`` / custom ``Exception`` from an enricher
    body is a code bug, not a transient backend issue — retrying
    won't fix it. The enrichment failure taxonomy is stricter than
    ``utils/retryable_errors.is_retryable_error`` (which defaults
    unknown to retryable for the LLM provider call path).
    """

    class WeirdException(Exception):
        pass

    assert classify_failure(WeirdException("?")) == RetryClass.NON_RETRYABLE
    assert classify_failure(RuntimeError("uncaught")) == RetryClass.NON_RETRYABLE


def test_classify_failure_http_shaped_exception_delegates_to_retryable_errors() -> None:
    """HTTP-shaped exceptions (with .response.status_code) defer to the shared helper.

    Cloud-LLM enrichers get the same retry/non-retry classification
    the providers already use — a 429 / 503 is retryable; a 401 / 404
    is not. The helper is consulted only when the exception is
    structurally HTTP-shaped; bare exceptions fall through.
    """

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class _HttpError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"HTTP {status_code}")
            self.response = _Response(status_code)

    # 429 (rate limit) → retryable.
    assert classify_failure(_HttpError(429)) == RetryClass.RETRYABLE
    # 503 (server overload) → retryable.
    assert classify_failure(_HttpError(503)) == RetryClass.RETRYABLE
    # 404 (not found) → non-retryable.
    assert classify_failure(_HttpError(404)) == RetryClass.NON_RETRYABLE


# ---------------------------------------------------------------------------
# Backoff computation
# ---------------------------------------------------------------------------


def test_compute_backoff_zero_for_deterministic() -> None:
    """Deterministic tier disables retries; backoff is always 0."""
    p = policy_for(EnricherTier.DETERMINISTIC)
    assert compute_backoff(1, p) == 0.0
    assert compute_backoff(5, p) == 0.0


def test_compute_backoff_grows_exponentially() -> None:
    """attempt 1 → ~initial; attempt 2 → ~initial*factor (with jitter ≤ 10%)."""
    p = policy_for(EnricherTier.EMBEDDING)  # 1s initial, 2x factor
    random.seed(42)
    b1 = compute_backoff(1, p)
    b2 = compute_backoff(2, p)
    b3 = compute_backoff(3, p)
    # Floor: initial * factor**(attempt-1); ceiling adds ≤ 10% jitter.
    assert 1.0 <= b1 <= 1.1
    assert 2.0 <= b2 <= 2.2
    assert 4.0 <= b3 <= 4.4


def test_compute_backoff_caps_at_max() -> None:
    """High attempt counts saturate at policy.max_backoff_s (+ jitter)."""
    p = policy_for(EnricherTier.LLM)  # initial 2s, factor 2x, cap 120s
    random.seed(0)
    b10 = compute_backoff(10, p)
    # base = 2 * 2**9 = 1024 → capped at 120; jitter is 10% of cap.
    assert 120.0 <= b10 <= 132.0


# ---------------------------------------------------------------------------
# EnricherCircuitState
# ---------------------------------------------------------------------------


def test_circuit_state_starts_closed() -> None:
    cs = EnricherCircuitState()
    assert cs.status is CircuitStatus.CLOSED
    assert cs.consecutive_failures == 0
    assert not cs.is_open


def test_circuit_state_records_failure_and_trips_after_threshold() -> None:
    p = policy_for(EnricherTier.ML)  # circuit_threshold=3
    cs = EnricherCircuitState()
    cs.record_failure(p)
    assert cs.consecutive_failures == 1
    assert cs.status is CircuitStatus.CLOSED
    cs.record_failure(p)
    cs.record_failure(p)
    assert cs.consecutive_failures == 3
    assert cs.is_open
    assert cs.opened_at is not None


def test_circuit_state_success_resets_consecutive_failures() -> None:
    p = policy_for(EnricherTier.ML)
    cs = EnricherCircuitState()
    cs.record_failure(p)
    cs.record_failure(p)
    cs.record_success()
    assert cs.consecutive_failures == 0
    assert cs.status is CircuitStatus.CLOSED
    assert cs.opened_at is None


def test_circuit_state_deterministic_tier_never_opens() -> None:
    """Deterministic tier has circuit_threshold=None; no number of failures opens."""
    p = policy_for(EnricherTier.DETERMINISTIC)
    cs = EnricherCircuitState()
    for _ in range(100):
        cs.record_failure(p)
    assert cs.status is CircuitStatus.CLOSED
    assert not cs.is_open


# ---------------------------------------------------------------------------
# HeartbeatWatchdog
# ---------------------------------------------------------------------------


def test_heartbeat_watchdog_starts_fresh() -> None:
    w = HeartbeatWatchdog("topic_similarity", expected_interval_s=5.0)
    assert not w.is_stalled(now=w.last_heartbeat_at + 1.0)


def test_heartbeat_watchdog_detects_stall_after_factor_x_interval() -> None:
    w = HeartbeatWatchdog("topic_similarity", expected_interval_s=2.0)
    base = w.last_heartbeat_at
    # 1.5x expected — not yet stalled (default factor=2).
    assert not w.is_stalled(now=base + 3.0)
    # 3x expected — definitely stalled.
    assert w.is_stalled(now=base + 7.0)


def test_heartbeat_watchdog_record_heartbeat_resets_clock() -> None:
    w = HeartbeatWatchdog("x", expected_interval_s=1.0)
    base = w.last_heartbeat_at
    assert w.is_stalled(now=base + 5.0)
    w.record_heartbeat(now=base + 4.5)
    assert not w.is_stalled(now=base + 5.0)


def test_heartbeat_watchdog_factor_param_tightens_detection() -> None:
    w = HeartbeatWatchdog("x", expected_interval_s=1.0)
    base = w.last_heartbeat_at
    # 1.5x expected; not stalled at factor=2, stalled at factor=1.
    assert not w.is_stalled(now=base + 1.5)
    assert w.is_stalled(now=base + 1.5, factor=1.0)


# ---------------------------------------------------------------------------
# CostCapState
# ---------------------------------------------------------------------------


def _manifest_with_cap(cap: float | None) -> EnricherManifest:
    return EnricherManifest(
        id="topic_consensus",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="nli.json",
        description="x",
        max_cost_usd_per_run=cap,
    )


def test_cost_cap_state_starts_zero() -> None:
    s = CostCapState()
    assert s.run_total_cost == 0.0
    assert s.per_enricher_cost == {}


def test_cost_cap_state_accumulates_per_enricher_and_run_wide() -> None:
    s = CostCapState()
    s.record_cost("topic_consensus", 0.10)
    s.record_cost("topic_consensus", 0.05)
    s.record_cost("query_synthesis", 0.20)
    assert s.per_enricher_cost == {
        "topic_consensus": pytest.approx(0.15),
        "query_synthesis": pytest.approx(0.20),
    }
    assert s.run_total_cost == pytest.approx(0.35)


def test_cost_cap_state_ignores_zero_and_negative_costs() -> None:
    s = CostCapState()
    s.record_cost("x", 0.0)
    s.record_cost("x", -0.1)
    assert s.per_enricher_cost == {}
    assert s.run_total_cost == 0.0


def test_per_enricher_cap_unbounded_when_manifest_field_is_none() -> None:
    s = CostCapState()
    s.record_cost("topic_consensus", 1000.0)
    assert not s.per_enricher_cap_exceeded("topic_consensus", _manifest_with_cap(None))


def test_per_enricher_cap_fires_at_threshold() -> None:
    s = CostCapState()
    m = _manifest_with_cap(0.50)
    s.record_cost("topic_consensus", 0.40)
    assert not s.per_enricher_cap_exceeded("topic_consensus", m)
    s.record_cost("topic_consensus", 0.15)
    assert s.per_enricher_cap_exceeded("topic_consensus", m)


def test_run_wide_cap_unbounded_when_arg_is_none() -> None:
    s = CostCapState()
    s.record_cost("x", 1000.0)
    assert not s.run_wide_cap_exceeded(None)


def test_run_wide_cap_fires_at_threshold() -> None:
    s = CostCapState()
    s.record_cost("a", 0.30)
    s.record_cost("b", 0.40)
    assert not s.run_wide_cap_exceeded(1.00)
    s.record_cost("c", 0.40)
    assert s.run_wide_cap_exceeded(1.00)


# ---------------------------------------------------------------------------
# status_for_exception
# ---------------------------------------------------------------------------


def test_status_for_exception_run_timeout_maps_to_timeout() -> None:
    assert status_for_exception(RunTimeoutError("boom")) == STATUS_TIMEOUT


@pytest.mark.parametrize(
    "exc",
    [
        EnvelopeShapeError("bad"),
        BadInputError("missing"),
        DependencyAccessError("lock"),
        ScorerTimeoutError("slow"),
        RuntimeError("?"),
    ],
)
def test_status_for_exception_others_map_to_failed(exc: BaseException) -> None:
    assert status_for_exception(exc) == STATUS_FAILED


# ---------------------------------------------------------------------------
# Cross-policy invariants
# ---------------------------------------------------------------------------


def test_policy_for_returns_frozen_tier_policy() -> None:
    p = policy_for(EnricherTier.LLM)
    assert isinstance(p, TierPolicy)
    with pytest.raises(Exception):
        p.max_retries = 99  # type: ignore[misc]


def test_auto_disable_thresholds_match_plan_table() -> None:
    """Codify the auto-disable thresholds the plan body table promises."""
    assert policy_for(EnricherTier.DETERMINISTIC).auto_disable_threshold == 5
    assert policy_for(EnricherTier.EMBEDDING).auto_disable_threshold == 3
    assert policy_for(EnricherTier.ML).auto_disable_threshold == 2
    assert policy_for(EnricherTier.LLM).auto_disable_threshold == 2
