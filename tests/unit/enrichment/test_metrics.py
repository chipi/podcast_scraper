"""Unit tests for ``enrichment.metrics`` — ``EnrichmentMetrics``."""

from __future__ import annotations

import json

import pytest

from podcast_scraper.enrichment.metrics import (
    EnrichmentMetrics,
    ERROR_SAMPLES_CAP,
    new_metrics_for,
)
from podcast_scraper.enrichment.protocol import (
    EnricherResult,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_QUARANTINED,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
)


def _ok(records: int = 1, duration_ms: int = 10) -> EnricherResult:
    return EnricherResult(
        status=STATUS_OK,
        data={"x": 1},
        duration_ms=duration_ms,
        records_written=records,
    )


def _failed(retry_count: int = 0) -> EnricherResult:
    return EnricherResult(
        status=STATUS_FAILED,
        error="boom",
        error_class="RuntimeError",
        retry_count=retry_count,
    )


def _make_metrics() -> EnrichmentMetrics:
    return new_metrics_for(
        enricher_id="topic_cooccurrence",
        enricher_version="1.0.0",
        scope="episode",
        tier="deterministic",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_new_metrics_for_starts_zero() -> None:
    m = _make_metrics()
    assert m.runs_total == 0
    assert m.runs_ok == 0
    assert m.duration_seconds == 0.0
    assert m.error_samples == []
    assert m.circuit_transitions == {}
    assert m.scorer_failures_total == {}


# ---------------------------------------------------------------------------
# record_result counter updates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "counter_attr"),
    [
        (STATUS_OK, "runs_ok"),
        (STATUS_FAILED, "runs_failed"),
        (STATUS_TIMEOUT, "runs_timeout"),
        (STATUS_QUARANTINED, "runs_quarantined"),
        (STATUS_CANCELLED, "runs_cancelled"),
        (STATUS_SKIPPED, "runs_skipped"),
    ],
)
def test_record_result_increments_per_status_counter(status: str, counter_attr: str) -> None:
    m = _make_metrics()
    if status == STATUS_OK:
        result = EnricherResult(status=status, data={})
    else:
        result = EnricherResult(status=status, error="boom")
    m.record_result(result, started_at="t0", finished_at="t1")
    assert getattr(m, counter_attr) == 1
    assert m.runs_total == 1


def test_record_result_accumulates_records_written_only_on_ok() -> None:
    m = _make_metrics()
    m.record_result(_ok(records=3), started_at="t0", finished_at="t1")
    m.record_result(_failed(), started_at="t2", finished_at="t3")
    m.record_result(_ok(records=2), started_at="t4", finished_at="t5")
    assert m.output_records_total == 5


def test_record_result_accumulates_duration_seconds() -> None:
    m = _make_metrics()
    m.record_result(_ok(duration_ms=100), started_at="t0", finished_at="t1")
    m.record_result(_ok(duration_ms=250), started_at="t2", finished_at="t3")
    assert m.duration_seconds == pytest.approx(0.350)


def test_record_result_explicit_duration_s_overrides_result_duration_ms() -> None:
    """Executor wall-clock measurement is authoritative when provided."""
    m = _make_metrics()
    m.record_result(_ok(duration_ms=100), started_at="t0", finished_at="t1", duration_s=2.0)
    assert m.duration_seconds == pytest.approx(2.0)


def test_record_result_tracks_retry_count() -> None:
    m = _make_metrics()
    m.record_result(_failed(retry_count=2), started_at="t0", finished_at="t1")
    m.record_result(_failed(retry_count=3), started_at="t2", finished_at="t3")
    assert m.retries_total == 5


def test_record_result_stamps_last_run_status_and_timestamps() -> None:
    m = _make_metrics()
    m.record_result(_ok(), started_at="t0", finished_at="t1")
    assert m.last_run_status == STATUS_OK
    assert m.last_run_started_at == "t0"
    assert m.last_run_finished_at == "t1"


# ---------------------------------------------------------------------------
# error_samples cap (per chunk-1 lock audit §I2)
# ---------------------------------------------------------------------------


def test_record_result_pushes_error_samples_on_failed_statuses() -> None:
    m = _make_metrics()
    m.record_result(_failed(), started_at="t0", finished_at="t1")
    assert len(m.error_samples) == 1
    sample = m.error_samples[0]
    assert sample["status"] == STATUS_FAILED
    assert sample["error"] == "boom"
    assert sample["error_class"] == "RuntimeError"


def test_record_result_does_not_push_error_samples_on_ok_or_skipped() -> None:
    m = _make_metrics()
    m.record_result(_ok(), started_at="t0", finished_at="t1")
    m.record_result(
        EnricherResult(status=STATUS_SKIPPED, error="auto-disabled"),
        started_at="t2",
        finished_at="t3",
    )
    m.record_result(
        EnricherResult(status=STATUS_CANCELLED, error="cancel-requested"),
        started_at="t4",
        finished_at="t5",
    )
    assert m.error_samples == []


def test_error_samples_capped_at_constant() -> None:
    m = _make_metrics()
    for i in range(ERROR_SAMPLES_CAP * 3):
        m.record_result(_failed(), started_at=f"t{i}", finished_at=f"t{i}+1")
    assert len(m.error_samples) == ERROR_SAMPLES_CAP


def test_error_samples_keep_newest_drop_oldest() -> None:
    m = _make_metrics()
    for i in range(ERROR_SAMPLES_CAP + 3):
        result = EnricherResult(status=STATUS_FAILED, error=f"boom-{i}", error_class="RuntimeError")
        m.record_result(result, started_at=f"t{i}", finished_at=f"t{i}+1")
    errors = [s["error"] for s in m.error_samples]
    # We dropped the first 3; kept the last ERROR_SAMPLES_CAP.
    assert errors == [f"boom-{i}" for i in range(3, ERROR_SAMPLES_CAP + 3)]


def test_post_init_truncates_loaded_oversized_error_samples() -> None:
    """When a loaded record has too many samples (rare), trim on construction."""
    m = EnrichmentMetrics(
        enricher_id="x",
        enricher_version="1",
        scope="episode",
        tier="deterministic",
        error_samples=[{"i": n} for n in range(20)],
    )
    assert len(m.error_samples) == ERROR_SAMPLES_CAP
    # Newest preserved.
    assert m.error_samples[-1] == {"i": 19}


# ---------------------------------------------------------------------------
# scorer + tokens + cost recording
# ---------------------------------------------------------------------------


def test_record_scorer_call_increments_and_breakdown_on_failure() -> None:
    m = _make_metrics()
    m.record_scorer_call()
    m.record_scorer_call(error_class="DependencyAccessError")
    m.record_scorer_call(error_class="DependencyAccessError")
    m.record_scorer_call(error_class="ScorerTimeoutError")
    assert m.scorer_calls_total == 4
    assert m.scorer_failures_total == {
        "DependencyAccessError": 2,
        "ScorerTimeoutError": 1,
    }


def test_record_tokens_accumulates() -> None:
    m = _make_metrics()
    m.record_tokens(tokens_in=100, tokens_out=20)
    m.record_tokens(tokens_in=50)
    m.record_tokens(tokens_out=15)
    assert m.tokens_in == 150
    assert m.tokens_out == 35


def test_record_tokens_ignores_zero() -> None:
    m = _make_metrics()
    m.record_tokens(tokens_in=0, tokens_out=0)
    assert m.tokens_in == 0
    assert m.tokens_out == 0


def test_record_cost_accumulates() -> None:
    m = _make_metrics()
    m.record_cost(cost_usd=0.10)
    m.record_cost(cost_usd=0.05)
    assert m.cost_usd == pytest.approx(0.15)


def test_record_cost_ignores_zero_and_negative() -> None:
    m = _make_metrics()
    m.record_cost(cost_usd=0.0)
    m.record_cost(cost_usd=-0.5)
    assert m.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Circuit transitions + model stamping
# ---------------------------------------------------------------------------


def test_record_circuit_transition_increments_by_pair() -> None:
    m = _make_metrics()
    m.record_circuit_transition(from_state="closed", to_state="open")
    m.record_circuit_transition(from_state="open", to_state="half_open")
    m.record_circuit_transition(from_state="closed", to_state="open")
    assert m.circuit_transitions == {
        "closed->open": 2,
        "open->half_open": 1,
    }


def test_set_model_stamps_id_and_version() -> None:
    m = _make_metrics()
    m.set_model(model_id="nli-deberta-v3-small", model_version="0.1.0")
    assert m.model_id == "nli-deberta-v3-small"
    assert m.model_version == "0.1.0"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_round_trips_through_json() -> None:
    m = _make_metrics()
    m.record_result(_ok(records=3), started_at="t0", finished_at="t1")
    m.record_tokens(tokens_in=100, tokens_out=20)
    m.record_cost(cost_usd=0.01)
    d = m.to_dict()
    encoded = json.dumps(d)
    decoded = json.loads(encoded)
    assert decoded["enricher_id"] == "topic_cooccurrence"
    assert decoded["runs_ok"] == 1
    assert decoded["tokens_in"] == 100
    assert decoded["cost_usd"] == pytest.approx(0.01)
