"""Unit tests for ``enrichment.run_summary`` — ``enrichments/run_summary.json``."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.enrichment.metrics import EnrichmentMetrics, new_metrics_for
from podcast_scraper.enrichment.protocol import EnricherResult, STATUS_FAILED, STATUS_OK
from podcast_scraper.enrichment.run_summary import (
    build_run_summary,
    read_run_summary,
    RUN_SUMMARY_SCHEMA_VERSION,
    write_run_summary,
)


def _populated_metrics(
    *,
    enricher_id: str,
    tier: str = "deterministic",
    duration_ms: int = 100,
    records: int = 3,
    cost_usd: float = 0.0,
) -> EnrichmentMetrics:
    m = new_metrics_for(
        enricher_id=enricher_id,
        enricher_version="1.0.0",
        scope="episode",
        tier=tier,
    )
    m.record_result(
        EnricherResult(
            status=STATUS_OK, data={"x": 1}, duration_ms=duration_ms, records_written=records
        ),
        started_at="2026-06-26T00:00:00Z",
        finished_at="2026-06-26T00:00:01Z",
    )
    m.record_cost(cost_usd=cost_usd)
    return m


# ---------------------------------------------------------------------------
# build_run_summary
# ---------------------------------------------------------------------------


def test_build_run_summary_carries_schema_version_and_envelope() -> None:
    summary = build_run_summary(
        run_id="job-1",
        parent_run_id="pipeline-1",
        profile="cloud_thin",
        started_at="2026-06-26T00:00:00Z",
        finished_at="2026-06-26T00:00:42Z",
        duration_ms=42000,
        status="ok",
        per_enricher={"topic_cooccurrence": _populated_metrics(enricher_id="topic_cooccurrence")},
    )
    assert summary["schema_version"] == RUN_SUMMARY_SCHEMA_VERSION
    assert summary["run_id"] == "job-1"
    assert summary["parent_run_id"] == "pipeline-1"
    assert summary["profile"] == "cloud_thin"
    assert summary["duration_ms"] == 42000
    assert summary["status"] == "ok"


def test_build_run_summary_includes_per_enricher_counters() -> None:
    m = _populated_metrics(enricher_id="topic_cooccurrence", records=5)
    summary = build_run_summary(
        run_id="job-1",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=100,
        status="ok",
        per_enricher={"topic_cooccurrence": m},
    )
    per_enr = summary["per_enricher"]["topic_cooccurrence"]
    assert per_enr["runs_total"] == 1
    assert per_enr["runs_ok"] == 1
    assert per_enr["records_written"] == 5
    assert per_enr["status"] == STATUS_OK


def test_build_run_summary_includes_token_and_cost_for_smart_tiers() -> None:
    m = _populated_metrics(enricher_id="topic_consensus", tier="ml", cost_usd=0.10)
    m.record_tokens(tokens_in=100, tokens_out=20)
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=1,
        status="ok",
        per_enricher={"topic_consensus": m},
    )
    per_enr = summary["per_enricher"]["topic_consensus"]
    assert per_enr["tokens_in"] == 100
    assert per_enr["tokens_out"] == 20
    assert per_enr["cost_usd"] == 0.10


def test_build_run_summary_carries_error_samples() -> None:
    m = new_metrics_for(
        enricher_id="x", enricher_version="1.0.0", scope="episode", tier="deterministic"
    )
    m.record_result(
        EnricherResult(status=STATUS_FAILED, error="boom", error_class="RuntimeError"),
        started_at="t0",
        finished_at="t1",
    )
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=1,
        status="failed",
        per_enricher={"x": m},
    )
    samples = summary["per_enricher"]["x"]["error_samples"]
    assert len(samples) == 1
    assert samples[0]["error_class"] == "RuntimeError"


def test_build_run_summary_includes_multiple_enrichers() -> None:
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=1,
        status="ok",
        per_enricher={
            "topic_cooccurrence": _populated_metrics(enricher_id="topic_cooccurrence"),
            "temporal_velocity": _populated_metrics(enricher_id="temporal_velocity"),
        },
    )
    assert set(summary["per_enricher"].keys()) == {"topic_cooccurrence", "temporal_velocity"}


# ---------------------------------------------------------------------------
# write + read round-trip
# ---------------------------------------------------------------------------


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    summary = build_run_summary(
        run_id="job-1",
        parent_run_id=None,
        profile="airgapped",
        started_at="t0",
        finished_at="t1",
        duration_ms=500,
        status="ok",
        per_enricher={"x": _populated_metrics(enricher_id="x")},
    )
    write_run_summary(tmp_path, summary)

    payload = read_run_summary(tmp_path)
    assert payload is not None
    assert payload["run_id"] == "job-1"
    assert payload["profile"] == "airgapped"
    assert "x" in payload["per_enricher"]


def test_read_run_summary_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_run_summary(tmp_path) is None


def test_read_run_summary_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    (tmp_path / "enrichments").mkdir()
    (tmp_path / "enrichments" / "run_summary.json").write_text("not valid json", encoding="utf-8")
    assert read_run_summary(tmp_path) is None


def test_write_run_summary_creates_enrichments_dir(tmp_path: Path) -> None:
    """Per chunk-1 lock audit §B6 spirit: enrichments/ created on first write."""
    assert not (tmp_path / "enrichments").exists()
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=0,
        status="ok",
        per_enricher={},
    )
    write_run_summary(tmp_path, summary)
    assert (tmp_path / "enrichments" / "run_summary.json").is_file()


def test_write_run_summary_atomic_no_leftover_tmp(tmp_path: Path) -> None:
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=0,
        status="ok",
        per_enricher={},
    )
    write_run_summary(tmp_path, summary)
    tmps = list((tmp_path / "enrichments").glob("*.tmp"))
    assert tmps == []


def test_write_run_summary_serializes_clean_json(tmp_path: Path) -> None:
    summary = build_run_summary(
        run_id="r",
        parent_run_id=None,
        profile=None,
        started_at="t0",
        finished_at="t1",
        duration_ms=0,
        status="ok",
        per_enricher={},
    )
    write_run_summary(tmp_path, summary)
    raw = (tmp_path / "enrichments" / "run_summary.json").read_text(encoding="utf-8")
    json.loads(raw)  # must parse
