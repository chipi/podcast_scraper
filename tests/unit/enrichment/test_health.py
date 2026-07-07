"""Unit tests for ``enrichment.health``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.health import (
    EnricherHealth,
    HEALTH_SCHEMA_VERSION,
    HealthRegistry,
)
from podcast_scraper.enrichment.protocol import EnricherTier
from podcast_scraper.enrichment.resilience import policy_for

# ---------------------------------------------------------------------------
# EnricherHealth defaults
# ---------------------------------------------------------------------------


def test_enricher_health_defaults_active() -> None:
    h = EnricherHealth()
    assert h.consecutive_failures == 0
    assert h.auto_disabled is False
    assert h.is_active()


def test_enricher_health_auto_disabled_is_inactive() -> None:
    h = EnricherHealth(auto_disabled=True)
    assert not h.is_active()


def test_enricher_health_cooldown_blocks_active_until_elapsed() -> None:
    # Cooldown in the past → active.
    h = EnricherHealth(cooldown_until="1970-01-01T00:00:00Z")
    assert h.is_active()
    # Cooldown in the far future → inactive.
    h2 = EnricherHealth(cooldown_until="2999-12-31T23:59:59Z")
    assert not h2.is_active()


# ---------------------------------------------------------------------------
# HealthRegistry: load / save round-trip
# ---------------------------------------------------------------------------


def test_health_registry_load_no_file_empty_state(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    reg.load()
    assert reg.all() == {}


def test_health_registry_save_creates_viewer_dir_and_file(tmp_path: Path) -> None:
    """Per chunk-1 lock audit §B6: ``.viewer/`` created on first write."""
    assert not (tmp_path / ".viewer").exists()
    reg = HealthRegistry(tmp_path)
    reg.update_after_run(
        "topic_similarity",
        run_id="job-1",
        status="ok",
        policy=policy_for(EnricherTier.EMBEDDING),
    )
    reg.save()
    assert (tmp_path / ".viewer" / "enrichment_health.json").is_file()


def test_health_registry_save_round_trip(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.EMBEDDING)
    reg.update_after_run("topic_similarity", run_id="job-1", status="ok", policy=p)
    reg.update_after_run("topic_consensus", run_id="job-1", status="failed", policy=p)
    reg.save()

    # Fresh registry → same state after load.
    reg2 = HealthRegistry(tmp_path)
    reg2.load()
    snap = reg2.all()
    assert set(snap.keys()) == {"topic_similarity", "topic_consensus"}
    assert snap["topic_similarity"].last_status == "ok"
    assert snap["topic_consensus"].consecutive_failures == 1


def test_health_registry_save_payload_carries_schema_version(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    reg.update_after_run(
        "x", run_id="r", status="ok", policy=policy_for(EnricherTier.DETERMINISTIC)
    )
    reg.save()
    payload = json.loads(reg.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == HEALTH_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# HealthRegistry: corrupt-file tolerance
# ---------------------------------------------------------------------------


def test_health_registry_load_corrupt_json_treats_as_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Write deliberately malformed JSON to the expected location.
    (tmp_path / ".viewer").mkdir()
    (tmp_path / ".viewer" / "enrichment_health.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    reg = HealthRegistry(tmp_path)
    with caplog.at_level("WARNING", logger="podcast_scraper.enrichment.health"):
        reg.load()
    assert reg.all() == {}
    assert any("could not be parsed" in r.message for r in caplog.records)


def test_health_registry_load_schema_mismatch_is_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".viewer").mkdir()
    (tmp_path / ".viewer" / "enrichment_health.json").write_text(
        json.dumps({"schema_version": "9999", "enrichers": {"x": {}}}),
        encoding="utf-8",
    )
    reg = HealthRegistry(tmp_path)
    with caplog.at_level("WARNING", logger="podcast_scraper.enrichment.health"):
        reg.load()
    assert reg.all() == {}


def test_health_registry_load_tolerates_extra_keys(tmp_path: Path) -> None:
    (tmp_path / ".viewer").mkdir()
    (tmp_path / ".viewer" / "enrichment_health.json").write_text(
        json.dumps(
            {
                "schema_version": HEALTH_SCHEMA_VERSION,
                "enrichers": {
                    "x": {
                        "consecutive_failures": 2,
                        "last_status": "failed",
                        "future_field_we_dont_know_about": "shrug",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    reg = HealthRegistry(tmp_path)
    reg.load()
    h = reg.get("x")
    assert h.consecutive_failures == 2
    assert h.last_status == "failed"


# ---------------------------------------------------------------------------
# HealthRegistry: update_after_run state machine
# ---------------------------------------------------------------------------


def test_update_after_run_ok_resets_counter(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.EMBEDDING)
    reg.update_after_run("x", run_id="r1", status="failed", policy=p)
    reg.update_after_run("x", run_id="r2", status="ok", policy=p)
    h = reg.get("x")
    assert h.consecutive_failures == 0
    assert h.auto_disabled is False


def test_update_after_run_failed_increments_counter(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.EMBEDDING)
    reg.update_after_run("x", run_id="r1", status="failed", policy=p)
    reg.update_after_run("x", run_id="r2", status="failed", policy=p)
    assert reg.get("x").consecutive_failures == 2


def test_update_after_run_triggers_auto_disable_at_threshold(tmp_path: Path) -> None:
    """ML tier auto-disable threshold is 2 runs."""
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    reg.update_after_run("nli", run_id="r1", status="failed", policy=p)
    assert not reg.get("nli").auto_disabled
    reg.update_after_run("nli", run_id="r2", status="failed", policy=p)
    h = reg.get("nli")
    assert h.auto_disabled is True
    assert h.auto_disabled_reason is not None
    assert "consecutive failed" in h.auto_disabled_reason


def test_update_after_run_timeout_counts_as_failure(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    reg.update_after_run("nli", run_id="r1", status="failed", policy=p)
    reg.update_after_run("nli", run_id="r2", status="timeout", policy=p)
    assert reg.get("nli").auto_disabled


def test_update_after_run_quarantined_counts_as_failure(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.EMBEDDING)
    for n in range(p.auto_disable_threshold):
        reg.update_after_run(
            "topic_similarity",
            run_id=f"r{n}",
            status="quarantined",
            policy=p,
        )
    assert reg.get("topic_similarity").auto_disabled


def test_update_after_run_cancelled_does_not_count_as_failure(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    for _ in range(10):
        reg.update_after_run("x", run_id="r", status="cancelled", policy=p)
    assert reg.get("x").consecutive_failures == 0
    assert reg.get("x").auto_disabled is False


def test_update_after_run_skipped_does_not_count_as_failure(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    reg.update_after_run("x", run_id="r1", status="skipped", policy=p)
    assert reg.get("x").consecutive_failures == 0


def test_update_after_run_records_run_id_and_timestamp(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    reg.update_after_run(
        "x", run_id="job-42", status="ok", policy=policy_for(EnricherTier.DETERMINISTIC)
    )
    h = reg.get("x")
    assert h.last_run_id == "job-42"
    assert h.last_run_at is not None
    assert h.last_run_at.endswith("Z")
    assert h.last_status == "ok"


# ---------------------------------------------------------------------------
# HealthRegistry: re_enable
# ---------------------------------------------------------------------------


def test_re_enable_clears_auto_disable_and_resets_counter(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    reg.update_after_run("x", run_id="r1", status="failed", policy=p)
    reg.update_after_run("x", run_id="r2", status="failed", policy=p)
    assert reg.get("x").auto_disabled

    reg.re_enable("x", reason="operator confirmed transient HF outage")
    h = reg.get("x")
    assert h.auto_disabled is False
    assert h.consecutive_failures == 0
    assert h.auto_disabled_reason is not None
    assert "re_enabled" in h.auto_disabled_reason


def test_re_enable_clears_cooldown_when_requested(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    h = reg.get("x")
    h.cooldown_until = "2999-01-01T00:00:00Z"
    h.circuit_state = "open"
    h.circuit_opened_at = "2026-06-01T00:00:00Z"
    reg.re_enable("x", reason="manual reset")
    h2 = reg.get("x")
    assert h2.cooldown_until is None
    assert h2.circuit_state == "closed"
    assert h2.circuit_opened_at is None


def test_re_enable_preserves_cooldown_when_clear_disabled(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    h = reg.get("x")
    h.cooldown_until = "2999-01-01T00:00:00Z"
    reg.re_enable("x", reason="r", clear_cooldown=False)
    assert reg.get("x").cooldown_until == "2999-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# HealthRegistry: is_active
# ---------------------------------------------------------------------------


def test_is_active_true_by_default(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    assert reg.is_active("any-enricher")


def test_is_active_false_after_auto_disable(tmp_path: Path) -> None:
    reg = HealthRegistry(tmp_path)
    p = policy_for(EnricherTier.ML)
    reg.update_after_run("x", run_id="r1", status="failed", policy=p)
    reg.update_after_run("x", run_id="r2", status="failed", policy=p)
    assert not reg.is_active("x")
