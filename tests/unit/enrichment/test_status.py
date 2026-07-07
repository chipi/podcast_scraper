"""Unit tests for ``enrichment.status`` — live-status writer."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.enrichment.status import (
    build_current_enricher_block,
    read_status,
    STATUS_SCHEMA_VERSION,
    write_idle,
    write_status,
)


def test_write_status_creates_viewer_dir(tmp_path: Path) -> None:
    """Per chunk-1 lock audit §B6: .viewer/ created on first write."""
    assert not (tmp_path / ".viewer").exists()
    write_status(
        tmp_path,
        run_id="job-1",
        started_at="2026-06-26T00:00:00Z",
        profile="cloud_thin",
        current_enricher=None,
        queue=[],
        completed=[],
    )
    assert (tmp_path / ".viewer" / "enrichment_status.json").is_file()


def test_write_status_serializes_full_envelope(tmp_path: Path) -> None:
    write_status(
        tmp_path,
        run_id="job-1",
        started_at="2026-06-26T00:00:00Z",
        profile="cloud_thin",
        current_enricher={
            "enricher_id": "topic_consensus",
            "scope": "corpus",
            "tier": "ml",
            "attempt": 1,
            "progress": {"items_done": 50, "items_total": 200, "eta_seconds": 142},
            "last_heartbeat_at": "2026-06-26T00:01:00Z",
        },
        queue=["topic_similarity"],
        completed=[
            {"enricher_id": "topic_cooccurrence_corpus", "status": "ok", "duration_ms": 412},
        ],
    )
    raw = (tmp_path / ".viewer" / "enrichment_status.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["schema_version"] == STATUS_SCHEMA_VERSION
    assert payload["run_id"] == "job-1"
    assert payload["profile"] == "cloud_thin"
    assert payload["current_enricher"]["enricher_id"] == "topic_consensus"
    assert payload["current_enricher"]["progress"]["items_done"] == 50
    assert payload["queue"] == ["topic_similarity"]
    assert payload["completed"][0]["duration_ms"] == 412


def test_write_idle_produces_empty_envelope(tmp_path: Path) -> None:
    write_idle(tmp_path)
    payload = read_status(tmp_path)
    assert payload is not None
    assert payload["run_id"] == ""
    assert payload["current_enricher"] is None
    assert payload["queue"] == []
    assert payload["completed"] == []


def test_read_status_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_status(tmp_path) is None


def test_read_status_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / ".viewer").mkdir()
    (tmp_path / ".viewer" / "enrichment_status.json").write_text("not valid json", encoding="utf-8")
    assert read_status(tmp_path) is None


def test_read_status_round_trips_with_write_status(tmp_path: Path) -> None:
    write_status(
        tmp_path,
        run_id="job-2",
        started_at="2026-06-26T00:00:00Z",
        profile=None,
        current_enricher=None,
        queue=["a", "b"],
        completed=[],
    )
    payload = read_status(tmp_path)
    assert payload is not None
    assert payload["run_id"] == "job-2"
    assert payload["profile"] is None
    assert payload["queue"] == ["a", "b"]


def test_build_current_enricher_block_shape() -> None:
    block = build_current_enricher_block(
        enricher_id="topic_consensus",
        scope="corpus",
        tier="ml",
        attempt=2,
        items_done=247,
        items_total=1000,
        eta_seconds=142,
        last_heartbeat_at="2026-06-26T15:03:14Z",
    )
    assert block["enricher_id"] == "topic_consensus"
    assert block["attempt"] == 2
    assert block["progress"] == {
        "items_done": 247,
        "items_total": 1000,
        "eta_seconds": 142,
    }
    assert block["last_heartbeat_at"] == "2026-06-26T15:03:14Z"


def test_build_current_enricher_block_optional_fields_default() -> None:
    block = build_current_enricher_block(
        enricher_id="x",
        scope="episode",
        tier="deterministic",
        attempt=1,
        items_done=0,
        items_total=None,
    )
    assert block["progress"]["items_total"] is None
    assert block["progress"]["eta_seconds"] is None
    # last_heartbeat_at defaults to ``utc_iso_now()`` — non-empty string.
    assert block["last_heartbeat_at"]
    assert block["last_heartbeat_at"].endswith("Z")


def test_write_status_atomic_replaces_existing(tmp_path: Path) -> None:
    """Second write replaces first; no leftover .tmp."""
    write_status(
        tmp_path,
        run_id="job-1",
        started_at="t0",
        profile=None,
        current_enricher=None,
        queue=["a"],
        completed=[],
    )
    write_status(
        tmp_path,
        run_id="job-2",
        started_at="t1",
        profile=None,
        current_enricher=None,
        queue=["b"],
        completed=[],
    )
    payload = read_status(tmp_path)
    assert payload is not None
    assert payload["run_id"] == "job-2"
    # No leftover .tmp file under .viewer/
    tmps = list((tmp_path / ".viewer").glob("*.tmp"))
    assert tmps == []
