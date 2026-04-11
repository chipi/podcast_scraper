"""Tests for pipeline status file I/O (RFC-065)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from podcast_scraper.monitor.status import (
    maybe_update_pipeline_status,
    pipeline_status_path,
    read_pipeline_status,
    write_pipeline_status_atomic,
)


def test_pipeline_status_path(tmp_path: Path) -> None:
    p = pipeline_status_path(tmp_path)
    assert p == tmp_path / ".pipeline_status.json"


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    path = pipeline_status_path(tmp_path)
    write_pipeline_status_atomic(
        path,
        {"pid": 1, "stage": "rss_feed_fetch", "started_at": 100.0, "stage_started_at": 100.0},
    )
    data = read_pipeline_status(tmp_path)
    assert data is not None
    assert data["stage"] == "rss_feed_fetch"
    assert data["pid"] == 1


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_pipeline_status(tmp_path) is None


def test_maybe_update_noop_when_monitor_off(tmp_path: Path) -> None:
    cfg = SimpleNamespace(monitor=False)
    maybe_update_pipeline_status(cfg, str(tmp_path), stage="rss_feed_fetch")
    assert not pipeline_status_path(tmp_path).exists()


def test_maybe_update_writes_when_monitor_on(tmp_path: Path) -> None:
    cfg = SimpleNamespace(monitor=True)
    maybe_update_pipeline_status(cfg, str(tmp_path), stage="rss_feed_fetch")
    raw = pipeline_status_path(tmp_path).read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["stage"] == "rss_feed_fetch"
    assert "pid" in data
    maybe_update_pipeline_status(cfg, str(tmp_path), stage="speaker_detection", episode_total=3)
    data2 = read_pipeline_status(tmp_path)
    assert data2 is not None
    assert data2["stage"] == "speaker_detection"
    assert data2["episode_total"] == 3
    assert data2["started_at"] == data["started_at"]
