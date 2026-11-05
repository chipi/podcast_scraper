"""Unit tests for operator write-path detection + audit log (#1071)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_operator_guard import is_operator_write


def test_is_operator_write_matches_mutating_operator_routes() -> None:
    assert is_operator_write("PUT", "/api/feeds")
    assert is_operator_write("PUT", "/api/operator-config")
    assert is_operator_write("POST", "/api/jobs")
    assert is_operator_write("POST", "/api/jobs/abc/cancel")
    assert is_operator_write("POST", "/api/jobs/reconcile")
    assert is_operator_write("POST", "/api/index/rebuild")
    assert is_operator_write("DELETE", "/api/corpus/runs/R1")
    assert is_operator_write("DELETE", "/api/corpus/episodes/ep-1")


def test_is_operator_write_ignores_reads_and_consumer_and_unrelated() -> None:
    assert not is_operator_write("GET", "/api/feeds")
    assert not is_operator_write("GET", "/api/jobs")
    assert not is_operator_write("POST", "/api/app/library")  # consumer surface
    assert not is_operator_write("GET", "/api/health")
    assert not is_operator_write("PUT", "/api/feedsX")  # not an exact/prefix match
    assert not is_operator_write("GET", "/api/corpus/episodes")


def test_append_audit_records_in_order(tmp_path: Path) -> None:
    from podcast_scraper.server.app_audit import read_audit

    path = tmp_path / "audit.jsonl"
    append_audit(path, {"method": "PUT", "path": "/api/feeds", "outcome": "allowed"})
    append_audit(path, {"method": "POST", "path": "/api/jobs", "outcome": "denied"})
    records = read_audit(path)
    assert len(records) == 2
    first = records[0]
    assert first["path"] == "/api/feeds" and first["outcome"] == "allowed" and "ts" in first
    assert not path.exists(), "with a database the trail is the audit_log table"


def test_append_audit_writes_jsonl_without_a_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.db import reset_engines
    from podcast_scraper.server.app_audit import read_audit

    reset_engines()
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    path = tmp_path / "audit.jsonl"
    append_audit(path, {"method": "PUT", "path": "/api/feeds", "outcome": "allowed"})
    first = json.loads(path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert first["path"] == "/api/feeds" and "ts" in first
    assert read_audit(path) == [first]


def test_append_audit_none_is_noop() -> None:
    append_audit(None, {"x": 1})  # must not raise
