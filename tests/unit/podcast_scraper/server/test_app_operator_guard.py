"""Unit tests for operator write-path detection + audit log (#1071)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.app_operator_guard import is_operator_write


def test_is_operator_write_matches_mutating_operator_routes() -> None:
    assert is_operator_write("PUT", "/api/feeds")
    assert is_operator_write("PUT", "/api/operator-config")
    assert is_operator_write("POST", "/api/jobs")
    assert is_operator_write("POST", "/api/jobs/abc/cancel")
    assert is_operator_write("POST", "/api/jobs/reconcile")


def test_is_operator_write_ignores_reads_and_consumer_and_unrelated() -> None:
    assert not is_operator_write("GET", "/api/feeds")
    assert not is_operator_write("GET", "/api/jobs")
    assert not is_operator_write("POST", "/api/app/library")  # consumer surface
    assert not is_operator_write("GET", "/api/health")
    assert not is_operator_write("PUT", "/api/feedsX")  # not an exact/prefix match


def test_append_audit_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    append_audit(path, {"method": "PUT", "path": "/api/feeds", "outcome": "allowed"})
    append_audit(path, {"method": "POST", "path": "/api/jobs", "outcome": "denied"})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["path"] == "/api/feeds" and first["outcome"] == "allowed" and "ts" in first


def test_append_audit_none_is_noop() -> None:
    append_audit(None, {"x": 1})  # must not raise
