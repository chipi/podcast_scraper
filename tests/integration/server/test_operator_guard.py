"""Integration tests for the operator write-path guard + audit (#1071)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _client(tmp_path: Path, *, key: str = "") -> TestClient:
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.operator_api_key = key
    app.state.audit_path = tmp_path / "audit.jsonl"
    return TestClient(app)


def test_write_requires_key_when_configured(tmp_path: Path) -> None:
    client = _client(tmp_path, key="secret")
    assert client.put("/api/feeds", json={"feeds": []}).status_code == 401
    with_key = client.put("/api/feeds", json={"feeds": []}, headers={"X-Operator-Key": "secret"})
    assert with_key.status_code != 401  # guard passed (route handles the body)
    # both attempts audited
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2


def test_write_allowed_when_no_key_configured(tmp_path: Path) -> None:
    client = _client(tmp_path, key="")
    assert client.put("/api/feeds", json={"feeds": []}).status_code != 401


def test_reads_are_not_gated(tmp_path: Path) -> None:
    client = _client(tmp_path, key="secret")
    assert client.get("/api/feeds").status_code != 401
