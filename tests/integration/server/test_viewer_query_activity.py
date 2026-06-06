"""Viewer API: GET /api/corpus/query-activity + /api/search logging (FR6.2, #888 follow-up).

Requires ``fastapi``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.search.query_log import append_query_event, query_log_path
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_query_activity_no_corpus_path() -> None:
    body = TestClient(create_app(None, static_dir=False)).get("/api/corpus/query-activity").json()
    assert body["error"] == "no_corpus_path"
    assert body["buckets"] == []


def test_query_activity_zero_filled_window(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get(
        "/api/corpus/query-activity", params={"path": str(tmp_path), "days": 3}
    ).json()
    assert body["total"] == 0
    assert len(body["buckets"]) == 3
    # oldest → newest, each zero.
    assert all(b["count"] == 0 for b in body["buckets"])


def test_query_activity_counts_logged_events_today(tmp_path: Path) -> None:
    # Two events dated "now" land in a 1-day window ending today.
    append_query_event(tmp_path, "semantic", now=datetime.now(timezone.utc))
    append_query_event(tmp_path, "entity_lookup", now=datetime.now(timezone.utc))
    client = TestClient(create_app(tmp_path, static_dir=False))
    body = client.get(
        "/api/corpus/query-activity", params={"path": str(tmp_path), "days": 1}
    ).json()
    assert body["total"] == 2


def test_search_appends_to_query_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> CorpusSearchOutcome:
        return CorpusSearchOutcome(results=[])

    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        fake_run,
    )
    client = TestClient(create_app(tmp_path, static_dir=False))
    assert not query_log_path(tmp_path).exists()
    r = client.get("/api/search", params={"q": "climate", "path": str(tmp_path)})
    assert r.status_code == 200
    assert query_log_path(tmp_path).exists()
    activity = client.get(
        "/api/corpus/query-activity", params={"path": str(tmp_path), "days": 1}
    ).json()
    assert activity["total"] == 1
