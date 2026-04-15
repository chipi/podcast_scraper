"""Integration tests for POST /api/corpus/resolve-episode-artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from tests.integration.server.test_cil_api import _bundle

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


@pytest.fixture()
def cil_corpus(tmp_path: Path) -> Path:
    meta = tmp_path / "metadata"
    _bundle(
        meta,
        "ep1",
        episode_id="episode:one",
        publish_date="2024-05-01",
        person="person:pat",
        topic="topic:science",
        metadata_episode_title="Science Monday",
    )
    return tmp_path


@pytest.fixture()
def client(cil_corpus: Path) -> TestClient:
    return TestClient(create_app(cil_corpus, static_dir=False))


def test_resolve_episode_artifacts(client: TestClient, cil_corpus: Path) -> None:
    resp = client.post(
        "/api/corpus/resolve-episode-artifacts",
        json={"episode_ids": ["episode:one", "missing"], "path": str(cil_corpus)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["resolved"]) == 1
    assert body["resolved"][0]["episode_id"] == "episode:one"
    assert body["resolved"][0]["gi_relative_path"]
    assert "missing" in body["missing_episode_ids"]
