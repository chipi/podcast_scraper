"""Integration tests for POST /api/corpus/node-episodes."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from tests.integration.server.test_cil_api import _bundle

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


@pytest.fixture()
def two_ep_corpus(tmp_path: Path) -> Path:
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
    _bundle(
        meta,
        "ep2",
        episode_id="episode:two",
        publish_date="2024-06-01",
        person="person:pat",
        topic="topic:science",
        metadata_episode_title="Science Tuesday",
    )
    return tmp_path


@pytest.fixture()
def client(two_ep_corpus: Path) -> TestClient:
    return TestClient(create_app(two_ep_corpus, static_dir=False))


def test_node_episodes_two_matches(client: TestClient, two_ep_corpus: Path) -> None:
    resp = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "g:topic:science", "path": str(two_ep_corpus)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == "topic:science"
    assert not body["truncated"]
    assert body["total_matched"] is None
    assert len(body["episodes"]) == 2
    gi_paths = {e["gi_relative_path"] for e in body["episodes"]}
    assert gi_paths == {"metadata/ep1.gi.json", "metadata/ep2.gi.json"}


def test_node_episodes_truncated(client: TestClient, two_ep_corpus: Path) -> None:
    resp = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "topic:science", "path": str(two_ep_corpus), "max_episodes": 1},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated"] is True
    assert body["total_matched"] == 2
    assert len(body["episodes"]) == 1
