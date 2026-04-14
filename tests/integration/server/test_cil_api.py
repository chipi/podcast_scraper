"""Integration tests for RFC-072 CIL query API (GitHub #527)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _bundle(
    directory: Path,
    stem: str,
    *,
    episode_id: str,
    publish_date: str,
    person: str,
    topic: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "identities": [
            {
                "id": person,
                "type": "person",
                "sources": {"gi": True, "kg": True},
                "display_name": "P",
                "aliases": [],
            },
            {
                "id": topic,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "T",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight-1",
                "type": "Insight",
                "properties": {"text": "Hello", "insight_type": "claim", "position_hint": 0.1},
            },
            {"id": "quote-1", "type": "Quote", "properties": {"text": "said"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote-1", "to": person},
            {"type": "SUPPORTED_BY", "from": "insight-1", "to": "quote-1"},
            {"type": "ABOUT", "from": "insight-1", "to": topic},
        ],
    }
    kg = {
        "nodes": [
            {
                "id": "ep1",
                "type": "Episode",
                "properties": {"publish_date": publish_date},
            }
        ],
        "edges": [],
    }
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


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
    )
    return tmp_path


@pytest.fixture()
def client(cil_corpus: Path) -> TestClient:
    return TestClient(create_app(cil_corpus, static_dir=False))


class TestCilApi:
    def test_positions(self, client: TestClient, cil_corpus: Path) -> None:
        pid = quote("person:pat", safe="")
        resp = client.get(
            f"/api/persons/{pid}/positions",
            params={"topic": "topic:science", "path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["person_id"] == "person:pat"
        assert body["topic_id"] == "topic:science"
        assert len(body["episodes"]) == 1
        assert body["episodes"][0]["episode_id"] == "episode:one"
        assert len(body["episodes"][0]["insights"]) == 1

    def test_brief(self, client: TestClient, cil_corpus: Path) -> None:
        pid = quote("person:pat", safe="")
        resp = client.get(
            f"/api/persons/{pid}/brief",
            params={"path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "topic:science" in body["topics"]
        assert len(body["quotes"]) == 1

    def test_timeline(self, client: TestClient, cil_corpus: Path) -> None:
        tid = quote("topic:science", safe="")
        resp = client.get(
            f"/api/topics/{tid}/timeline",
            params={"path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        assert len(resp.json()["episodes"]) == 1

    def test_person_topics(self, client: TestClient, cil_corpus: Path) -> None:
        pid = quote("person:pat", safe="")
        resp = client.get(
            f"/api/persons/{pid}/topics",
            params={"path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        assert resp.json()["ids"] == ["topic:science"]

    def test_topic_persons(self, client: TestClient, cil_corpus: Path) -> None:
        tid = quote("topic:science", safe="")
        resp = client.get(
            f"/api/topics/{tid}/persons",
            params={"path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        assert resp.json()["ids"] == ["person:pat"]

    def test_positions_insight_types_wildcards(self, client: TestClient, cil_corpus: Path) -> None:
        pid = quote("person:pat", safe="")
        for raw in ("all", "*", ""):
            resp = client.get(
                f"/api/persons/{pid}/positions",
                params={"topic": "topic:science", "path": str(cil_corpus), "insight_types": raw},
            )
            assert resp.status_code == 200
            assert len(resp.json()["episodes"]) == 1

    def test_positions_default_output_dir_no_path_query(self, cil_corpus: Path) -> None:
        client = TestClient(create_app(cil_corpus, static_dir=False))
        pid = quote("person:pat", safe="")
        resp = client.get(
            f"/api/persons/{pid}/positions",
            params={"topic": "topic:science"},
        )
        assert resp.status_code == 200
        assert resp.json()["path"]

    def test_require_path_when_no_default_output_dir(self) -> None:
        client = TestClient(create_app(None, static_dir=False))
        pid = quote("person:pat", safe="")
        resp = client.get(
            f"/api/persons/{pid}/positions",
            params={"topic": "topic:science"},
        )
        assert resp.status_code == 400
        assert "path" in resp.json()["detail"].lower()
