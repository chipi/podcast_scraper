"""Integration tests for CIL query API (GitHub #527)."""

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
    metadata_episode_title: str | None = None,
    metadata_feed_title: str | None = None,
    metadata_episode_number: int | None = None,
    metadata_episode_image_url: str | None = None,
    metadata_feed_image_url: str | None = None,
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
    meta_doc: dict = {}
    feed_meta: dict = {"feed_id": "feed:test-fixture"}
    if metadata_feed_title is not None:
        feed_meta["title"] = metadata_feed_title
    if metadata_feed_image_url is not None:
        feed_meta["image_url"] = metadata_feed_image_url
    meta_doc["feed"] = feed_meta
    ep_meta: dict = {"episode_id": episode_id}
    if metadata_episode_title is not None:
        ep_meta["title"] = metadata_episode_title
    if metadata_episode_number is not None:
        ep_meta["episode_number"] = metadata_episode_number
    if metadata_episode_image_url is not None:
        ep_meta["image_url"] = metadata_episode_image_url
    meta_doc["episode"] = ep_meta
    (directory / f"{stem}.metadata.json").write_text(
        json.dumps(meta_doc),
        encoding="utf-8",
    )


def _bundle_two_topics_same_episode(
    directory: Path,
    stem: str,
    *,
    episode_id: str,
    publish_date: str,
    person: str,
    topic_a: str,
    topic_b: str,
) -> None:
    """One episode with two topics and two insights (merged cluster timeline)."""
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
                "id": topic_a,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "TA",
                "aliases": [],
            },
            {
                "id": topic_b,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "TB",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight-a",
                "type": "Insight",
                "properties": {"text": "About A", "insight_type": "claim", "position_hint": 0.1},
            },
            {
                "id": "insight-b",
                "type": "Insight",
                "properties": {"text": "About B", "insight_type": "claim", "position_hint": 0.2},
            },
            {"id": "quote-a", "type": "Quote", "properties": {"text": "said a"}},
            {"id": "quote-b", "type": "Quote", "properties": {"text": "said b"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote-a", "to": person},
            {"type": "SPOKEN_BY", "from": "quote-b", "to": person},
            {"type": "SUPPORTED_BY", "from": "insight-a", "to": "quote-a"},
            {"type": "SUPPORTED_BY", "from": "insight-b", "to": "quote-b"},
            {"type": "ABOUT", "from": "insight-a", "to": topic_a},
            {"type": "ABOUT", "from": "insight-b", "to": topic_b},
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
    meta_doc = {
        "feed": {"feed_id": "feed:test-fixture"},
        "episode": {"episode_id": episode_id},
    }
    (directory / f"{stem}.metadata.json").write_text(
        json.dumps(meta_doc),
        encoding="utf-8",
    )


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
        metadata_feed_title="Mock Feed Show",
        metadata_episode_number=101,
        metadata_episode_image_url="https://example.com/ep-cover.jpg",
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
        body = resp.json()
        assert len(body["episodes"]) == 1
        ep0 = body["episodes"][0]
        assert ep0["episode_title"] == "Science Monday"
        assert ep0["feed_title"] == "Mock Feed Show"
        assert ep0["episode_number"] == 101
        assert ep0["episode_image_url"] == "https://example.com/ep-cover.jpg"

    def test_timeline_merge_post(self, client: TestClient, cil_corpus: Path) -> None:
        resp = client.post(
            "/api/topics/timeline",
            json={
                "topic_ids": ["topic:science", "topic:science"],
                "path": str(cil_corpus),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["topic_ids"] == ["topic:science"]
        assert len(body["episodes"]) == 1
        assert body["episodes"][0]["episode_id"] == "episode:one"

    def test_timeline_merge_two_topics_one_episode(self, tmp_path: Path) -> None:
        meta = tmp_path / "metadata"
        _bundle_two_topics_same_episode(
            meta,
            "ep1",
            episode_id="episode:one",
            publish_date="2024-05-01",
            person="person:pat",
            topic_a="topic:science",
            topic_b="topic:economics",
        )
        client = TestClient(create_app(tmp_path, static_dir=False))
        resp = client.post(
            "/api/topics/timeline",
            json={
                "topic_ids": ["topic:science", "topic:economics"],
                "path": str(tmp_path),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["topic_ids"] == ["topic:science", "topic:economics"]
        assert len(body["episodes"]) == 1
        assert body["episodes"][0]["episode_id"] == "episode:one"
        assert len(body["episodes"][0]["insights"]) == 2

    def test_timeline_normalizes_viewer_g_prefixed_topic_id(
        self, client: TestClient, cil_corpus: Path
    ) -> None:
        """GI+KG merged graph node ids use ``g:`` prefix; API matches on-disk bridge ids."""
        tid = quote("g:topic:science", safe="")
        resp = client.get(
            f"/api/topics/{tid}/timeline",
            params={"path": str(cil_corpus)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["topic_id"] == "topic:science"
        assert len(body["episodes"]) == 1

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
