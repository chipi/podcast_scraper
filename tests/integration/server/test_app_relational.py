"""Integration tests for the consumer knowledge-card routes (#1095/#1096/#1097).

GET /api/app/persons/{id}, /api/app/topics/{id} (KG co-occurrence cards) and
/api/app/entities/search (exact/near-exact entity resolution) over a real fixture corpus via
TestClient. Mounted at /api/app, separate from the operator API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]],
    topics: list[tuple[str, str]],
    published: str = "2024-03-10T00:00:00",
) -> None:
    """Write one episode (metadata + KG with the given person/topic nodes)."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons]
    nodes += [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    kg = {"episode_id": episode_id, "nodes": nodes}
    (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _write_clusters(root: Path) -> None:
    """A topic_clusters.json making topic:ai and topic:ml siblings of one 'AI' cluster."""
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "Artificial Intelligence",
                "members": [
                    {"topic_id": "topic:ai", "label": "AI"},
                    {"topic_id": "topic:ml", "label": "Machine Learning"},
                ],
            }
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _two_episode_corpus(root: Path) -> None:
    # ep1: Jane + Bob discuss AI & ML.  ep2: Jane + Carol discuss AI.
    _write_episode(
        root,
        stem="0001-a",
        episode_id="ep1",
        persons=[("person:jane-doe", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002-b",
        episode_id="ep2",
        persons=[("person:jane-doe", "Jane Doe"), ("person:carol", "Carol")],
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def test_person_card_aggregates_episodes_and_related(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:jane-doe")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "person:jane-doe"
    assert body["label"] == "Jane Doe"
    assert body["episode_count"] == 2
    # newest-first: ep2 (June) before ep1 (Jan).
    assert [e["title"] for e in body["episodes"]] == ["Episode ep2", "Episode ep1"]
    related_ids = {p["id"] for p in body["related_people"]}
    assert related_ids == {"person:bob", "person:carol"}  # self excluded
    topic_ids = {t["id"] for t in body["related_topics"]}
    assert topic_ids == {"topic:ai", "topic:ml"}


def test_person_card_related_topics_carry_cluster_info(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    body = _client(tmp_path).get("/api/app/persons/person:jane-doe").json()
    ai = next(t for t in body["related_topics"] if t["id"] == "topic:ai")
    assert ai["cluster_id"] == "tc:ai"
    assert ai["cluster_label"] == "Artificial Intelligence"
    assert ai["cluster_size"] == 2


def test_person_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:nobody")
    assert resp.status_code == 404


def test_topic_card_episodes_siblings_and_people(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    resp = _client(tmp_path).get("/api/app/topics/topic:ai")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "topic:ai"
    assert body["label"] == "AI"
    assert body["cluster_id"] == "tc:ai"
    assert body["cluster_size"] == 2
    assert body["episode_count"] == 2  # both episodes discuss AI
    sibling_ids = {s["id"] for s in body["sibling_topics"]}
    assert sibling_ids == {"topic:ml"}  # cluster co-member, self excluded
    people_ids = {p["id"] for p in body["related_people"]}
    assert people_ids == {"person:jane-doe", "person:bob", "person:carol"}


def test_topic_card_without_clusters_has_no_siblings(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/topics/topic:ml").json()
    assert body["episode_count"] == 1  # only ep1 discusses ML
    assert body["sibling_topics"] == []
    assert body["cluster_id"] is None


def test_topic_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/topics/topic:nope").status_code == 404


def test_entity_search_resolves_person_exact_and_near_exact(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    client = _client(tmp_path)
    exact = client.get("/api/app/entities/search", params={"q": "Jane Doe"}).json()
    assert exact["entity"] == {"id": "person:jane-doe", "kind": "person", "label": "Jane Doe"}
    # Case/punctuation-insensitive ("near-exact"): hyphen + lowercase still resolves.
    near = client.get("/api/app/entities/search", params={"q": "jane-doe"}).json()
    assert near["entity"]["id"] == "person:jane-doe"


def test_entity_search_resolves_topic(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = (
        _client(tmp_path).get("/api/app/entities/search", params={"q": "machine learning"}).json()
    )
    assert body["entity"] == {"id": "topic:ml", "kind": "topic", "label": "Machine Learning"}


def test_entity_search_no_match_returns_null(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = (
        _client(tmp_path)
        .get("/api/app/entities/search", params={"q": "quantum chromodynamics"})
        .json()
    )
    assert body["query"] == "quantum chromodynamics"
    assert body["entity"] is None


def test_entity_search_prefers_person_over_topic_on_collision(tmp_path: Path) -> None:
    # A person and a topic share the name "Focus" → the person card wins.
    _write_episode(
        tmp_path,
        stem="0001-c",
        episode_id="ep1",
        persons=[("person:focus", "Focus")],
        topics=[("topic:focus", "Focus")],
    )
    body = _client(tmp_path).get("/api/app/entities/search", params={"q": "focus"}).json()
    assert body["entity"]["kind"] == "person"
    assert body["entity"]["id"] == "person:focus"


def _write_arc_bundle(root: Path, stem: str, publish: str, insight_id: str, label: str) -> None:
    """One bridge+gi+kg episode + an insight_sentiment sidecar (drives the CIL conversation arc)."""
    md = root / "metadata"
    md.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "3.0",
        "episode_id": f"ep-{stem}",
        "identities": [
            {"id": "person:a", "type": "person", "sources": {"gi": True}, "display_name": "A"},
            {"id": "topic:ai", "type": "topic", "sources": {"gi": True}, "display_name": "AI"},
        ],
    }
    gi = {
        "schema_version": "3.0",
        "episode_id": f"ep-{stem}",
        "nodes": [
            {"id": insight_id, "type": "Insight", "properties": {"text": "an AI take"}},
            {"id": f"q-{stem}", "type": "Quote", "properties": {"text": "q"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": f"q-{stem}", "to": "person:a"},
            {"type": "SUPPORTED_BY", "from": insight_id, "to": f"q-{stem}"},
            {"type": "ABOUT", "from": insight_id, "to": "topic:ai"},
        ],
    }
    kg = {"nodes": [{"id": "kg:ep", "type": "Episode", "properties": {"publish_date": publish}}]}
    (md / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (md / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (md / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    (md / "enrichments").mkdir(parents=True, exist_ok=True)
    (md / "enrichments" / f"{stem}.insight_sentiment.json").write_text(
        json.dumps(
            {"data": {"insights": [{"insight_id": insight_id, "compound": 0.7, "label": label}]}}
        ),
        encoding="utf-8",
    )


def test_topic_conversation_arc_route(tmp_path: Path) -> None:
    _write_arc_bundle(tmp_path, "0001-a", "2024-01-15", "i1", "positive")
    _write_arc_bundle(tmp_path, "0002-b", "2024-01-16", "i2", "negative")
    resp = _client(tmp_path).get("/api/app/topics/topic:ai/conversation-arc")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["topic_id"] == "topic:ai"
    # Both dates are ISO week 2024-W03 → one bucket, volume 2, 1 pos + 1 neg.
    assert len(body["weeks"]) == 1
    wk = body["weeks"][0]
    assert wk["week"] == "2024-W03"
    assert wk["volume"] == 2 and wk["positive"] == 1 and wk["negative"] == 1
