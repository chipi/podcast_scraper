"""Integration tests for the consumer knowledge-card routes (#1095/#1096/#1097).

GET /api/app/persons/{id}, /api/app/topics/{id} (KG co-occurrence cards) and
/api/app/entities/search (exact/near-exact entity resolution) over a real fixture corpus via
TestClient. Mounted at /api/app, separate from the operator API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

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
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_person_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:nobody")
    assert resp.status_code == 404


def test_topic_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/topics/topic:nope").status_code == 404


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


def _fixture_client(tmp_path: Path) -> TestClient:
    """A client over the COMMITTED app-validation corpus, writing its app-data to ``tmp_path``.

    NOT ``_client(fixture_root)``: that derives ``data_dir`` as ``<root>/_appdata``, so pointing it
    at the tracked corpus creates ``tests/fixtures/app-validation-corpus/v3/_appdata/`` and leaves
    the fixture dirty after every run. `.gitignore` force-includes that tree, so it shows up as an
    untracked directory in every later `git status` — which is how a committed fixture quietly grows
    a users/ directory nobody meant to add.
    """
    fixture = Path("tests/fixtures/app-validation-corpus/v3")
    if not fixture.is_dir():
        pytest.skip(f"fixture corpus missing: {fixture}")
    data_dir = tmp_path / "_appdata"
    app = create_app(fixture, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_grouping_perspectives_404_on_an_absence_not_a_fault(tmp_path: Path) -> None:
    """A grouping nobody speaks to is a 404, which the client renders as nothing at all.

    `tc:lifelong-learning`'s members are abstract labels ("lifelong learning", "expert interviews")
    that nobody utters aloud, so no member has a speaker-attributable insight. That is an ABSENCE,
    and the section renders nothing rather than an error.
    """
    client = _fixture_client(tmp_path)

    assert client.get("/api/app/themes/tc:lifelong-learning/perspectives").status_code == 404
    assert client.get("/api/app/themes/tc:nope/perspectives").status_code == 404
    # A THEME id is not a storyline — the kind is explicit, never sniffed from the prefix.
    assert client.get("/api/app/storylines/tc:safety-practices/perspectives").status_code == 404


def _search(client: TestClient, kind: str, q: str) -> list[dict]:
    resp = client.get("/api/app/interests/search", params={"kind": kind, "q": q})
    assert resp.status_code == 200, resp.text
    items: list[dict] = resp.json()["items"]
    return items
