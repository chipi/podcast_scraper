"""Route tests for the personal-corpus surface /api/app/corpus (RFC-114 Phase 1, #1470)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import episode_slug
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _corpus(root: Path) -> str:
    """One episode with KG + a known duration; return its slug."""
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = "metadata/0001.metadata.json"
    doc = {
        "feed": {"feed_id": "fa", "title": "Show", "url": "https://p/f.xml"},
        "episode": {
            "episode_id": "e1",
            "title": "Ep",
            "published_date": "2024-01-01T00:00:00",
            "duration_seconds": 1000,
        },
        "summary": {"title": "S", "bullets": ["a"]},
        "content": {"transcript_file_path": "transcripts/0001.txt"},
    }
    (meta / "0001.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / "0001.txt").write_text("hi", encoding="utf-8")
    (meta / "0001.kg.json").write_text(
        json.dumps(
            {
                "episode_id": "e1",
                "nodes": [
                    {"id": "person:jane", "type": "Person", "properties": {"name": "Jane"}},
                    {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
                ],
            }
        ),
        encoding="utf-8",
    )
    return episode_slug("fa", "e1", rel)


def _authed(tmp_path: Path):
    root = tmp_path / "corpus"
    data_dir = tmp_path / "app"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="google", subject="s", email="u@g.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client, root, data_dir, user.user_id


def test_summary_facets_and_correction(tmp_path: Path) -> None:
    client, root, data_dir, uid = _authed(tmp_path)
    slug = _corpus(root)
    # heard episode → experienced; a separate favorited episode → saved (NOT experienced/recall)
    app_user_state.set_playback(data_dir, uid, slug, 500.0, 1)  # 50% → heard
    app_user_state.add_favorite(data_dir, uid, {"kind": "episode", "ref": "some-unheard-ep"})

    body = client.get("/api/app/corpus").json()
    assert body["experienced_count"] == 1
    assert body["saved_count"] == 1
    assert body["revision"] >= 1
    assert any(e["kind"] == "topic" and e["label"] == "AI" for e in body["top_entities"])

    exp = client.get("/api/app/corpus/episodes", params={"facet": "experienced"}).json()
    assert exp["slugs"] == [slug]
    sav = client.get("/api/app/corpus/episodes", params={"facet": "saved"}).json()
    assert sav["slugs"] == ["some-unheard-ep"]


def test_changes_delta_and_tombstone(tmp_path: Path) -> None:
    client, root, data_dir, uid = _authed(tmp_path)
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": "ep-a", "kind": "span", "created_at": 1}
    )
    c1 = client.get("/api/app/corpus/changes", params={"since": 0}).json()
    assert c1["revision"] == 1
    assert c1["events"] == [{"seq": 1, "kind": "added", "facet": "experienced", "ref": "ep-a"}]

    # delete the highlight → ep-a leaves experienced → a tombstone in the next delta
    app_user_state.remove_highlight(data_dir, uid, "h1")
    c2 = client.get("/api/app/corpus/changes", params={"since": 1}).json()
    assert c2["events"] == [{"seq": 2, "kind": "removed", "facet": "experienced", "ref": "ep-a"}]


def test_requires_auth(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _corpus(root)
    app = create_app(root, static_dir=False)
    app.state.app_data_dir = tmp_path / "app"
    app.state.session_secret = "s"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    assert TestClient(app).get("/api/app/corpus").status_code in (401, 403)


def test_ranked_by_strength(tmp_path: Path) -> None:
    client, root, data_dir, uid = _authed(tmp_path)
    slug = _corpus(root)
    # heard the episode + a highlight on it → it ranks; a second experienced ep via a note only.
    app_user_state.set_playback(data_dir, uid, slug, 1000.0, 1)  # fully heard
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h1", "episode_slug": slug, "kind": "span", "created_at": 1}
    )
    app_user_state.add_highlight(
        data_dir, uid, {"id": "h2", "episode_slug": "ep-note-only", "kind": "span", "created_at": 1}
    )
    ranked = client.get("/api/app/corpus/ranked").json()["items"]
    slugs = [r["slug"] for r in ranked]
    assert slug in slugs and "ep-note-only" in slugs
    # the fully-heard + highlighted episode outranks the capture-only one
    by = {r["slug"]: r["strength"] for r in ranked}
    assert by[slug] > by["ep-note-only"]
    assert ranked == sorted(ranked, key=lambda r: -r["strength"])  # strongest first
