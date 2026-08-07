"""Tier-3 end-to-end: the in-app "Your Week" surface over a REAL synthetic corpus (#1412).

Unlike ``test_app_your_week_routes.py`` (which monkeypatches the assembler to test route wiring),
this drives ``GET /api/app/your-week`` through the ACTUAL assembler + corpus readers — real KG/GI
resolution, real follows/heard state — and asserts graph-carrying content flows through the HTTP
layer. This is the parity test with the digest email's own corpus e2e (test_app_digest_e2e.py):
one assembler, and here we prove the *route* (not just the function) surfaces the real rollup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_comms_store, app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_slugs import episode_slug
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = pytest.mark.integration


def _write_ep(
    root: Path,
    *,
    stem: str,
    feed_id: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    gi_text: str | None = None,
    published: str = "2024-03-10T00:00:00",
) -> str:
    """Write a corpus episode (metadata + KG [+ GI]); return its slug. Mirrors the digest e2e."""
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    rel = f"metadata/{stem}.metadata.json"
    doc = {
        "feed": {
            "feed_id": feed_id,
            "title": f"Show {feed_id}",
            "url": "https://p/f.xml",
            "image_url": f"https://img.example/{feed_id}.jpg",
        },
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "S", "bullets": ["a"]},
        "content": {
            "transcript_file_path": f"transcripts/{stem}.txt",
            "media_url": "https://cdn.example/a.mp3",
            "media_type": "audio/mpeg",
        },
    }
    (meta_dir / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello world", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": la}} for tid, la in topics]
    (meta_dir / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if gi_text is not None:
        gi = {
            "episode_id": episode_id,
            "nodes": [
                {
                    "id": "i1",
                    "type": "Insight",
                    "properties": {"text": gi_text, "grounded": True, "salience": 1.0},
                },
                {
                    "id": "q1",
                    "type": "Quote",
                    "properties": {"text": "quote", "timestamp_start_ms": 60000},
                },
            ],
            "edges": [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
        }
        (meta_dir / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    return episode_slug(feed_id, episode_id, rel)


def _client(root: Path, data_dir: Path, user_id: str) -> TestClient:
    app = create_app(root, static_dir=False)  # output_dir = the corpus root
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _kinds(body: dict) -> dict[str, list]:
    return {s["kind"]: s["items"] for s in body["sections"]}


def test_your_week_route_surfaces_real_graph_content(tmp_path: Path) -> None:
    """A heard-but-uncaptured episode with GI + a followed show → the ROUTE returns both sections
    with real graph_refs/quote/deep_link — no monkeypatching."""
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    user = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    )

    heard = _write_ep(
        root,
        stem="0001",
        feed_id="fa",
        episode_id="e1",
        topics=[("topic:ai", "AI")],
        gi_text="a grounded point",
    )
    app_user_state.set_playback(data_dir, user.user_id, heard, 500.0, 1)  # ≥30% → heard
    followed = _write_ep(
        root, stem="0002", feed_id="fb", episode_id="e2", topics=[("topic:ml", "ML")]
    )
    app_user_state.add_subscription(data_dir, user.user_id, {"feed_id": "fb"})  # unheard follow

    resp = _client(root, data_dir, user.user_id).get("/api/app/your-week")
    assert resp.status_code == 200
    body = resp.json()
    kinds = _kinds(body)

    revisit = kinds["revisit"][0]
    assert revisit["source"] == "auto"
    assert revisit["quote"] == "a grounded point"
    assert revisit["graph_refs"] == [{"id": "topic:ai", "kind": "topic", "label": "AI"}]
    assert revisit["deep_link"] == f"/player/{heard}?t=60"
    # In-app route enriches items with the show/episode art for the card backdrop.
    assert revisit["image_url"] == "https://img.example/fa.jpg"

    nif = kinds["new_in_follows"][0]
    assert nif["episode_slug"] == followed
    assert nif["graph_refs"] == [{"id": "topic:ml", "kind": "topic", "label": "ML"}]

    assert body["period_label"] and body["generated_at"].endswith("Z")


def test_your_week_route_ignores_email_consent_with_real_content(tmp_path: Path) -> None:
    """Real content still flows through the route when the email digest is OFF (consent-decoupled)."""
    root, data_dir = tmp_path / "corpus", tmp_path / "app"
    user = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    )
    heard = _write_ep(
        root,
        stem="0001",
        feed_id="fa",
        episode_id="e1",
        topics=[("topic:ai", "AI")],
        gi_text="a grounded point",
    )
    app_user_state.set_playback(data_dir, user.user_id, heard, 500.0, 1)
    app_comms_store.set_comms(data_dir, user.user_id, digest={"enabled": False})  # email OFF

    resp = _client(root, data_dir, user.user_id).get("/api/app/your-week")
    assert resp.status_code == 200
    assert _kinds(resp.json())["revisit"][0]["quote"] == "a grounded point"
