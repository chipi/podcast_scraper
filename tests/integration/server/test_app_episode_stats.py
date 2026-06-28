"""Integration tests for GET /api/app/episodes/{slug}/stats (UXS-014).

Public, no-auth cross-user reach for one episode: distinct listeners, total opens,
grounded-insight count, and a zero-filled daily opens sparkline. Listener/open counts
are aggregated by scanning every user's ``listen_events.jsonl`` under the app data dir,
so the test seeds those logs directly via :func:`app_user_state.append_listen_event`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_stats, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _write_corpus(root: Path, *, stem: str = "0001-hello", episode_id: str = "ep1") -> None:
    """One KG+GI episode (the GI carries a single grounded Insight node)."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": "Hello",
            "published_date": "2024-03-10T00:00:00",
            "duration_seconds": 4823,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {"text": "Big claim.", "grounded": True, "insight_type": "claim"},
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "text": "verbatim quote",
                    "speaker_id": "SPEAKER_00",
                    "timestamp_start_ms": 1000,
                    "timestamp_end_ms": 2000,
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    kg = {
        "episode_id": episode_id,
        "nodes": [{"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}}],
    }
    (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _client(root: Path) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.app_data_dir = root / "appdata"
    return TestClient(app)


def _only_slug(root: Path) -> str:
    rows = build_catalog_rows_cumulative(root)
    assert len(rows) == 1
    return slug_for_row(rows[0])


def _seed_user_opens(data_dir: Path, *, subject: str, slug: str, opens: int) -> None:
    """Create a user and append ``opens`` listen events for ``slug``."""
    user = get_or_create_user(
        data_dir, provider="stub", subject=subject, email=f"{subject}@x.com", name=subject
    )
    now = int(time.time())
    for i in range(opens):
        app_user_state.append_listen_event(data_dir, user.user_id, slug, "myfeed", now - i)


def test_stats_aggregates_listeners_opens_and_insights(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    data_dir = tmp_path / "appdata"
    # Two distinct users open the episode; user A twice, user B once → 3 total opens.
    _seed_user_opens(data_dir, subject="a", slug=slug, opens=2)
    _seed_user_opens(data_dir, subject="b", slug=slug, opens=1)

    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/stats")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slug"] == slug
    assert body["listeners"] == 2  # distinct people
    assert body["opens"] == 3  # total opens across everyone
    assert body["insights"] == 1  # one grounded Insight in the GI
    # Zero-filled daily sparkline of fixed length, total counts == opens.
    assert len(body["daily"]) == app_stats.SERIES_DAYS
    assert sum(point["count"] for point in body["daily"]) == 3


def test_stats_zero_state_for_unopened_episode(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    data_dir = tmp_path / "appdata"
    # A user exists and has events, but for a DIFFERENT slug → nobody opened this one.
    _seed_user_opens(data_dir, subject="a", slug="some-other-slug", opens=3)

    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/stats").json()
    assert body["listeners"] == 0
    assert body["opens"] == 0
    assert body["insights"] == 1  # insight count is independent of listening reach
    assert len(body["daily"]) == app_stats.SERIES_DAYS
    assert all(point["count"] == 0 for point in body["daily"])


def test_stats_unknown_slug_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/episodes/does-not-exist/stats").status_code == 404
