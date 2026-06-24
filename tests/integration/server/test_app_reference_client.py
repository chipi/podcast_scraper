"""End-to-end reference-client test — proves the whole /api/app spine (#1072)."""

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
from podcast_scraper.server.app_reference_client import walk_app_contract
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _corpus(root: Path) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    stem = "0001-hello"
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/f.xml"},
        "episode": {
            "episode_id": "ep1",
            "title": "Hello",
            "published_date": "2024-03-10T00:00:00",
            "duration_seconds": 4823,
        },
        "summary": {"title": "S", "bullets": ["a"]},
        "content": {
            "transcript_file": f"transcripts/{stem}.txt",
            "media_url": "https://cdn.example/ep1.mp3",
            "media_type": "audio/mpeg",
            "media_id": "sha256:x",
        },
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.segments.json").write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 2.5, "text": "Hi", "speaker_label": "A"}]),
        encoding="utf-8",
    )
    (root / "metadata" / f"{stem}.gi.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "insight:1",
                        "type": "Insight",
                        "properties": {"text": "Claim", "grounded": True},
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps(
            {"nodes": [{"id": "person:jane", "type": "Person", "properties": {"name": "Jane"}}]}
        ),
        encoding="utf-8",
    )


def test_reference_client_walks_full_spine(tmp_path: Path) -> None:
    _corpus(tmp_path)
    data_dir = tmp_path / "appdata"
    app = create_app(tmp_path, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    client = TestClient(app)
    client.cookies.set(
        app_sessions.SESSION_COOKIE,
        app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret"),
    )
    slug = slug_for_row(build_catalog_rows_cumulative(tmp_path)[0])

    summary = walk_app_contract(client, slug)
    assert summary["user"] == "j@x.com"
    assert summary["title"] == "Hello"
    assert summary["segments"] == 1
    assert summary["audio_url"] == "https://cdn.example/ep1.mp3"
    assert summary["insights"] == 1
    assert summary["persons"] == 1
    assert summary["resume_seconds"] == 12.0
    assert summary["queue"] == [slug]
    assert summary["library"] == 1
