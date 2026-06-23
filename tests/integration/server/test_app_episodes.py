"""Integration tests for the consumer platform episode routes (#1067/#1070).

GET /api/app/episodes/{slug}/segments  and  /audio-source, against a real fixture
corpus via TestClient. Slug-addressed, mounted only when enable_platform=True.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

pytestmark = [pytest.mark.integration]


def _write_corpus(
    root: Path,
    *,
    stem: str = "0001-hello",
    episode_id: str | None = "ep1",
    media_url: str | None = "https://cdn.example/ep1.mp3",
    with_segments: bool = True,
) -> None:
    """Build a minimal flat corpus: one episode metadata + transcript segments."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)

    content: dict = {"transcript_file": f"transcripts/{stem}.txt"}
    if media_url is not None:
        content.update(
            {"media_url": media_url, "media_type": "audio/mpeg", "media_id": "sha256:abc"}
        )

    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": "Hello",
            "published_date": "2024-03-10T00:00:00",
            "duration_seconds": 4823,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b"]},
        "content": content,
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    if with_segments:
        segs = [
            {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world.", "speaker_label": "Alice"},
            {"id": 1, "start": 2.5, "end": 5.0, "text": "Second.", "speaker_id": "SPEAKER_01"},
        ]
        (root / "transcripts" / f"{stem}.segments.json").write_text(
            json.dumps(segs), encoding="utf-8"
        )


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def _only_slug(root: Path) -> str:
    rows = build_catalog_rows_cumulative(root)
    assert len(rows) == 1
    return slug_for_row(rows[0])


def test_segments_endpoint_returns_contract(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["version"] == "1.0"
    assert body["episode_slug"] == slug
    assert [s["id"] for s in body["segments"]] == ["seg_0000", "seg_0001"]
    assert body["segments"][0] == {
        "id": "seg_0000",
        "start": 0.0,
        "end": 2.5,
        "text": "Hello world.",
        "speaker": "Alice",
    }
    assert body["segments"][1]["speaker"] == "SPEAKER_01"


def test_audio_source_returns_origin_url(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/audio-source")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {
        "episode_slug": slug,
        "url": "https://cdn.example/ep1.mp3",
        "mime": "audio/mpeg",
        "duration_seconds": 4823,
        "media_id": "sha256:abc",
        "strategy": "direct",
    }


def test_unknown_slug_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = _client(tmp_path)
    assert client.get("/api/app/episodes/does-not-exist/segments").status_code == 404
    assert client.get("/api/app/episodes/does-not-exist/audio-source").status_code == 404


def test_no_segments_file_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path, with_segments=False)
    slug = _only_slug(tmp_path)
    assert _client(tmp_path).get(f"/api/app/episodes/{slug}/segments").status_code == 404


def test_no_media_url_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path, media_url=None)
    slug = _only_slug(tmp_path)
    assert _client(tmp_path).get(f"/api/app/episodes/{slug}/audio-source").status_code == 404
