"""Integration tests for the consumer platform episode routes (#1067/#1068/#1070).

GET /api/app/episodes/{slug} (detail), /insights, /entities, /segments, /audio-source,
against a real fixture corpus via TestClient. Slug-addressed; routes mounted at /api/app.
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
    with_gi: bool = True,
    with_kg: bool = True,
) -> None:
    """Build a minimal flat corpus: one episode metadata + transcript/segments + GI/KG."""
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

    if with_gi:
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

    if with_kg:
        kg = {
            "episode_id": episode_id,
            "nodes": [
                {"id": "person:jane-doe", "type": "Person", "properties": {"name": "Jane Doe"}},
                {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
            ],
        }
        (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


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


def test_detail_returns_metadata_and_flags(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}").json()
    assert body["slug"] == slug
    assert body["title"] == "Hello"
    assert body["feed_id"] == "myfeed"
    assert body["podcast_title"] == "My Show"
    assert body["duration_seconds"] == 4823
    assert body["summary_bullets"] == ["a", "b"]
    assert body["has_transcript"] is True
    assert body["has_summary"] is True
    assert body["has_gi"] is True
    assert body["has_kg"] is True


def test_insights_endpoint_returns_grounded(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/insights").json()
    assert body["episode_slug"] == slug
    assert len(body["insights"]) == 1
    ins = body["insights"][0]
    assert ins["id"] == "insight:1"
    assert ins["text"] == "Big claim."
    assert ins["grounded"] is True
    assert ins["insight_type"] == "claim"
    assert ins["quotes"][0]["text"] == "verbatim quote"
    assert ins["quotes"][0]["speaker"] == "SPEAKER_00"
    assert ins["quotes"][0]["start_ms"] == 1000


def test_entities_endpoint_returns_persons_and_topics(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/entities").json()
    assert body["episode_slug"] == slug
    assert [p["id"] for p in body["persons"]] == ["person:jane-doe"]
    assert body["persons"][0]["name"] == "Jane Doe"
    assert [t["id"] for t in body["topics"]] == ["topic:ai"]


def test_insights_empty_when_no_gi(tmp_path: Path) -> None:
    _write_corpus(tmp_path, with_gi=False)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/insights").json()
    assert body == {"episode_slug": slug, "insights": []}


def test_entities_empty_when_no_kg(tmp_path: Path) -> None:
    _write_corpus(tmp_path, with_kg=False)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/entities").json()
    assert body == {"episode_slug": slug, "persons": [], "orgs": [], "topics": []}


def test_detail_unknown_slug_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/episodes/does-not-exist").status_code == 404
