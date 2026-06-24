"""Integration tests for the consumer platform episode routes (#1067/#1068/#1070).

GET /api/app/episodes/{slug} (detail), /insights, /entities, /segments, /audio-source,
against a real fixture corpus via TestClient. Slug-addressed; routes mounted at /api/app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
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

    content: dict = {"transcript_file_path": f"transcripts/{stem}.txt"}
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
        "resolved_url": None,
        "verified": None,
        "content_length": None,
    }


def test_unknown_slug_404(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    client = _client(tmp_path)
    assert client.get("/api/app/episodes/does-not-exist/segments").status_code == 404
    assert client.get("/api/app/episodes/does-not-exist/audio-source").status_code == 404


def test_related_empty_without_index(tmp_path: Path) -> None:
    # "More like this" needs the vector index; without one it degrades to 200 + empty
    # (graceful — the panel section simply hides). No real index is built in CI.
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    client = _client(tmp_path)
    resp = client.get(f"/api/app/episodes/{slug}/related")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["items"] == []
    assert body["total"] == 0
    # Unknown slug still 404s.
    assert client.get("/api/app/episodes/nope/related").status_code == 404


def test_segments_resolve_in_nested_run_layout(tmp_path: Path) -> None:
    # Prod layout: feeds/<feed>/run_<R>/{metadata,transcripts}; transcript_file_path is
    # relative to the RUN dir (regression guard for the segments path-resolution bug).
    run = tmp_path / "feeds" / "myfeed" / "run_20240101_x"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir(parents=True)
    (run / "transcripts" / "0001.txt").write_text("hi", encoding="utf-8")
    (run / "transcripts" / "0001.segments.json").write_text(
        json.dumps([{"id": 0, "start": 0.0, "end": 1.0, "text": "Hello.", "speaker_label": "A"}]),
        encoding="utf-8",
    )
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show"},
        "episode": {"episode_id": "ep1", "title": "Hello", "published_date": "2024-01-01T00:00:00"},
        "content": {"transcript_file_path": "transcripts/0001.txt"},
    }
    (run / "metadata" / "0001.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    slug = _only_slug(tmp_path)
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments")
    assert resp.status_code == 200, resp.text
    assert resp.json()["segments"][0]["text"] == "Hello."


def test_player_serves_raw_segments_not_adfree(tmp_path: Path) -> None:
    # The player streams the ORIGINAL audio (ads in), so transcript-sync must use the raw
    # canonical segments (original timeline), NOT the ad-free ones (minutes shorter → drift).
    _write_corpus(tmp_path, stem="0001-ep")
    # Add an ad-free variant alongside the raw one, with distinct text + later timeline.
    (tmp_path / "transcripts" / "0001-ep.adfree.segments.json").write_text(
        json.dumps(
            [{"id": 0, "start": 0.0, "end": 9.0, "text": "AD-FREE timeline.", "speaker_label": "X"}]
        ),
        encoding="utf-8",
    )
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/segments").json()
    texts = [s["text"] for s in body["segments"]]
    assert "Hello world." in texts  # raw canonical (from _write_corpus)
    assert "AD-FREE timeline." not in texts  # ad-free must NOT be served to the player


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


def test_audio_source_validate_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.server import app_audio_bridge

    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    monkeypatch.setattr(
        app_audio_bridge,
        "_head_request",
        lambda url, timeout: (
            200,
            "https://cdn.example/final.mp3",
            {"content-type": "audio/mpeg", "content-length": "999"},
        ),
    )
    body = (
        _client(tmp_path)
        .get(f"/api/app/episodes/{slug}/audio-source", params={"validate": "true"})
        .json()
    )
    assert body["verified"] is True
    assert body["resolved_url"] == "https://cdn.example/final.mp3"
    assert body["content_length"] == 999


def test_audio_source_default_skips_validation(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/audio-source").json()
    assert body["verified"] is None
    assert body["resolved_url"] is None


def test_audio_source_validate_network_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    from podcast_scraper.server import app_audio_bridge

    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)

    def boom(url: str, timeout: float):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(app_audio_bridge, "_head_request", boom)
    body = (
        _client(tmp_path)
        .get(f"/api/app/episodes/{slug}/audio-source", params={"validate": "true"})
        .json()
    )
    assert body["verified"] is False
    assert body["resolved_url"] == "https://cdn.example/ep1.mp3"  # falls back to the original URL


def test_episode_search_no_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_corpus(tmp_path)
    slug = _only_slug(tmp_path)
    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        lambda output_dir, query, **kw: CorpusSearchOutcome(error="no_index", detail="no index"),
    )
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/search", params={"q": "x"}).json()
    assert body["error"] == "no_index"
    assert body["results"] == []


def test_resolve_slug_fallback_without_episode_id(tmp_path: Path) -> None:
    _write_corpus(tmp_path, episode_id=None)  # exercises the metadata-relpath slug fallback
    slug = _only_slug(tmp_path)
    resp = _client(tmp_path).get(f"/api/app/episodes/{slug}")
    assert resp.status_code == 200
    assert resp.json()["slug"] == slug


def test_episode_search_filters_to_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_corpus(tmp_path)  # feed "myfeed", episode "ep1"
    slug = _only_slug(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(output_dir: Path, query: str, **kwargs: Any) -> CorpusSearchOutcome:
        captured["feed"] = kwargs.get("feed")
        return CorpusSearchOutcome(
            results=[
                {
                    "doc_id": "a",
                    "score": 0.9,
                    "metadata": {"doc_type": "insight", "episode_id": "ep1"},
                    "text": "mine",
                },
                {
                    "doc_id": "b",
                    "score": 0.8,
                    "metadata": {"doc_type": "insight", "episode_id": "other"},
                    "text": "theirs",
                },
            ]
        )

    monkeypatch.setattr("podcast_scraper.search.capability.run_corpus_search", fake_run)
    body = _client(tmp_path).get(f"/api/app/episodes/{slug}/search", params={"q": "foo"}).json()
    assert captured["feed"] == "myfeed"  # over-fetched scoped by feed
    assert [r["doc_id"] for r in body["results"]] == ["a"]  # narrowed to this episode
