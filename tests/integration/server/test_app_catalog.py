"""Integration tests for the consumer catalog list routes (#1078).

GET /api/app/episodes and /api/app/podcasts/{feed_id}/episodes against a real fixture
corpus via TestClient. Reads are open (no auth), consistent with the other /api/app reads.
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
    feed_id: str,
    feed_title: str,
    episode_id: str,
    title: str,
    published: str,
    with_transcript: bool = True,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    content: dict = {}
    if with_transcript:
        (root / "transcripts").mkdir(parents=True, exist_ok=True)
        (root / "transcripts" / f"{stem}.txt").write_text("hi", encoding="utf-8")
        content["transcript_file_path"] = f"transcripts/{stem}.txt"
    doc = {
        "feed": {"feed_id": feed_id, "title": feed_title, "url": f"https://{feed_id}.example/f"},
        "episode": {
            "episode_id": episode_id,
            "title": title,
            "published_date": published,
            "duration_seconds": 1800,
        },
        "summary": {"title": f"{title} sum", "bullets": ["alpha", "beta"]},
        "content": content,
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _corpus(root: Path) -> None:
    _write_episode(
        root,
        stem="0001",
        feed_id="showa",
        feed_title="Show A",
        episode_id="a1",
        title="A One",
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002",
        feed_id="showa",
        feed_title="Show A",
        episode_id="a2",
        title="A Two",
        published="2024-03-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0003",
        feed_id="showb",
        feed_title="Show B",
        episode_id="b1",
        title="B One",
        published="2024-02-01T00:00:00",
        with_transcript=False,
    )


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def test_global_catalog_lists_newest_first(tmp_path: Path) -> None:
    _corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/episodes")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 3
    assert body["page"] == 1
    assert body["has_more"] is False
    assert [i["title"] for i in body["items"]] == ["A Two", "B One", "A One"]
    # Card carries slug + degradation flags + preview.
    first = body["items"][0]
    assert first["slug"]
    assert first["status"] == "ready"
    assert first["summary_preview"]
    assert first["topics"]


def test_global_catalog_pagination_has_more(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    p1 = client.get("/api/app/episodes", params={"page": 1, "page_size": 2}).json()
    assert [i["title"] for i in p1["items"]] == ["A Two", "B One"]
    assert p1["has_more"] is True and p1["total"] == 3
    p2 = client.get("/api/app/episodes", params={"page": 2, "page_size": 2}).json()
    assert [i["title"] for i in p2["items"]] == ["A One"]
    assert p2["has_more"] is False


def test_podcast_scoped_catalog(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/podcasts/showa/episodes").json()
    assert body["total"] == 2
    assert {i["feed_id"] for i in body["items"]} == {"showa"}
    assert [i["title"] for i in body["items"]] == ["A Two", "A One"]


def test_status_filter(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    pending = client.get("/api/app/episodes", params={"status": "pending"}).json()
    assert [i["title"] for i in pending["items"]] == ["B One"]
    ready = client.get("/api/app/episodes", params={"status": "ready"}).json()
    assert {i["title"] for i in ready["items"]} == {"A One", "A Two"}


def test_empty_corpus_returns_empty_page(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir(parents=True, exist_ok=True)
    body = _client(tmp_path).get("/api/app/episodes").json()
    assert body == {"items": [], "page": 1, "page_size": 20, "total": 0, "has_more": False}


def test_page_size_bounds_enforced(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    assert client.get("/api/app/episodes", params={"page_size": 0}).status_code == 422
    assert client.get("/api/app/episodes", params={"page_size": 101}).status_code == 422
    assert client.get("/api/app/episodes", params={"page": 0}).status_code == 422
