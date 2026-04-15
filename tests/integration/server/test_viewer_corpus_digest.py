"""Integration tests for GET /api/corpus/digest (RFC-068)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _episode_doc(
    *,
    feed_id: str = "feed_a",
    episode_id: str = "ep1",
    episode_title: str = "Hello",
    published: str,
) -> dict:
    return {
        "feed": {"feed_id": feed_id, "title": "Show"},
        "episode": {
            "episode_id": episode_id,
            "title": episode_title,
            "published_date": published,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b", "c"]},
    }


def test_digest_compact_diversity_and_omit_topics(tmp_path: Path) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    for i in range(3):
        (meta / f"a{i}.metadata.json").write_text(
            json.dumps(
                _episode_doc(
                    feed_id="feed_a",
                    episode_id=f"a{i}",
                    episode_title=f"A{i}",
                    published=f"{today}T12:00:00Z",
                ),
            ),
            encoding="utf-8",
        )
    for i in range(3):
        (meta / f"b{i}.metadata.json").write_text(
            json.dumps(
                _episode_doc(
                    feed_id="feed_b",
                    episode_id=f"b{i}",
                    episode_title=f"B{i}",
                    published=f"{today}T12:00:00Z",
                ),
            ),
            encoding="utf-8",
        )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "compact": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["compact"] is True
    assert body["window"] == "24h"
    assert body["topics"] == []
    rows = body["rows"]
    feed_a = sum(1 for row in rows if row["feed_id"] == "feed_a")
    feed_b = sum(1 for row in rows if row["feed_id"] == "feed_b")
    assert feed_a <= 2
    assert feed_b <= 2
    assert len(rows) == 4
    for row in rows:
        assert len(row["summary_bullets_preview"]) <= 4
        assert len(row["summary_bullet_graph_topic_ids"]) == len(row["summary_bullets_preview"])


def test_digest_full_window_and_topics_no_index(tmp_path: Path) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(
            _episode_doc(
                published=f"{today}T12:00:00Z",
            ),
        ),
        encoding="utf-8",
    )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "7d"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["window"] == "7d"
    assert len(body["rows"]) == 1
    assert body["rows"][0]["episode_title"] == "Hello"
    assert body["topics"] == []
    assert body.get("topics_unavailable_reason") == "no_index"


def test_digest_window_1mo_ok(tmp_path: Path) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(
            _episode_doc(
                published=f"{today}T12:00:00Z",
            ),
        ),
        encoding="utf-8",
    )
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "1mo"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["window"] == "1mo"
    assert "window_start_utc" in body
    assert "window_end_utc" in body


def test_digest_since_requires_param(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "since"},
    )
    assert r.status_code == 400


def test_digest_rows_include_visual_metadata(tmp_path: Path) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    d = _episode_doc(published=f"{today}T12:00:00Z")
    d["feed"]["image_url"] = "https://cdn.example/f.png"
    d["episode"]["image_url"] = "https://cdn.example/e.png"
    d["episode"]["duration_seconds"] = 61
    d["episode"]["episode_number"] = 5
    (meta / "one.metadata.json").write_text(json.dumps(d), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "7d"},
    )
    assert r.status_code == 200
    row = r.json()["rows"][0]
    assert row["feed_image_url"] == "https://cdn.example/f.png"
    assert row["episode_image_url"] == "https://cdn.example/e.png"
    assert row["duration_seconds"] == 61
    assert row["episode_number"] == 5
    assert row["feed_display_title"] == "Show"
    assert row["summary_preview"] == "Sum — a · b"
    assert len(row["summary_bullet_graph_topic_ids"]) == len(row["summary_bullets_preview"])


def test_digest_feed_display_title_from_sibling_when_feed_title_omitted(
    tmp_path: Path,
) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.metadata.json").write_text(
        json.dumps(
            _episode_doc(episode_id="ep1", published=f"{today}T12:00:00Z"),
        ),
        encoding="utf-8",
    )
    bare = _episode_doc(
        episode_id="ep2",
        episode_title="Other ep",
        published=f"{today}T12:05:00Z",
    )
    bare["feed"] = {"feed_id": "feed_a"}
    (meta / "b.metadata.json").write_text(json.dumps(bare), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/digest", params={"path": str(tmp_path), "window": "7d"})
    assert r.status_code == 200
    rows = r.json()["rows"]
    other = next(x for x in rows if x["episode_title"] == "Other ep")
    assert other["feed_display_title"] == "Show"


def test_digest_row_includes_feed_rss_and_description(tmp_path: Path) -> None:
    today = date.today().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    d = _episode_doc(published=f"{today}T12:00:00Z")
    d["feed"]["url"] = "https://feeds.example/pod.xml"
    d["feed"]["description"] = "Digest feed desc"
    (meta / "one.metadata.json").write_text(json.dumps(d), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "7d"},
    )
    assert r.status_code == 200
    row = r.json()["rows"][0]
    assert row["feed_rss_url"] == "https://feeds.example/pod.xml"
    assert row["feed_description"] == "Digest feed desc"
