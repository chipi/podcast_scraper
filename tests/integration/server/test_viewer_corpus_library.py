"""Integration tests for GET /api/corpus/* (RFC-067)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]


def _episode_doc(
    *,
    feed_id: str = "myfeed",
    feed_title: str = "My Show",
    episode_id: str = "ep1",
    episode_title: str = "Hello",
    published: str = "2024-03-10T00:00:00",
) -> dict:
    return {
        "feed": {"feed_id": feed_id, "title": feed_title},
        "episode": {
            "episode_id": episode_id,
            "title": episode_title,
            "published_date": published,
        },
        "summary": {
            "title": "Sum",
            "bullets": ["a", "b"],
            "short_summary": "Full paragraph summary.",
        },
    }


def test_corpus_feeds_and_episodes_flat_layout(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(_episode_doc()),
        encoding="utf-8",
    )
    (meta / "one.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    fr = client.get("/api/corpus/feeds", params={"path": str(tmp_path)})
    assert fr.status_code == 200
    body = fr.json()
    assert body["path"] == str(tmp_path.resolve())
    assert len(body["feeds"]) == 1
    assert body["feeds"][0]["feed_id"] == "myfeed"
    assert body["feeds"][0]["episode_count"] == 1

    doc = json.loads((meta / "one.metadata.json").read_text(encoding="utf-8"))
    doc["feed"]["image_url"] = "https://cdn.example/feed-art.png"
    doc["episode"]["image_url"] = "https://cdn.example/ep-art.png"
    doc["episode"]["duration_seconds"] = 90
    doc["episode"]["episode_number"] = 3
    (meta / "one.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    fr2 = client.get("/api/corpus/feeds", params={"path": str(tmp_path)})
    assert fr2.status_code == 200
    assert fr2.json()["feeds"][0]["image_url"] == "https://cdn.example/feed-art.png"

    er = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert er.status_code == 200
    ep = er.json()
    assert len(ep["items"]) == 1
    assert ep["items"][0]["episode_title"] == "Hello"
    assert ep["items"][0]["feed_display_title"] == "My Show"
    assert ep["items"][0]["topics"] == ["a", "b"]
    assert ep["items"][0]["metadata_relative_path"].endswith("one.metadata.json")
    item0 = ep["items"][0]
    assert item0["feed_image_url"] == "https://cdn.example/feed-art.png"
    assert item0["episode_image_url"] == "https://cdn.example/ep-art.png"
    assert item0["duration_seconds"] == 90
    assert item0["episode_number"] == 3
    assert item0["summary_preview"] == "Sum — a · b"
    assert item0["summary_title"] == "Sum"
    assert item0["summary_bullets_preview"] == ["a", "b"]

    rel = ep["items"][0]["metadata_relative_path"]
    dr = client.get(
        "/api/corpus/episodes/detail",
        params={"path": str(tmp_path), "metadata_relpath": rel},
    )
    assert dr.status_code == 200
    detail = dr.json()
    assert detail["summary_bullets"] == ["a", "b"]
    assert detail["summary_text"] == "Full paragraph summary."
    assert detail["has_gi"] is True
    assert detail["has_kg"] is False
    assert detail["has_bridge"] is False
    assert detail["bridge_relative_path"].endswith("one.bridge.json")
    assert detail["feed_image_url"] == "https://cdn.example/feed-art.png"
    assert detail["episode_image_url"] == "https://cdn.example/ep-art.png"
    assert detail["duration_seconds"] == 90
    assert detail["episode_number"] == 3


def test_corpus_feeds_and_episodes_include_rss_and_description(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    doc = _episode_doc()
    doc["feed"]["url"] = "https://pod.example/feed.xml"
    doc["feed"]["description"] = "Weekly tech chat"
    (meta / "one.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    fr = client.get("/api/corpus/feeds", params={"path": str(tmp_path)})
    assert fr.status_code == 200
    f0 = fr.json()["feeds"][0]
    assert f0["rss_url"] == "https://pod.example/feed.xml"
    assert f0["description"] == "Weekly tech chat"

    er = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert er.status_code == 200
    item = er.json()["items"][0]
    assert item["feed_rss_url"] == "https://pod.example/feed.xml"
    assert item["feed_description"] == "Weekly tech chat"

    dr = client.get(
        "/api/corpus/episodes/detail",
        params={
            "path": str(tmp_path),
            "metadata_relpath": "metadata/one.metadata.json",
        },
    )
    assert dr.status_code == 200
    det = dr.json()
    assert det["feed_rss_url"] == "https://pod.example/feed.xml"
    assert det["feed_description"] == "Weekly tech chat"


def test_corpus_episodes_feed_display_title_from_sibling(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "a.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="e1", episode_title="First")),
        encoding="utf-8",
    )
    bare = _episode_doc(episode_id="e2", episode_title="Second")
    bare["feed"] = {"feed_id": "myfeed"}
    (meta / "b.metadata.json").write_text(json.dumps(bare), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert r.status_code == 200
    second = next(x for x in r.json()["items"] if x["episode_title"] == "Second")
    assert second["feed_display_title"] == "My Show"


def test_corpus_episodes_topic_q_filters(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "match.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="m1", episode_title="Match ep")),
        encoding="utf-8",
    )
    other = _episode_doc(episode_id="o1", episode_title="Other ep")
    other["summary"] = {"title": "Other headline", "bullets": ["unique-bbb-only", "ccc"]}
    (meta / "other.metadata.json").write_text(json.dumps(other), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r_all = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert r_all.status_code == 200
    assert len(r_all.json()["items"]) == 2

    r_topic = client.get(
        "/api/corpus/episodes",
        params={"path": str(tmp_path), "limit": 10, "topic_q": "unique-bbb"},
    )
    assert r_topic.status_code == 200
    body = r_topic.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["episode_id"] == "o1"


def test_corpus_episodes_pagination_cursor(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    for i in range(3):
        month = i + 1
        (meta / f"e{i}.metadata.json").write_text(
            json.dumps(
                _episode_doc(
                    episode_id=f"id{i}",
                    episode_title=f"T{i}",
                    published=f"2024-{month:02d}-15T00:00:00",
                ),
            ),
            encoding="utf-8",
        )

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r1 = client.get(
        "/api/corpus/episodes",
        params={"path": str(tmp_path), "limit": 2},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert len(b1["items"]) == 2
    assert b1["next_cursor"]

    r2 = client.get(
        "/api/corpus/episodes",
        params={"path": str(tmp_path), "limit": 2, "cursor": b1["next_cursor"]},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert len(b2["items"]) == 1
    assert b2["next_cursor"] is None


def test_corpus_detail_rejects_traversal(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episodes/detail",
        params={"path": str(tmp_path), "metadata_relpath": "../outside.metadata.json"},
    )
    assert r.status_code == 400


def test_corpus_similar_no_index_returns_soft_error(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    doc = _episode_doc()
    doc["summary"] = {
        "title": "Summary headline",
        "bullets": ["First longer bullet point", "Second point here"],
    }
    (meta / "one.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    er = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    rel = er.json()["items"][0]["metadata_relative_path"]
    sr = client.get(
        "/api/corpus/episodes/similar",
        params={"path": str(tmp_path), "metadata_relpath": rel},
    )
    assert sr.status_code == 200
    body = sr.json()
    assert body["error"] == "no_index"
    assert body["items"] == []
    assert body["source_metadata_relative_path"] == rel
    assert "Summary headline" in body["query_used"]


def test_corpus_similar_insufficient_text(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    short = {
        "feed": {"feed_id": "myfeed", "title": "My Show"},
        "episode": {
            "episode_id": "ep1",
            "title": "Hi",
            "published_date": "2024-03-10T00:00:00",
        },
    }
    (meta / "one.metadata.json").write_text(json.dumps(short), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    sr = client.get(
        "/api/corpus/episodes/similar",
        params={"path": str(tmp_path), "metadata_relpath": "metadata/one.metadata.json"},
    )
    assert sr.status_code == 200
    assert sr.json()["error"] == "insufficient_text"


def test_corpus_binary_serves_artwork_under_allowlisted_prefix(tmp_path: Path) -> None:
    art_rel = ".podcast_scraper/corpus-art/sha256/de/ad/deadbeef.jpg"
    art_file = tmp_path / art_rel.replace("/", "/")
    art_file.parent.mkdir(parents=True, exist_ok=True)
    art_file.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/binary",
        params={"path": str(tmp_path), "relpath": art_rel},
    )
    assert r.status_code == 200
    assert r.content.startswith(b"\xff\xd8\xff")


def test_corpus_binary_rejects_non_artwork_path(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/binary",
        params={"path": str(tmp_path), "relpath": "metadata/secret.jpg"},
    )
    assert r.status_code == 400


def test_corpus_binary_rejects_path_traversal(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    bad = ".podcast_scraper/corpus-art/sha256/../metadata/x.jpg"
    r = client.get(
        "/api/corpus/binary",
        params={"path": str(tmp_path), "relpath": bad},
    )
    assert r.status_code == 400


def test_corpus_episodes_includes_verified_local_artwork_paths(tmp_path: Path) -> None:
    art_rel = ".podcast_scraper/corpus-art/sha256/ab/cd/abc123.jpg"
    art_file = tmp_path / art_rel.replace("/", "/")
    art_file.parent.mkdir(parents=True, exist_ok=True)
    art_file.write_bytes(b"x")

    meta = tmp_path / "metadata"
    meta.mkdir()
    doc = _episode_doc()
    doc["feed"]["image_url"] = "https://cdn.example/feed.png"
    doc["feed"]["image_local_relpath"] = art_rel
    doc["episode"]["image_local_relpath"] = art_rel
    (meta / "one.metadata.json").write_text(json.dumps(doc), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    er = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert er.status_code == 200
    item = er.json()["items"][0]
    assert item["feed_image_local_relpath"] == art_rel
    assert item["episode_image_local_relpath"] == art_rel

    rel = item["metadata_relative_path"]
    dr = client.get(
        "/api/corpus/episodes/detail",
        params={"path": str(tmp_path), "metadata_relpath": rel},
    )
    assert dr.status_code == 200
    detail = dr.json()
    assert detail["feed_image_local_relpath"] == art_rel
    assert detail["episode_image_local_relpath"] == art_rel
