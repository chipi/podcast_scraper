"""Integration tests for GET /api/corpus/*."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


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


def test_corpus_episodes_topic_cluster_only_filter(tmp_path: Path) -> None:
    """``topic_cluster_only`` keeps rows listed on a cluster member for a bridge topic."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    search = tmp_path / "search"
    search.mkdir()
    clusters = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ab",
                "canonical_label": "AB",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:alpha", "episode_ids": ["e_cluster"]},
                    {"topic_id": "topic:beta", "episode_ids": ["e_other"]},
                ],
            }
        ],
    }
    (search / "topic_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")

    (meta / "clustered.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="e_cluster", episode_title="Clustered ep")),
        encoding="utf-8",
    )
    (meta / "clustered.gi.json").write_text("{}", encoding="utf-8")
    bridge = {"identities": [{"id": "topic:alpha", "display_name": "Alpha"}]}
    (meta / "clustered.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")

    (meta / "plain.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="e_plain", episode_title="Plain ep")),
        encoding="utf-8",
    )
    (meta / "plain.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r_all = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert r_all.status_code == 200
    assert len(r_all.json()["items"]) == 2

    r_tc = client.get(
        "/api/corpus/episodes",
        params={"path": str(tmp_path), "limit": 10, "topic_cluster_only": True},
    )
    assert r_tc.status_code == 200
    body = r_tc.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["episode_title"] == "Clustered ep"


def test_corpus_episodes_topic_cluster_only_respects_member_episode_ids(tmp_path: Path) -> None:
    """Two episodes share a clustered bridge topic; only one is listed on the member row."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    search = tmp_path / "search"
    search.mkdir()
    clusters = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ab",
                "canonical_label": "AB",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:alpha", "episode_ids": ["e_listed"]},
                    {"topic_id": "topic:beta", "episode_ids": ["e_other"]},
                ],
            }
        ],
    }
    (search / "topic_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")

    bridge = {"identities": [{"id": "topic:alpha", "display_name": "Alpha"}]}
    (meta / "listed.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="e_listed", episode_title="Listed on cluster member")),
        encoding="utf-8",
    )
    (meta / "listed.gi.json").write_text("{}", encoding="utf-8")
    (meta / "listed.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")

    (meta / "absent.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="e_absent", episode_title="Same topic not on member")),
        encoding="utf-8",
    )
    (meta / "absent.gi.json").write_text("{}", encoding="utf-8")
    (meta / "absent.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r_all = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 10})
    assert r_all.status_code == 200
    titles = {row["episode_title"] for row in r_all.json()["items"]}
    assert titles == {"Listed on cluster member", "Same topic not on member"}

    r_tc = client.get(
        "/api/corpus/episodes",
        params={"path": str(tmp_path), "limit": 10, "topic_cluster_only": True},
    )
    assert r_tc.status_code == 200
    body = r_tc.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["episode_id"] == "e_listed"
    assert body["items"][0]["episode_title"] == "Listed on cluster member"


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


# #678 PR-C6: POST routes coverage gaps surfaced by the test review audit.
# Both routes had only indirect coverage via the larger viewer flow tests;
# these add explicit request/response contract tests.


def test_corpus_resolve_episode_artifacts_returns_resolved_and_missing(tmp_path: Path) -> None:
    """POST /api/corpus/resolve-episode-artifacts maps episode_ids to
    GI/KG/bridge relative paths via catalog scan; unknown ids land in
    missing_episode_ids."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="ep1")),
        encoding="utf-8",
    )
    (meta / "ep1.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r = client.post(
        "/api/corpus/resolve-episode-artifacts",
        json={"path": str(tmp_path), "episode_ids": ["ep1", "never-seen"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == str(tmp_path.resolve())
    assert len(body["resolved"]) == 1
    assert body["resolved"][0]["episode_id"] == "ep1"
    assert body["resolved"][0]["gi_relative_path"].endswith("ep1.gi.json")
    assert body["missing_episode_ids"] == ["never-seen"]


def test_corpus_resolve_episode_artifacts_dedupes_and_rejects_empty(tmp_path: Path) -> None:
    """Duplicate episode_ids in the request are deduped; empty list rejects with 422."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="ep1")),
        encoding="utf-8",
    )
    (meta / "ep1.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    # Duplicate ids
    r = client.post(
        "/api/corpus/resolve-episode-artifacts",
        json={"path": str(tmp_path), "episode_ids": ["ep1", "ep1", "ep1"]},
    )
    assert r.status_code == 200
    assert len(r.json()["resolved"]) == 1

    # Empty list — schema constraint min_length=1 → 422
    r2 = client.post(
        "/api/corpus/resolve-episode-artifacts",
        json={"path": str(tmp_path), "episode_ids": []},
    )
    assert r2.status_code == 422


def test_corpus_node_episodes_returns_empty_for_unknown_node(tmp_path: Path) -> None:
    """POST /api/corpus/node-episodes with an unknown node_id returns an
    empty episodes list (not a 404). The endpoint is content-addressed."""
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "ep1.metadata.json").write_text(
        json.dumps(_episode_doc(episode_id="ep1")),
        encoding="utf-8",
    )
    (meta / "ep1.gi.json").write_text("{}", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)

    r = client.post(
        "/api/corpus/node-episodes",
        json={"path": str(tmp_path), "node_id": "nonexistent::node"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == str(tmp_path.resolve())
    # node_id is normalised through canonical_cil_entity_id; just verify it's a string.
    assert isinstance(body["node_id"], str)
    assert body["episodes"] == []
