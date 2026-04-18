"""Integration tests for CIL topic pills on digest and library APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _write_minimal_corpus(
    root: Path,
    *,
    episode_id: str = "ep-1",
    feed_id: str = "feed-a",
) -> None:
    meta = {
        "feed": {"feed_id": feed_id, "title": "Feed A"},
        "episode": {
            "episode_id": episode_id,
            "title": "Episode One",
            "published_date": "2024-06-15T12:00:00Z",
        },
        "summary": {"title": "Sum", "bullets": ["Bullet one"]},
    }
    mdir = root / "metadata"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "one.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (mdir / "one.gi.json").write_text("{}", encoding="utf-8")
    (mdir / "one.kg.json").write_text("{}", encoding="utf-8")
    bridge = {
        "identities": [
            {"id": "topic:alpha", "display_name": "Alpha"},
            {"id": "topic:beta", "display_name": "Beta"},
        ]
    }
    (mdir / "one.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    search = root / "search"
    search.mkdir(parents=True, exist_ok=True)
    clusters = {
        "schema_version": "2",
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ab",
                "canonical_label": "AB",
                "member_count": 2,
                "members": [
                    {"topic_id": "topic:alpha", "episode_ids": [episode_id]},
                    {"topic_id": "topic:gamma", "episode_ids": [episode_id]},
                ],
            }
        ],
    }
    (search / "topic_clusters.json").write_text(json.dumps(clusters), encoding="utf-8")


def test_digest_includes_cil_digest_topics(tmp_path: Path) -> None:
    _write_minimal_corpus(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get("/api/corpus/digest", params={"path": str(tmp_path), "window": "all"})
    assert r.status_code == 200
    body = r.json()
    rows = body.get("rows") or []
    assert len(rows) >= 1
    row0 = rows[0]
    pills = row0.get("cil_digest_topics") or []
    assert isinstance(pills, list)
    assert len(pills) >= 1
    alpha = next((p for p in pills if p.get("topic_id") == "topic:alpha"), None)
    assert alpha is not None
    assert alpha.get("in_topic_cluster") is True
    assert alpha.get("label") == "Alpha"


def test_episodes_detail_includes_cil_digest_topics(tmp_path: Path) -> None:
    _write_minimal_corpus(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/episodes/detail",
        params={"path": str(tmp_path), "metadata_relpath": "metadata/one.metadata.json"},
    )
    assert r.status_code == 200
    body = r.json()
    pills = body.get("cil_digest_topics") or []
    assert any(p.get("topic_id") == "topic:alpha" for p in pills)


def test_episodes_list_omits_cil_digest_topics(tmp_path: Path) -> None:
    """List rows do not load bridge/CIL pills (detail + digest only)."""
    _write_minimal_corpus(tmp_path)
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r_all = client.get("/api/corpus/episodes", params={"path": str(tmp_path), "limit": 50})
    assert r_all.status_code == 200
    items = r_all.json().get("items") or []
    assert len(items) >= 1
    for it in items:
        assert it.get("cil_digest_topics") == []
