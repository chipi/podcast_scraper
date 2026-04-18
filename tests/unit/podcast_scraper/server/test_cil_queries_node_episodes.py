"""Unit tests for RFC-076 bridge node -> episode listing (``cil_queries``)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.cil_queries import episodes_for_bridge_node_id


def _write_bundle(meta: Path, stem: str, episode_id: str, topic: str) -> None:
    meta.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "identities": [
            {
                "id": topic,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "T",
                "aliases": [],
            },
        ],
    }
    gi = {"episode_id": episode_id, "nodes": [], "edges": []}
    kg = {
        "nodes": [{"id": "ep", "type": "Episode", "properties": {}}],
        "edges": [],
    }
    (meta / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (meta / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (meta / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def test_episodes_for_bridge_node_id_two_matches(tmp_path: Path) -> None:
    root = tmp_path
    meta = root / "metadata"
    _write_bundle(meta, "a", "episode:a", "topic:shared")
    _write_bundle(meta, "b", "episode:b", "topic:shared")
    anchor = str(root.resolve())
    rows, truncated, total = episodes_for_bridge_node_id(
        anchor,
        anchor,
        "g:k:topic:shared",
        max_episodes=None,
    )
    assert not truncated
    assert total is None
    assert len(rows) == 2
    paths = {r["gi_relative_path"] for r in rows}
    assert paths == {"metadata/a.gi.json", "metadata/b.gi.json"}


def test_episodes_for_bridge_node_id_max_cap(tmp_path: Path) -> None:
    root = tmp_path
    meta = root / "metadata"
    _write_bundle(meta, "a", "episode:a", "topic:x")
    _write_bundle(meta, "b", "episode:b", "topic:x")
    anchor = str(root.resolve())
    rows, truncated, total = episodes_for_bridge_node_id(
        anchor,
        anchor,
        "topic:x",
        max_episodes=1,
    )
    assert truncated
    assert total == 2
    assert len(rows) == 1
    assert rows[0]["gi_relative_path"] == "metadata/a.gi.json"


def test_episodes_for_bridge_node_id_unknown_topic(tmp_path: Path) -> None:
    root = tmp_path
    meta = root / "metadata"
    _write_bundle(meta, "a", "episode:a", "topic:only")
    anchor = str(root.resolve())
    rows, truncated, total = episodes_for_bridge_node_id(
        anchor,
        anchor,
        "topic:missing",
        max_episodes=None,
    )
    assert rows == []
    assert not truncated
    assert total is None
