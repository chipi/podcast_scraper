"""Unit tests for RFC-072 ``cil_queries`` (GitHub #527)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from podcast_scraper.server import cil_queries


def _write_bundle(
    directory: Path,
    stem: str,
    *,
    episode_id: str,
    publish_date: str,
    person: str,
    topic: str,
    insight_id: str,
    quote_id: str,
    insight_text: str,
    insight_type: str = "claim",
    position_hint: float = 0.5,
    metadata_episode_title: str | None = None,
    metadata_feed_title: str | None = None,
    metadata_episode_number: int | None = None,
    metadata_episode_image_url: str | None = None,
    metadata_feed_image_url: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "identities": [
            {
                "id": person,
                "type": "person",
                "sources": {"gi": True, "kg": True},
                "display_name": "P",
                "aliases": [],
            },
            {
                "id": topic,
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "T",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": insight_id,
                "type": "Insight",
                "properties": {
                    "text": insight_text,
                    "insight_type": insight_type,
                    "position_hint": position_hint,
                },
            },
            {"id": quote_id, "type": "Quote", "properties": {"text": "quote body"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": quote_id, "to": person},
            {"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id},
            {"type": "ABOUT", "from": insight_id, "to": topic},
        ],
    }
    kg = {
        "nodes": [
            {
                "id": "kg:episode:x",
                "type": "Episode",
                "properties": {"publish_date": publish_date},
            }
        ],
        "edges": [],
    }
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    if (
        metadata_episode_title is not None
        or metadata_feed_title is not None
        or metadata_episode_number is not None
        or metadata_episode_image_url is not None
        or metadata_feed_image_url is not None
    ):
        meta_doc: dict[str, Any] = {}
        feed_meta: dict[str, Any] = {}
        if metadata_feed_title is not None:
            feed_meta["title"] = metadata_feed_title
        if metadata_feed_image_url is not None:
            feed_meta["image_url"] = metadata_feed_image_url
        if feed_meta:
            meta_doc["feed"] = feed_meta
        ep_meta: dict[str, Any] = {}
        if metadata_episode_title is not None:
            ep_meta["title"] = metadata_episode_title
        if metadata_episode_number is not None:
            ep_meta["episode_number"] = metadata_episode_number
        if metadata_episode_image_url is not None:
            ep_meta["image_url"] = metadata_episode_image_url
        if ep_meta:
            meta_doc["episode"] = ep_meta
        (directory / f"{stem}.metadata.json").write_text(
            json.dumps(meta_doc),
            encoding="utf-8",
        )


def test_position_arc_default_claim_filter(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-01-15",
        person="person:alice",
        topic="topic:climate",
        insight_id="insight-a",
        quote_id="quote-a",
        insight_text="We should act",
        insight_type="claim",
        metadata_episode_title="Episode A human title",
        metadata_feed_title="Climate Cast",
    )
    _write_bundle(
        meta,
        "b",
        episode_id="episode:b",
        publish_date="2024-02-01",
        person="person:alice",
        topic="topic:climate",
        insight_id="insight-b",
        quote_id="quote-b",
        insight_text="Just wondering",
        insight_type="observation",
    )
    root = str(tmp_path)
    arc = cil_queries.position_arc(root, root, "person:alice", "topic:climate")
    assert len(arc) == 1
    assert arc[0]["episode_id"] == "episode:a"
    assert arc[0]["episode_title"] == "Episode A human title"
    assert arc[0]["feed_title"] == "Climate Cast"
    assert len(arc[0]["insights"]) == 1
    assert arc[0]["insights"][0]["properties"]["text"] == "We should act"

    arc_all = cil_queries.position_arc(
        root,
        root,
        "person:alice",
        "topic:climate",
        insight_types=None,
    )
    assert len(arc_all) == 2


def test_person_profile_and_person_topics(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-01-10",
        person="person:bob",
        topic="topic:ai",
        insight_id="i1",
        quote_id="q1",
        insight_text="AI is big",
    )
    root = str(tmp_path)
    brief = cil_queries.person_profile(root, root, "person:bob")
    assert brief["person_id"] == "person:bob"
    assert "topic:ai" in brief["topics"]
    assert len(brief["quotes"]) == 1
    topics = cil_queries.person_topic_ids(root, root, "person:bob")
    assert topics == ["topic:ai"]


def test_topic_timeline_accepts_viewer_g_prefixed_topic_id(tmp_path: Path) -> None:
    """Merged GI+KG graph uses ``g:`` prefix on GI ids; CIL reads raw JSON."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "g:topic:tax")
    assert len(tl) == 1
    assert tl[0]["episode_id"] == "episode:x"
    assert cil_queries.topic_person_ids(root, root, "g:topic:tax") == ["person:carol"]


def test_topic_timeline_resolves_episode_title_from_metadata(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
        metadata_episode_title="March tax episode",
        metadata_feed_title="Tax Show",
        metadata_episode_number=42,
        metadata_episode_image_url="https://example.com/art.png",
        metadata_feed_image_url="https://example.com/feed.png",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:tax")
    assert len(tl) == 1
    assert tl[0]["episode_title"] == "March tax episode"
    assert tl[0]["feed_title"] == "Tax Show"
    assert tl[0]["episode_number"] == 42
    assert tl[0]["episode_image_url"] == "https://example.com/art.png"
    assert tl[0]["feed_image_url"] == "https://example.com/feed.png"


def test_topic_timeline_metadata_feed_without_episode_title(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
        metadata_feed_title="Only Feed",
        metadata_episode_number=7,
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:tax")
    assert tl[0]["episode_title"] is None
    assert tl[0]["feed_title"] == "Only Feed"
    assert tl[0]["episode_number"] == 7


def test_topic_timeline_and_topic_persons(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:tax")
    assert len(tl) == 1
    assert tl[0]["episode_id"] == "episode:x"
    persons = cil_queries.topic_person_ids(root, root, "topic:tax")
    assert persons == ["person:carol"]


def test_skips_incomplete_triple(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "orphan.bridge.json").write_text(
        '{"schema_version":"1.0","identities":[]}', encoding="utf-8"
    )
    root = str(tmp_path)
    assert cil_queries.position_arc(root, root, "person:x", "topic:y") == []


def test_skips_corrupt_gi_json(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "ok",
        episode_id="episode:z",
        publish_date="2024-01-01",
        person="person:z",
        topic="topic:z",
        insight_id="iz",
        quote_id="qz",
        insight_text="Z",
    )
    gi_path = meta / "ok.gi.json"
    gi_path.write_text("{not json", encoding="utf-8")
    root = str(tmp_path)
    assert cil_queries.position_arc(root, root, "person:z", "topic:z") == []


def test_iter_rejects_root_outside_anchor(tmp_path: Path) -> None:
    anchor = str(tmp_path / "out")
    outside = str(tmp_path / "other")
    assert list(cil_queries.iter_cil_episode_bundles(outside, anchor)) == []


def test_topic_timeline_merged_matches_single_topic(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
    )
    root = str(tmp_path)
    single = cil_queries.topic_timeline(root, root, "topic:tax")
    merged = cil_queries.topic_timeline_merged(root, root, ["topic:tax"])
    assert merged == single


def test_topic_timeline_merged_empty_ids(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert cil_queries.topic_timeline_merged(root, root, []) == []
    assert cil_queries.topic_timeline_merged(root, root, ["", "  "]) == []


def test_topic_timeline_merged_matches_prefixed_about_to(tmp_path: Path) -> None:
    """GI may store ``g:topic:…`` on ABOUT edges; bridge ids stay ``topic:…``."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
    )
    gi_path = meta / "x.gi.json"
    gi = json.loads(gi_path.read_text(encoding="utf-8"))
    for e in gi.get("edges") or []:
        if isinstance(e, dict) and str(e.get("type")) == "ABOUT":
            e["to"] = "g:topic:tax"
    gi_path.write_text(json.dumps(gi), encoding="utf-8")

    root = str(tmp_path)
    single = cil_queries.topic_timeline(root, root, "topic:tax")
    merged = cil_queries.topic_timeline_merged(root, root, ["topic:tax"])
    assert len(single) == 1
    assert merged == single


def test_topic_timeline_merged_accepts_k_and_gk_layer_topic_ids(tmp_path: Path) -> None:
    """Merged graph uses ``k:topic:…`` / ``g:k:topic:…``; bridge identities stay ``topic:…``."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "x",
        episode_id="episode:x",
        publish_date="2024-03-01",
        person="person:carol",
        topic="topic:tax",
        insight_id="ix",
        quote_id="qx",
        insight_text="Tax is complex",
    )
    root = str(tmp_path)
    baseline = cil_queries.topic_timeline_merged(root, root, ["topic:tax"])
    assert cil_queries.topic_timeline_merged(root, root, ["k:topic:tax"]) == baseline
    assert cil_queries.topic_timeline_merged(root, root, ["g:k:topic:tax"]) == baseline
    assert cil_queries.topic_timeline_merged(root, root, ["g:topic:tax"]) == baseline


def test_topic_timeline_insight_type_filter(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "t",
        episode_id="episode:t",
        publish_date="2024-01-01",
        person="person:p",
        topic="topic:t",
        insight_id="ins",
        quote_id="quo",
        insight_text="Claim text",
        insight_type="claim",
    )
    _write_bundle(
        meta,
        "u",
        episode_id="episode:u",
        publish_date="2024-02-01",
        person="person:p2",
        topic="topic:t2",
        insight_id="ins2",
        quote_id="quo2",
        insight_text="Observation",
        insight_type="observation",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:t", insight_types=("claim",))
    assert len(tl) == 1
    assert tl[0]["episode_id"] == "episode:t"


def test_topic_person_ids_skips_non_person_spoken_by(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    directory = meta
    directory.mkdir(parents=True, exist_ok=True)
    episode_id = "episode:sp"
    bridge = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "identities": [
            {
                "id": "topic:tx",
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "Tx",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "ins_x",
                "type": "Insight",
                "properties": {"text": "x", "insight_type": "claim"},
            },
            {
                "id": "quo_x",
                "type": "Quote",
                "properties": {"char_start": 0, "char_end": 10},
            },
        ],
        "edges": [
            {"type": "ABOUT", "from": "ins_x", "to": "topic:tx"},
            {"type": "SUPPORTED_BY", "from": "ins_x", "to": "quo_x"},
            {"type": "SPOKEN_BY", "from": "quo_x", "to": "host:anon"},
        ],
    }
    kg = {
        "nodes": [
            {
                "id": "kg:ep",
                "type": "Episode",
                "properties": {"publish_date": "2024-01-01"},
            }
        ],
        "edges": [],
    }
    stem = "edge"
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    root = str(tmp_path)
    assert cil_queries.topic_person_ids(root, root, "topic:tx") == []


def test_position_arc_skips_empty_episode_id(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    directory = meta
    directory.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": "",
        "identities": [
            {
                "id": "person:a",
                "type": "person",
                "sources": {"gi": True, "kg": True},
                "display_name": "A",
                "aliases": [],
            },
            {
                "id": "topic:t",
                "type": "topic",
                "sources": {"gi": True, "kg": True},
                "display_name": "T",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": "",
        "nodes": [
            {
                "id": "i",
                "type": "Insight",
                "properties": {"text": "x", "insight_type": "claim"},
            },
            {"id": "q", "type": "Quote", "properties": {}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "q", "to": "person:a"},
            {"type": "SUPPORTED_BY", "from": "i", "to": "q"},
            {"type": "ABOUT", "from": "i", "to": "topic:t"},
        ],
    }
    kg: dict[str, Any] = {"nodes": [], "edges": []}
    for stem in ("bad",):
        (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
        (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
        (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    root = str(tmp_path)
    assert cil_queries.position_arc(root, root, "person:a", "topic:t") == []


def test_topic_timeline_kg_only_topic_finds_gi_insights_via_bridge(
    tmp_path: Path,
) -> None:
    """KG-only topic in bridge + GI ABOUT edges targeting GI topic ids.

    ``topic_clusters.json`` clusters use KG-sourced short slugs (e.g.
    ``topic:economic-struggles``).  GI ABOUT edges target GI-sourced
    sentence-style ids.  The bridge lists both.  The timeline must expand
    the requested KG topic id to include GI topic ids from the same bridge
    so insights are found.
    """
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": "ep:iran",
        "identities": [
            {
                "id": "topic:economic-struggles",
                "type": "topic",
                "sources": {"gi": False, "kg": True},
                "display_name": "Economic Struggles",
                "aliases": [],
            },
            {
                "id": "topic:iran-economic-pain-long-slug",
                "type": "topic",
                "sources": {"gi": True, "kg": False},
                "display_name": "Iran economic pain (GI)",
                "aliases": [],
            },
            {
                "id": "person:ali",
                "type": "person",
                "sources": {"gi": False, "kg": True},
                "display_name": "Ali",
                "aliases": [],
            },
        ],
    }
    gi = {
        "episode_id": "ep:iran",
        "nodes": [
            {
                "id": "ins:a",
                "type": "Insight",
                "properties": {
                    "text": "Sanctions hurt",
                    "insight_type": "claim",
                    "position_hint": 0.3,
                },
            },
        ],
        "edges": [
            {
                "type": "ABOUT",
                "from": "ins:a",
                "to": "topic:iran-economic-pain-long-slug",
            },
        ],
    }
    kg = {
        "nodes": [
            {
                "id": "kg:episode:iran",
                "type": "Episode",
                "properties": {"publish_date": "2026-02-07"},
            }
        ],
        "edges": [],
    }
    (meta / "iran.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (meta / "iran.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (meta / "iran.kg.json").write_text(json.dumps(kg), encoding="utf-8")

    root = str(tmp_path)

    # Requesting the KG-only topic id should find insights via GI ABOUT
    tl_single = cil_queries.topic_timeline(root, root, "topic:economic-struggles")
    assert len(tl_single) == 1, f"expected 1 episode, got {len(tl_single)}"
    assert tl_single[0]["episode_id"] == "ep:iran"
    assert len(tl_single[0]["insights"]) == 1

    # Merged endpoint too
    tl_merged = cil_queries.topic_timeline_merged(root, root, ["topic:economic-struggles"])
    assert len(tl_merged) == 1
    assert tl_merged[0]["insights"][0]["id"] == "ins:a"


def test_skips_when_kg_missing(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    bridge = {
        "schema_version": "1.0",
        "episode_id": "e",
        "identities": [
            {
                "id": "person:x",
                "type": "person",
                "sources": {"gi": True, "kg": True},
                "display_name": "P",
                "aliases": [],
            },
        ],
    }
    gi = {"episode_id": "e", "nodes": [], "edges": []}
    (meta / "solo.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (meta / "solo.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    root = str(tmp_path)
    assert cil_queries.person_profile(root, root, "person:x")["topics"] == {}
