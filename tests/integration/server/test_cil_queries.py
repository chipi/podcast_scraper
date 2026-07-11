"""Unit tests for ``cil_queries`` (GitHub #527)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import cil_queries

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm] or viewer HTTP
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


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
    metadata_summary_title: str | None = None,
    metadata_summary_text: str | None = None,
    # RFC-097 v3.0 chunk-4 additions — typed MENTIONS family + Org / Podcast
    # nodes. ``mention_org`` adds a ``MENTIONS_ORG`` edge from the insight to
    # a synthetic Organization node and emits the Organization in kg.json.
    # ``mention_person_typed`` emits a ``MENTIONS_PERSON`` edge (the typed
    # variant — by default ``_write_bundle`` only emits SPOKEN_BY for the
    # speaker, not a typed mention). ``podcast_id`` adds a Podcast node +
    # HAS_EPISODE edge to kg.json.
    mention_org: str | None = None,
    mention_person_typed: str | None = None,
    podcast_id: str | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "3.0",
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
    gi_nodes: list = [
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
    ]
    gi_edges: list = [
        {"type": "SPOKEN_BY", "from": quote_id, "to": person},
        {"type": "SUPPORTED_BY", "from": insight_id, "to": quote_id},
        {"type": "ABOUT", "from": insight_id, "to": topic},
    ]
    # Typed MENTIONS edges (RFC-097 v3.0). Caller opt-in.
    if mention_person_typed is not None:
        gi_edges.append({"type": "MENTIONS_PERSON", "from": insight_id, "to": mention_person_typed})
    if mention_org is not None:
        gi_edges.append({"type": "MENTIONS_ORG", "from": insight_id, "to": mention_org})

    gi = {
        "schema_version": "3.0",
        "episode_id": episode_id,
        "nodes": gi_nodes,
        "edges": gi_edges,
    }
    kg_nodes: list = [
        {
            "id": "kg:episode:x",
            "type": "Episode",
            "properties": {"publish_date": publish_date},
        }
    ]
    kg_edges: list = []
    # RFC-097 v3.0: Person / Organization / Podcast as first-class KG nodes.
    if mention_person_typed is not None:
        kg_nodes.append(
            {
                "id": mention_person_typed,
                "type": "Person",
                "properties": {"name": "Mentioned Person"},
            }
        )
    if mention_org is not None:
        kg_nodes.append(
            {
                "id": mention_org,
                "type": "Organization",
                "properties": {"name": "Mentioned Org"},
            }
        )
    if podcast_id is not None:
        kg_nodes.append(
            {
                "id": podcast_id,
                "type": "Podcast",
                "properties": {"title": "Test Podcast"},
            }
        )
        kg_edges.append({"type": "HAS_EPISODE", "from": podcast_id, "to": "kg:episode:x"})

    kg = {"schema_version": "2.0", "nodes": kg_nodes, "edges": kg_edges}
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    if (
        metadata_episode_title is not None
        or metadata_feed_title is not None
        or metadata_episode_number is not None
        or metadata_episode_image_url is not None
        or metadata_feed_image_url is not None
        or metadata_summary_title is not None
        or metadata_summary_text is not None
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
        summary_meta: dict[str, Any] = {}
        if metadata_summary_title is not None:
            summary_meta["title"] = metadata_summary_title
        if metadata_summary_text is not None:
            summary_meta["raw_text"] = metadata_summary_text
        if summary_meta:
            meta_doc["summary"] = summary_meta
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


def test_person_profile_aggregates_recurring_guest_across_episodes(tmp_path: Path) -> None:
    """#909: a recurring guest (person:liam) appearing in multiple episodes resolves to ONE
    canonical person whose quotes + topics aggregate from ALL their episodes — the
    cross-episode identity payoff unlocked once #875/#876 give corpus-wide SPOKEN_BY."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "ep1",
        episode_id="episode:ep1",
        publish_date="2024-01-10",
        person="person:liam",
        topic="topic:spacex",
        insight_id="i1",
        quote_id="q1",
        insight_text="SpaceX will IPO.",
    )
    _write_bundle(
        meta,
        "ep2",
        episode_id="episode:ep2",
        publish_date="2024-03-22",
        person="person:liam",
        topic="topic:ai",
        insight_id="i2",
        quote_id="q2",
        insight_text="AI regulation lags.",
    )
    root = str(tmp_path)
    brief = cil_queries.person_profile(root, root, "person:liam")
    assert brief["person_id"] == "person:liam"
    # Quotes aggregate across BOTH episodes (not just one), under the single person.
    assert len(brief["quotes"]) == 2
    assert set(brief["topics"]) == {"topic:spacex", "topic:ai"}
    assert set(cil_queries.person_topic_ids(root, root, "person:liam")) == {
        "topic:spacex",
        "topic:ai",
    }


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


def test_topic_timeline_includes_summary_title_and_text(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-03-01",
        person="person:alice",
        topic="topic:taxes",
        insight_id="insight-a",
        quote_id="quote-a",
        insight_text="Taxes rose",
        metadata_episode_title="March tax episode",
        metadata_summary_title="A one-line recap",
        metadata_summary_text="The long-form summary prose for this episode.",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:taxes")
    assert len(tl) == 1
    assert tl[0]["summary_title"] == "A one-line recap"
    assert tl[0]["summary_text"] == "The long-form summary prose for this episode."


def test_topic_timeline_summary_fields_none_when_absent(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-03-01",
        person="person:alice",
        topic="topic:taxes",
        insight_id="insight-a",
        quote_id="quote-a",
        insight_text="Taxes rose",
        metadata_episode_title="March tax episode",
    )
    root = str(tmp_path)
    tl = cil_queries.topic_timeline(root, root, "topic:taxes")
    assert len(tl) == 1
    assert tl[0]["summary_title"] is None
    assert tl[0]["summary_text"] is None


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
        '{"schema_version": "3.0","identities":[]}', encoding="utf-8"
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
        "schema_version": "3.0",
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
        "schema_version": "3.0",
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
        "schema_version": "3.0",
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
        "schema_version": "3.0",
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


def test_cil_queries_apply_cross_episode_canonical_map(tmp_path: Path, monkeypatch) -> None:
    # #852 regression guard: ep1 uses spelling variants (person:tracy / topic:cargil),
    # ep2 uses the canonical ids. The bridge queries must unify them the same way
    # get_corpus_graph does -- previously they matched the raw id and under-counted.
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-01-10",
        person="person:tracy",
        topic="topic:cargil",
        insight_id="i1",
        quote_id="q1",
        insight_text="variant episode",
    )
    _write_bundle(
        meta,
        "b",
        episode_id="episode:b",
        publish_date="2024-02-10",
        person="person:tracey-alloway",
        topic="topic:cargill",
        insight_id="i2",
        quote_id="q2",
        insight_text="canonical episode",
    )

    id_map = {"person:tracy": "person:tracey-alloway", "topic:cargil": "topic:cargill"}
    monkeypatch.setattr(cil_queries, "_cil_entity_id_map", lambda root: id_map)

    root = str(tmp_path)

    rows, _trunc, _total = cil_queries.episodes_for_bridge_node_id(root, root, "topic:cargill")
    assert len(rows) == 2, "node-episodes listing must include the variant episode"

    tl = cil_queries.topic_timeline(root, root, "topic:cargill")
    assert {r["episode_id"] for r in tl} == {"episode:a", "episode:b"}

    brief = cil_queries.person_profile(root, root, "person:tracey-alloway")
    assert {q["episode_id"] for q in brief["quotes"]} == {"episode:a", "episode:b"}

    # Querying the VARIANT id resolves to the same canonical and unifies too.
    brief_variant = cil_queries.person_profile(root, root, "person:tracy")
    assert {q["episode_id"] for q in brief_variant["quotes"]} == {"episode:a", "episode:b"}

    persons = cil_queries.topic_person_ids(root, root, "topic:cargill")
    assert set(persons) == {"person:tracy", "person:tracey-alloway"}


# ---------------------------------------------------------------------------
# RFC-097 v3.0 phase-3 "3.4" — full-vocabulary cross-layer fixture (no
# deferrals; every data point asserted in one canonical flow per
# operator direction 2026-06-22). The fixture exercises the complete
# v3.0 vocabulary in artifacts: typed MENTIONS family, insight_type enum,
# position_hint, Person + Organization + Podcast + Topic + Episode +
# Insight + Quote node types, ABOUT + MENTIONS_PERSON + MENTIONS_ORG +
# SUPPORTED_BY + SPOKEN_BY + HAS_EPISODE edges.
# ---------------------------------------------------------------------------


def _build_v3_full_vocabulary_fixture(meta: Path) -> None:
    """Write a synthetic v3.0 bundle exercising every new data point.

    Three episodes, four insight types (claim / recommendation / observation
    / question) across them, two distinct topics, one person, one mentioned
    Organization, one Podcast that owns all three episodes. Designed so a
    single query of the corpus surfaces every node and edge type in the
    v3.0 vocabulary at least once.
    """
    _write_bundle(
        meta,
        "v3a",
        episode_id="episode:v3a",
        publish_date="2026-01-15",
        person="person:ada",
        topic="topic:reliability",
        insight_id="ins:v3a-1",
        quote_id="q:v3a-1",
        insight_text="The team should consider running on-call drills monthly.",
        insight_type="recommendation",
        position_hint=0.25,
        mention_person_typed="person:mentioned-bob",
        mention_org="org:acme-platform",
        podcast_id="podcast:practical-systems",
    )
    _write_bundle(
        meta,
        "v3b",
        episode_id="episode:v3b",
        publish_date="2026-02-20",
        person="person:ada",
        topic="topic:reliability",
        insight_id="ins:v3b-1",
        quote_id="q:v3b-1",
        insight_text="Acme proved error budgets reduce regressions by 30%.",
        insight_type="claim",
        position_hint=0.5,
        mention_org="org:acme-platform",
        podcast_id="podcast:practical-systems",
    )
    _write_bundle(
        meta,
        "v3c",
        episode_id="episode:v3c",
        publish_date="2026-03-10",
        person="person:ada",
        topic="topic:hiring",
        insight_id="ins:v3c-1",
        quote_id="q:v3c-1",
        insight_text="Should small teams hire senior or train junior?",
        insight_type="question",
        position_hint=0.8,
        mention_person_typed="person:mentioned-bob",
        podcast_id="podcast:practical-systems",
    )


def test_v3_vocabulary_full_loop_fixture_shape(tmp_path: Path) -> None:
    """Phase-3 verification surface: synthetic v3.0 fixture covers every data point.

    This is the structural assertion — the gold-fixture writer plus the
    on-disk artifact shape contract. Validates:

    - Every v3.0 node type (Person, Organization, Topic, Podcast, Episode,
      Insight, Quote) is materialised across the three episodes.
    - Every v3.0 cross-layer edge (MENTIONS_PERSON, MENTIONS_ORG, ABOUT,
      SUPPORTED_BY, SPOKEN_BY, HAS_EPISODE) appears at least once.
    - ``insight_type`` covers ≥3 distinct enum buckets (recommendation,
      claim, question — observation is the floor and lands in 3.4-style
      runs alongside).
    - ``position_hint`` values are present on every Insight and within the
      schema-mandated ``[0.0, 1.0]`` range.
    """
    meta = tmp_path / "metadata"
    _build_v3_full_vocabulary_fixture(meta)

    # Walk every artifact and collect node types, edge types, insight_type
    # values, and position_hint values.
    node_types: set[str] = set()
    edge_types: set[str] = set()
    insight_types: set[str] = set()
    position_hints: list[float] = []
    for gi_path in sorted(meta.glob("*.gi.json")):
        gi = json.loads(gi_path.read_text(encoding="utf-8"))
        for n in gi.get("nodes") or []:
            t = n.get("type")
            if isinstance(t, str):
                node_types.add(t)
            if t == "Insight":
                props = n.get("properties") or {}
                itype = props.get("insight_type")
                if isinstance(itype, str):
                    insight_types.add(itype)
                ph = props.get("position_hint")
                if isinstance(ph, (int, float)):
                    position_hints.append(float(ph))
        for e in gi.get("edges") or []:
            t = e.get("type")
            if isinstance(t, str):
                edge_types.add(t)
    for kg_path in sorted(meta.glob("*.kg.json")):
        kg = json.loads(kg_path.read_text(encoding="utf-8"))
        for n in kg.get("nodes") or []:
            t = n.get("type")
            if isinstance(t, str):
                node_types.add(t)
        for e in kg.get("edges") or []:
            t = e.get("type")
            if isinstance(t, str):
                edge_types.add(t)

    expected_node_types = {"Insight", "Quote", "Episode", "Person", "Organization", "Podcast"}
    missing_nodes = expected_node_types - node_types
    assert not missing_nodes, (
        f"v3.0 vocabulary missing node types: {missing_nodes} " f"(found: {sorted(node_types)})"
    )

    expected_edge_types = {
        "MENTIONS_PERSON",
        "MENTIONS_ORG",
        "ABOUT",
        "SUPPORTED_BY",
        "SPOKEN_BY",
        "HAS_EPISODE",
    }
    missing_edges = expected_edge_types - edge_types
    assert not missing_edges, (
        f"v3.0 vocabulary missing edge types: {missing_edges} " f"(found: {sorted(edge_types)})"
    )

    expected_insight_buckets = {"claim", "recommendation", "question"}
    missing_buckets = expected_insight_buckets - insight_types
    assert not missing_buckets, (
        f"insight_type vocabulary missing buckets: {missing_buckets} "
        f"(found: {sorted(insight_types)})"
    )

    assert len(position_hints) == 3, "expected one position_hint per insight"
    for ph in position_hints:
        assert 0.0 <= ph <= 1.0, f"position_hint {ph} out of schema range"
    # Strict: the three position_hints should be DISTINCT (the fixture sets
    # 0.25 / 0.5 / 0.8 per episode). A waterfall that defaulted everything
    # to a single value would pass the range check above; this catches that.
    assert sorted(position_hints) == [0.25, 0.5, 0.8], (
        f"position_hint values diverged from the fixture spec "
        f"(0.25 / 0.5 / 0.8). Got: {sorted(position_hints)}. "
        f"Indicates a regression where the fixture writer or the artifact "
        f"shape stopped preserving per-insight position_hint values."
    )
    # The classifier must produce more than ONE bucket across the corpus
    # (a single bucket would mean classifier collapsed). The fixture
    # exercises claim / recommendation / question; assert ≥3 distinct
    # non-"unknown" buckets actually landed.
    non_unknown = insight_types - {"unknown"}
    assert len(non_unknown) >= 3, (
        f"insight_type classifier collapsed to <3 distinct buckets: "
        f"{sorted(insight_types)}. Indicates classifier broke or the "
        f"fixture writer stopped diversifying."
    )


def test_v3_vocabulary_full_loop_query_layer(tmp_path: Path) -> None:
    """Phase-3 verification surface: the cil_queries layer surfaces every
    data point from the v3.0 fixture, end-to-end.

    Where the *_shape test asserts the artifact-shape contract, this test
    asserts the query-result contract — the data flows from disk through
    cil_queries' graph composition into the API-shape rows the viewer
    consumes. If any wiring drops a vocabulary item between artifact and
    query, this fails.
    """
    meta = tmp_path / "metadata"
    _build_v3_full_vocabulary_fixture(meta)
    root = str(tmp_path)

    # 1. person_profile aggregates quotes + topics across all three episodes
    #    for the canonical speaker.
    brief = cil_queries.person_profile(root, root, "person:ada")
    assert brief["person_id"] == "person:ada"
    assert len(brief["quotes"]) == 3  # one quote per episode
    assert set(brief["topics"]) == {"topic:reliability", "topic:hiring"}

    # 2. topic_timeline filters by insight_type — exercises the classifier
    #    output (3.2) reaching the query layer.
    tl_claim = cil_queries.topic_timeline(root, root, "topic:reliability", insight_types=("claim",))
    assert len(tl_claim) == 1
    assert tl_claim[0]["episode_id"] == "episode:v3b"
    tl_rec = cil_queries.topic_timeline(
        root, root, "topic:reliability", insight_types=("recommendation",)
    )
    assert len(tl_rec) == 1
    assert tl_rec[0]["episode_id"] == "episode:v3a"

    # 3. position_arc on (person:ada, topic:reliability) returns episodes in
    #    publish_date order; each result carries an insight with
    #    ``position_hint`` propagated from the artifact.
    arc = cil_queries.position_arc(
        root, root, "person:ada", "topic:reliability", insight_types=None
    )
    assert len(arc) == 2
    arc_eps = [r["episode_id"] for r in arc]
    assert arc_eps == ["episode:v3a", "episode:v3b"]
    for row in arc:
        for ins in row["insights"]:
            ph = (ins.get("properties") or {}).get("position_hint")
            assert isinstance(
                ph, (int, float)
            ), f"position_hint not propagated to query layer: {ins!r}"
            assert 0.0 <= float(ph) <= 1.0


def test_topic_perspectives_groups_insights_by_speaker(tmp_path: Path) -> None:
    """#1146 — insights ABOUT a topic are grouped per speaker, most-insights first."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-01-01",
        person="person:alice",
        topic="topic:ai",
        insight_id="ia",
        quote_id="qa",
        insight_text="Alice on AI",
    )
    _write_bundle(
        meta,
        "b",
        episode_id="episode:b",
        publish_date="2024-02-01",
        person="person:bob",
        topic="topic:ai",
        insight_id="ib",
        quote_id="qb",
        insight_text="Bob on AI",
    )
    root = str(tmp_path)
    persp = cil_queries.topic_perspectives(root, root, "topic:ai")
    by_person = {p["person_id"]: p for p in persp}
    assert set(by_person) == {"person:alice", "person:bob"}
    assert by_person["person:alice"]["insight_count"] == 1
    assert by_person["person:alice"]["episode_count"] == 1
    assert by_person["person:alice"]["insights"][0]["properties"]["text"] == "Alice on AI"


def test_topic_perspectives_scope_filters_by_episode(tmp_path: Path) -> None:
    """#1149 — keep_episode_ids restricts perspectives to the given episode set."""
    meta = tmp_path / "metadata"
    _write_bundle(
        meta,
        "a",
        episode_id="episode:a",
        publish_date="2024-01-01",
        person="person:alice",
        topic="topic:ai",
        insight_id="ia",
        quote_id="qa",
        insight_text="Alice on AI",
    )
    _write_bundle(
        meta,
        "b",
        episode_id="episode:b",
        publish_date="2024-02-01",
        person="person:bob",
        topic="topic:ai",
        insight_id="ib",
        quote_id="qb",
        insight_text="Bob on AI",
    )
    root = str(tmp_path)
    only_a = cil_queries.topic_perspectives(root, root, "topic:ai", keep_episode_ids={"episode:a"})
    assert {p["person_id"] for p in only_a} == {"person:alice"}
    assert cil_queries.topic_perspectives(root, root, "topic:ai", keep_episode_ids=set()) == []


def _write_sentiment(directory: Path, stem: str, rows: list) -> None:
    """Write an episode's insight_sentiment sidecar next to its bridge (enrichments/ subdir)."""
    ed = directory / "enrichments"
    ed.mkdir(parents=True, exist_ok=True)
    (ed / f"{stem}.insight_sentiment.json").write_text(
        json.dumps({"data": {"insights": rows}}), encoding="utf-8"
    )


def test_conversation_arc_and_timeline_sentiment(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    _write_bundle(
        corpus,
        "e1",
        episode_id="ep1",
        publish_date="2024-01-15",
        person="person:a",
        topic="topic:ai",
        insight_id="i1",
        quote_id="q1",
        insight_text="a great breakthrough",
    )
    _write_sentiment(corpus, "e1", [{"insight_id": "i1", "compound": 0.8, "label": "positive"}])
    _write_bundle(
        corpus,
        "e2",
        episode_id="ep2",
        publish_date="2024-01-16",
        person="person:b",
        topic="topic:ai",
        insight_id="i2",
        quote_id="q2",
        insight_text="a serious risk",
    )
    _write_sentiment(corpus, "e2", [{"insight_id": "i2", "compound": -0.6, "label": "negative"}])
    root = str(corpus)

    # topic_timeline now tags each Insight with sentiment (join by insight_id).
    tl = cil_queries.topic_timeline(root, root, "topic:ai", insight_types=None)
    labels = {n.get("sentiment", {}).get("label") for b in tl for n in b["insights"]}
    assert {"positive", "negative"} <= labels

    # conversation arc: both dates are ISO week 2024-W03 → one bucket, 1 pos + 1 neg.
    arc = cil_queries.topic_conversation_arc(root, root, "topic:ai", insight_types=None)
    assert len(arc) == 1
    wk = arc[0]
    assert wk["week"] == "2024-W03"
    assert wk["volume"] == 2 and wk["positive"] == 1 and wk["negative"] == 1
    assert abs(wk["avg_compound"] - 0.1) < 1e-6  # (0.8 + -0.6) / 2


def test_conversation_arc_drops_insights_with_unparsable_dates(tmp_path: Path) -> None:
    """An episode with a malformed publish_date yields no ISO week (``_iso_week`` → None), so its
    insight is dropped from the arc rather than crashing the weekly aggregation."""
    corpus = tmp_path / "c"
    _write_bundle(
        corpus,
        "e1",
        episode_id="ep1",
        publish_date="not-a-date",
        person="person:a",
        topic="topic:ai",
        insight_id="i1",
        quote_id="q1",
        insight_text="undated take",
    )
    _write_sentiment(corpus, "e1", [{"insight_id": "i1", "compound": 0.5, "label": "positive"}])
    root = str(corpus)
    assert cil_queries.topic_conversation_arc(root, root, "topic:ai", insight_types=None) == []


def test_timeline_dedups_to_latest_run_per_feed(tmp_path: Path) -> None:
    """When a feed has been re-run, the CIL timeline uses only the latest run's insights (matching
    the enrichment / indexer latest-run-per-feed dedup) so superseded runs don't double-count."""
    corpus = tmp_path / "c"
    feed = corpus / "feeds" / "f1"
    old_run = feed / "run_20260101_000000" / "metadata"
    new_run = feed / "run_20260201_000000" / "metadata"
    _write_bundle(
        old_run,
        "e1",
        episode_id="ep1",
        publish_date="2024-01-15",
        person="person:a",
        topic="topic:ai",
        insight_id="i_old",
        quote_id="q_old",
        insight_text="old run take",
    )
    _write_bundle(
        new_run,
        "e1",
        episode_id="ep1",
        publish_date="2024-01-15",
        person="person:a",
        topic="topic:ai",
        insight_id="i_new",
        quote_id="q_new",
        insight_text="new run take",
    )
    root = str(corpus)
    tl = cil_queries.topic_timeline(root, root, "topic:ai", insight_types=None)
    ids = {n.get("id") for b in tl for n in b["insights"]}
    assert "i_new" in ids
    assert "i_old" not in ids  # the superseded run is dropped


def test_timeline_sentiment_missing_sidecar_leaves_insights_untinted(tmp_path: Path) -> None:
    """With no ``insight_sentiment`` sidecar, timeline insights come back with no ``sentiment`` key
    (surfaces render un-tinted) rather than raising."""
    corpus = tmp_path / "c"
    _write_bundle(
        corpus,
        "e1",
        episode_id="ep1",
        publish_date="2024-01-15",
        person="person:a",
        topic="topic:ai",
        insight_id="i1",
        quote_id="q1",
        insight_text="a take",
    )
    # Deliberately no _write_sentiment: the sidecar is absent.
    root = str(corpus)
    tl = cil_queries.topic_timeline(root, root, "topic:ai", insight_types=None)
    assert tl, "timeline still returns the episode even without a sentiment sidecar"
    assert all("sentiment" not in n for b in tl for n in b["insights"])
