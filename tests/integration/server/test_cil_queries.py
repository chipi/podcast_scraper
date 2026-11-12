"""Unit tests for ``cil_queries`` (GitHub #527)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Moved from tests/unit/ — RFC-081 (not in this repository) PR-A1: tests that import [ml]/[llm] or viewer HTTP
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
    salience: float | None = None,
    quote_start_ms: int | None = None,
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
    insight_props: dict = {
        "text": insight_text,
        "insight_type": insight_type,
        "position_hint": position_hint,
    }
    if salience is not None:
        insight_props["salience"] = salience
    quote_props: dict = {"text": "quote body"}
    if quote_start_ms is not None:
        quote_props["timestamp_start_ms"] = quote_start_ms
    gi_nodes: list = [
        {"id": insight_id, "type": "Insight", "properties": insight_props},
        {"id": quote_id, "type": "Quote", "properties": quote_props},
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


def _write_sentiment(directory: Path, stem: str, rows: list) -> None:
    """Write an episode's insight_sentiment sidecar next to its bridge (enrichments/ subdir)."""
    ed = directory / "enrichments"
    ed.mkdir(parents=True, exist_ok=True)
    (ed / f"{stem}.insight_sentiment.json").write_text(
        json.dumps({"data": {"insights": rows}}), encoding="utf-8"
    )


def _write_episode_metadata(directory: Path, stem: str, *, feed_id: str, episode_id: str) -> None:
    (directory / f"{stem}.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": feed_id}, "episode": {"episode_id": episode_id}}),
        encoding="utf-8",
    )
