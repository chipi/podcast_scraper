"""Unit tests for RFC-072 ``cil_queries`` (GitHub #527)."""

from __future__ import annotations

import json
from pathlib import Path

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
    arc = cil_queries.position_arc(tmp_path, "person:alice", "topic:climate")
    assert len(arc) == 1
    assert arc[0]["episode_id"] == "episode:a"
    assert len(arc[0]["insights"]) == 1
    assert arc[0]["insights"][0]["properties"]["text"] == "We should act"

    arc_all = cil_queries.position_arc(
        tmp_path,
        "person:alice",
        "topic:climate",
        insight_types=None,
    )
    assert len(arc_all) == 2


def test_guest_brief_and_person_topics(tmp_path: Path) -> None:
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
    brief = cil_queries.guest_brief(tmp_path, "person:bob")
    assert brief["person_id"] == "person:bob"
    assert "topic:ai" in brief["topics"]
    assert len(brief["quotes"]) == 1
    topics = cil_queries.person_topic_ids(tmp_path, "person:bob")
    assert topics == ["topic:ai"]


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
    tl = cil_queries.topic_timeline(tmp_path, "topic:tax")
    assert len(tl) == 1
    assert tl[0]["episode_id"] == "episode:x"
    persons = cil_queries.topic_person_ids(tmp_path, "topic:tax")
    assert persons == ["person:carol"]


def test_skips_incomplete_triple(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    (meta / "orphan.bridge.json").write_text(
        '{"schema_version":"1.0","identities":[]}', encoding="utf-8"
    )
    assert cil_queries.position_arc(tmp_path, "person:x", "topic:y") == []
