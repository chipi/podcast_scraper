"""Write one episode's gi/kg/bridge/metadata files for server tests (a fixture writer)."""

from __future__ import annotations

import json
from pathlib import Path


def _bundle(
    directory: Path,
    stem: str,
    *,
    episode_id: str,
    publish_date: str,
    person: str,
    topic: str,
    metadata_episode_title: str | None = None,
    metadata_feed_title: str | None = None,
    metadata_episode_number: int | None = None,
    metadata_episode_image_url: str | None = None,
    metadata_feed_image_url: str | None = None,
    # RFC-097 v3.0 vocabulary opt-ins (caller passes when the test
    # asserts on the typed-MENTIONS family or first-class Org/Podcast).
    # Defaults preserve existing tests' fixtures unchanged.
    mention_person_typed: str | None = None,
    mention_org: str | None = None,
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
    gi_edges: list = [
        {"type": "SPOKEN_BY", "from": "quote-1", "to": person},
        {"type": "SUPPORTED_BY", "from": "insight-1", "to": "quote-1"},
        {"type": "ABOUT", "from": "insight-1", "to": topic},
    ]
    # RFC-097 v3.0 typed MENTIONS family — opt-in.
    if mention_person_typed is not None:
        gi_edges.append(
            {"type": "MENTIONS_PERSON", "from": "insight-1", "to": mention_person_typed}
        )
    if mention_org is not None:
        gi_edges.append({"type": "MENTIONS_ORG", "from": "insight-1", "to": mention_org})

    gi = {
        "schema_version": "3.0",
        "episode_id": episode_id,
        "nodes": [
            {
                "id": "insight-1",
                "type": "Insight",
                "properties": {"text": "Hello", "insight_type": "claim", "position_hint": 0.1},
            },
            {"id": "quote-1", "type": "Quote", "properties": {"text": "said"}},
        ],
        "edges": gi_edges,
    }
    kg_nodes: list = [
        {
            "id": "ep1",
            "type": "Episode",
            "properties": {"publish_date": publish_date},
        }
    ]
    kg_edges: list = []
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
        kg_nodes.append({"id": podcast_id, "type": "Podcast", "properties": {"title": "Test Show"}})
        kg_edges.append({"type": "HAS_EPISODE", "from": podcast_id, "to": "ep1"})
    kg = {"schema_version": "2.0", "nodes": kg_nodes, "edges": kg_edges}
    (directory / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (directory / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (directory / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    meta_doc: dict = {}
    feed_meta: dict = {"feed_id": "feed:test-fixture"}
    if metadata_feed_title is not None:
        feed_meta["title"] = metadata_feed_title
    if metadata_feed_image_url is not None:
        feed_meta["image_url"] = metadata_feed_image_url
    meta_doc["feed"] = feed_meta
    ep_meta: dict = {"episode_id": episode_id}
    if metadata_episode_title is not None:
        ep_meta["title"] = metadata_episode_title
    if metadata_episode_number is not None:
        ep_meta["episode_number"] = metadata_episode_number
    if metadata_episode_image_url is not None:
        ep_meta["image_url"] = metadata_episode_image_url
    meta_doc["episode"] = ep_meta
    (directory / f"{stem}.metadata.json").write_text(
        json.dumps(meta_doc),
        encoding="utf-8",
    )
