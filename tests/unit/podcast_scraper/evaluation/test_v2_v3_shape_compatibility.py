"""RFC-097 chunk 7 contract: scorers handle v2.0 KG + v3.0 GI artifact shapes.

The KG and GI scorers operate at the counts / grounding-contract level — they
don't enumerate node types. v2.0 / v3.0 shape changes (Entity → Person /
Organization, MENTIONS → MENTIONS_PERSON / MENTIONS_ORG, insight_type +
position_hint) therefore pass through them unchanged. This test locks that
contract in so a future refactor of the scorers can't silently regress it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.kg_scorer import compute_kg_prediction_stats

pytestmark = pytest.mark.unit


def _v2_kg_payload() -> dict:
    """A v2.0 KG artifact carrying Person + Organization + Podcast + HAS_EPISODE."""
    return {
        "schema_version": "2.0",
        "episode_id": "e1",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2026-06-20T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "podcast:show",
                "type": "Podcast",
                "properties": {"title": "Show"},
            },
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "podcast:show",
                    "title": "Ep",
                    "publish_date": "2026-06-20T00:00:00Z",
                },
            },
            {
                "id": "person:alice",
                "type": "Person",
                "properties": {"name": "Alice", "role": "host"},
            },
            {
                "id": "org:acme",
                "type": "Organization",
                "properties": {"name": "ACME"},
            },
        ],
        "edges": [
            {"type": "HAS_EPISODE", "from": "podcast:show", "to": "episode:e1"},
            {"type": "MENTIONS", "from": "person:alice", "to": "episode:e1"},
            {"type": "MENTIONS", "from": "org:acme", "to": "episode:e1"},
        ],
    }


def test_kg_scorer_handles_v2_shape_unchanged():
    """v2.0 nodes/edges count the same way v1.x did — scorers are shape-agnostic."""
    preds = [{"episode_id": "e1", "output": {"kg": _v2_kg_payload()}}]
    stats = compute_kg_prediction_stats(preds)
    assert stats["episodes_with_kg"] == 1
    assert stats["avg_nodes"] == 4.0  # Podcast + Episode + Person + Organization
    assert stats["avg_edges"] == 3.0  # HAS_EPISODE + 2 MENTIONS


def test_kg_scorer_v1_and_v2_payloads_aggregate_consistently():
    """Mixed corpus (legacy Entity + new Person/Org) aggregates uniformly."""
    legacy = {
        "schema_version": "1.2",
        "episode_id": "e_legacy",
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "t.txt",
        },
        "nodes": [
            {
                "id": "episode:e_legacy",
                "type": "Episode",
                "properties": {
                    "podcast_id": "p",
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": "person:bob",
                "type": "Entity",
                "properties": {"name": "Bob", "kind": "person"},
            },
        ],
        "edges": [{"type": "MENTIONS", "from": "person:bob", "to": "episode:e_legacy"}],
    }
    preds = [
        {"episode_id": "e1", "output": {"kg": _v2_kg_payload()}},
        {"episode_id": "e_legacy", "output": {"kg": legacy}},
    ]
    stats = compute_kg_prediction_stats(preds)
    assert stats["episodes_with_kg"] == 2
    # Mean across (4 nodes, 2 nodes) = 3.0; (3 edges, 1 edge) = 2.0.
    assert stats["avg_nodes"] == 3.0
    assert stats["avg_edges"] == 2.0


def test_v3_gi_shape_strict_validates():
    """A v3.0 GI artifact with Organization + MENTIONS_PERSON/_ORG + insight_type
    + position_hint passes strict schema validation (chunk-7 sufficiency check
    that the silver shape regen targets is locked in)."""
    from podcast_scraper.gi.schema import validate_artifact as validate_gi

    art = {
        "schema_version": "3.0",
        "episode_id": "e1",
        "model_version": "stub",
        "prompt_version": "v1",
        "nodes": [
            {
                "id": "episode:e1",
                "type": "Episode",
                "properties": {
                    "podcast_id": "podcast:show",
                    "title": "Ep",
                    "publish_date": "2026-06-20T00:00:00Z",
                    "duration_ms": 600000,
                },
            },
            {"id": "person:alice", "type": "Person", "properties": {"name": "Alice"}},
            {"id": "org:acme", "type": "Organization", "properties": {"name": "ACME"}},
            {
                "id": "topic:ai",
                "type": "Topic",
                "properties": {"label": "AI"},
            },
            {
                "id": "insight:i1",
                "type": "Insight",
                "properties": {
                    "text": "Alice from ACME observed AI.",
                    "episode_id": "e1",
                    "grounded": True,
                    "insight_type": "observation",
                    "position_hint": 0.42,
                },
            },
            {
                "id": "quote:q1",
                "type": "Quote",
                "properties": {
                    "text": "Alice from ACME observed AI.",
                    "episode_id": "e1",
                    "char_start": 0,
                    "char_end": 28,
                    "timestamp_start_ms": 250000,
                    "timestamp_end_ms": 260000,
                    "transcript_ref": "t.txt",
                },
            },
        ],
        "edges": [
            {"type": "HAS_INSIGHT", "from": "episode:e1", "to": "insight:i1"},
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {
                "type": "MENTIONS_PERSON",
                "from": "insight:i1",
                "to": "person:alice",
                "properties": {"confidence": 0.9},
            },
            {
                "type": "MENTIONS_ORG",
                "from": "insight:i1",
                "to": "org:acme",
                "properties": {"confidence": 0.85},
            },
            {
                "type": "ABOUT",
                "from": "insight:i1",
                "to": "topic:ai",
                "properties": {"confidence": 0.75},
            },
        ],
    }
    validate_gi(art, strict=True)
