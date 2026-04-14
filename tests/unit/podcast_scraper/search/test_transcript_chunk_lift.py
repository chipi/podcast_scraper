"""Unit tests for transcript chunk-to-Insight lift (#528)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.cil_lift_overrides import CilLiftOverrides
from podcast_scraper.search.transcript_chunk_lift import (
    bridge_path_next_to_gi,
    lift_row_if_transcript,
    TranscriptLiftGiCache,
    try_lift_transcript_chunk_from_gi,
)


def test_bridge_path_next_to_gi() -> None:
    p = Path("/x/metadata/foo.gi.json")
    assert bridge_path_next_to_gi(p) == Path("/x/metadata/foo.bridge.json")


def test_try_lift_transcript_chunk_from_gi(tmp_path: Path) -> None:
    gi_path = tmp_path / "ep.gi.json"
    bridge_path = tmp_path / "ep.bridge.json"
    gi = {
        "nodes": [
            {
                "id": "insight:1",
                "type": "Insight",
                "properties": {
                    "text": "Regulation lags.",
                    "grounded": True,
                    "insight_type": "claim",
                    "position_hint": 0.5,
                },
            },
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {
                    "char_start": 10,
                    "char_end": 80,
                    "timestamp_start_ms": 1000,
                    "timestamp_end_ms": 2000,
                },
            },
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
            {"type": "ABOUT", "from": "insight:1", "to": "topic:ai-regulation"},
            {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:alex"},
        ],
    }
    bridge = {
        "identities": [
            {"id": "person:alex", "display_name": "Alex Host"},
            {"id": "topic:ai-regulation", "display_name": "AI Regulation"},
        ]
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    lifted = try_lift_transcript_chunk_from_gi(
        gi,
        tmp_path,
        gi_path,
        char_start=20,
        char_end=40,
    )
    assert lifted is not None
    assert lifted["insight"]["id"] == "insight:1"
    assert lifted["insight"]["text"] == "Regulation lags."
    assert lifted["speaker"]["id"] == "person:alex"
    assert lifted["speaker"]["display_name"] == "Alex Host"
    assert lifted["topic"]["id"] == "topic:ai-regulation"
    assert lifted["topic"]["display_name"] == "AI Regulation"
    assert lifted["quote"]["timestamp_start_ms"] == 1000


def test_lift_row_if_transcript_sets_lifted(tmp_path: Path) -> None:
    gi_path = tmp_path / "ep2.gi.json"
    gi = {
        "nodes": [
            {
                "id": "i2",
                "type": "Insight",
                "properties": {"text": "T", "grounded": True},
            },
            {
                "id": "q2",
                "type": "Quote",
                "properties": {
                    "char_start": 0,
                    "char_end": 50,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                },
            },
        ],
        "edges": [
            {"type": "supported_by", "from": "i2", "to": "q2"},
        ],
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    row: dict = {
        "doc_id": "chunk:x:0",
        "score": 0.9,
        "metadata": {
            "doc_type": "transcript",
            "episode_id": "episode:z",
            "char_start": 5,
            "char_end": 15,
        },
        "text": "hello",
    }
    cache = TranscriptLiftGiCache()
    lift_row_if_transcript(row, tmp_path, gi_path, cache)
    assert "lifted" in row
    assert row["lifted"]["insight"]["id"] == "i2"


def test_lift_row_applies_transcript_char_shift(tmp_path: Path) -> None:
    """Positive shift aligns index chunk spans with GI Quote char space."""
    gi_path = tmp_path / "ep_shift.gi.json"
    gi = {
        "nodes": [
            {
                "id": "ins",
                "type": "Insight",
                "properties": {"text": "Shifted.", "grounded": True},
            },
            {
                "id": "quo",
                "type": "Quote",
                "properties": {
                    "char_start": 100,
                    "char_end": 200,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "ins", "to": "quo"}],
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    row: dict = {
        "doc_id": "chunk",
        "score": 0.5,
        "metadata": {
            "doc_type": "transcript",
            "episode_id": "e1",
            "char_start": 0,
            "char_end": 50,
        },
        "text": "x",
    }
    cache = TranscriptLiftGiCache()
    lift_row_if_transcript(
        row,
        tmp_path,
        gi_path,
        cache,
        CilLiftOverrides(transcript_char_shift=100),
    )
    assert row.get("lifted", {}).get("insight", {}).get("id") == "ins"


def test_entity_alias_uses_bridge_display_for_target_id(tmp_path: Path) -> None:
    gi_path = tmp_path / "ep_alias.gi.json"
    bridge_path = tmp_path / "ep_alias.bridge.json"
    gi = {
        "nodes": [
            {
                "id": "ins",
                "type": "Insight",
                "properties": {"text": "A", "grounded": True},
            },
            {
                "id": "quo",
                "type": "Quote",
                "properties": {
                    "char_start": 0,
                    "char_end": 50,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                },
            },
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "ins", "to": "quo"},
            {"type": "SPOKEN_BY", "from": "quo", "to": "person:legacy"},
        ],
    }
    bridge = {"identities": [{"id": "person:canonical", "display_name": "Canonical Name"}]}
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    overrides = CilLiftOverrides(entity_id_aliases={"person:legacy": "person:canonical"})
    lifted = try_lift_transcript_chunk_from_gi(
        gi,
        tmp_path,
        gi_path,
        char_start=10,
        char_end=20,
        overrides=overrides,
    )
    assert lifted is not None
    assert lifted["speaker"]["id"] == "person:canonical"
    assert lifted["speaker"]["display_name"] == "Canonical Name"


def test_transcript_lift_cache_rejects_non_object_json(tmp_path: Path) -> None:
    gi_path = tmp_path / "list.gi.json"
    gi_path.write_text("[1,2,3]", encoding="utf-8")
    cache = TranscriptLiftGiCache()
    assert cache.get(gi_path) is None


def test_lift_row_skips_non_dict_metadata(tmp_path: Path) -> None:
    row: dict = {"metadata": "bad"}
    cache = TranscriptLiftGiCache()
    lift_row_if_transcript(row, tmp_path, tmp_path / "x.gi.json", cache)
    assert "lifted" not in row


def test_lift_row_skips_when_shift_collapses_span(tmp_path: Path) -> None:
    gi_path = tmp_path / "s.gi.json"
    gi_path.write_text("{}", encoding="utf-8")
    row: dict = {
        "metadata": {
            "doc_type": "transcript",
            "episode_id": "e",
            "char_start": 0,
            "char_end": 10,
        },
        "text": "t",
    }
    cache = TranscriptLiftGiCache()
    lift_row_if_transcript(
        row,
        tmp_path,
        gi_path,
        cache,
        CilLiftOverrides(transcript_char_shift=-100),
    )
    assert "lifted" not in row


def test_no_lift_when_no_overlap(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {
                "id": "i",
                "type": "Insight",
                "properties": {"text": "T", "grounded": True},
            },
            {
                "id": "q",
                "type": "Quote",
                "properties": {
                    "char_start": 100,
                    "char_end": 120,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                },
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "i", "to": "q"}],
    }
    p = tmp_path / "unused.gi.json"
    assert try_lift_transcript_chunk_from_gi(gi, tmp_path, p, char_start=0, char_end=10) is None


def test_lift_ignores_malformed_bridge_json(tmp_path: Path) -> None:
    gi_path = tmp_path / "ep_br_mal.gi.json"
    gi = {
        "nodes": [
            {
                "id": "ins",
                "type": "Insight",
                "properties": {"text": "X", "grounded": True},
            },
            {
                "id": "quo",
                "type": "Quote",
                "properties": {
                    "char_start": 0,
                    "char_end": 40,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 0,
                },
            },
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "ins", "to": "quo"},
            {"type": "ABOUT", "from": "ins", "to": "topic:t"},
            {"type": "SPOKEN_BY", "from": "quo", "to": "person:p"},
        ],
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    (tmp_path / "ep_br_mal.bridge.json").write_text("{not-json", encoding="utf-8")
    lifted = try_lift_transcript_chunk_from_gi(gi, tmp_path, gi_path, char_start=5, char_end=25)
    assert lifted is not None
    assert lifted["speaker"]["display_name"] == ""
    assert lifted["topic"]["display_name"] == ""
