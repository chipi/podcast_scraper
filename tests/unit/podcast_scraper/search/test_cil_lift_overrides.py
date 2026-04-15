"""Unit tests for optional ``cil_lift_overrides.json`` (#528)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.cil_lift_overrides import (
    CilLiftOverrides,
    load_cil_lift_overrides,
    resolve_id_alias,
    write_cil_lift_overrides_merged_topic_id_aliases,
)


def test_load_missing_file_returns_defaults(tmp_path: Path) -> None:
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == 0
    assert o.entity_id_aliases == {}
    assert o.topic_id_aliases == {}


def test_load_invalid_json_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / "cil_lift_overrides.json").write_text("{broken", encoding="utf-8")
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == 0
    assert o.entity_id_aliases == {}


def test_load_valid_file(tmp_path: Path) -> None:
    (tmp_path / "cil_lift_overrides.json").write_text(
        json.dumps(
            {
                "transcript_char_shift": -12,
                "entity_id_aliases": {"person:a": "person:b"},
                "topic_id_aliases": {"topic:x": "topic:y"},
            }
        ),
        encoding="utf-8",
    )
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == -12
    assert o.entity_id_aliases == {"person:a": "person:b"}
    assert o.topic_id_aliases == {"topic:x": "topic:y"}


def test_resolve_id_alias_chain() -> None:
    aliases = {"person:a": "person:b", "person:b": "person:c"}
    assert resolve_id_alias("person:a", aliases) == "person:c"


def test_resolve_id_alias_cycle_safe() -> None:
    aliases = {"person:a": "person:b", "person:b": "person:a"}
    assert resolve_id_alias("person:a", aliases) == "person:a"


def test_frozen_overrides_dataclass() -> None:
    o = CilLiftOverrides(transcript_char_shift=3)
    assert o.transcript_char_shift == 3


def test_load_top_level_array_returns_defaults(tmp_path: Path) -> None:
    (tmp_path / "cil_lift_overrides.json").write_text("[1,2]", encoding="utf-8")
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == 0


def test_load_invalid_shift_coerces_to_zero(tmp_path: Path) -> None:
    (tmp_path / "cil_lift_overrides.json").write_text(
        json.dumps({"transcript_char_shift": "nope"}),
        encoding="utf-8",
    )
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == 0


def test_write_merged_topic_id_aliases_creates_file(tmp_path: Path) -> None:
    merged = write_cil_lift_overrides_merged_topic_id_aliases(tmp_path, {"topic:x": "topic:y"})
    assert merged == {"topic:x": "topic:y"}
    o = load_cil_lift_overrides(tmp_path)
    assert o.topic_id_aliases == {"topic:x": "topic:y"}


def test_write_merged_topic_id_aliases_existing_file_wins(tmp_path: Path) -> None:
    (tmp_path / "cil_lift_overrides.json").write_text(
        json.dumps(
            {
                "transcript_char_shift": 5,
                "entity_id_aliases": {"person:a": "person:b"},
                "topic_id_aliases": {"topic:x": "topic:manual"},
            }
        ),
        encoding="utf-8",
    )
    auto = {"topic:x": "topic:auto", "topic:z": "topic:w"}
    merged = write_cil_lift_overrides_merged_topic_id_aliases(tmp_path, auto)
    assert merged["topic:x"] == "topic:manual"
    assert merged["topic:z"] == "topic:w"
    o = load_cil_lift_overrides(tmp_path)
    assert o.transcript_char_shift == 5
    assert o.entity_id_aliases == {"person:a": "person:b"}
