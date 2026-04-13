"""Unit tests for optional ``cil_lift_overrides.json`` (#528)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.search.cil_lift_overrides import (
    CilLiftOverrides,
    load_cil_lift_overrides,
    resolve_id_alias,
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
