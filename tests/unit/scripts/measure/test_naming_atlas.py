"""scripts/measure/naming_atlas.py: where a gold name was available, and the summary buckets."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "naming_atlas.py"
_spec = importlib.util.spec_from_file_location("naming_atlas", _SCRIPT)
assert _spec and _spec.loader
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)


def test_where_a_name_was_available() -> None:
    inputs = {
        "known_hosts": ["Ann Host"],
        "metadata_named": ["Bo Guest"],
        "llm_voice_names": {"S1": "Cy Model"},
        "stated_voice_names": {},
    }
    assert na.where_available("Ann Host", inputs, "", "") == ["known_host"]
    assert na.where_available("Bo Guest", inputs, "", "") == ["metadata"]
    assert na.where_available("Cy Model", inputs, "", "") == ["llm"]
    assert na.where_available("Di Titled", inputs, "x", "Ep 1 with Di Titled") == ["title"]
    assert na.where_available("Ed Heard", inputs, "so Heard said", "") == ["surname_in_transcript"]
    assert na.where_available("Fay Gone", inputs, "", "") == []
    assert na.availability([]) == "name nowhere"
    assert na.availability(["title", "surname_in_transcript"]) == "name in the title only"
    assert na.availability(["surname_in_transcript", "metadata"]) == "name stated or proposed"


def test_a_wrong_name_whose_owner_is_on_another_voice_is_a_swap() -> None:
    row = {
        "score": "wrong_name",
        "label": {"role": "host"},
        "where": ["known_host"],
        "label_name_on_other_voice": ["S2"],
    }
    other = dict(row, label_name_on_other_voice=[])
    assert na.summary([row, other]) == {
        "wrong_name (swap) | host | name stated or proposed": 1,
        "wrong_name | host | name stated or proposed": 1,
    }
