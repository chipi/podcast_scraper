"""Isolated unit tests for speaker_detectors.entities (E1, RFC-059).

Imports the submodule directly (not via the providers.ml.speaker_detection facade)
and drives it with a fake spaCy ``nlp`` so PERSON extraction, separator splitting,
and the pattern-based fallback are exercised without loading a real model.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from podcast_scraper.speaker_detectors.constants import PATTERN_BASED_CONFIDENCE_SCORE
from podcast_scraper.speaker_detectors.entities import (
    _pattern_based_fallback,
    _split_text_on_separators,
    extract_person_entities,
)

pytestmark = pytest.mark.unit


class _FakeEnt:
    def __init__(self, text: str, label: str = "PERSON") -> None:
        self.text = text
        self.label_ = label


class _FakeDoc:
    def __init__(self, ents: List[_FakeEnt]) -> None:
        self.ents = ents


class _FakeNlp:
    """Callable that maps a text fragment to a predefined list of entities."""

    def __init__(self, by_text: dict[str, List[_FakeEnt]]) -> None:
        self._by_text = by_text

    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(self._by_text.get(text, []))


def _names(persons: List[Tuple[str, float]]) -> List[str]:
    return [name for name, _score in persons]


def test_empty_text_or_missing_nlp_returns_empty() -> None:
    assert extract_person_entities("", _FakeNlp({})) == []
    assert extract_person_entities("Some title", None) == []


def test_extracts_person_entities_and_dedups() -> None:
    nlp = _FakeNlp(
        {
            "Jane Doe and Jane Doe talk": [
                _FakeEnt("Jane Doe"),
                _FakeEnt("Jane Doe"),  # duplicate raw → deduped
                _FakeEnt("ACME Corp", label="ORG"),  # non-PERSON → ignored
            ]
        }
    )
    persons = extract_person_entities("Jane Doe and Jane Doe talk", nlp)
    assert _names(persons) == ["Jane Doe"]


def test_falls_back_to_last_segment_when_full_text_has_no_person() -> None:
    # Full text yields nothing; the last separator segment carries the name.
    nlp = _FakeNlp(
        {
            "Episode 12 | John Smith": [],
            "Episode 12": [],
            "John Smith": [_FakeEnt("John Smith")],
        }
    )
    persons = extract_person_entities("Episode 12 | John Smith", nlp)
    assert _names(persons) == ["John Smith"]


def test_split_text_on_separators_prefers_first_present_separator() -> None:
    segments, last = _split_text_on_separators("A | B | C")
    assert segments == ["A", "B", "C"]
    assert last == "C"
    # No separator → whole string, no last segment.
    assert _split_text_on_separators("No separators here") == (["No separators here"], None)


def test_pattern_based_fallback_accepts_name_shape() -> None:
    result = _pattern_based_fallback("Mary Johnson", set())
    assert result == ("Mary Johnson", PATTERN_BASED_CONFIDENCE_SCORE)


def test_pattern_based_fallback_rejects_common_phrases_and_seen_names() -> None:
    # Contains a common phrase ("Guest") → rejected even though it looks like a name.
    assert _pattern_based_fallback("Special Guest", set()) is None
    # Already-seen name → rejected.
    assert _pattern_based_fallback("Mary Johnson", {"mary johnson"}) is None
    # Not a First-Last name shape → rejected.
    assert _pattern_based_fallback("lowercase words", set()) is None


def test_extraction_swallows_nlp_errors() -> None:
    class _Boom:
        def __call__(self, _text: str):
            raise RuntimeError("model exploded")

    assert extract_person_entities("anything", _Boom()) == []
