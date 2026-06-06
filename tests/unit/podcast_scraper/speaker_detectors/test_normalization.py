"""Isolated unit tests for speaker_detectors.normalization (E1, RFC-059).

Imports the submodule directly (not via the providers.ml.speaker_detection facade)
to realize the refactor's stated benefit: each module is testable on its own.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.normalization import (
    _sanitize_person_name,
    _validate_person_entity,
    filter_default_speaker_names,
    is_default_speaker_name,
)

pytestmark = pytest.mark.unit


def test_is_default_speaker_name_detects_placeholders() -> None:
    assert is_default_speaker_name("Host") is True
    assert is_default_speaker_name("Jane Doe") is False


def test_filter_default_speaker_names_drops_placeholders() -> None:
    names = ["Host", "Ada Lovelace", "unknown_guest_1"]
    filtered = filter_default_speaker_names(names)
    assert "Ada Lovelace" in filtered
    assert "Host" not in filtered


def test_sanitize_person_name_strips_parentheticals_and_punctuation() -> None:
    assert _sanitize_person_name("Jane Doe (host),") == "Jane Doe"


@pytest.mark.parametrize("bad", ["", "123", "!!", "()"])
def test_sanitize_person_name_rejects_non_names(bad: str) -> None:
    assert _sanitize_person_name(bad) is None


@pytest.mark.parametrize("bad", ["", "a", "42", "na<me>"])
def test_validate_person_entity_rejects_invalid(bad: str) -> None:
    assert _validate_person_entity(bad) is False


def test_validate_person_entity_accepts_plausible_name() -> None:
    assert _validate_person_entity("Grace Hopper") is True
