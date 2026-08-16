"""ASR name-canonicalization in transcript body text (#1285)."""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.name_canonicalization import (
    build_canonical_map,
    canonicalize_text,
)

pytestmark = pytest.mark.unit


def test_snaps_garbled_surname_to_speaking_person() -> None:
    # "Kevin Russo" / "Kevin Roos" -> "Kevin Roose" when Kevin Roose is a speaking voice.
    text = "So Kevin Russo said, and later Kevin Roos agreed."
    out, fixes = canonicalize_text(text, ["Kevin Roose", "Casey Newton"])
    assert out == "So Kevin Roose said, and later Kevin Roose agreed."
    assert ("Kevin Russo", "Kevin Roose") in fixes
    assert ("Kevin Roos", "Kevin Roose") in fixes


def test_leaves_already_correct_name_untouched() -> None:
    text = "Kevin Roose and Casey Newton host the show."
    out, fixes = canonicalize_text(text, ["Kevin Roose", "Casey Newton"])
    assert out == text
    assert fixes == []


def test_abstains_on_first_name_collision() -> None:
    # Two speaking "Eric"s -> ambiguous, must NOT rewrite (the Schmidt/Schmitt hazard).
    text = "Then Eric Schmidt weighed in."
    out, fixes = canonicalize_text(text, ["Eric Schmitt", "Eric Adams"])
    assert out == text
    assert fixes == []


def test_requires_a_speaking_person_not_a_mere_mention() -> None:
    # A garbled name whose canonical is NOT a speaking voice is left alone (no reference to trust).
    text = "They discussed Elon Muskk at length."
    out, fixes = canonicalize_text(text, ["Kevin Roose"])
    assert out == text
    assert fixes == []


def test_requires_phonetic_match_not_just_shared_first_name() -> None:
    # Same first name but a phonetically-unrelated surname is a DIFFERENT person — never rewrite.
    text = "Here's Kevin Durant."
    out, fixes = canonicalize_text(text, ["Kevin Roose"])
    assert out == text
    assert fixes == []


def test_first_name_must_match_exactly() -> None:
    text = "That was Kevan Roose maybe."  # first name garbled, not surname
    out, fixes = canonicalize_text(text, ["Kevin Roose"])
    # "Kevan" != "Kevin" exactly -> no map entry -> untouched (surname garbles only).
    assert out == text
    assert fixes == []


def test_build_canonical_map_drops_collisions_keeps_singletons() -> None:
    m = build_canonical_map(["Kevin Roose", "Eric Schmitt", "Eric Adams", "Casey"])
    assert m["kevin"] == "Kevin Roose"
    assert "eric" not in m  # collision dropped
    assert "casey" not in m  # mononym dropped


def test_idempotent() -> None:
    text = "Kevin Russo spoke."
    once, _ = canonicalize_text(text, ["Kevin Roose"])
    twice, fixes2 = canonicalize_text(once, ["Kevin Roose"])
    assert once == twice == "Kevin Roose spoke."
    assert fixes2 == []
