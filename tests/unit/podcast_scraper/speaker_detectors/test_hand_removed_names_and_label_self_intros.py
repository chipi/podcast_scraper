"""0033's two rules: the publish gate refuses the names 0015 removed by hand, and a one-word
self-introduction followed by a lone capital letter names a label, not the speaker.

Prod, 2026-10-14: a reprocess of The Long Run read "I'm Generation X, I suppose" as the guest
introducing himself as "Generation" and published it on 58 insights; 0015 had removed that exact
name on 2026-10-03, but only from the corpus as it was then.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    distinct_self_introductions,
    extract_self_introduced_host,
    HAND_REMOVED_SPEAKER_NAMES,
    is_publishable_speaker_name,
)

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("name", sorted(HAND_REMOVED_SPEAKER_NAMES))
@pytest.mark.parametrize("shape", [True, False])
def test_the_gate_refuses_every_hand_removed_name(name: str, shape: bool) -> None:
    assert not is_publishable_speaker_name(name, require_person_shape=shape)
    assert not is_publishable_speaker_name(f"  {name.upper()} ", require_person_shape=shape)
    assert not is_publishable_speaker_name(name, require_person_shape=shape, language="es")


@pytest.mark.parametrize("name", ["Gene Sykes", "Moral Hazard Smith", "Colin", "Generations"])
def test_the_gate_refuses_only_the_whole_name(name: str) -> None:
    assert is_publishable_speaker_name(name, require_person_shape=False)


@pytest.mark.parametrize(
    "text",
    [
        "So yes. I'm Generation X, I suppose, without giving my age completely away.",
        "Well, I'm Gen Z and proud of it.",
        "Honestly I'm Plan B here.",
        "I'm Type A, always have been.",
    ],
)
def test_a_word_and_letter_label_is_not_a_self_introduction(text: str) -> None:
    assert extract_self_introduced_host(text, intro_chars=5000) is None
    assert distinct_self_introductions(text, intro_chars=5000) == []


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Hi, I'm Brandon. Welcome to the show.", "Brandon"),
        ("I'm Faith, your host.", "Faith"),
        ("I'm Robin, and this is the podcast.", "Robin"),
        ("I'm Steve Jobs.", "Steve Jobs"),
        ("I'm Jack. X marks the spot.", "Jack"),
    ],
)
def test_a_name_without_the_lone_letter_is_kept(text: str, name: str) -> None:
    assert extract_self_introduced_host(text, intro_chars=5000) == name
    assert distinct_self_introductions(text, intro_chars=5000) == [name]


def test_the_label_does_not_hide_a_real_introduction_after_it() -> None:
    text = "I'm Generation X, I suppose. Anyway, I'm Brandon and this is the show."
    assert extract_self_introduced_host(text, intro_chars=5000) == "Brandon"
