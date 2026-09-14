"""Speaker detection must not be killed by a publisher's show notes (#2011).

``detect_speakers`` sends title + description + known hosts and asks for 300 output tokens. No
transcript. So the only thing that can overflow the context window is the description — and some
publishers put a full show-notes dump in it. Measured 2026-09-13 on the live corpus: Latent Space
ships episode descriptions up to 137,398 chars (~39,000 tokens) against a 32,768-token window,
while every feed that has never overflowed sits under 4,100.

The failure mode is not a shorter answer, it is a 400 and ZERO speaker attribution for that
episode. Clipping is strictly better, and #2011's acceptance criteria names truncating input as
the first acceptable resolution.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.openai.openai_provider import (
    _clip_speaker_description,
    _SPEAKER_DESCRIPTION_MAX_CHARS,
)

pytestmark = pytest.mark.unit


class TestClipLeavesNormalFeedsAlone:
    """Nothing that works today may change."""

    @pytest.mark.parametrize(
        "size,feed",
        [
            (4_069, "Peter Attia — largest observed"),
            (4_024, "Dwarkesh — largest observed"),
            (1_919, "In Moscow's Shadows — largest observed"),
            (221, "ChinaTalk"),
            (0, "empty description"),
        ],
    )
    def test_descriptions_that_never_overflowed_are_untouched(self, size: int, feed: str) -> None:
        text = "x" * size
        assert _clip_speaker_description(text) == text, feed

    def test_none_becomes_empty_string_not_the_literal_none(self) -> None:
        # The template interpolates this; "None" in a prompt is a real bug, not a cosmetic one.
        assert _clip_speaker_description(None) == ""


class TestClipSavesTheEpisodeThatWouldOtherwise400:
    def test_a_latent_space_sized_description_is_bounded(self) -> None:
        # The real shape: 137,398 chars is ~39,000 tokens against a 32,768 window.
        clipped = _clip_speaker_description("y" * 137_398, "Some Latent Space Episode")
        assert len(clipped) == _SPEAKER_DESCRIPTION_MAX_CHARS

    def test_the_bound_leaves_room_for_the_rest_of_the_prompt(self) -> None:
        # 300 output tokens are requested against a 32,768 window. At a pessimistic 3.5
        # chars/token the clipped description must be a small fraction of that, so the system
        # prompt, title and hosts cannot push the request over.
        assert _SPEAKER_DESCRIPTION_MAX_CHARS / 3.5 < 3_000

    def test_the_kept_portion_is_the_head_where_the_names_are(self) -> None:
        # Show notes lead with the guest; the tail is timestamps, sponsors and links.
        text = "Guest today is Dr. Jane Doe. " + ("filler " * 50_000)
        assert _clip_speaker_description(text).startswith("Guest today is Dr. Jane Doe.")


class TestClipIsObservable:
    def test_clipping_says_so(self, caplog: pytest.LogCaptureFixture) -> None:
        # Silent truncation is how you lose a stage's quality without anyone noticing.
        with caplog.at_level("INFO"):
            _clip_speaker_description("z" * 50_000, "An Episode")
        assert any("clipped" in r.message for r in caplog.records)

    def test_no_log_when_nothing_was_clipped(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _clip_speaker_description("short")
        assert not [r for r in caplog.records if "clipped" in r.message]
