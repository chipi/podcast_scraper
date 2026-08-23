"""A summary must never be the prompt's own few-shot example (#1179).

qwen3.5:35b did exactly this on the DGX pilot: an episode about Tim Cook's retirement came back
summarized as "Speed gains come from braking earlier and smoother...", a verbatim line from the
prompt's style examples. The prompts state plainly that the examples are style references only —
Gemini obeys, a 35B local model does not.

A silent success with fabricated content is worse than a loud failure, so the leak is detected and
(#1386) REJECTED (RecoverableSummarizationError) so the fabricated summary is dropped, not shipped.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.exceptions import RecoverableSummarizationError
from podcast_scraper.workflow.metadata_generation import _reject_if_prompt_examples_leaked

pytestmark = pytest.mark.unit


def test_detects_and_rejects_a_copied_example(caplog) -> None:
    # #1386: a poisoned summary is now REJECTED (RecoverableSummarizationError), not just logged.
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RecoverableSummarizationError):
            _reject_if_prompt_examples_leaked(
                1,
                "Tim Cook's Legacy",
                [
                    "Speed gains come from braking earlier and smoother rather than taking bigger "
                    "risks — a counterintuitive but reliable principle for riders at any level."
                ],
            )
    assert "SUMMARY POISONED" in caplog.text


def test_detects_a_leak_in_the_title_too(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RecoverableSummarizationError):
            _reject_if_prompt_examples_leaked(1, "Most underwater stress stems from surprise", [])
    assert "SUMMARY POISONED" in caplog.text


def test_a_real_summary_is_silent(caplog) -> None:
    """The guard must not cry wolf on a genuine summary (no raise, no log)."""
    with caplog.at_level(logging.ERROR):
        _reject_if_prompt_examples_leaked(
            1,
            "Tim Cook's Legacy and the Return of UBI",
            [
                "Apple's market cap grew tenfold under Tim Cook, driven by hardware pivots like "
                "Apple Silicon and the Apple Watch.",
                "John Ternus's appointment as CEO signals a strategic shift toward core hardware.",
            ],
        )
    assert "POISONED" not in caplog.text


def test_empty_input_is_safe(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        _reject_if_prompt_examples_leaked(1, None, None)
    assert "POISONED" not in caplog.text


class TestCleaningDestructionGuard:
    """Cleaning removes ads — never the episode (#1179).

    On the DGX pilot the LLM cleaner returned ~150 characters of a 75 000-char transcript (and
    sometimes nothing). It did not fail; it returned a plausible fragment, and every downstream
    stage worked perfectly on it — so all ten episodes were summarized from their own outro,
    silently, and the run reported green.
    """

    def test_a_destroyed_transcript_falls_back_to_the_raw_text(self, caplog) -> None:
        from podcast_scraper.workflow.metadata_generation import _reject_destroyed_cleaning

        raw = "word " * 15_000  # ~75k chars, a real episode
        destroyed = "Thanks for listening. Email us at show@example.com."

        with caplog.at_level(logging.ERROR):
            result = _reject_destroyed_cleaning(raw, destroyed, episode_idx=1)

        assert result == raw, "must summarize the episode, not the remnant"
        assert "CLEANING DESTROYED THE TRANSCRIPT" in caplog.text

    def test_an_empty_result_falls_back(self) -> None:
        from podcast_scraper.workflow.metadata_generation import _reject_destroyed_cleaning

        raw = "word " * 15_000
        assert _reject_destroyed_cleaning(raw, "", episode_idx=1) == raw

    def test_a_normal_clean_is_kept(self, caplog) -> None:
        """Pattern cleaning trims ads and intros — a real, modest reduction must pass through."""
        from podcast_scraper.workflow.metadata_generation import _reject_destroyed_cleaning

        raw = "word " * 10_000
        cleaned = "word " * 8_500  # 85% — ads removed, episode intact

        with caplog.at_level(logging.ERROR):
            result = _reject_destroyed_cleaning(raw, cleaned, episode_idx=1)

        assert result == cleaned
        assert "DESTROYED" not in caplog.text


# --- 2026-08-16: the guard used to reject on VOCABULARY, which broke real episodes -------------
#
# The original rule matched two-word fragments ("braking earlier"), justified by the claim that the
# examples are "about motorcycling, software architecture and scuba diving — subjects no podcast
# episode we ingest is about". That is false. The app-validation corpus has a mountain-biking show,
# a software show and a scuba show; p01_e02 ("Enduro Skills Without the Hype") is literally about
# braking technique and lost its summary on every attempt, on every retry.
#
# The guard now measures how much of a line is one contiguous run lifted from an example, so a
# summary may share an example's subject without being treated as a copy of it.


class TestSubjectOverlapIsNotACopy:
    """Genuine summaries whose episode is ABOUT the example's subject must survive."""

    @pytest.mark.parametrize(
        "bullet",
        [
            # The p01_e02 case: an episode about braking technique.
            "Braking earlier into a corner preserves grip on loose surfaces, which matters more "
            "on off-camber trails than raw suspension travel.",
            "Riders at any level benefit from setting tire pressure before touching suspension "
            "clickers.",
            # p03, scuba — collides with the diving example.
            "Most underwater stress comes from task loading on the first dive of a trip, not "
            "from depth.",
            "Rehearsal of valve drills is what separates a manageable failure from a panicked "
            "ascent.",
            # p02, software — collides with the architecture example.
            "Reliability is a property of the whole system rather than any single component — "
            "perfect services still compose into an unreliable whole.",
        ],
    )
    def test_genuine_bullet_sharing_example_vocabulary_is_kept(self, bullet: str, caplog) -> None:
        with caplog.at_level(logging.ERROR):
            # The contract is "does not raise". Asserting `is None` proved nothing: the function
            # is declared `-> None`, so that comparison holds however it behaves.
            _reject_if_prompt_examples_leaked(1, "A real episode title", [bullet])
        assert "SUMMARY POISONED" not in caplog.text

    def test_the_actual_p01_e02_leak_is_still_rejected(self, caplog) -> None:
        """What homelab-flash-0731 really returned — reworded tail, lifted body. Still a copy."""
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RecoverableSummarizationError):
                _reject_if_prompt_examples_leaked(
                    1,
                    "Enduro Skills Without the Hype",
                    [
                        "Speed comes from braking earlier and smoother rather than taking bigger "
                        "risks — a counterintuitive but reliable principle observed across three "
                        "different teams."
                    ],
                )
        assert "SUMMARY POISONED" in caplog.text

    def test_a_truncated_copy_is_still_rejected(self) -> None:
        """A model that emits only the example's opening has still copied it."""
        with pytest.raises(RecoverableSummarizationError):
            _reject_if_prompt_examples_leaked(
                1, "Ep", ["Speed gains come from braking earlier and smoother"]
            )

    def test_short_incidental_overlap_is_not_a_copy(self) -> None:
        """A stock phrase shorter than the minimum run must not trip the guard on its own."""
        # Must not raise — see the note above on why `is None` asserted nothing.
        _reject_if_prompt_examples_leaked(1, "Trail Care", ["Braking earlier helps."])
