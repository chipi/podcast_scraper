"""A summary must never be the prompt's own few-shot example (#1179).

qwen3.5:35b did exactly this on the DGX pilot: an episode about Tim Cook's retirement came back
summarized as "Speed gains come from braking earlier and smoother...", a verbatim line from the
prompt's style examples. The prompts state plainly that the examples are style references only —
Gemini obeys, a 35B local model does not.

A silent success with fabricated content is worse than a loud failure, so the leak is detected
rather than trusted not to happen.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.workflow.metadata_generation import _warn_if_prompt_examples_leaked

pytestmark = pytest.mark.unit


def test_detects_a_copied_example(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        _warn_if_prompt_examples_leaked(
            "Tim Cook's Legacy",
            [
                "Speed gains come from braking earlier and smoother rather than taking bigger "
                "risks — a counterintuitive but reliable principle for riders at any level."
            ],
        )
    assert "SUMMARY POISONED" in caplog.text


def test_detects_a_leak_in_the_title_too(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        _warn_if_prompt_examples_leaked("Most underwater stress stems from surprise", [])
    assert "SUMMARY POISONED" in caplog.text


def test_a_real_summary_is_silent(caplog) -> None:
    """The guard must not cry wolf on a genuine summary."""
    with caplog.at_level(logging.ERROR):
        _warn_if_prompt_examples_leaked(
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
        _warn_if_prompt_examples_leaked(None, None)
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
