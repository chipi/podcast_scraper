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
