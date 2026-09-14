"""Fit a transcript by COUNTING tokens, not by guessing a chars-per-token rate (#2050).

Every context-overflow in this repo came from the same shape: a budget in tokens, a transcript in
characters, and a guessed conversion between them. The guess is content-dependent, so it is always
wrong somewhere — and wrong in the optimistic direction costs the whole stage for that episode.

The number that settled it came from production, not from a fixture: vLLM rejected the same
106,905-char prompt 508 times in 30 days, each time reporting 30,721 input tokens against a 32,768
limit. Over by ONE token, deterministically. A server that can tokenize ends the argument.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from podcast_scraper.providers.common.token_budget import (
    FALLBACK_EFFECTIVE_CHARS_PER_TOKEN,
    fit_text_to_token_budget,
)

pytestmark = pytest.mark.unit


def _counter(chars_per_token: float, calls: Optional[List[str]] = None):
    """A tokenizer whose rate we control, so a test can model real content (3.48) or fixtures."""

    def count(text: str) -> Optional[int]:
        if calls is not None:
            calls.append(text)
        return max(1, int(len(text) / chars_per_token))

    return count


class TestItFitsByMeasuring:
    def test_text_that_already_fits_is_untouched(self) -> None:
        text = "word " * 100
        assert fit_text_to_token_budget(text, 10_000, _counter(3.48)) == text

    def test_it_asks_the_tokenizer_rather_than_assuming(self) -> None:
        calls: List[str] = []
        fit_text_to_token_budget("x" * 200_000, 1_000, _counter(3.48, calls))
        assert calls, "the whole point is that it measures"

    @pytest.mark.parametrize("rate", [3.0, 3.48, 4.0, 4.44, 6.0])
    def test_the_result_fits_whatever_the_content_tokenises_at(self, rate: float) -> None:
        """The property a constant cannot have: correct across every content density."""
        count = _counter(rate)
        fitted = fit_text_to_token_budget("x" * 300_000, 29_696, count)
        measured = count(fitted)
        assert measured is not None and measured <= 29_696

    def test_the_production_case_would_not_have_overflowed(self) -> None:
        # 106,905 chars at the real 3.48 = 30,721 tokens; +2,048 reply = 32,769 of 32,768.
        # Counting instead of estimating must land under, not one over.
        count = _counter(106_905 / 30_721)
        fitted = fit_text_to_token_budget("x" * 106_905, 32_768 - 2_048 - 1_024, count)
        measured = count(fitted)
        assert measured is not None and measured + 2_048 + 1_024 <= 32_768


class TestItDegradesInsteadOfFailing:
    """A budgeting helper that can take down a stage is worse than a slightly wrong budget."""

    def test_no_tokenizer_falls_back_to_the_pessimistic_estimate(self) -> None:
        fitted = fit_text_to_token_budget("x" * 200_000, 1_000, None)
        assert len(fitted) == int(1_000 * FALLBACK_EFFECTIVE_CHARS_PER_TOKEN)

    def test_a_tokenizer_outage_falls_back_rather_than_raising(self) -> None:
        fitted = fit_text_to_token_budget("x" * 200_000, 1_000, lambda _t: None)
        assert len(fitted) == int(1_000 * FALLBACK_EFFECTIVE_CHARS_PER_TOKEN)

    def test_an_outage_partway_through_still_returns_something_safe(self) -> None:
        state = {"n": 0}

        def flaky(text: str) -> Optional[int]:
            state["n"] += 1
            if state["n"] == 1:
                return 99_999  # far too big, forces a shrink round
            return None  # then the tokenizer dies

        fitted = fit_text_to_token_budget("x" * 200_000, 1_000, flaky)
        assert len(fitted) == int(1_000 * FALLBACK_EFFECTIVE_CHARS_PER_TOKEN)

    def test_the_fallback_is_pessimistic_not_optimistic(self) -> None:
        # Real corpus transcripts run 3.48 chars/token (106,905 / 30,721, from 508 production
        # rejections). The effective fallback must sit BELOW that, or it re-creates the exact bug:
        # a budget satisfied and the request still rejected.
        assert FALLBACK_EFFECTIVE_CHARS_PER_TOKEN < 106_905 / 30_721


class TestDegenerateInputs:
    @pytest.mark.parametrize("budget", [0, -1, -5000])
    def test_a_nonpositive_budget_yields_empty_not_a_reversed_slice(self, budget: int) -> None:
        # text[:negative] silently returns the HEAD minus the tail — the model would receive a
        # truncated episode with no indication anything was wrong.
        assert fit_text_to_token_budget("x" * 1000, budget, _counter(3.48)) == ""

    def test_empty_text_is_returned_as_is(self) -> None:
        assert fit_text_to_token_budget("", 1000, _counter(3.48)) == ""

    def test_it_terminates_even_when_the_tokenizer_never_reports_a_fit(self) -> None:
        # A pathological tokenizer must not loop forever; it must land on the safe estimate.
        fitted = fit_text_to_token_budget("x" * 200_000, 1_000, lambda _t: 10**9)
        assert len(fitted) == int(1_000 * FALLBACK_EFFECTIVE_CHARS_PER_TOKEN)
