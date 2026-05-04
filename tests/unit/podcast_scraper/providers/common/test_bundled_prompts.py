"""Unit tests for shared bundled-mode prompt builders (#698).

Covers ``providers/common/bundled_prompts.py`` — the pure-function prompt
fragments + token-budget helpers that every provider's
``extract_quotes_bundled`` / ``score_entailment_bundled`` shares. Pure-function
coverage; no provider SDK or HTTP involvement.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.common.bundled_prompts import (
    extract_quotes_bundled_max_tokens,
    EXTRACT_QUOTES_BUNDLED_SYSTEM,
    extract_quotes_bundled_user,
    score_entailment_bundled_max_tokens,
    SCORE_ENTAILMENT_BUNDLED_SYSTEM,
    score_entailment_bundled_user,
    transcript_clip,
)


class TestSystemPrompts:
    """The two SYSTEM constants must mention the JSON contract the parsers expect."""

    def test_extract_system_mentions_json_contract(self) -> None:
        # Parser expects {"<idx_str>": ["quote", ...]} so the prompt must say so.
        s = EXTRACT_QUOTES_BUNDLED_SYSTEM
        assert "JSON" in s
        assert '"0"' in s  # explicit example of integer-as-string keys

    def test_extract_system_mentions_distinct_quotes(self) -> None:
        # The matrix found that "different passages" wording produced 3-5 distinct
        # quotes; without it the model repeats. Don't regress.
        assert "different" in EXTRACT_QUOTES_BUNDLED_SYSTEM.lower()

    def test_extract_system_mentions_empty_array_for_missing(self) -> None:
        # The parser interprets [] as "no candidates"; prompt must teach the model.
        assert "empty array" in EXTRACT_QUOTES_BUNDLED_SYSTEM.lower()

    def test_score_system_mentions_json_contract(self) -> None:
        s = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        assert "JSON" in s
        assert '"0"' in s

    def test_score_system_mentions_0_to_1_range(self) -> None:
        # The parser clips outside this range; prompt sets the model's expectation.
        s = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        assert "0" in s and "1" in s


class TestExtractQuotesBundledUser:
    def test_includes_transcript_and_each_insight(self) -> None:
        out = extract_quotes_bundled_user("alpha beta gamma", ["i one", "i two"])
        assert "alpha beta gamma" in out
        assert "i one" in out
        assert "i two" in out

    def test_numbers_insights_zero_indexed(self) -> None:
        out = extract_quotes_bundled_user("t", ["a", "b", "c"])
        # Parser keys are integer-as-string; user message must number from 0.
        assert "0: a" in out
        assert "1: b" in out
        assert "2: c" in out

    def test_strips_transcript_and_insight_whitespace(self) -> None:
        out = extract_quotes_bundled_user("   transcript   \n", ["  insight  "])
        assert "   transcript   " not in out
        assert "transcript" in out
        assert "  insight  " not in out
        assert "insight" in out

    def test_demands_json_only_response(self) -> None:
        out = extract_quotes_bundled_user("t", ["i"])
        assert "Return JSON only." in out

    def test_handles_empty_insight_list(self) -> None:
        # Edge case: caller may pass [] when episode has no insights.
        out = extract_quotes_bundled_user("t", [])
        assert "Insights:" in out
        assert "Return JSON only." in out


class TestScoreEntailmentBundledUser:
    def test_includes_each_premise_and_hypothesis(self) -> None:
        pairs = [("p1", "h1"), ("p2", "h2")]
        out = score_entailment_bundled_user(pairs)
        assert "p1" in out
        assert "h1" in out
        assert "p2" in out
        assert "h2" in out

    def test_numbers_pairs_zero_indexed(self) -> None:
        pairs = [("a", "b"), ("c", "d")]
        out = score_entailment_bundled_user(pairs)
        # Parser keys are integer-as-string; first pair must be "0:" not "1:".
        assert "0:" in out
        assert "1:" in out

    def test_labels_premise_and_hypothesis_distinctly(self) -> None:
        out = score_entailment_bundled_user([("alpha", "beta")])
        assert "premise: alpha" in out
        assert "hypothesis: beta" in out

    def test_strips_pair_whitespace(self) -> None:
        out = score_entailment_bundled_user([("  alpha  ", "  beta  ")])
        assert "premise: alpha" in out
        assert "hypothesis: beta" in out

    def test_demands_json_only_response(self) -> None:
        out = score_entailment_bundled_user([("p", "h")])
        assert "Return JSON only." in out

    def test_handles_empty_pairs(self) -> None:
        out = score_entailment_bundled_user([])
        assert "Pairs:" in out
        assert "Return JSON only." in out


class TestExtractQuotesBundledMaxTokens:
    def test_floor_is_1024(self) -> None:
        assert extract_quotes_bundled_max_tokens(1) == 1024
        assert extract_quotes_bundled_max_tokens(0) == 1024

    def test_scales_linearly_with_insight_count(self) -> None:
        # 256 * N until floor / cap.
        assert extract_quotes_bundled_max_tokens(8) == 2048
        assert extract_quotes_bundled_max_tokens(16) == 4096

    def test_cap_is_8192(self) -> None:
        assert extract_quotes_bundled_max_tokens(100) == 8192
        assert extract_quotes_bundled_max_tokens(32) == 8192

    @pytest.mark.parametrize("n", [-1, -100])
    def test_negative_treated_as_one(self, n: int) -> None:
        # max(1, n) makes negatives floor to the 1-insight default.
        assert extract_quotes_bundled_max_tokens(n) == 1024


class TestScoreEntailmentBundledMaxTokens:
    def test_floor_is_256(self) -> None:
        assert score_entailment_bundled_max_tokens(1) == 256
        assert score_entailment_bundled_max_tokens(0) == 256

    def test_scales_linearly_with_chunk_size(self) -> None:
        # 30 * chunk_size until floor / cap.
        assert score_entailment_bundled_max_tokens(15) == 450
        assert score_entailment_bundled_max_tokens(100) == 3000

    def test_cap_is_8192(self) -> None:
        assert score_entailment_bundled_max_tokens(1000) == 8192

    @pytest.mark.parametrize("n", [-1, -100])
    def test_negative_treated_as_one(self, n: int) -> None:
        assert score_entailment_bundled_max_tokens(n) == 256


class TestTranscriptClip:
    def test_default_50k_chars(self) -> None:
        long = "x" * 100_000
        clipped = transcript_clip(long)
        assert len(clipped) == 50_000

    def test_strips_leading_trailing_whitespace_before_clip(self) -> None:
        # "   xxx   " → strip → "xxx", then clip — strip happens first.
        clipped = transcript_clip("   alpha   ")
        assert clipped == "alpha"

    def test_short_transcript_unchanged(self) -> None:
        clipped = transcript_clip("short")
        assert clipped == "short"

    def test_custom_max_chars(self) -> None:
        # Smaller models pass tighter budgets.
        clipped = transcript_clip("abcdefghij", max_chars=4)
        assert clipped == "abcd"

    def test_empty_string(self) -> None:
        assert transcript_clip("") == ""
