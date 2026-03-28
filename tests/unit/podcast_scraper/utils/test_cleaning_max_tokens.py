"""Tests for transcript cleaning max_tokens clamping."""

from __future__ import annotations

import unittest

from podcast_scraper.utils.cleaning_max_tokens import (
    ANTHROPIC_CLEANING_MAX_TOKENS,
    clamp_cleaning_max_tokens,
    DEEPSEEK_CLEANING_MAX_TOKENS,
    estimate_cleaning_output_tokens,
    GEMINI_CLEANING_MAX_OUTPUT_TOKENS,
    GROK_CLEANING_MAX_TOKENS,
    MISTRAL_CLEANING_MAX_TOKENS,
    OLLAMA_CLEANING_MAX_TOKENS,
    OPENAI_CLEANING_MAX_TOKENS,
)


class TestCleaningMaxTokens(unittest.TestCase):
    def test_estimate_minimum_one(self) -> None:
        self.assertEqual(estimate_cleaning_output_tokens(0), 1)
        self.assertEqual(estimate_cleaning_output_tokens(1), 1)

    def test_estimate_scales_with_word_count(self) -> None:
        # 100 words * 0.85 * 1.3 = 110.5 -> 110
        self.assertEqual(estimate_cleaning_output_tokens(100), 110)

    def test_clamp_respects_cap(self) -> None:
        huge = 1_000_000
        self.assertEqual(
            clamp_cleaning_max_tokens(huge, OPENAI_CLEANING_MAX_TOKENS),
            OPENAI_CLEANING_MAX_TOKENS,
        )
        self.assertEqual(
            clamp_cleaning_max_tokens(huge, DEEPSEEK_CLEANING_MAX_TOKENS),
            DEEPSEEK_CLEANING_MAX_TOKENS,
        )

    def test_clamp_floor_one(self) -> None:
        self.assertEqual(clamp_cleaning_max_tokens(0, 4096), 1)

    def test_clamp_within_range(self) -> None:
        self.assertEqual(clamp_cleaning_max_tokens(500, 8192), 500)

    def test_provider_caps_are_positive_ints(self) -> None:
        caps = (
            OPENAI_CLEANING_MAX_TOKENS,
            DEEPSEEK_CLEANING_MAX_TOKENS,
            ANTHROPIC_CLEANING_MAX_TOKENS,
            MISTRAL_CLEANING_MAX_TOKENS,
            GROK_CLEANING_MAX_TOKENS,
            OLLAMA_CLEANING_MAX_TOKENS,
            GEMINI_CLEANING_MAX_OUTPUT_TOKENS,
        )
        for c in caps:
            self.assertIsInstance(c, int)
            self.assertGreaterEqual(c, 4096)


if __name__ == "__main__":
    unittest.main()
