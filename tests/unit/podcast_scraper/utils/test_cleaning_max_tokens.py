"""Tests for transcript cleaning max_tokens clamping."""

from __future__ import annotations

import unittest

from podcast_scraper.utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    DEEPSEEK_CLEANING_MAX_TOKENS,
    estimate_cleaning_output_tokens,
    OPENAI_CLEANING_MAX_TOKENS,
)


class TestCleaningMaxTokens(unittest.TestCase):
    def test_estimate_minimum_one(self) -> None:
        self.assertEqual(estimate_cleaning_output_tokens(0), 1)
        self.assertEqual(estimate_cleaning_output_tokens(1), 1)

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


if __name__ == "__main__":
    unittest.main()
