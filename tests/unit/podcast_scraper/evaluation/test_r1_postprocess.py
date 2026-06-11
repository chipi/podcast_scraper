"""Tests for the #961 R1-Distill reasoning-preamble strip helper."""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.r1_postprocess import strip_r1_reasoning


class TestSummaryTagExtraction:
    def test_clean_summary_block_is_extracted(self) -> None:
        text = "<summary>The episode covered renewable energy.</summary>"
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_summary_block_after_think_block(self) -> None:
        text = (
            "<think>Okay let me reason about this...</think>\n"
            "<summary>The episode covered renewable energy.</summary>"
        )
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_summary_block_after_freeform_preamble(self) -> None:
        text = (
            "Okay, so I need to summarize this. Let me start.\n\n"
            "<summary>The episode covered renewable energy.</summary>"
        )
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_multiline_summary_block(self) -> None:
        text = "<summary>\nFirst paragraph here.\n\nSecond paragraph here.\n</summary>"
        assert strip_r1_reasoning(text) == "First paragraph here.\n\nSecond paragraph here."

    def test_summary_tag_case_insensitive(self) -> None:
        text = "<SUMMARY>Renewable energy.</SUMMARY>"
        assert strip_r1_reasoning(text) == "Renewable energy."


class TestForgottenClosingTag:
    def test_open_summary_only_returns_everything_after(self) -> None:
        text = "<summary>The episode covered renewable energy."
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_open_summary_after_think_returns_everything_after(self) -> None:
        text = "<think>reasoning...</think>\n<summary>The episode covered renewable energy."
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."


class TestNoSummaryTagFallbacks:
    def test_complete_think_block_is_stripped(self) -> None:
        text = "<think>Let me think about this.</think>\nThe episode covered renewable energy."
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_unclosed_think_block_returns_empty(self) -> None:
        text = "<think>Let me think about this. (model hit max_tokens mid-reasoning)"
        assert strip_r1_reasoning(text) == ""

    def test_preamble_opener_drops_first_paragraph(self) -> None:
        text = (
            "Okay, so I need to summarize this episode.\n\n" "The episode covered renewable energy."
        )
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."

    def test_preamble_opener_without_break_returns_as_is(self) -> None:
        # If there's no clear paragraph break we'd rather keep the text
        # than aggressively trim — caller can audit.
        text = "Okay so the episode covered renewable energy."
        assert strip_r1_reasoning(text) == text.strip()


class TestDegenerateCases:
    @pytest.mark.parametrize("value", ["", "   ", "\n\n"])
    def test_empty_or_whitespace_returns_empty(self, value: str) -> None:
        assert strip_r1_reasoning(value) == ""

    def test_clean_summary_without_any_tags_or_preamble(self) -> None:
        text = "The episode covered renewable energy."
        assert strip_r1_reasoning(text) == text

    def test_trailing_text_after_summary_block_is_dropped(self) -> None:
        text = (
            "<summary>The episode covered renewable energy.</summary>\n"
            "\nFinal answer: the summary above is correct."
        )
        assert strip_r1_reasoning(text) == "The episode covered renewable energy."
