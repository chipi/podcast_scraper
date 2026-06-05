"""Unit tests for CommercialDetector Phase 1."""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.commercial import CommercialDetector

pytestmark = pytest.mark.unit


class TestCommercialDetector:
    def test_removes_intro_sponsor_block(self) -> None:
        text = (
            "Host: Welcome to the show.\n\n"
            "This episode is brought to you by Stripe. Visit stripe.com/podcast for details.\n\n"
            "Host: Let's talk about design systems."
        )
        cleaned = CommercialDetector(confidence_threshold=0.65).remove(text)
        assert "stripe.com" not in cleaned.lower()
        assert "design systems" in cleaned.lower()

    def test_keeps_non_sponsor_conversation(self) -> None:
        text = "Host: We discuss Stripe the company history without any sponsor CTA."
        cleaned = CommercialDetector(confidence_threshold=0.65).remove(text)
        assert cleaned == text

    def test_detect_returns_candidates_with_confidence(self) -> None:
        text = "This episode is sponsored by Figma. Visit figma.com/start today."
        detector = CommercialDetector(confidence_threshold=0.5)
        candidates = detector.detect(text)
        assert candidates
        assert candidates[0].confidence >= 0.65

    def test_legacy_four_phrase_still_removed(self) -> None:
        text = "Intro\n\nOur sponsors today are Notion and Linear.\n\nMain content here."
        cleaned = CommercialDetector().remove(text)
        assert "notion" not in cleaned.lower()
        assert "main content" in cleaned.lower()

    def test_podcast_intro_welcome_back_not_removed(self) -> None:
        text = (
            "Maya: Welcome back to Singletrack Sessions. Today we're talking about trails.\n"
            "Liam: Thanks, Maya. Let's dive into maintenance routines."
        )
        cleaned = CommercialDetector().remove(text)
        assert cleaned == text
