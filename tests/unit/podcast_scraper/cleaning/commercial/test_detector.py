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

    def test_confidence_threshold_is_tunable(self) -> None:
        """A higher threshold keeps a borderline block that the default would remove (B3)."""
        text = (
            "Host: welcome.\n\n"
            "A quick word from our sponsor today.\n\n"
            "Host: back to the main topic now."
        )
        # Default (0.65) removes the intro-sponsor block...
        assert "sponsor" not in CommercialDetector(confidence_threshold=0.65).remove(text).lower()
        # ...but a strict threshold keeps it.
        assert "sponsor" in CommercialDetector(confidence_threshold=0.99).remove(text).lower()

    def test_uncorroborated_inline_cta_not_detected(self) -> None:
        """A bare URL in ordinary speech (no brand/promo/intro nearby) is left alone (B2)."""
        text = "Host: you should really check out github.com, it's great for hosting code."
        detector = CommercialDetector(confidence_threshold=0.55)
        assert detector.detect(text) == []

    def test_corroborated_inline_cta_is_detected(self) -> None:
        """A known brand near the inline CTA corroborates it -> detected (B2)."""
        body = "We were deep in distributed consensus and how partitions get handled. " * 3
        text = (
            f"Host: {body}\n\n"
            "Quick break: check out figma.com for your design work.\n\n"
            f"Host: {body}"
        )
        detector = CommercialDetector(confidence_threshold=0.55)
        candidates = detector.detect(text)
        assert candidates
        assert any("figma.com" in text[c.start : c.end] for c in candidates)

    def test_diarization_guest_speaker_skips_candidate(self) -> None:
        text = "Intro\nSponsored by Acme\nOutro"
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Intro", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 35.0, "text": "Sponsored by Acme", "speaker": "SPEAKER_01"},
        ]
        cleaned = CommercialDetector(
            confidence_threshold=0.5,
            diarization_segments=segments,
            host_speaker_id="SPEAKER_00",
        ).remove(text)
        assert "Sponsored by Acme" in cleaned
