"""Unit tests for CommercialDetector Phase 1."""

from __future__ import annotations

from pathlib import Path

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
        # Use the bare brand (no dotted domain) — CodeQL's URL-substring rule flags
        # "figma.com" in <str> as incomplete host validation; this is a span check.
        assert any("figma" in text[c.start : c.end] for c in candidates)

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


class TestTotalRemovalCeiling:
    """#1641-#1645 — the UNION of sponsor spans may not swallow the episode.

    Every per-candidate cap bounds ONE span: ``_span_too_large_for_confidence``,
    ``_SPONSOR_BLOCK_MAX_CHARS``, the inline 2000-char clamp. All are applied before
    merging, so none can see the union. Overlapping spans then merge transitively and a
    run of ordinary sponsor mentions — each individually legal — chains into a single
    span covering the episode.

    Measured on this fixture before the fix: seven candidates, the largest 1521 chars
    (44% of the transcript), merging into one 2942-char span = 86% of the text. Six of
    the 36 app-validation episodes retained only 12-16%, and what survived was the
    intro, an ad, and the outro — the episode itself was gone. Downstream that trips
    ``_reject_destroyed_cleaning``, which throws away ALL cleaning and summarizes the
    raw transcript, ads included.
    """

    # Screenplay-shaped transcript (single newlines between turns, no blank lines) with
    # sponsor reads spread through the body — the shape that makes every boundary walk
    # run past the end of the ad and into content.
    FIXTURE = (
        Path(__file__).resolve().parents[4]
        / "fixtures/app-validation-corpus/v3/feeds/p09/run_20260101_000000"
        / "transcripts/p09_e03.txt"
    )

    def _text(self) -> str:
        assert self.FIXTURE.exists(), f"fixture moved: {self.FIXTURE}"
        return self.FIXTURE.read_text(encoding="utf-8")

    def test_union_of_spans_cannot_eat_the_episode(self) -> None:
        text = self._text()
        assert len(text) >= 2000, "guard only applies above the short-text floor"
        cleaned = CommercialDetector(confidence_threshold=0.65).remove(text)
        assert len(cleaned) >= 0.5 * len(text), (
            f"cleaner kept only {len(cleaned)}/{len(text)} chars "
            f"({len(cleaned) / len(text):.1%}) — it removed the episode, not the ads"
        )

    def test_episode_body_survives_while_the_sponsor_read_goes(self) -> None:
        cleaned = CommercialDetector(confidence_threshold=0.65).remove(self._text()).lower()
        # Body — the substance of the conversation.
        assert "labor markets are where macro stops being abstract" in cleaned
        # Ad — the pre-roll sponsor read.
        assert "today's episode is sponsored by" not in cleaned

    def test_short_text_may_still_be_mostly_ad(self) -> None:
        """The ceiling must not fire below the floor: a snippet can legitimately be half ad."""
        text = (
            "Host: Welcome to the show.\n\n"
            "This episode is brought to you by Stripe. Visit stripe.com/podcast for details.\n\n"
            "Host: Let's talk about design systems."
        )
        assert len(text) < 2000
        cleaned = CommercialDetector(confidence_threshold=0.65).remove(text)
        assert "stripe" not in cleaned.lower()
        assert "design systems" in cleaned.lower()
