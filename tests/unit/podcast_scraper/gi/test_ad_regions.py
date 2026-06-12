"""Unit tests for gi.ad_regions pre-extraction ad-region excision (#663)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.ad_regions import (
    AdRegionMetadata,
    detect_postroll_ad_start,
    detect_preroll_ad_end,
    excise_ad_regions,
)

# Representative pre-roll: 4 distinct ad patterns concentrated in ~1800 chars,
# followed by clear content. Mirrors the Invest-Like-the-Best style (#663).
_PREROLL_AD_TEXT = (
    "Ramp understands that no one wants to spend hours chasing receipts, "
    "reviewing expense reports, and checking for policy violations. "
    "So they built their tools to give that time back, using AI to automate "
    "85% of expense reviews with 99% accuracy. "
    "And since Ramp saves companies 5%, it's no wonder Shopify runs on Ramp. "
    "To see what happens when you eliminate the busy work, "
    "check out ramp.com slash invest. "
    "OpenAI, cursor, anthropic, perplexity, and VersaL all have something "
    "in common. They all use WorkOS. "
    "To achieve enterprise adoption at scale, you have to deliver on core "
    "capabilities like SSO, SCIM, RBAC, and audit logs. "
    "That's where WorkOS comes in. "
    "Visit WorkOS.com to get started. "
    "Felix by Rogo is a personal finance agent that turns a single prompt "
    "into finished client-ready work using your firm's own templates. "
    "Learn more at rogo.ai slash Felix. "
)
_CONTENT_BLOCK = (
    "Hello and welcome everyone. I'm the host. "
    "Today we are talking about GLP-1 medicines and the bioscience boom. "
    "Our guest has spent twenty years in the healthcare investing space "
    "and has unusually clear views on drug commercialization strategy. "
    "Let's dive in. " * 20
)
# Representative post-roll: 3 ad patterns + CTA sentences at the end.
_POSTROLL_AD_TEXT = (
    " Your finance team isn't losing money on big mistakes. "
    "It's leaking through a thousand tiny decisions nobody's watching. "
    "Ramp puts guardrails on spending before it happens. "
    "Try it at ramp.com slash invest. "
    "As your business grows, Vanta scales with you, automating compliance. "
    "Learn more at vanta.com slash invest. "
    "Check out ridgelineapps.com to see what they can unlock for your firm. "
)


@pytest.mark.unit
class TestDetectPrerollAdEnd:
    def test_detects_clustered_ad_patterns_and_snaps_to_sentence_end(self):
        text = _PREROLL_AD_TEXT + _CONTENT_BLOCK
        end = detect_preroll_ad_end(text)
        assert end is not None
        # Cut should land inside or at the end of the ad block, before content.
        assert end <= len(_PREROLL_AD_TEXT) + 5  # small tolerance for sentence snap
        assert end >= 500  # not right at char 0

    def test_returns_none_when_pattern_hits_below_threshold(self):
        # Only 1 pattern hit — well below default threshold of 3.
        text = (
            "Welcome to the podcast. Today we talk about oil prices. "
            "Our guest is a researcher at Goldman Sachs. "
            "Visit goldmansachs.com for more. "
        ) + ("This is interview content. " * 200)
        assert detect_preroll_ad_end(text) is None

    def test_returns_none_on_empty_text(self):
        assert detect_preroll_ad_end("") is None


@pytest.mark.unit
class TestDetectPostrollAdStart:
    def test_detects_clustered_ads_at_tail_and_snaps_back_past_block(self):
        text = _CONTENT_BLOCK + _POSTROLL_AD_TEXT
        start = detect_postroll_ad_start(text)
        assert start is not None
        # Cut should land inside the content/ad transition, never at 0.
        assert start > len(_CONTENT_BLOCK) - 2000
        assert start < len(text)

    def test_returns_none_when_tail_is_clean(self):
        text = _CONTENT_BLOCK + (
            "That's our show. Until next time, stay curious. " "We appreciate you listening. "
        )
        assert detect_postroll_ad_start(text) is None

    def test_respects_scan_chars_window(self):
        # Put an ad block well outside the default 5000-char window —
        # should NOT be detected as post-roll.
        text = _POSTROLL_AD_TEXT + _CONTENT_BLOCK * 10
        # Post-roll scans the LAST scan_chars; ads are at the start.
        assert detect_postroll_ad_start(text, scan_chars=500) is None


@pytest.mark.unit
class TestExciseAdRegions:
    def test_excises_preroll_only(self):
        text = _PREROLL_AD_TEXT + _CONTENT_BLOCK
        cleaned, _, meta = excise_ad_regions(text)
        # Pre-roll cut, content preserved.
        assert meta.preroll_cut_end is not None
        assert meta.postroll_cut_start is None
        # Fixture's pre-roll is ~856 chars; expect the bulk of it gone.
        assert meta.chars_removed >= 800
        # Cleaned text should start with content, not Ramp/WorkOS.
        assert "Ramp" not in cleaned[:200]
        assert "Hello and welcome" in cleaned[:100]

    def test_excises_postroll_only(self):
        text = _CONTENT_BLOCK + _POSTROLL_AD_TEXT
        cleaned, _, meta = excise_ad_regions(text)
        assert meta.postroll_cut_start is not None
        assert meta.chars_removed > 100
        # Cleaned text should not end with ad CTA.
        assert "ramp.com slash" not in cleaned.lower()
        assert "vanta.com slash" not in cleaned.lower()

    def test_excises_both(self):
        text = _PREROLL_AD_TEXT + _CONTENT_BLOCK + _POSTROLL_AD_TEXT
        cleaned, _, meta = excise_ad_regions(text)
        assert meta.preroll_cut_end is not None
        assert meta.postroll_cut_start is not None
        assert meta.chars_removed >= 1000
        assert "WorkOS" not in cleaned
        assert "ramp.com slash" not in cleaned.lower()

    def test_dry_run_returns_source_but_populates_metadata(self):
        text = _PREROLL_AD_TEXT + _CONTENT_BLOCK + _POSTROLL_AD_TEXT
        cleaned, _, meta = excise_ad_regions(text, dry_run=True)
        assert cleaned == text  # not mutated
        # Metadata still populated — operators can audit what *would* be cut.
        assert meta.preroll_cut_end is not None
        assert meta.postroll_cut_start is not None
        assert meta.chars_removed > 0

    def test_no_change_when_no_ads(self):
        text = _CONTENT_BLOCK * 3
        cleaned, _, meta = excise_ad_regions(text)
        assert cleaned == text
        assert meta.chars_removed == 0
        assert meta.preroll_cut_end is None
        assert meta.postroll_cut_start is None
        assert isinstance(meta, AdRegionMetadata)

    def test_empty_text_safe(self):
        cleaned, segments, meta = excise_ad_regions("")
        assert cleaned == ""
        assert segments is None
        assert meta.chars_removed == 0

    def test_segments_realigned_by_drop(self):
        text = _PREROLL_AD_TEXT + _CONTENT_BLOCK
        # Simulate word-level segments where each dict has a 'text' key.
        segments = []
        pos = 0
        # Break into ~100-char chunks to simulate utterance segments.
        while pos < len(text):
            chunk = text[pos : pos + 100]
            segments.append({"text": chunk, "start": pos / 100.0, "end": (pos + 100) / 100.0})
            pos += 100
        original_count = len(segments)
        cleaned_text, cleaned_segments, meta = excise_ad_regions(text, segments=segments)
        assert cleaned_segments is not None
        # Segments inside the excised range should be dropped.
        assert len(cleaned_segments) < original_count
        # Surviving segment text should NOT mention Ramp (the ad-region content).
        assert not any("Ramp understands" in str(s.get("text", "")) for s in cleaned_segments)

    def test_excise_preserves_middle_content_verbatim(self):
        """Middle content must pass through byte-identical — no whitespace
        re-joining or subtle corruption when only the edges are cut."""
        middle = "This is the middle. " * 100
        text = _PREROLL_AD_TEXT + middle + _POSTROLL_AD_TEXT
        cleaned, _, _ = excise_ad_regions(text)
        assert middle in cleaned


def _sentence_segments(text: str, speaker: str, t0: float = 0.0):
    """Split ``text`` into ~sentence segments tagged with a speaker (for screenplay)."""
    segs = []
    t = t0
    for piece in text.replace("! ", ". ").replace("? ", ". ").split(". "):
        piece = piece.strip()
        if not piece:
            continue
        segs.append({"start": t, "end": t + 1.0, "text": piece, "speaker_label": speaker})
        t += 1.0
    return segs


@pytest.mark.unit
class TestExciseAdRegionsWithOffsets:
    """#974 Fault B: ad-free screenplay segments must carry exact char ranges."""

    def _build(self):
        from podcast_scraper.providers.ml.diarization.formatting import (
            format_diarized_screenplay_with_offsets,
        )

        segs = (
            _sentence_segments(_PREROLL_AD_TEXT, "Host", 0.0)
            + _sentence_segments(_CONTENT_BLOCK, "Host", 100.0)
            + _sentence_segments("And what do you think? I think it is great.", "Guest", 200.0)
            + _sentence_segments(_POSTROLL_AD_TEXT, "Host", 300.0)
        )
        text, offset_segs = format_diarized_screenplay_with_offsets(segs)
        return text, offset_segs

    def test_survivors_slice_back_to_their_text_exactly(self):
        from podcast_scraper.gi.ad_regions import excise_ad_regions_with_offsets

        text, offset_segs = self._build()
        # Sanity: the formatter's own offsets are exact in the raw screenplay.
        for s in offset_segs:
            assert text[s["char_start"] : s["char_end"]] == s["text"]

        cleaned, survivors, meta = excise_ad_regions_with_offsets(text, offset_segs)
        assert meta.chars_removed > 0
        assert meta.preroll_cut_end is not None
        # Every surviving segment maps EXACTLY into the ad-free text — no guard needed.
        for s in survivors:
            assert cleaned[s["char_start"] : s["char_end"]] == s["text"]
        # Ad segments dropped; content + speaker labels preserved.
        assert len(survivors) < len(offset_segs)
        assert not any("Ramp understands" in s["text"] for s in survivors)
        assert any(s["speaker_label"] == "Guest" for s in survivors)

    def test_no_ads_returns_segments_unchanged(self):
        from podcast_scraper.gi.ad_regions import excise_ad_regions_with_offsets
        from podcast_scraper.providers.ml.diarization.formatting import (
            format_diarized_screenplay_with_offsets,
        )

        segs = _sentence_segments(_CONTENT_BLOCK * 3, "Host", 0.0)
        text, offset_segs = format_diarized_screenplay_with_offsets(segs)
        cleaned, survivors, meta = excise_ad_regions_with_offsets(text, offset_segs)
        assert meta.chars_removed == 0
        assert cleaned == text
        assert len(survivors) == len(offset_segs)
        for s in survivors:
            assert cleaned[s["char_start"] : s["char_end"]] == s["text"]
