"""Unit tests for #652 Part B — GI insight filters (ad + dialogue)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.filters import (
    apply_insight_filters,
    insight_looks_like_ad,
    insight_looks_like_dialogue,
)

pytestmark = [pytest.mark.unit]


class TestAdFilter:
    def test_two_pattern_hits_trigger_drop(self):
        # "brought to you by" + "promo code" → 2 hits in same text.
        source = "This episode is brought to you by Acme. " "Use promo code ACME20 for 20% off."
        assert insight_looks_like_ad("irrelevant insight text", source) is True

    def test_single_pattern_does_not_trigger(self):
        # Only "go to example.com/offer" — could appear in legitimate content.
        source = "Go to example.com/research for the dataset."
        assert insight_looks_like_ad("insight", source) is False

    def test_no_source_text_scans_insight(self):
        # When we only have the insight text itself, two hits in THAT text
        # are still enough.
        text = "Ramp is brought to you by our sponsor. Use code RAMP30 for 30% off."
        assert insight_looks_like_ad(text) is True

    def test_clean_insight_passes(self):
        assert insight_looks_like_ad("AI regulation is accelerating") is False

    def test_real_spoken_form_ads_trigger_drop(self):
        """#652 stabilization: patterns must catch spoken-form ad reads.

        Pre-stabilization the regex assumed literal URLs (``.com/path``)
        which NEVER appear in Whisper transcripts. All 8 original patterns
        caught 0/1200 insights on the 100-ep real corpus. These cases are
        drawn from real ad reads in that corpus.
        """
        # Bloomberg cross-promo (real): "Bloomberg dot com slash odd Lots"
        assert (
            insight_looks_like_ad(
                "Go to Bloomberg dot com slash odd Lots for the daily newsletter."
            )
            is True
        )
        # Classic sponsor disclosure + spoken URL:
        assert (
            insight_looks_like_ad(
                "This show is brought to you by Ramp, go to ramp dot com slash podcast."
            )
            is True
        )
        # Multi-signal ad bullet (LLM hallucination scenario):
        assert insight_looks_like_ad("Use promo code ACME20 for 20% off your first order.") is True
        # Canonical "visit X dot com + free trial + limited time":
        assert (
            insight_looks_like_ad("Visit indeed dot com slash hire for a limited time free trial.")
            is True
        )
        # "Sponsored by" + "our sponsors" (two disclosures in one sentence):
        assert insight_looks_like_ad("This episode is sponsored by our sponsors at Workos.") is True

    def test_substantive_content_with_weak_markers_passes(self):
        """Real negatives — substance that MENTIONS one ad-adjacent phrase
        but isn't actually an ad. The ≥ 2 distinct-pattern threshold keeps
        false positives low."""
        assert (
            insight_looks_like_ad("Go to refinery to understand how crude oil gets processed.")
            is False
        )
        assert insight_looks_like_ad("Taxes dropped by 15% off the 2020 peak.") is False
        assert insight_looks_like_ad("Visit the research paper for more detail.") is False


class TestDialogueFilter:
    def test_filler_prefix_triggers_drop(self):
        assert insight_looks_like_dialogue("Yeah, that's a great point.") is True
        assert insight_looks_like_dialogue("Okay, so we should think about…") is True
        assert insight_looks_like_dialogue("Well, it depends.") is True

    def test_compound_filler_prefix(self):
        assert insight_looks_like_dialogue("You know, that's a great point.") is True
        assert insight_looks_like_dialogue("I mean, we should consider…") is True

    def test_first_person_density_triggers_drop(self):
        """Density threshold is 0.25 post-#652 audit (was 0.15 — too aggressive).

        The text below has 4 first-person pronouns (I, we, our, I) in 10
        tokens = 0.4 density, well above the 0.25 threshold.
        """
        text = "I think we should probably tell our audience what I did."
        assert insight_looks_like_dialogue(text) is True

    def test_substantive_first_person_below_threshold_passes(self):
        """#652 stabilization: genuine first-person analysis / CEO claims must
        NOT be dropped. Pronoun density 0.15-0.24 range passes under the new
        0.25 threshold.

        "We made our first investment before the AI trade started" — 2
        pronouns (we, our) in 10 tokens = 0.2 density. Below 0.25.
        """
        assert (
            insight_looks_like_dialogue("We made our first investment before the AI trade started.")
            is False
        )

    def test_quote_coverage_triggers_drop(self):
        insight = "It's a deal to our let's do it"  # 31 chars
        quote = "It's a deal to our let's do it"  # 100 % coverage
        assert insight_looks_like_dialogue(insight, quote) is True

    def test_quote_partial_coverage_does_not_trigger(self):
        insight = "The speaker emphasised that prediction markets face regulatory hurdles"
        quote = "prediction markets"  # small fraction of insight
        assert insight_looks_like_dialogue(insight, quote) is False

    def test_clean_insight_passes(self):
        assert insight_looks_like_dialogue("Prediction markets face regulatory hurdles") is False

    def test_empty_string_is_not_dialogue(self):
        assert insight_looks_like_dialogue("") is False


class TestApplyInsightFilters:
    def test_drops_ads_and_dialogue_independently(self):
        insights = [
            {"text": "AI regulation is accelerating"},
            {"text": "Yeah, that's cool."},  # dialogue
            {"text": "Clean insight 2"},
        ]
        windows = {
            0: "Clean transcript window",
            1: "…",
            2: "…",
        }
        kept, ads, dialogue = apply_insight_filters(insights, transcript_window_by_index=windows)
        assert len(kept) == 2
        assert ads == 0
        assert dialogue == 1
        assert all(i["text"] != "Yeah, that's cool." for i in kept)

    def test_ad_in_transcript_window_drops_insight(self):
        insights = [{"text": "Some distilled claim"}]
        windows = {
            0: ("This episode is brought to you by Acme. " "Use promo code ACME20 for 20% off.")
        }
        kept, ads, dialogue = apply_insight_filters(insights, transcript_window_by_index=windows)
        assert len(kept) == 0
        assert ads == 1
        assert dialogue == 0

    def test_empty_text_insights_pass_through(self):
        """Empty-text insights aren't the filter's job — downstream validation
        handles schema violations. Don't double-count them here."""
        insights = [{"text": ""}, {"text": "Valid claim"}]
        kept, ads, dialogue = apply_insight_filters(insights)
        assert len(kept) == 2
        assert ads == 0
        assert dialogue == 0
