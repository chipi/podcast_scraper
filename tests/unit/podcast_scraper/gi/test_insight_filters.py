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


class TestDialogueFilter:
    def test_filler_prefix_triggers_drop(self):
        assert insight_looks_like_dialogue("Yeah, that's a great point.") is True
        assert insight_looks_like_dialogue("Okay, so we should think about…") is True
        assert insight_looks_like_dialogue("Well, it depends.") is True

    def test_compound_filler_prefix(self):
        assert insight_looks_like_dialogue("You know, that's a great point.") is True
        assert insight_looks_like_dialogue("I mean, we should consider…") is True

    def test_first_person_density_triggers_drop(self):
        # 4 pronouns out of 10 tokens = 0.4 density → above 0.15 threshold.
        text = "I think we should probably tell our audience what I did."
        assert insight_looks_like_dialogue(text) is True

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
