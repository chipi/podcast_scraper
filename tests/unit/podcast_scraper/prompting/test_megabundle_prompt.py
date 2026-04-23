"""Unit tests for mega-bundle prompt builder (#643)."""

from __future__ import annotations

from podcast_scraper.prompting.megabundle import (
    build_extraction_bundle_prompt,
    build_megabundle_prompt,
    DEFAULT_MEGA_BUNDLE_ENTITIES_MAX,
    DEFAULT_MEGA_BUNDLE_INSIGHTS,
    DEFAULT_MEGA_BUNDLE_TOPICS,
)


class TestBuildMegaBundlePrompt:
    def test_includes_all_required_fields(self):
        system, user = build_megabundle_prompt("Some transcript text.")
        assert "JSON object" in system
        assert '"title"' in user
        assert '"summary"' in user
        assert '"bullets"' in user
        assert '"insights"' in user
        assert '"topics"' in user
        assert '"entities"' in user

    def test_includes_research_sweet_spots_by_default(self):
        _, user = build_megabundle_prompt("T")
        assert f"EXACTLY {DEFAULT_MEGA_BUNDLE_INSIGHTS}" in user
        assert f"EXACTLY {DEFAULT_MEGA_BUNDLE_TOPICS}" in user
        assert f"up to {DEFAULT_MEGA_BUNDLE_ENTITIES_MAX}" in user

    def test_custom_counts_propagate(self):
        _, user = build_megabundle_prompt(
            "T",
            num_insights=8,
            num_topics=6,
            max_entities=10,
        )
        assert "EXACTLY 8" in user
        assert "EXACTLY 6" in user
        assert "up to 10" in user

    def test_language_hint_default_en(self):
        _, user = build_megabundle_prompt("T")
        assert "Language: en" in user

    def test_language_hint_can_be_disabled(self):
        _, user = build_megabundle_prompt("T", language=None)
        assert "Language:" not in user

    def test_language_hint_custom(self):
        _, user = build_megabundle_prompt("T", language="fr")
        assert "Language: fr" in user

    def test_transcript_truncated_to_cap(self):
        long_t = "x" * 50_000
        _, user = build_megabundle_prompt(long_t, max_transcript_chars=1000)
        # The user prompt should contain the truncated 1000-char transcript.
        # Count occurrences of 'x' in user; rough bound.
        assert user.count("x") <= 1000 + 50  # small slack for prompt scaffolding

    def test_noun_phrase_instruction_preserved(self):
        _, user = build_megabundle_prompt("T")
        # KG v3 noun-phrase discipline is load-bearing for topic quality.
        # #652 tightened to 2–3 word canonical forms (repeatable across
        # episodes) rather than strictly "unique" — check for the new wording.
        assert "noun phrase" in user.lower()
        assert "canonical" in user.lower()
        assert "2–3 word" in user or "2-3 word" in user


class TestBuildExtractionBundlePrompt:
    def test_omits_summary_bullets_title(self):
        _, user = build_extraction_bundle_prompt("T")
        assert '"summary"' not in user
        assert '"bullets"' not in user
        assert '"title"' not in user

    def test_keeps_extraction_fields(self):
        _, user = build_extraction_bundle_prompt("T")
        assert '"insights"' in user
        assert '"topics"' in user
        assert '"entities"' in user

    def test_research_sweet_spots_by_default(self):
        _, user = build_extraction_bundle_prompt("T")
        assert f"EXACTLY {DEFAULT_MEGA_BUNDLE_INSIGHTS}" in user
        assert f"EXACTLY {DEFAULT_MEGA_BUNDLE_TOPICS}" in user
