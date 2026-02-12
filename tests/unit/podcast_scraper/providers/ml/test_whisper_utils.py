"""Unit tests for podcast_scraper.providers.ml.whisper_utils module."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.whisper_utils import normalize_whisper_model_name


@pytest.mark.unit
class TestNormalizeWhisperModelName:
    """Tests for normalize_whisper_model_name."""

    def test_english_prefers_en_variant(self):
        """For language en, model gets .en suffix when in WHISPER_MODELS_WITH_EN_VARIANT."""
        name, chain = normalize_whisper_model_name("base", "en")
        assert name == "base.en"
        assert "base.en" in chain
        assert "tiny.en" in chain

    def test_english_tiny_en_unchanged(self):
        """tiny.en with en stays tiny.en."""
        name, chain = normalize_whisper_model_name("tiny.en", "en")
        assert name == "tiny.en"
        assert chain == ["tiny.en"] or "tiny.en" in chain

    def test_non_english_strips_en_suffix(self):
        """For non-English language, .en suffix is removed."""
        name, chain = normalize_whisper_model_name("base.en", "fr")
        assert name == "base"
        assert "base" in chain

    def test_fallback_chain_ordered_largest_to_smallest(self):
        """Fallback chain includes smaller models after the requested one."""
        name, chain = normalize_whisper_model_name("large", "en")
        assert name == "large"
        assert chain[0] == "large"
        assert len(chain) > 1

    def test_no_duplicates_in_chain(self):
        """Fallback chain has no duplicate model names."""
        _, chain = normalize_whisper_model_name("base", "en")
        assert len(chain) == len(set(chain))
