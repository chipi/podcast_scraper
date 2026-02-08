"""Integration tests for protocol verification in provider factories."""

from __future__ import annotations

import unittest

import pytest

from podcast_scraper import config
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
class TestProtocolVerificationIntegration(unittest.TestCase):
    """Integration tests for protocol verification in real provider creation."""

    def test_transcription_provider_verification(self):
        """Test that transcription providers are verified at creation."""
        cfg = config.Config(
            rss_url="https://example.com",
            transcription_provider="whisper",
        )

        # Provider creation should include verification (in __debug__ mode)
        provider = create_transcription_provider(cfg)

        # Provider should implement the protocol correctly
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_speaker_detector_verification(self):
        """Test that speaker detectors are verified at creation."""
        cfg = config.Config(
            rss_url="https://example.com",
            speaker_detector_provider="spacy",
        )

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        provider = create_speaker_detector(cfg)

        # Provider should implement the protocol correctly
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_summarization_provider_verification(self):
        """Test that summarization providers are verified at creation."""
        cfg = config.Config(
            rss_url="https://example.com",
            summary_provider="transformers",
        )

        from podcast_scraper.summarization.factory import create_summarization_provider

        provider = create_summarization_provider(cfg)

        # Provider should implement the protocol correctly
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


if __name__ == "__main__":
    unittest.main()
