#!/usr/bin/env python3
"""Tests for protocol definitions (unit tests).

These tests verify that Protocol definitions are valid and type-checkable.
This is a unit test because it only checks protocol definitions, not component interactions.
"""

import unittest
from typing import get_type_hints

from podcast_scraper.speaker_detectors.base import SpeakerDetector
from podcast_scraper.summarization.base import SummarizationProvider
from podcast_scraper.transcription.base import TranscriptionProvider


class TestProtocolDefinitions(unittest.TestCase):
    """Test that Protocol definitions are valid and type-checkable."""

    def test_speaker_detector_protocol_exists(self):
        """Test that SpeakerDetector protocol is defined."""
        # Verify protocol has required methods
        hints = get_type_hints(SpeakerDetector.detect_speakers)
        self.assertIn("episode_title", hints)
        self.assertIn("episode_description", hints)
        self.assertIn("known_hosts", hints)

    def test_transcription_provider_protocol_exists(self):
        """Test that TranscriptionProvider protocol is defined."""
        # Verify protocol has required methods
        hints = get_type_hints(TranscriptionProvider.transcribe)
        self.assertIn("audio_path", hints)
        self.assertIn("language", hints)

    def test_summarization_provider_protocol_exists(self):
        """Test that SummarizationProvider protocol is defined."""
        # Verify protocol has required methods
        hints = get_type_hints(SummarizationProvider.summarize)
        self.assertIn("text", hints)
        self.assertIn("episode_title", hints)
        self.assertIn("episode_description", hints)
        self.assertIn("params", hints)


if __name__ == "__main__":
    unittest.main()
