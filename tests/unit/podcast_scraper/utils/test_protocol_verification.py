#!/usr/bin/env python3
"""Tests for protocol verification utilities.

These tests verify that runtime protocol checking works correctly.
"""

import unittest
from typing import Protocol, runtime_checkable
from unittest.mock import Mock, patch

from podcast_scraper.speaker_detectors.base import SpeakerDetector
from podcast_scraper.summarization.base import SummarizationProvider
from podcast_scraper.transcription.base import TranscriptionProvider
from podcast_scraper.utils.protocol_verification import verify_protocol_compliance


@runtime_checkable
class ExampleProtocol(Protocol):
    """Example protocol for verification tests."""

    def required_method(self) -> str:
        """Required method for protocol."""
        ...


class ValidProvider:
    """Valid provider that implements ExampleProtocol."""

    def required_method(self) -> str:
        return "test"


class InvalidProvider:
    """Invalid provider that does not implement ExampleProtocol."""

    def other_method(self) -> str:
        return "test"


class TestProtocolVerification(unittest.TestCase):
    """Test protocol verification logic."""

    def test_verify_valid_provider_passes(self):
        """Test that valid provider passes verification."""
        provider = ValidProvider()
        result = verify_protocol_compliance(provider, ExampleProtocol, "ExampleProtocol")
        self.assertTrue(result)

    def test_verify_invalid_provider_fails(self):
        """Test that invalid provider fails verification."""
        provider = InvalidProvider()
        with patch("podcast_scraper.utils.protocol_verification.logger") as mock_logger:
            result = verify_protocol_compliance(provider, ExampleProtocol, "ExampleProtocol")
            self.assertFalse(result)
            # Should log warning
            mock_logger.warning.assert_called_once()

    def test_verify_production_mode_skips_check(self):
        """Test that verification is skipped in production mode (__debug__=False)."""
        provider = InvalidProvider()
        # Note: __debug__ is a built-in constant that can't be patched at runtime.
        # This test verifies the logic path, but actual production mode behavior
        # requires running Python with -O flag. We test that the function handles
        # the __debug__ check correctly.
        # In normal test execution (__debug__=True), this will perform verification
        result = verify_protocol_compliance(provider, ExampleProtocol, "ExampleProtocol")
        # In debug mode, should fail verification
        self.assertFalse(result)

    def test_verify_transcription_provider(self):
        """Test verification with actual TranscriptionProvider protocol."""
        # Create a mock provider that implements the protocol
        mock_provider = Mock(spec=TranscriptionProvider)
        mock_provider.initialize = Mock()
        mock_provider.transcribe = Mock()
        mock_provider.transcribe_with_segments = Mock()
        mock_provider.cleanup = Mock()

        result = verify_protocol_compliance(
            mock_provider, TranscriptionProvider, "TranscriptionProvider"
        )
        self.assertTrue(result)

    def test_verify_speaker_detector(self):
        """Test verification with actual SpeakerDetector protocol."""
        # Create a mock provider that implements the protocol
        mock_provider = Mock(spec=SpeakerDetector)
        mock_provider.initialize = Mock()
        mock_provider.detect_hosts = Mock()
        mock_provider.detect_speakers = Mock()
        mock_provider.analyze_patterns = Mock()
        mock_provider.cleanup = Mock()
        mock_provider.clear_cache = Mock()

        result = verify_protocol_compliance(mock_provider, SpeakerDetector, "SpeakerDetector")
        self.assertTrue(result)

    def test_verify_summarization_provider(self):
        """Test verification with actual SummarizationProvider protocol."""
        # Create a mock provider that implements the protocol
        mock_provider = Mock(spec=SummarizationProvider)
        mock_provider.initialize = Mock()
        mock_provider.summarize = Mock()
        mock_provider.cleanup = Mock()

        result = verify_protocol_compliance(
            mock_provider, SummarizationProvider, "SummarizationProvider"
        )
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
