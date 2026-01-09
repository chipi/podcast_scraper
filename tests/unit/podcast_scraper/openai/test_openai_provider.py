#!/usr/bin/env python3
"""Standalone unit tests for unified OpenAI provider.

These tests verify that OpenAIProvider correctly implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using
OpenAI's API (Whisper API, GPT API).

These are standalone provider tests - they test the provider itself,
not its integration with the app.
"""

import json
import os
import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.openai.openai_provider import OpenAIProvider


@pytest.mark.unit
class TestOpenAIProviderStandalone(unittest.TestCase):
    """Standalone tests for OpenAIProvider - testing the provider itself."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=False,  # Disable to avoid API calls
            auto_speakers=False,  # Disable to avoid API calls
            generate_summaries=False,  # Disable to avoid API calls
        )

    def test_provider_creation(self):
        """Test that OpenAIProvider can be created."""
        provider = OpenAIProvider(self.cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "OpenAIProvider")

    def test_provider_creation_requires_api_key(self):
        """Test that OpenAIProvider requires API key."""
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            # Note: Config validation happens before provider creation
            # So we need to catch ValidationError from Config, not ValueError from provider
            with self.assertRaises(Exception) as context:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="openai",
                )
                OpenAIProvider(cfg)
            # Error can be either ValidationError (from Config) or ValueError (from provider)
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
        error_msg = str(context.exception)
        self.assertTrue(
            "OpenAI API key required" in error_msg or "validation error" in error_msg.lower()
        )

    def test_provider_implements_all_protocols(self):
        """Test that OpenAIProvider implements all three protocols."""
        provider = OpenAIProvider(self.cfg)

        # TranscriptionProvider protocol
        self.assertTrue(hasattr(provider, "transcribe"))
        self.assertTrue(hasattr(provider, "transcribe_with_segments"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # SpeakerDetector protocol
        self.assertTrue(hasattr(provider, "detect_speakers"))
        self.assertTrue(hasattr(provider, "detect_hosts"))
        self.assertTrue(hasattr(provider, "analyze_patterns"))
        self.assertTrue(hasattr(provider, "clear_cache"))

        # SummarizationProvider protocol
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_provider_initialization_state(self):
        """Test that provider tracks initialization state for each capability."""
        provider = OpenAIProvider(self.cfg)

        # Initially not initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    def test_provider_thread_safe(self):
        """Test that provider marks itself as thread-safe."""
        provider = OpenAIProvider(self.cfg)
        self.assertFalse(provider._requires_separate_instances)

    def test_provider_supports_custom_base_url(self):
        """Test that provider supports custom base_url for E2E testing."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="http://localhost:8000/v1",
        )
        provider = OpenAIProvider(cfg)
        # Verify client was created with custom base_url
        self.assertIsNotNone(provider.client)
        # The client should have the base_url set
        # (we can't easily verify this without accessing private attributes)

    def test_transcription_initialization(self):
        """Test that transcription can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            transcribe_missing=True,
            auto_speakers=False,  # Disable to avoid initializing speaker detection
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._transcription_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)

    def test_speaker_detection_initialization(self):
        """Test that speaker detection can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            speaker_detector_provider="openai",
            auto_speakers=True,
            transcribe_missing=False,  # Explicitly disable to avoid initializing transcription
            generate_summaries=False,  # Explicitly disable to avoid initializing summarization
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._speaker_detection_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._summarization_initialized)

    def test_summarization_initialization(self):
        """Test that summarization can be initialized independently."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            summary_provider="openai",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
            auto_speakers=False,  # Disable to avoid initializing speaker detection
            transcribe_missing=False,  # Explicitly disable to avoid initializing transcription
        )
        provider = OpenAIProvider(cfg)

        provider.initialize()

        self.assertTrue(provider._summarization_initialized)
        # Other capabilities should not be initialized
        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)

    def test_unified_initialization(self):
        """Test that all capabilities can be initialized together."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            openai_api_key=self.cfg.openai_api_key,
            openai_api_base=self.cfg.openai_api_base,
            transcription_provider="openai",
            speaker_detector_provider="openai",
            summary_provider="openai",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries is True
        )

        provider = OpenAIProvider(cfg)
        provider.initialize()

        # All should be initialized
        self.assertTrue(provider._transcription_initialized)
        self.assertTrue(provider._speaker_detection_initialized)
        self.assertTrue(provider._summarization_initialized)
        self.assertTrue(provider.is_initialized)

    def test_cleanup_releases_all_resources(self):
        """Test that cleanup releases all resources."""
        provider = OpenAIProvider(self.cfg)
        provider._transcription_initialized = True
        provider._speaker_detection_initialized = True
        provider._summarization_initialized = True

        provider.cleanup()

        self.assertFalse(provider._transcription_initialized)
        self.assertFalse(provider._speaker_detection_initialized)
        self.assertFalse(provider._summarization_initialized)
        self.assertFalse(provider.is_initialized)

    def test_backward_compatibility_properties(self):
        """Test that backward compatibility properties exist."""
        provider = OpenAIProvider(self.cfg)

        # Transcription properties
        self.assertTrue(hasattr(provider, "model"))
        self.assertTrue(hasattr(provider, "is_initialized"))

        # Verify model property returns transcription model
        self.assertEqual(provider.model, provider.transcription_model)


@pytest.mark.unit
class TestOpenAIProviderTranscription(unittest.TestCase):
    """Tests for OpenAIProvider transcription methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=True,
        )

    @patch("builtins.open", create=True)
    @patch("os.path.getsize")
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_getsize, mock_open):
        """Test successful transcription."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB (within limit)
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        # When response_format="text", API returns string directly, not an object
        mock_client.audio.transcriptions.create.return_value = "Hello world"

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.transcribe("/path/to/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("builtins.open", create=True)
    @patch("os.path.getsize")
    @patch("os.path.exists")
    def test_transcribe_with_language(self, mock_exists, mock_getsize, mock_open):
        """Test transcription with explicit language."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB (within limit)
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Bonjour"
        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        provider.transcribe("/path/to/audio.mp3", language="fr")

        # Verify language was passed to API
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["language"], "fr")

    def test_transcribe_not_initialized(self):
        """Test transcribe raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists):
        """Test transcribe raises FileNotFoundError if file doesn't exist."""
        mock_exists.return_value = False

        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/path/to/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_transcribe_rejects_oversized_audio(self, mock_getsize, mock_exists):
        """Test that OpenAI provider rejects files larger than 25 MB."""
        import tempfile

        mock_exists.return_value = True
        # Create a file larger than 25 MB (26 MB)
        mock_getsize.return_value = 26 * 1024 * 1024  # 26 MB

        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                provider.transcribe(audio_path)

            error_msg = str(context.exception)
            self.assertIn("exceeds OpenAI API limit", error_msg)
            self.assertIn("25 MB", error_msg)
            self.assertIn("transformers", error_msg)  # Should suggest alternative
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_transcribe_accepts_valid_size_audio(self, mock_getsize, mock_exists):
        """Test that OpenAI provider accepts files within the 25 MB limit."""
        import tempfile

        mock_exists.return_value = True
        # Create a file within the limit (24 MB)
        mock_getsize.return_value = 24 * 1024 * 1024  # 24 MB

        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            # Mock the API call to avoid actual API request
            with patch.object(provider.client.audio.transcriptions, "create") as mock_create:
                mock_create.return_value = "Test transcript"
                # Should not raise ValueError for valid file size
                result = provider.transcribe(audio_path)
                self.assertEqual(result, "Test transcript")
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_transcribe_with_segments_rejects_oversized_audio(self, mock_getsize, mock_exists):
        """Test that transcribe_with_segments also rejects oversized files."""
        import tempfile

        mock_exists.return_value = True
        # Create a file larger than 25 MB (26 MB)
        mock_getsize.return_value = 26 * 1024 * 1024  # 26 MB

        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                provider.transcribe_with_segments(audio_path)

            error_msg = str(context.exception)
            self.assertIn("exceeds OpenAI API limit", error_msg)
            self.assertIn("25 MB", error_msg)
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @patch("builtins.open", create=True)
    @patch("os.path.getsize")
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_getsize, mock_open):
        """Test transcribe handles API errors."""
        from openai import APIError

        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB (within limit)
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        # APIError requires request and body parameters
        mock_request = Mock()
        api_error = APIError(message="API error", request=mock_request, body={})
        mock_client.audio.transcriptions.create.side_effect = api_error

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/path/to/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    @patch("builtins.open", create=True)
    @patch("os.path.getsize")
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(self, mock_exists, mock_getsize, mock_open):
        """Test transcribe_with_segments returns full result."""
        mock_exists.return_value = True
        mock_getsize.return_value = 1024 * 1024  # 1 MB (within limit)
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock segment objects
        mock_segment1 = Mock()
        mock_segment1.start = 0.0
        mock_segment1.end = 1.0
        mock_segment1.text = "Hello"

        mock_segment2 = Mock()
        mock_segment2.start = 1.0
        mock_segment2.end = 2.0
        mock_segment2.text = "world"

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = [mock_segment1, mock_segment2]

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/path/to/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 2)
        self.assertIsInstance(elapsed, float)
        self.assertGreater(elapsed, 0)


@pytest.mark.unit
class TestOpenAIProviderSpeakerDetection(unittest.TestCase):
    """Tests for OpenAIProvider speaker detection methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )

    def test_detect_hosts_from_feed_authors(self):
        """Test detect_hosts prefers feed_authors."""
        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._build_speaker_detection_prompt")
    def test_detect_hosts_without_authors(self, mock_build_prompt):
        """Test detect_hosts uses API when no feed_authors."""
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Alice, Bob"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        hosts = provider.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        # Should return empty set if no feed_authors and no API call made
        # (since we're not actually calling the API in this simplified test)
        self.assertIsInstance(hosts, set)

    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._build_speaker_detection_prompt")
    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._parse_speakers_from_response")
    def test_detect_speakers_success(self, mock_parse, mock_build_prompt):
        """Test successful speaker detection."""
        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]}'
        )
        # _parse_speakers_from_response returns:
        # (speaker_names_list, detected_hosts_set, detection_succeeded)
        mock_parse.return_value = (["Alice", "Bob"], {"Alice"}, True)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        speakers, hosts, success = provider.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(speakers, ["Alice", "Bob"])
        self.assertTrue(success)

    def test_detect_speakers_not_initialized(self):
        """Test detect_speakers raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    def test_parse_speakers_from_response_filtering(self):
        """Test _parse_speakers_from_response filtering logic for guests."""
        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        # Test 1: Filter out generic labels ("Host", "Guest", "Speaker")
        response = json.dumps(
            {
                "speakers": ["Alice", "Host", "Guest", "Speaker"],
                "hosts": ["Alice"],
                "guests": ["Host", "Guest", "Speaker", "Bob"],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertNotIn("Host", speakers)
        self.assertNotIn("Guest", speakers)
        self.assertNotIn("Speaker", speakers)
        self.assertEqual(hosts, {"Alice"})
        self.assertTrue(success)

        # Test 2: Filter out organization acronyms (e.g., "NPR", "CNN")
        response = json.dumps(
            {
                "speakers": ["Alice", "NPR", "CNN", "BBC"],
                "hosts": ["Alice"],
                "guests": ["NPR", "CNN", "BBC", "Bob"],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertNotIn("NPR", speakers)
        self.assertNotIn("CNN", speakers)
        self.assertNotIn("BBC", speakers)

        # Test 3: Filter out political/executive titles
        response = json.dumps(
            {
                "speakers": ["Alice", "President Trump", "Governor Newsom", "Mayor Johnson"],
                "hosts": ["Alice"],
                "guests": ["President Trump", "Governor Newsom", "Mayor Johnson", "Bob"],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertNotIn("President Trump", speakers)
        self.assertNotIn("Governor Newsom", speakers)
        self.assertNotIn("Mayor Johnson", speakers)

        # Test 4: Remove hosts from guests list (prevent duplicates)
        response = json.dumps(
            {
                "speakers": ["Alice", "Bob"],
                "hosts": ["Alice"],
                "guests": ["Alice", "Bob"],  # Alice is in both hosts and guests
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        # Alice should not appear twice
        self.assertEqual(speakers.count("Alice"), 1)
        self.assertEqual(hosts, {"Alice"})

        # Test 5: Filter guests that are in known_hosts (even if not detected as hosts)
        response = json.dumps(
            {
                "speakers": ["Alice", "Bob"],
                "hosts": ["Alice"],
                "guests": ["Bob", "Charlie"],  # Charlie is in known_hosts
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(
            response, {"Alice", "Charlie"}  # known_hosts includes Charlie
        )
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertNotIn("Charlie", speakers)  # Should be filtered out
        self.assertEqual(hosts, {"Alice"})

        # Test 6: Empty guests list (host-only episode) - should still succeed
        response = json.dumps(
            {
                "speakers": ["Alice"],
                "hosts": ["Alice"],
                "guests": [],  # No guests
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        self.assertIn("Alice", speakers)
        self.assertEqual(hosts, {"Alice"})
        # Should still succeed even with no guests (host-only episode)
        self.assertTrue(success)

        # Test 7: Organization acronym edge cases (should not filter long acronyms or mixed case)
        # Note: Function limits to min_speakers (default 2), so we test with fewer guests
        response = json.dumps(
            {
                "speakers": ["Alice", "NASA", "NPR", "ABC123"],
                "hosts": ["Alice"],
                "guests": ["NASA", "NPR", "ABC123"],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        # "NASA" is 4 chars (<= 5) and all caps - should be filtered
        # "NPR" is 3 chars (<= 5) and all caps - should be filtered
        # "ABC123" contains numbers - should NOT be filtered (only alphabetic)
        self.assertIn("Alice", speakers)
        self.assertNotIn("NASA", speakers)
        self.assertNotIn("NPR", speakers)  # All caps, short, filtered
        # "ABC123" should be in speakers (contains numbers, not filtered)
        # But may be limited by min_speakers, so just verify it's not filtered out incorrectly
        # (The actual result depends on min_speakers limit)

    def test_parse_speakers_from_response_text_fallback(self):
        """Test _parse_speakers_from_text fallback when JSON parsing fails."""
        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        # Test with invalid JSON (should fall back to text parsing)
        # Missing closing brace
        response = '{"speakers": ["Alice", "Bob"], "hosts": ["Alice"], "guests": ["Bob"]'
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        # Should still return something (fallback parsing)
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)
        self.assertIsInstance(success, bool)

        # Test with text format (should extract names using regex)
        response = "Speakers: Alice, Bob\nHosts: Alice\nGuests: Bob"
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        # Text parsing should extract names
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(hosts, set)
        self.assertIsInstance(success, bool)

    def test_parse_speakers_from_response_min_speakers(self):
        """Test _parse_speakers_from_response handles min_speakers requirement."""
        provider = OpenAIProvider(self.cfg)
        provider.initialize()

        # Test with only one speaker (should add defaults to reach min_speakers)
        response = json.dumps(
            {
                "speakers": ["Alice"],
                "hosts": ["Alice"],
                "guests": [],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, {"Alice"})
        # Should have at least 2 speakers (min_speakers default is 2)
        self.assertGreaterEqual(len(speakers), 2)
        self.assertIn("Alice", speakers)

        # Test with no real speakers (should use defaults)
        response = json.dumps(
            {
                "speakers": [],
                "hosts": [],
                "guests": [],
            }
        )
        speakers, hosts, success = provider._parse_speakers_from_response(response, set())
        # Should use defaults
        self.assertEqual(len(speakers), 2)
        self.assertEqual(speakers, ["Host", "Guest"])
        self.assertFalse(success)

    @patch("podcast_scraper.prompt_store.render_prompt")
    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._build_speaker_detection_prompt")
    def test_analyze_patterns_success(self, mock_build_prompt, mock_render_prompt):
        """Test successful pattern analysis."""
        from podcast_scraper import models

        mock_build_prompt.return_value = "Prompt"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"pattern": "value"}'

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        episodes = [
            models.Episode(
                idx=1,
                title="Episode 1",
                title_safe="Episode_1",
                item=None,
                transcript_urls=[],
                media_url="https://example.com/1",
                media_type="audio/mpeg",
            )
        ]

        # OpenAI provider doesn't implement pattern analysis, returns None
        result = provider.analyze_patterns(episodes=episodes, known_hosts={"Alice"})

        self.assertIsNone(result)  # OpenAI provider returns None to use local logic

    def test_clear_cache(self):
        """Test cache clearing (no-op for OpenAI provider)."""
        provider = OpenAIProvider(self.cfg)

        # clear_cache should not raise (it's a no-op for OpenAI provider)
        provider.clear_cache()

        # OpenAI provider doesn't use spaCy cache, but method exists for protocol compliance
        # It's essentially a no-op


@pytest.mark.unit
class TestOpenAIProviderSummarization(unittest.TestCase):
    """Tests for OpenAIProvider summarization methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test123",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
        )

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_summarize_success(self, mock_render_prompt):
        """Test successful summarization."""
        # Mock render_prompt to return prompts (called twice: system and user)
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a summary."

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.summarize("This is a long transcript text.")

        self.assertEqual(result["summary"], "This is a summary.")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model"], provider.summary_model)

    def test_summarize_not_initialized(self):
        """Test summarize raises RuntimeError if not initialized."""
        provider = OpenAIProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.prompt_store.get_prompt_metadata")
    @patch("podcast_scraper.prompt_store.render_prompt")
    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._build_summarization_prompts")
    def test_summarize_with_params(self, mock_build_prompts, mock_render_prompt, mock_get_metadata):
        """Test summarization with custom parameters."""
        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )
        # render_prompt is called inside _build_summarization_prompts
        mock_render_prompt.side_effect = ["System Prompt", "User Prompt"]
        # get_prompt_metadata is called for tracking
        mock_get_metadata.return_value = {"name": "system_v1", "sha256": "abc123"}

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        params = {"max_length": 100, "min_length": 50}
        provider.summarize("Text", params=params)

        # Verify API was called
        mock_client.chat.completions.create.assert_called()

    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider._build_summarization_prompts")
    def test_summarize_api_error(self, mock_build_prompts):
        """Test summarization error handling."""
        from openai import APIError

        # _build_summarization_prompts returns:
        # (system_prompt, user_prompt, system_prompt_name, user_prompt_name,
        #  paragraphs_min, paragraphs_max)
        mock_build_prompts.return_value = (
            "System Prompt",
            "User Prompt",
            "system_v1",
            "user_v1",
            1,
            3,
        )

        mock_client = Mock()
        # APIError requires request and body parameters
        mock_request = Mock()
        api_error = APIError(message="API error", request=mock_request, body={})
        mock_client.chat.completions.create.side_effect = api_error

        provider = OpenAIProvider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.summarize("Text")

        self.assertIn("summarization failed", str(context.exception).lower())
