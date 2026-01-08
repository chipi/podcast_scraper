#!/usr/bin/env python3
"""Unit tests for OpenAISpeakerDetector class.

These tests verify the OpenAI API-based speaker detection provider implementation
with edge cases and error handling.
"""

import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import from parent conftest explicitly to avoid conflicts
import importlib.util
from pathlib import Path

parent_tests_dir = Path(__file__).parent.parent.parent.parent
if str(parent_tests_dir) not in sys.path:
    sys.path.insert(0, str(parent_tests_dir))

parent_conftest_path = parent_tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode

from podcast_scraper import config  # noqa: E402
from podcast_scraper.speaker_detectors.factory import create_speaker_detector  # noqa: E402


class TestOpenAISpeakerDetector(unittest.TestCase):
    """Tests for OpenAISpeakerDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=True,
        )

    def test_init_success(self):
        """Test successful initialization."""
        detector = create_speaker_detector(self.cfg)
        self.assertEqual(detector.cfg, self.cfg)
        self.assertIsNotNone(detector.client)
        self.assertEqual(detector.speaker_model, config.PROD_DEFAULT_OPENAI_SPEAKER_MODEL)
        self.assertEqual(detector.speaker_temperature, 0.3)
        self.assertFalse(detector._speaker_detection_initialized)

    def test_init_missing_api_key(self):
        """Test initialization raises error when API key is missing."""
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.speaker_detector_provider = "openai"
        mock_cfg.openai_api_key = None
        mock_cfg.openai_api_base = None

        with self.assertRaises(ValueError) as context:
            create_speaker_detector(mock_cfg)

        self.assertIn("OpenAI API key required", str(context.exception))

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base_url."""
        cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="https://api.example.com/v1",
        )
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector.client)

    def test_init_with_custom_model(self):
        """Test initialization with custom speaker model."""
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.speaker_detector_provider = "openai"
        mock_cfg.openai_api_key = "sk-test123"
        mock_cfg.openai_api_base = None
        type(mock_cfg).openai_speaker_model = property(lambda self: "gpt-4")
        type(mock_cfg).openai_temperature = property(lambda self: 0.5)

        detector = create_speaker_detector(mock_cfg)
        self.assertEqual(detector.speaker_model, "gpt-4")
        self.assertEqual(detector.speaker_temperature, 0.5)

    def test_initialize_success(self):
        """Test successful initialization."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        self.assertTrue(detector._speaker_detection_initialized)

    def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()
        # Call again
        detector.initialize()

        self.assertTrue(detector._speaker_detection_initialized)

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_success(self, mock_render_prompt):
        """Test successful speaker detection."""
        # render_prompt is called twice: once for system prompt, once for user prompt
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["Alice", "Bob"],
                "hosts": ["Alice"],
                "guests": ["Bob"],
            }
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Alice interviews Bob",
            episode_description="A great conversation",
            known_hosts={"Alice"},
        )

        self.assertEqual(len(speakers), 2)
        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertEqual(detected_hosts, {"Alice"})
        self.assertTrue(success)
        # Verify render_prompt was called (at least for system prompt)
        self.assertGreaterEqual(mock_render_prompt.call_count, 1)

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_auto_speakers_disabled(self, mock_render_prompt):
        """Test detect_speakers returns defaults when auto_speakers is disabled."""
        cfg = create_test_config(
            speaker_detector_provider="openai",
            openai_api_key="sk-test123",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts=set(),
        )

        self.assertEqual(speakers, ["Host", "Guest"])
        self.assertEqual(detected_hosts, set())
        self.assertFalse(success)
        mock_render_prompt.assert_not_called()

    def test_detect_speakers_not_initialized(self):
        """Test detect_speakers raises error when not initialized."""
        detector = create_speaker_detector(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            detector.detect_speakers("Title", "Description", set())

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_empty_response(self, mock_render_prompt):
        """Test detect_speakers handles empty API response."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts=set(),
        )

        self.assertEqual(speakers, ["Host", "Guest"])
        self.assertFalse(success)

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_invalid_json(self, mock_render_prompt):
        """Test detect_speakers handles invalid JSON response."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Not valid JSON"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts=set(),
        )

        # Should fall back to defaults
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(detected_hosts, set)

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_api_error(self, mock_render_prompt):
        """Test detect_speakers handles API errors."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        with self.assertRaises(ValueError) as context:
            detector.detect_speakers("Title", "Description", set())

        self.assertIn("speaker detection failed", str(context.exception))

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_detect_speakers_min_speakers_enforced(self, mock_render_prompt):
        """Test detect_speakers enforces minimum number of speakers."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "speakers": ["Alice"],  # Only one speaker
                "hosts": ["Alice"],
                "guests": [],
            }
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts={"Alice"},
        )

        # Should have at least 2 speakers (min from config, which defaults to 2)
        # The config's screenplay_num_speakers defaults to 2
        self.assertGreaterEqual(len(speakers), 2)

    def test_detect_hosts_from_feed_authors(self):
        """Test detect_hosts prefers feed_authors."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        hosts = detector.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=["Alice", "Bob"],
        )

        self.assertEqual(hosts, {"Alice", "Bob"})

    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider.detect_speakers")
    def test_detect_hosts_without_authors(self, mock_detect_speakers):
        """Test detect_hosts uses API when no feed_authors."""
        mock_detect_speakers.return_value = (["Alice", "Bob"], {"Alice"}, True)

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        hosts = detector.detect_hosts(
            feed_title="The Podcast",
            feed_description="A great podcast",
            feed_authors=None,
        )

        self.assertEqual(hosts, {"Alice"})
        mock_detect_speakers.assert_called_once()

    def test_detect_hosts_no_title(self):
        """Test detect_hosts returns empty set when no title."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        hosts = detector.detect_hosts(
            feed_title=None,
            feed_description="Description",
            feed_authors=None,
        )

        self.assertEqual(hosts, set())

    @patch("podcast_scraper.openai.openai_provider.OpenAIProvider.detect_speakers")
    def test_detect_hosts_handles_exception(self, mock_detect_speakers):
        """Test detect_hosts handles exceptions gracefully."""
        mock_detect_speakers.side_effect = Exception("API error")

        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        hosts = detector.detect_hosts(
            feed_title="The Podcast",
            feed_description="Description",
            feed_authors=None,
        )

        # Should return empty set on error
        self.assertEqual(hosts, set())

    def test_detect_hosts_not_initialized(self):
        """Test detect_hosts raises error when not initialized."""
        detector = create_speaker_detector(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            detector.detect_hosts("Title", "Description", None)

        self.assertIn("not initialized", str(context.exception))

    def test_analyze_patterns(self):
        """Test analyze_patterns returns None (not implemented)."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        episodes = [create_test_episode(idx=1, title="Episode 1")]
        result = detector.analyze_patterns(episodes, {"Host1"})

        self.assertIsNone(result)

    def test_cleanup(self):
        """Test cleanup method resets initialization state."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()
        self.assertTrue(detector._speaker_detection_initialized)

        # Should not raise
        detector.cleanup()
        # Cleanup resets initialization state
        self.assertFalse(detector._speaker_detection_initialized)

    def test_clear_cache(self):
        """Test clear_cache method (no-op for API provider)."""
        detector = create_speaker_detector(self.cfg)
        detector.initialize()

        # Should not raise
        detector.clear_cache()

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_parse_speakers_from_response_success(self, mock_render_prompt):
        """Test _parse_speakers_from_response with valid JSON."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        response_text = json.dumps(
            {
                "speakers": ["Alice", "Bob", "Charlie"],
                "hosts": ["Alice"],
                "guests": ["Bob", "Charlie"],
            }
        )

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_text

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts={"Alice"},
        )

        self.assertIn("Alice", speakers)
        self.assertIn("Bob", speakers)
        self.assertEqual(detected_hosts, {"Alice"})
        self.assertTrue(success)

    @patch("podcast_scraper.prompt_store.render_prompt")
    def test_parse_speakers_from_text_fallback(self, mock_render_prompt):
        """Test _parse_speakers_from_text fallback parsing."""
        mock_render_prompt.side_effect = ["System prompt", "User prompt"]

        # Invalid JSON but has text patterns
        response_text = 'Speakers: ["Alice", "Bob"]\nHosts: Alice\nGuests: Bob'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = response_text

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        detector = create_speaker_detector(self.cfg)
        detector.client = mock_client
        detector.initialize()

        speakers, detected_hosts, success = detector.detect_speakers(
            episode_title="Test",
            episode_description="Test",
            known_hosts={"Alice"},
        )

        # Should parse from text patterns
        self.assertIsInstance(speakers, list)
        self.assertIsInstance(detected_hosts, set)


if __name__ == "__main__":
    unittest.main()
