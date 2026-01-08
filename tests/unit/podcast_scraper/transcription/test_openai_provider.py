#!/usr/bin/env python3
"""Unit tests for OpenAIProvider transcription (via factory).

These tests verify the OpenAI Whisper API-based transcription provider implementation
using the unified OpenAIProvider returned by the factory.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

from podcast_scraper import config  # noqa: E402
from podcast_scraper.transcription.factory import create_transcription_provider  # noqa: E402


class TestOpenAITranscriptionProvider(unittest.TestCase):
    """Tests for OpenAIProvider transcription (via factory)."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcription_provider="openai",
            openai_api_key="sk-test123",
            transcribe_missing=True,
        )

    def test_init_success(self):
        """Test successful initialization."""
        provider = create_transcription_provider(self.cfg)
        self.assertEqual(provider.cfg, self.cfg)
        self.assertIsNotNone(provider.client)
        self.assertEqual(
            provider.transcription_model, config.TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
        )
        self.assertFalse(provider._transcription_initialized)

    def test_init_missing_api_key(self):
        """Test initialization raises error when API key is missing."""
        # Create config without API key by using a mock config
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.transcription_provider = "openai"
        mock_cfg.openai_api_key = None
        mock_cfg.openai_api_base = None
        # getattr should return None for openai_transcription_model
        type(mock_cfg).openai_transcription_model = property(
            lambda self: getattr(self, "_openai_transcription_model", None)
        )

        with self.assertRaises(ValueError) as context:
            create_transcription_provider(mock_cfg)

        self.assertIn("OpenAI API key required", str(context.exception))

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base_url."""
        cfg = create_test_config(
            transcription_provider="openai",
            openai_api_key="sk-test123",
            openai_api_base="https://api.example.com/v1",
        )
        provider = create_transcription_provider(cfg)
        # Verify client was created with custom base_url
        self.assertIsNotNone(provider.client)

    def test_init_with_custom_model(self):
        """Test initialization with custom transcription model."""
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.transcription_provider = "openai"
        mock_cfg.openai_api_key = "sk-test123"
        mock_cfg.openai_api_base = None
        # Use getattr to simulate openai_transcription_model attribute
        type(mock_cfg).openai_transcription_model = property(lambda self: "whisper-2")

        provider = create_transcription_provider(mock_cfg)
        self.assertEqual(provider.transcription_model, "whisper-2")

    def test_initialize_success(self):
        """Test successful initialization."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        self.assertTrue(provider._transcription_initialized)

    def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()
        # Call again
        provider.initialize()

        self.assertTrue(provider._transcription_initialized)

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open):
        """Test successful transcription."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_file.read.return_value = b"fake audio data"
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = "Hello world"
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.transcribe("/tmp/audio.mp3")

        self.assertEqual(result, "Hello world")
        mock_client.audio.transcriptions.create.assert_called_once()

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(self, mock_exists, mock_open):
        """Test transcription with explicit language."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = "Bonjour"
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result = provider.transcribe("/tmp/audio.mp3", language="fr")

        self.assertEqual(result, "Bonjour")
        # Verify language was passed to API
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["language"], "fr")

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_config_language(self, mock_exists, mock_open):
        """Test transcription uses config language when not provided."""
        cfg = create_test_config(
            transcription_provider="openai",
            openai_api_key="sk-test123",
            language="es",
            transcribe_missing=True,
        )
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = "Hola"
        provider = create_transcription_provider(cfg)
        provider.client = mock_client
        provider.initialize()

        provider.transcribe("/tmp/audio.mp3")

        # Verify language from config was used
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["language"], "es")

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_auto_detects_language(self, mock_exists, mock_open):
        """Test transcription auto-detects language when not specified."""
        # Create config without language (or with language=None)
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.transcription_provider = "openai"
        mock_cfg.openai_api_key = "sk-test123"
        mock_cfg.openai_api_base = None
        mock_cfg.language = None  # No default language
        type(mock_cfg).openai_transcription_model = property(
            lambda self: config.TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
        )

        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = "Hello"
        provider = create_transcription_provider(mock_cfg)
        provider.client = mock_client
        provider.initialize()

        provider.transcribe("/tmp/audio.mp3", language=None)

        # Verify language was not passed (auto-detect)
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertNotIn("language", call_kwargs)

    def test_transcribe_not_initialized(self):
        """Test transcribe raises error when not initialized."""
        provider = create_transcription_provider(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists):
        """Test transcribe raises error when file doesn't exist."""
        mock_exists.return_value = False

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError) as context:
            provider.transcribe("/tmp/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error(self, mock_exists, mock_open):
        """Test transcribe handles API errors."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe("/tmp/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_success(self, mock_exists, mock_open):
        """Test transcribe_with_segments returns full result."""
        mock_exists.return_value = True
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
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/tmp/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 2)
        self.assertEqual(result_dict["segments"][0]["start"], 0.0)
        self.assertEqual(result_dict["segments"][0]["end"], 1.0)
        self.assertEqual(result_dict["segments"][0]["text"], "Hello")
        self.assertIsInstance(elapsed, float)
        self.assertGreaterEqual(elapsed, 0)

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_dict_segments(self, mock_exists, mock_open):
        """Test transcribe_with_segments handles dict-like segments."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        # Mock response with dict-like segments
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/tmp/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(len(result_dict["segments"]), 2)
        self.assertEqual(result_dict["segments"][0]["start"], 0.0)

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_no_segments(self, mock_exists, mock_open):
        """Test transcribe_with_segments handles response without segments."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        result_dict, elapsed = provider.transcribe_with_segments("/tmp/audio.mp3")

        self.assertEqual(result_dict["text"], "Hello world")
        self.assertEqual(result_dict["segments"], [])

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_api_error(self, mock_exists, mock_open):
        """Test transcribe_with_segments handles API errors."""
        mock_exists.return_value = True
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.side_effect = Exception("API error")
        provider = create_transcription_provider(self.cfg)
        provider.client = mock_client
        provider.initialize()

        with self.assertRaises(ValueError) as context:
            provider.transcribe_with_segments("/tmp/audio.mp3")

        self.assertIn("transcription failed", str(context.exception))

    def test_transcribe_with_segments_not_initialized(self):
        """Test transcribe_with_segments raises error when not initialized."""
        provider = create_transcription_provider(self.cfg)

        with self.assertRaises(RuntimeError) as context:
            provider.transcribe_with_segments("/tmp/audio.mp3")

        self.assertIn("not initialized", str(context.exception))

    @patch("os.path.exists")
    def test_transcribe_with_segments_file_not_found(self, mock_exists):
        """Test transcribe_with_segments raises error when file doesn't exist."""
        mock_exists.return_value = False

        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        with self.assertRaises(FileNotFoundError) as context:
            provider.transcribe_with_segments("/tmp/nonexistent.mp3")

        self.assertIn("not found", str(context.exception))

    def test_cleanup(self):
        """Test cleanup method resets initialization state."""
        provider = create_transcription_provider(self.cfg)
        provider.initialize()

        # Should not raise
        provider.cleanup()
        # Cleanup resets initialization state
        self.assertFalse(provider._transcription_initialized)


if __name__ == "__main__":
    unittest.main()
