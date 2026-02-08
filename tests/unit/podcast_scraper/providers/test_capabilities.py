"""Unit tests for provider capability contract system."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, Mock, patch

# Mock openai before importing modules that require it
# Unit tests run without openai package installed
# Use patch.dict without 'with' to avoid context manager conflicts with @patch decorators
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.providers.capabilities import (
    get_provider_capabilities,
    ProviderCapabilities,
)


class TestProviderCapabilities(unittest.TestCase):
    """Test ProviderCapabilities dataclass."""

    def test_capabilities_creation(self):
        """Test creating a ProviderCapabilities instance."""
        caps = ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_audio_input=True,
            supports_json_mode=True,
            max_context_tokens=128000,
            provider_name="test",
        )
        self.assertTrue(caps.supports_transcription)
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertTrue(caps.supports_audio_input)
        self.assertTrue(caps.supports_json_mode)
        self.assertEqual(caps.max_context_tokens, 128000)
        self.assertEqual(caps.provider_name, "test")

    def test_capabilities_defaults(self):
        """Test ProviderCapabilities default values."""
        caps = ProviderCapabilities(
            supports_transcription=False,
            supports_speaker_detection=False,
            supports_summarization=False,
            supports_audio_input=False,
            supports_json_mode=False,
            max_context_tokens=0,
        )
        # Default values
        self.assertTrue(caps.supports_tool_calls)  # Default True
        self.assertTrue(caps.supports_system_prompt)  # Default True
        self.assertFalse(caps.supports_streaming)  # Default False
        self.assertEqual(caps.provider_name, "unknown")  # Default

    def test_capabilities_string_representation(self):
        """Test string representation of capabilities."""
        caps = ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_audio_input=True,
            supports_json_mode=True,
            max_context_tokens=128000,
            provider_name="test",
        )
        str_repr = str(caps)
        self.assertIn("test", str_repr)
        self.assertIn("transcription", str_repr)
        self.assertIn("speaker_detection", str_repr)
        self.assertIn("summarization", str_repr)
        self.assertIn("json_mode", str_repr)
        self.assertIn("max_tokens=128000", str_repr)

    def test_capabilities_frozen(self):
        """Test that ProviderCapabilities is frozen (immutable)."""
        caps = ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=False,
            supports_summarization=False,
            supports_audio_input=False,
            supports_json_mode=False,
            max_context_tokens=0,
        )
        # Attempting to modify should raise AttributeError
        with self.assertRaises(Exception):  # dataclass frozen raises FrozenInstanceError
            caps.supports_transcription = False


class TestGetProviderCapabilities(unittest.TestCase):
    """Test get_provider_capabilities() function."""

    def test_explicit_get_capabilities_method(self):
        """Test that explicit get_capabilities() method is called if available."""
        mock_provider = Mock()
        expected_caps = ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_audio_input=True,
            supports_json_mode=True,
            max_context_tokens=128000,
            provider_name="mock",
        )
        mock_provider.get_capabilities.return_value = expected_caps

        result = get_provider_capabilities(mock_provider)
        self.assertEqual(result, expected_caps)
        mock_provider.get_capabilities.assert_called_once()

    def test_fallback_to_introspection(self):
        """Test fallback to introspection when get_capabilities() not available."""
        # Create a mock that doesn't have get_capabilities method
        # Use spec=[] to prevent auto-creation of methods
        mock_provider = Mock(spec=[])
        mock_provider.transcribe = Mock()  # Has transcription
        mock_provider.detect_speakers = Mock()  # Has speaker detection
        mock_provider.summarize = Mock()  # Has summarization
        mock_provider.max_context_tokens = 128000
        # Set __name__ on the type, not the instance
        mock_provider.__class__.__name__ = "TestProvider"

        result = get_provider_capabilities(mock_provider)
        self.assertIsInstance(result, ProviderCapabilities)
        self.assertTrue(result.supports_transcription)
        self.assertTrue(result.supports_speaker_detection)
        self.assertTrue(result.supports_summarization)
        self.assertEqual(result.max_context_tokens, 128000)

    def test_get_capabilities_error_fallback(self):
        """Test that errors in get_capabilities() fall back to introspection."""
        mock_provider = Mock()
        mock_provider.get_capabilities.side_effect = ValueError("Error")
        mock_provider.transcribe = Mock()
        mock_provider.max_context_tokens = 0
        type(mock_provider).__name__ = "TestProvider"

        # Should not raise, should fall back to introspection
        result = get_provider_capabilities(mock_provider)
        self.assertIsInstance(result, ProviderCapabilities)

    def test_ml_provider_capabilities(self):
        """Test MLProvider capabilities."""
        cfg = config.Config(rss_url="https://example.com", transcription_provider="whisper")
        from podcast_scraper.providers.ml.ml_provider import MLProvider

        provider = MLProvider(cfg)
        caps = get_provider_capabilities(provider)

        self.assertTrue(caps.supports_transcription)
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertTrue(caps.supports_audio_input)
        self.assertFalse(caps.supports_json_mode)  # ML providers don't support JSON mode
        self.assertFalse(caps.supports_tool_calls)  # ML models don't support tool calls
        self.assertFalse(caps.supports_system_prompt)  # ML models don't use system prompts
        self.assertEqual(caps.provider_name, "ml")

    def test_openai_provider_capabilities(self):
        """Test OpenAIProvider capabilities."""
        cfg = config.Config(
            rss_url="https://example.com",
            transcription_provider="openai",
            openai_api_key="test-key",
        )
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

        provider = OpenAIProvider(cfg)
        caps = get_provider_capabilities(provider)

        self.assertTrue(caps.supports_transcription)
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertTrue(caps.supports_audio_input)
        self.assertTrue(caps.supports_json_mode)
        self.assertTrue(caps.supports_tool_calls)
        self.assertTrue(caps.supports_system_prompt)
        self.assertEqual(caps.provider_name, "openai")
        self.assertEqual(caps.max_context_tokens, 128000)

    def test_gemini_provider_capabilities(self):
        """Test GeminiProvider capabilities."""
        cfg = config.Config(
            rss_url="https://example.com",
            transcription_provider="gemini",
            gemini_api_key="test-key",
        )
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        provider = GeminiProvider(cfg)
        caps = get_provider_capabilities(provider)

        self.assertTrue(caps.supports_transcription)
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertTrue(caps.supports_audio_input)
        self.assertTrue(caps.supports_json_mode)
        self.assertTrue(caps.supports_tool_calls)
        self.assertTrue(caps.supports_system_prompt)
        self.assertEqual(caps.provider_name, "gemini")
        self.assertEqual(caps.max_context_tokens, 2000000)

    def test_anthropic_provider_capabilities(self):
        """Test AnthropicProvider capabilities."""
        cfg = config.Config(
            rss_url="https://example.com",
            speaker_detector_provider="anthropic",
            anthropic_api_key="test-key",
        )
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider(cfg)
        caps = get_provider_capabilities(provider)

        self.assertFalse(caps.supports_transcription)  # Anthropic doesn't support transcription
        self.assertTrue(caps.supports_speaker_detection)
        self.assertTrue(caps.supports_summarization)
        self.assertFalse(caps.supports_audio_input)  # Anthropic doesn't accept audio
        self.assertTrue(caps.supports_json_mode)
        self.assertTrue(caps.supports_tool_calls)
        self.assertTrue(caps.supports_system_prompt)
        self.assertEqual(caps.provider_name, "anthropic")
        self.assertEqual(caps.max_context_tokens, 200000)


if __name__ == "__main__":
    unittest.main()
