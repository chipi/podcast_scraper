#!/usr/bin/env python3
"""Tests for speaker detector factory experiment mode (all providers).

This module tests experiment-style parameter passing for all speaker detector
providers to improve code coverage.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config
from podcast_scraper.providers.params import SpeakerDetectionParams
from podcast_scraper.speaker_detectors.factory import create_speaker_detector


class TestSpeakerDetectorFactoryExperimentModeAllProviders(unittest.TestCase):
    """Test speaker detector factory experiment mode for all providers."""

    def test_experiment_mode_spacy_with_params(self):
        """Test experiment mode with spaCy provider and params."""
        params = SpeakerDetectionParams(model_name="en_core_web_sm", temperature=0.5)
        detector = create_speaker_detector("spacy", params)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "MLProvider")

    def test_experiment_mode_spacy_without_params(self):
        """Test experiment mode with spaCy provider without params (uses defaults)."""
        detector = create_speaker_detector("spacy")
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "MLProvider")

    def test_experiment_mode_openai_with_params(self):
        """Test experiment mode with OpenAI provider and params."""
        params = SpeakerDetectionParams(model_name="gpt-4o-mini", temperature=0.7)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            detector = create_speaker_detector("openai", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "OpenAIProvider")

    def test_experiment_mode_openai_without_params(self):
        """Test experiment mode with OpenAI provider without params."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            detector = create_speaker_detector("openai")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "OpenAIProvider")

    def test_experiment_mode_gemini_with_params(self):
        """Test experiment mode with Gemini provider and params."""
        params = SpeakerDetectionParams(model_name="gemini-2.0-flash", temperature=0.5)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            detector = create_speaker_detector("gemini", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "GeminiProvider")

    def test_experiment_mode_gemini_without_params(self):
        """Test experiment mode with Gemini provider without params."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            detector = create_speaker_detector("gemini")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "GeminiProvider")

    def test_experiment_mode_mistral_with_params(self):
        """Test experiment mode with Mistral provider and params."""
        params = SpeakerDetectionParams(model_name="mistral-small-latest", temperature=0.6)
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            detector = create_speaker_detector("mistral", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "MistralProvider")

    def test_experiment_mode_mistral_without_params(self):
        """Test experiment mode with Mistral provider without params."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            detector = create_speaker_detector("mistral")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "MistralProvider")

    def test_experiment_mode_grok_with_params(self):
        """Test experiment mode with Grok provider and params."""
        params = SpeakerDetectionParams(model_name="grok-2", temperature=0.5)
        with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}):
            detector = create_speaker_detector("grok", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "GrokProvider")

    def test_experiment_mode_grok_without_params(self):
        """Test experiment mode with Grok provider without params."""
        with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}):
            detector = create_speaker_detector("grok")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "GrokProvider")

    def test_experiment_mode_deepseek_with_params(self):
        """Test experiment mode with DeepSeek provider and params."""
        params = SpeakerDetectionParams(model_name="deepseek-chat", temperature=0.5)
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            detector = create_speaker_detector("deepseek", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "DeepSeekProvider")

    def test_experiment_mode_deepseek_without_params(self):
        """Test experiment mode with DeepSeek provider without params."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            detector = create_speaker_detector("deepseek")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "DeepSeekProvider")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_experiment_mode_ollama_with_params(self, mock_httpx):
        """Test experiment mode with Ollama provider and params."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response
        params = SpeakerDetectionParams(model_name="llama3.3:latest", temperature=0.5)
        with patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434/v1"}):
            detector = create_speaker_detector("ollama", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "OllamaProvider")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_experiment_mode_ollama_without_params(self, mock_httpx):
        """Test experiment mode with Ollama provider without params."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response
        with patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434/v1"}):
            detector = create_speaker_detector("ollama")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "OllamaProvider")

    def test_experiment_mode_anthropic_with_params(self):
        """Test experiment mode with Anthropic provider and params."""
        params = SpeakerDetectionParams(model_name="claude-3-5-sonnet-20241022", temperature=0.5)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            detector = create_speaker_detector("anthropic", params)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "AnthropicProvider")

    def test_experiment_mode_anthropic_without_params(self):
        """Test experiment mode with Anthropic provider without params."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            detector = create_speaker_detector("anthropic")
            self.assertIsNotNone(detector)
            self.assertEqual(detector.__class__.__name__, "AnthropicProvider")

    def test_experiment_mode_with_params_dict(self):
        """Test experiment mode with params as dict."""
        params = {"model_name": "en_core_web_sm", "temperature": 0.5}
        detector = create_speaker_detector("spacy", params)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "MLProvider")

    def test_experiment_mode_invalid_provider_type(self):
        """Test experiment mode raises ValueError for invalid provider type."""
        with self.assertRaises(ValueError) as context:
            create_speaker_detector("invalid_provider")
        self.assertIn("Invalid provider type", str(context.exception))

    def test_experiment_mode_invalid_params_type(self):
        """Test experiment mode raises TypeError for invalid params type."""
        with self.assertRaises(TypeError) as context:
            create_speaker_detector("spacy", "invalid_params")
        self.assertIn("params must be SpeakerDetectionParams or dict", str(context.exception))

    def test_config_mode_with_params_raises_typeerror(self):
        """Test that providing params with Config raises TypeError."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", speaker_detector_provider="spacy"
        )
        params = SpeakerDetectionParams(model_name="en_core_web_sm")
        with self.assertRaises(TypeError) as context:
            create_speaker_detector(cfg, params)
        self.assertIn("Cannot provide params when using Config-based mode", str(context.exception))
