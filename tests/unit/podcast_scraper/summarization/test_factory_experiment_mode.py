#!/usr/bin/env python3
"""Tests for summarization factory experiment mode (all providers).

This module tests experiment-style parameter passing for all summarization
providers to improve code coverage.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config
from podcast_scraper.providers.params import SummarizationParams
from podcast_scraper.summarization.factory import create_summarization_provider


class TestSummarizationFactoryExperimentModeAllProviders(unittest.TestCase):
    """Test summarization factory experiment mode for all providers."""

    def test_experiment_mode_transformers_with_params(self):
        """Test experiment mode with transformers provider and params."""
        params = SummarizationParams(model_name="facebook/bart-base", max_length=200, device="cpu")
        provider = create_summarization_provider("transformers", params)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_transformers_without_params(self):
        """Test experiment mode with transformers provider without params."""
        provider = create_summarization_provider("transformers")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_openai_with_params(self):
        """Test experiment mode with OpenAI provider and params."""
        params = SummarizationParams(model_name="gpt-4o-mini", temperature=0.7, max_length=500)
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_summarization_provider("openai", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "OpenAIProvider")

    def test_experiment_mode_openai_without_params(self):
        """Test experiment mode with OpenAI provider without params."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_summarization_provider("openai")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "OpenAIProvider")

    def test_experiment_mode_gemini_with_params(self):
        """Test experiment mode with Gemini provider and params."""
        params = SummarizationParams(model_name="gemini-2.0-flash", temperature=0.5, max_length=500)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            provider = create_summarization_provider("gemini", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "GeminiProvider")

    def test_experiment_mode_gemini_without_params(self):
        """Test experiment mode with Gemini provider without params."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            provider = create_summarization_provider("gemini")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "GeminiProvider")

    def test_experiment_mode_mistral_with_params(self):
        """Test experiment mode with Mistral provider and params."""
        params = SummarizationParams(
            model_name="mistral-small-latest", temperature=0.6, max_length=500
        )
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = create_summarization_provider("mistral", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "MistralProvider")

    def test_experiment_mode_mistral_without_params(self):
        """Test experiment mode with Mistral provider without params."""
        with patch.dict("os.environ", {"MISTRAL_API_KEY": "test-key"}):
            provider = create_summarization_provider("mistral")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "MistralProvider")

    def test_experiment_mode_grok_with_params(self):
        """Test experiment mode with Grok provider and params."""
        params = SummarizationParams(model_name="grok-2", temperature=0.5, max_length=500)
        with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}):
            provider = create_summarization_provider("grok", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "GrokProvider")

    def test_experiment_mode_grok_without_params(self):
        """Test experiment mode with Grok provider without params."""
        with patch.dict("os.environ", {"GROK_API_KEY": "test-key"}):
            provider = create_summarization_provider("grok")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "GrokProvider")

    def test_experiment_mode_deepseek_with_params(self):
        """Test experiment mode with DeepSeek provider and params."""
        params = SummarizationParams(model_name="deepseek-chat", temperature=0.5, max_length=500)
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            provider = create_summarization_provider("deepseek", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "DeepSeekProvider")

    def test_experiment_mode_deepseek_without_params(self):
        """Test experiment mode with DeepSeek provider without params."""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            provider = create_summarization_provider("deepseek")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "DeepSeekProvider")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_experiment_mode_ollama_with_params(self, mock_httpx):
        """Test experiment mode with Ollama provider and params."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response
        params = SummarizationParams(model_name="llama3.3:latest", temperature=0.5, max_length=500)
        with patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434/v1"}):
            provider = create_summarization_provider("ollama", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "OllamaProvider")

    @patch("podcast_scraper.providers.ollama.ollama_provider.httpx")
    def test_experiment_mode_ollama_without_params(self, mock_httpx):
        """Test experiment mode with Ollama provider without params."""
        # Mock health check
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_httpx.get.return_value = mock_response
        with patch.dict("os.environ", {"OLLAMA_API_BASE": "http://localhost:11434/v1"}):
            provider = create_summarization_provider("ollama")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "OllamaProvider")

    def test_experiment_mode_anthropic_with_params(self):
        """Test experiment mode with Anthropic provider and params."""
        params = SummarizationParams(
            model_name="claude-3-5-sonnet-20241022", temperature=0.5, max_length=500
        )
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_summarization_provider("anthropic", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "AnthropicProvider")

    def test_experiment_mode_anthropic_without_params(self):
        """Test experiment mode with Anthropic provider without params."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = create_summarization_provider("anthropic")
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "AnthropicProvider")

    def test_experiment_mode_with_params_dict(self):
        """Test experiment mode with params as dict."""
        params = {
            "model_name": "facebook/bart-base",
            "max_length": 200,
            "device": "cpu",
        }
        provider = create_summarization_provider("transformers", params)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_invalid_provider_type(self):
        """Test experiment mode raises ValueError for invalid provider type."""
        with self.assertRaises(ValueError) as context:
            create_summarization_provider("invalid_provider")
        self.assertIn("Invalid provider type", str(context.exception))

    def test_experiment_mode_invalid_params_type(self):
        """Test experiment mode raises TypeError for invalid params type."""
        with self.assertRaises(TypeError) as context:
            create_summarization_provider("transformers", "invalid_params")
        self.assertIn("params must be SummarizationParams or dict", str(context.exception))

    def test_config_mode_with_params_raises_typeerror(self):
        """Test that providing params with Config raises TypeError."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="transformers")
        params = SummarizationParams(model_name="facebook/bart-base")
        with self.assertRaises(TypeError) as context:
            create_summarization_provider(cfg, params)
        self.assertIn("Cannot provide params when using Config-based mode", str(context.exception))
