#!/usr/bin/env python3
"""Tests for summarization factory experiment mode (ML providers only).

This module tests experiment-style parameter passing for ML-based summarization
providers (transformers). Other providers (OpenAI, Gemini, Anthropic, Mistral,
Grok, DeepSeek, Ollama) have their own dedicated test files (test_*_providers.py)
that test both Config-based and experiment-style usage.
"""

from __future__ import annotations

import unittest

from podcast_scraper import config
from podcast_scraper.providers.params import SummarizationParams
from podcast_scraper.summarization.factory import create_summarization_provider


class TestSummarizationFactoryExperimentModeMLProviders(unittest.TestCase):
    """Test summarization factory experiment mode for ML providers.

    Note: Other providers (OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama)
    are tested in their dedicated test files (test_*_providers.py).
    """

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
