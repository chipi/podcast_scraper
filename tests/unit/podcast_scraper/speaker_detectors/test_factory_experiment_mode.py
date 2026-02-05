#!/usr/bin/env python3
"""Tests for speaker detector factory experiment mode (ML providers only).

This module tests experiment-style parameter passing for ML-based speaker detector
providers (spaCy). Other providers (OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama)
have their own dedicated test files (test_*_providers.py) that test both Config-based
and experiment-style usage.
"""

from __future__ import annotations

import unittest

from podcast_scraper import config
from podcast_scraper.providers.params import SpeakerDetectionParams
from podcast_scraper.speaker_detectors.factory import create_speaker_detector


class TestSpeakerDetectorFactoryExperimentModeMLProviders(unittest.TestCase):
    """Test speaker detector factory experiment mode for ML providers.

    Note: Other providers (OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama)
    are tested in their dedicated test files (test_*_providers.py).
    """

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
