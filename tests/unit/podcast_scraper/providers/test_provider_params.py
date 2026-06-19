"""Tests for provider parameter models and experiment-style factory usage.

This module tests the new experiment-style parameter passing functionality
for provider factories, ensuring backward compatibility with Config-based usage.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock, Mock, patch

# Mock openai for factory imports; unit-only pytest (``make test-ci-fast``).
mock_openai = MagicMock()
mock_openai.OpenAI = Mock()
# Give the mock a truthy __spec__ so importlib.util.find_spec("openai") doesn't
# crash when a later test probes package availability. patch.dict.start() without
# matching .stop() leaves this mock in sys.modules for the rest of the session.
mock_openai.__spec__ = importlib.util.spec_from_loader("openai", loader=None)
_patch_openai = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
    },
)
_patch_openai.start()

from podcast_scraper import config
from podcast_scraper.providers.params import (
    SpeakerDetectionParams,
    SummarizationParams,
    TranscriptionParams,
)
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


class TestProviderParamsModels(unittest.TestCase):
    """Test ProviderParams Pydantic models."""

    def test_summarization_params_creation(self):
        """Test SummarizationParams can be created with required fields."""
        params = SummarizationParams(model_name="facebook/bart-large-cnn")
        self.assertEqual(params.model_name, "facebook/bart-large-cnn")
        self.assertEqual(params.max_length, 150)  # Default
        self.assertEqual(params.min_length, 30)  # Default

    def test_summarization_params_validation(self):
        """Test SummarizationParams validation."""
        # Valid params
        params = SummarizationParams(
            model_name="facebook/bart-large-cnn",
            max_length=200,
            min_length=50,
            device="mps",
        )
        self.assertEqual(params.max_length, 200)
        self.assertEqual(params.device, "mps")

        # Invalid device
        with self.assertRaises(ValueError):
            SummarizationParams(model_name="test", device="invalid")

        # Invalid temperature range
        with self.assertRaises(ValueError):
            SummarizationParams(model_name="test", temperature=3.0)

    def test_summarization_params_top_p_default_is_none(self):
        """#108 — top_p is opt-in; default None means 'use provider default'."""
        params = SummarizationParams(model_name="test")
        self.assertIsNone(params.top_p)

    def test_summarization_params_top_p_valid_range(self):
        """#108 — top_p accepts the [0.0, 1.0] nucleus-sampling range."""
        for valid in (0.0, 0.5, 0.8, 1.0):
            params = SummarizationParams(model_name="test", top_p=valid)
            self.assertEqual(params.top_p, valid)

    def test_summarization_params_top_p_rejects_out_of_range(self):
        """#108 — top_p < 0 or > 1 fails Pydantic validation."""
        with self.assertRaises(ValueError):
            SummarizationParams(model_name="test", top_p=-0.1)
        with self.assertRaises(ValueError):
            SummarizationParams(model_name="test", top_p=1.5)

    def test_summarization_params_response_format_default_is_none(self):
        """#108 — response_format default None means plain text."""
        params = SummarizationParams(model_name="test")
        self.assertIsNone(params.response_format)

    def test_summarization_params_response_format_accepts_json_object(self):
        """#108 — common OpenAI structured-JSON spec round-trips."""
        spec = {"type": "json_object"}
        params = SummarizationParams(model_name="test", response_format=spec)
        self.assertEqual(params.response_format, spec)

    def test_summarization_params_response_format_accepts_json_schema(self):
        """#108 — full JSON schema spec round-trips (Kimi-Linear v3_json shape)."""
        spec = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary",
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        }
        params = SummarizationParams(model_name="test", response_format=spec)
        self.assertEqual(params.response_format, spec)

    def test_summarization_params_model_dump_omits_none_top_p_and_response_format(self):
        """#108 — model_dump (overridden in params.py) strips None values, so
        configs that don't set top_p / response_format don't materialize them
        in serialized payloads. Important for provider-side compat."""
        params = SummarizationParams(model_name="test")
        dumped = params.model_dump()
        self.assertNotIn("top_p", dumped)
        self.assertNotIn("response_format", dumped)
        self.assertEqual(dumped["model_name"], "test")

    def test_summarization_params_model_dump_includes_set_top_p(self):
        """#108 — when top_p is set, it surfaces in model_dump for downstream
        providers (or extra_body wiring) to pick up."""
        params = SummarizationParams(
            model_name="test", top_p=0.8, response_format={"type": "json_object"}
        )
        dumped = params.model_dump()
        self.assertEqual(dumped["top_p"], 0.8)
        self.assertEqual(dumped["response_format"], {"type": "json_object"})

    def test_transcription_params_creation(self):
        """Test TranscriptionParams can be created."""
        params = TranscriptionParams(model_name="base.en")
        self.assertEqual(params.model_name, "base.en")
        self.assertIsNone(params.device)  # Default
        self.assertIsNone(params.language)  # Default

    def test_speaker_detection_params_creation(self):
        """Test SpeakerDetectionParams can be created."""
        params = SpeakerDetectionParams(model_name="en_core_web_sm")
        self.assertEqual(params.model_name, "en_core_web_sm")
        self.assertEqual(params.temperature, 0.3)  # Default


class TestSummarizationFactoryExperimentMode(unittest.TestCase):
    """Test summarization factory with experiment-style params."""

    def test_config_mode_backward_compatible(self):
        """Test Config-based mode still works (backward compatibility)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_with_params_dict(self):
        """Test experiment mode with params as dict."""
        params = {
            "model_name": "facebook/bart-large-cnn",
            "max_length": 200,
            "device": "cpu",
        }
        provider = create_summarization_provider("transformers", params)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_with_params_object(self):
        """Test experiment mode with SummarizationParams object."""
        params = SummarizationParams(
            model_name="facebook/bart-large-cnn",
            max_length=200,
            device="cpu",
        )
        provider = create_summarization_provider("transformers", params)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_without_params(self):
        """Test experiment mode without params (uses defaults)."""
        provider = create_summarization_provider("transformers")
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_openai(self):
        """Test experiment mode with OpenAI provider."""
        params = SummarizationParams(
            model_name="gpt-4o-mini",
            temperature=0.7,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = create_summarization_provider("openai", params)
            self.assertIsNotNone(provider)
            self.assertEqual(provider.__class__.__name__, "OpenAIProvider")

    def test_error_params_with_config(self):
        """Test error when params provided with Config."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        params = SummarizationParams(model_name="test")
        with self.assertRaises(TypeError):
            create_summarization_provider(cfg, params)

    def test_error_invalid_params_type(self):
        """Test error when params is invalid type."""
        with self.assertRaises(TypeError):
            create_summarization_provider("transformers", "invalid")


class TestTranscriptionFactoryExperimentMode(unittest.TestCase):
    """Test transcription factory with experiment-style params."""

    def test_config_mode_backward_compatible(self):
        """Test Config-based mode still works."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
        )
        provider = create_transcription_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_experiment_mode_with_params(self):
        """Test experiment mode with params."""
        params = TranscriptionParams(
            model_name="base.en",
            device="cpu",
            language="en",
        )
        provider = create_transcription_provider("whisper", params)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "MLProvider")

    def test_error_params_with_config(self):
        """Test error when params provided with Config."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        params = TranscriptionParams(model_name="test")
        with self.assertRaises(TypeError):
            create_transcription_provider(cfg, params)


class TestSpeakerDetectionFactoryExperimentMode(unittest.TestCase):
    """Test speaker detection factory with experiment-style params."""

    def test_config_mode_backward_compatible(self):
        """Test Config-based mode still works."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
        )
        detector = create_speaker_detector(cfg)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "MLProvider")

    def test_experiment_mode_with_params(self):
        """Test experiment mode with params."""
        params = SpeakerDetectionParams(
            model_name="en_core_web_sm",
            temperature=0.5,
        )
        detector = create_speaker_detector("spacy", params)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.__class__.__name__, "MLProvider")

    def test_error_params_with_config(self):
        """Test error when params provided with Config."""
        cfg = config.Config(rss_url="https://example.com/feed.xml")
        params = SpeakerDetectionParams(model_name="test")
        with self.assertRaises(TypeError):
            create_speaker_detector(cfg, params)
