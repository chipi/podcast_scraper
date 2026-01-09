#!/usr/bin/env python3
"""Tests for summarization provider (Stage 4).

These tests verify that the summarization provider pattern works correctly.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

# Mock ML dependencies before importing modules that require them
# Unit tests run without ML dependencies installed
with patch.dict("sys.modules", {"torch": MagicMock(), "transformers": MagicMock()}):
    from podcast_scraper import config
    from podcast_scraper.exceptions import ProviderNotInitializedError
    from podcast_scraper.summarization.factory import create_summarization_provider


class TestSummarizationProviderFactory(unittest.TestCase):
    """Test summarization provider factory."""

    def test_create_local_provider(self):
        """Test that factory creates MLProvider for 'transformers'."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_create_invalid_provider_raises_error(self):
        """Test that factory raises ValueError for unsupported provider."""
        # Note: Config validation rejects invalid providers at creation time,
        # so we test with a mock config that bypasses validation
        from unittest.mock import MagicMock

        mock_cfg = MagicMock()
        mock_cfg.summary_provider = "invalid"

        with self.assertRaises(ValueError) as context:
            create_summarization_provider(mock_cfg)

        self.assertIn("Unsupported summarization provider: invalid", str(context.exception))
        self.assertIn("Supported providers: 'transformers', 'openai'", str(context.exception))

    def test_create_invalid_provider(self):
        """Test that factory raises ValueError for invalid provider type."""
        with self.assertRaises(ValueError) as context:
            from podcast_scraper.summarization.factory import create_summarization_provider

            class MockConfig:
                summary_provider = "invalid"

            create_summarization_provider(MockConfig())  # type: ignore[arg-type]

        self.assertIn("Unsupported summarization provider", str(context.exception))

    def test_factory_returns_provider_instance(self):
        """Test that factory returns a provider instance."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_summarization_provider(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


class TestMLProviderSummarizationViaFactory(unittest.TestCase):
    """Test MLProvider summarization capability via factory."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,  # Disable to avoid loading models
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
        )

    def test_provider_creation_via_factory(self):
        """Test that provider can be created via factory."""
        provider = create_summarization_provider(self.cfg)
        self.assertIsNotNone(provider)
        # Verify it's the unified ML provider
        self.assertEqual(provider.__class__.__name__, "MLProvider")
        # Verify protocol compliance
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

    def test_provider_initialization_state(self):
        """Test that provider tracks initialization state."""
        provider = create_summarization_provider(self.cfg)
        # Initially not initialized (generate_summaries is False)
        self.assertFalse(provider._transformers_initialized)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_initialize_loads_models(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that initialize() loads MAP and REDUCE models via factory."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )

        mock_map_model = Mock()
        mock_map_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_map_model.device = "cpu"
        mock_reduce_model = Mock()
        mock_reduce_model.model_name = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_reduce_model.device = "cpu"
        mock_summary_model.side_effect = [mock_map_model, mock_reduce_model]
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

        provider = create_summarization_provider(cfg)
        provider.initialize()

        self.assertTrue(provider._transformers_initialized)
        self.assertEqual(provider.map_model, mock_map_model)
        self.assertEqual(provider.reduce_model, mock_reduce_model)

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_initialize_same_model(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that initialize() reuses MAP model for REDUCE when same."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )

        mock_map_model = Mock()
        mock_map_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL  # Same model

        provider = create_summarization_provider(cfg)
        provider.initialize()

        self.assertTrue(provider._transformers_initialized)
        self.assertEqual(provider.map_model, mock_map_model)
        self.assertEqual(provider.reduce_model, mock_map_model)  # Same instance

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_summarize(
        self, mock_summary_model, mock_select_map, mock_select_reduce, mock_summarize_long
    ):
        """Test that summarize() calls summarize_long_text() via factory."""
        from podcast_scraper import config

        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )

        mock_map_model = Mock()
        mock_map_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summarize_long.return_value = "Test summary"

        provider = create_summarization_provider(cfg)
        provider.initialize()

        result = provider.summarize("Test transcript text")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Test summary")
        self.assertIsNone(result["summary_short"])
        self.assertIn("metadata", result)
        from podcast_scraper import config

        self.assertEqual(result["metadata"]["model_used"], config.TEST_DEFAULT_SUMMARY_MODEL)
        mock_summarize_long.assert_called_once()

    def test_provider_summarize_not_initialized(self):
        """Test that summarize() raises RuntimeError if not initialized."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )
        provider = create_summarization_provider(cfg)
        # Don't call initialize()

        with self.assertRaises(ProviderNotInitializedError) as context:
            provider.summarize("Test text")

        self.assertIn("not initialized", str(context.exception))
        self.assertEqual(context.exception.provider, "MLProvider/Transformers")

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_cleanup(self, mock_summary_model, mock_select_map, mock_select_reduce):
        """Test that cleanup() resets state."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )

        mock_map_model = Mock()
        mock_map_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

        provider = create_summarization_provider(cfg)
        provider.initialize()
        self.assertTrue(provider._transformers_initialized)

        provider.cleanup()

        self.assertFalse(provider._transformers_initialized)
        self.assertIsNone(provider.map_model)
        self.assertIsNone(provider.reduce_model)

    @patch("podcast_scraper.ml.ml_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_provider_summarize_with_params(
        self, mock_summary_model, mock_select_map, mock_select_reduce, mock_summarize_long
    ):
        """Test that summarize() passes params correctly."""
        cfg = config.Config(
            rss_url=self.cfg.rss_url,
            summary_provider=self.cfg.summary_provider,
            generate_summaries=True,
            auto_speakers=False,  # Disable to avoid loading spaCy
            generate_metadata=True,  # Required when generate_summaries is True
            summary_model=self.cfg.summary_model,
        )

        mock_map_model = Mock()
        mock_map_model.model_name = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summarize_long.return_value = "Test summary"

        provider = create_summarization_provider(cfg)
        provider.initialize()

        params = {
            "max_length": 200,
            "min_length": 50,
            "chunk_size": 512,
            "chunk_parallelism": 4,  # Use chunk_parallelism instead of batch_size
            "prompt": "Custom prompt",
        }
        provider.summarize("Test transcript", params=params)

        # Verify summarize_long_text was called with correct params
        call_args = mock_summarize_long.call_args
        self.assertEqual(call_args[1]["max_length"], 200)
        self.assertEqual(call_args[1]["min_length"], 50)
        self.assertEqual(call_args[1]["chunk_size"], 512)
        # chunk_parallelism is converted to batch_size internally (for CPU device)
        self.assertEqual(call_args[1]["batch_size"], 4)
        self.assertEqual(call_args[1]["prompt"], "Custom prompt")


class TestSummarizationProviderProtocol(unittest.TestCase):
    """Test that MLProvider implements SummarizationProvider protocol (via factory)."""

    def test_provider_implements_protocol(self):
        """Test that MLProvider implements SummarizationProvider protocol."""

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
            auto_speakers=False,  # Disable to avoid loading spaCy
        )
        provider = create_summarization_provider(cfg)

        # Check that provider has required protocol methods
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))

        # Protocol requires summarize(text, episode_title, episode_description, params)
        import inspect

        sig = inspect.signature(provider.summarize)
        params = list(sig.parameters.keys())
        self.assertIn("text", params)
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("params", params)
