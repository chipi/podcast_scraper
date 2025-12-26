#!/usr/bin/env python3
"""Tests for summarization provider (Stage 4).

These tests verify that the summarization provider pattern works correctly.
"""

import unittest
from unittest.mock import Mock, patch

from podcast_scraper import config
from podcast_scraper.summarization.factory import create_summarization_provider


class TestSummarizationProviderFactory(unittest.TestCase):
    """Test summarization provider factory."""

    def test_create_local_provider(self):
        """Test that factory creates TransformersSummarizationProvider for 'local'."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)
        self.assertIsNotNone(provider)
        self.assertEqual(provider.__class__.__name__, "TransformersSummarizationProvider")

    def test_create_anthropic_provider_not_implemented(self):
        """Test that factory raises NotImplementedError for 'anthropic' provider."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="anthropic",
            generate_summaries=False,
        )
        with self.assertRaises(NotImplementedError) as context:
            create_summarization_provider(cfg)

        self.assertIn(
            "Anthropic summarization provider is not yet implemented", str(context.exception)
        )
        self.assertIn("Currently supported providers: 'local', 'openai'", str(context.exception))

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
            summary_provider="local",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)
        # Verify it has the expected methods
        self.assertTrue(hasattr(provider, "summarize"))
        self.assertTrue(hasattr(provider, "initialize"))
        self.assertTrue(hasattr(provider, "cleanup"))


class TestTransformersSummarizationProvider(unittest.TestCase):
    """Test TransformersSummarizationProvider implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_model="facebook/bart-base",
        )

    def test_provider_initialization(self):
        """Test that provider can be initialized."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        provider = TransformersSummarizationProvider(self.cfg)
        self.assertFalse(provider.is_initialized)
        self.assertIsNone(provider.map_model)
        self.assertIsNone(provider.reduce_model)

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_initialize_loads_models(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that initialize() loads MAP and REDUCE models."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        mock_map_model = Mock()
        mock_reduce_model = Mock()
        mock_summary_model.side_effect = [mock_map_model, mock_reduce_model]
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-large"

        provider = TransformersSummarizationProvider(self.cfg)
        provider.initialize()

        self.assertTrue(provider.is_initialized)
        self.assertEqual(provider.map_model, mock_map_model)
        self.assertEqual(provider.reduce_model, mock_reduce_model)
        self.assertEqual(mock_summary_model.call_count, 2)

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_initialize_same_model(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that initialize() reuses MAP model for REDUCE when same."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        mock_map_model = Mock()
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"  # Same model

        provider = TransformersSummarizationProvider(self.cfg)
        provider.initialize()

        self.assertTrue(provider.is_initialized)
        self.assertEqual(provider.map_model, mock_map_model)
        self.assertEqual(provider.reduce_model, mock_map_model)  # Same instance
        self.assertEqual(mock_summary_model.call_count, 1)  # Only called once

    @patch("podcast_scraper.summarization.local_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_summarize(
        self, mock_summary_model, mock_select_map, mock_select_reduce, mock_summarize_long
    ):
        """Test that summarize() calls summarize_long_text()."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-base"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summarize_long.return_value = "Test summary"

        provider = TransformersSummarizationProvider(self.cfg)
        provider.initialize()

        result = provider.summarize("Test transcript text")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Test summary")
        self.assertIsNone(result["summary_short"])
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["model_used"], "facebook/bart-base")
        mock_summarize_long.assert_called_once()

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_summarize_not_initialized(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that summarize() raises RuntimeError if not initialized."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        provider = TransformersSummarizationProvider(self.cfg)
        # Don't call initialize()

        with self.assertRaises(RuntimeError) as context:
            provider.summarize("Test text")

        self.assertIn("not initialized", str(context.exception))

    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_cleanup(self, mock_summary_model, mock_select_map, mock_select_reduce):
        """Test that cleanup() resets state."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        mock_map_model = Mock()
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"

        provider = TransformersSummarizationProvider(self.cfg)
        provider.initialize()
        self.assertTrue(provider.is_initialized)

        provider.cleanup()

        self.assertFalse(provider.is_initialized)
        self.assertIsNone(provider.map_model)
        self.assertIsNone(provider.reduce_model)

    @patch("podcast_scraper.summarization.local_provider.summarizer.summarize_long_text")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_provider_summarize_with_params(
        self, mock_summary_model, mock_select_map, mock_select_reduce, mock_summarize_long
    ):
        """Test that summarize() passes params correctly."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        mock_map_model = Mock()
        mock_map_model.model_name = "facebook/bart-base"
        mock_map_model.device = "cpu"
        mock_summary_model.return_value = mock_map_model
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summarize_long.return_value = "Test summary"

        provider = TransformersSummarizationProvider(self.cfg)
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
    """Test that TransformersSummarizationProvider implements SummarizationProvider protocol."""

    def test_provider_implements_protocol(self):
        """Test that TransformersSummarizationProvider implements SummarizationProvider protocol."""
        from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="local",
            generate_summaries=False,
        )
        provider = TransformersSummarizationProvider(cfg)

        # Check that provider has required protocol methods
        self.assertTrue(hasattr(provider, "summarize"))
        # Protocol requires summarize(text, episode_title, episode_description, params)
        import inspect

        sig = inspect.signature(provider.summarize)
        params = list(sig.parameters.keys())
        self.assertIn("text", params)
        self.assertIn("episode_title", params)
        self.assertIn("episode_description", params)
        self.assertIn("params", params)
