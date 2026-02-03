"""Unit tests for podcast_scraper.workflow.orchestration module.

Tests for parallelism logging and configuration.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import orchestration


@pytest.mark.unit
class TestLogParallelismConfiguration(unittest.TestCase):
    """Tests for log_parallelism_configuration function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            summary_provider="transformers",
        )

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_with_cpu_device(self, mock_logger):
        """Test parallelism logging with CPU device."""
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "cpu"

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify log was called with CPU device
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        cpu_log_found = any("device=cpu" in str(call) for call in log_calls)
        self.assertTrue(cpu_log_found, "Should log CPU device")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_with_mps_device(self, mock_logger):
        """Test parallelism logging with MPS device."""
        # Config is frozen, so create new one with mps_exclusive
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="transformers",
            mps_exclusive=True,
        )
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "mps"
        summary_provider._map_model._summarize_lock = MagicMock()

        orchestration._log_effective_parallelism(cfg, summary_provider)

        # Verify log was called with MPS device and serialization
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        mps_log_found = any("device=mps" in str(call) for call in log_calls)
        serialized_log_found = any("serialized" in str(call) for call in log_calls)
        self.assertTrue(mps_log_found, "Should log MPS device")
        self.assertTrue(serialized_log_found, "Should log serialization status")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_with_cuda_device(self, mock_logger):
        """Test parallelism logging with CUDA device."""
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "cuda"
        summary_provider._map_model._summarize_lock = MagicMock()

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify log was called with CUDA device
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        cuda_log_found = any("device=cuda" in str(call) for call in log_calls)
        self.assertTrue(cuda_log_found, "Should log CUDA device")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_with_mps_exclusive_serialization(self, mock_logger):
        """Test parallelism logging includes mps_exclusive in serialization reasons."""
        # Config is frozen, so create new one with mps_exclusive
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="transformers",
            mps_exclusive=True,
        )
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "mps"
        summary_provider._map_model._summarize_lock = MagicMock()

        orchestration._log_effective_parallelism(cfg, summary_provider)

        # Verify serialization includes mps_exclusive
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        serialized_log = next((str(call) for call in log_calls if "serialized" in str(call)), None)
        self.assertIsNotNone(serialized_log, "Should have serialization log")
        self.assertIn("mps_exclusive", serialized_log, "Should include mps_exclusive")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_with_tokenizer_lock_serialization(self, mock_logger):
        """Test parallelism logging includes tokenizer_lock in serialization reasons."""
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "mps"
        summary_provider._map_model._summarize_lock = MagicMock()

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify serialization includes tokenizer_lock
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        serialized_log = next((str(call) for call in log_calls if "serialized" in str(call)), None)
        self.assertIsNotNone(serialized_log, "Should have serialization log")
        self.assertIn("tokenizer_lock", serialized_log, "Should include tokenizer_lock")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_fallback_to_reduce_model(self, mock_logger):
        """Test parallelism logging falls back to _reduce_model when _map_model not available."""
        summary_provider = Mock()
        summary_provider._map_model = None
        summary_provider._reduce_model = Mock()
        summary_provider._reduce_model.device = "cpu"

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify log was called
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        cpu_log_found = any("device=cpu" in str(call) for call in log_calls)
        self.assertTrue(cpu_log_found, "Should log CPU device from reduce model")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_fallback_to_config_device(self, mock_logger):
        """Test parallelism logging falls back to config.summary_device."""
        summary_provider = Mock()
        summary_provider._map_model = None
        summary_provider._reduce_model = None
        # Config is frozen, so create new one with summary_device
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
            summary_provider="transformers",
            summary_device="mps",
        )

        orchestration._log_effective_parallelism(cfg, summary_provider)

        # Verify log was called with device from config
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        mps_log_found = any("device=mps" in str(call) for call in log_calls)
        self.assertTrue(mps_log_found, "Should log device from config")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_fallback_to_provider_device(self, mock_logger):
        """Test parallelism logging falls back to provider.device attribute."""
        summary_provider = Mock()
        summary_provider._map_model = None
        summary_provider._reduce_model = None
        summary_provider.device = "cpu"

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify log was called
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        cpu_log_found = any("device=cpu" in str(call) for call in log_calls)
        self.assertTrue(cpu_log_found, "Should log device from provider attribute")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_summarization_disabled(self, mock_logger):
        """Test parallelism logging when summarization is disabled."""
        # Config is frozen, so create new one without summarization
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=False,
        )

        orchestration._log_effective_parallelism(cfg, None)

        # Verify log shows N/A
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        na_log_found = any("N/A" in str(call) and "disabled" in str(call) for call in log_calls)
        self.assertTrue(na_log_found, "Should log N/A when summarization disabled")

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_log_parallelism_no_serialization_reasons(self, mock_logger):
        """Test parallelism logging with no serialization reasons."""
        summary_provider = Mock()
        summary_provider._map_model = Mock()
        summary_provider._map_model.device = "cpu"
        # No mps_exclusive, no _summarize_lock

        orchestration._log_effective_parallelism(self.cfg, summary_provider)

        # Verify no serialization status in log
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        serialized_log = next((str(call) for call in log_calls if "serialized" in str(call)), None)
        self.assertIsNone(serialized_log, "Should not have serialization log for CPU")
