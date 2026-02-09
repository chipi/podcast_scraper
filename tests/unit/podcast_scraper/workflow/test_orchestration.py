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


@pytest.mark.unit
class TestEpisodeMetricsInitialization(unittest.TestCase):
    """Tests for episode metrics initialization in run_pipeline."""

    def test_episode_metrics_initialized_before_update(self):
        """Test that episode metrics can be initialized before update to prevent warnings."""
        from podcast_scraper.workflow import metrics

        pipeline_metrics = metrics.Metrics()
        episode_id = "test_episode_123"
        episode_number = 1

        # Initialize metrics upfront (like run_pipeline does)
        pipeline_metrics.get_or_create_episode_metrics(
            episode_id=episode_id, episode_number=episode_number
        )

        # Now update metrics - should not trigger warning
        with patch("podcast_scraper.workflow.metrics.logger") as mock_logger:
            pipeline_metrics.update_episode_metrics(
                episode_id=episode_id, audio_sec=100.0, transcribe_sec=5.0
            )
            # Verify no debug log about creating new entry (metrics should exist)
            debug_calls = [
                str(call)
                for call in mock_logger.debug.call_args_list
                if "creating new entry" in str(call)
            ]
            self.assertEqual(len(debug_calls), 0, "Should not log debug message when metrics exist")


@pytest.mark.unit
class TestGetFactoryFunction(unittest.TestCase):
    """Tests for _get_factory_function helper function."""

    @patch("sys.modules")
    def test_get_factory_function_returns_default(self, mock_modules):
        """Test that default factory is returned when no mock is found."""
        mock_workflow = Mock()
        mock_modules.get.return_value = mock_workflow
        del mock_workflow.create_transcription_provider  # Attribute doesn't exist

        default_factory = Mock()
        result = orchestration._get_factory_function(
            "create_transcription_provider", default_factory
        )

        self.assertEqual(result, default_factory)

    @patch("sys.modules")
    def test_get_factory_function_returns_mock(self, mock_modules):
        """Test that mock factory is returned when mock is found."""
        mock_factory = Mock()
        mock_workflow = Mock()
        mock_workflow.create_transcription_provider = mock_factory
        mock_modules.get.return_value = mock_workflow

        default_factory = Mock()
        result = orchestration._get_factory_function(
            "create_transcription_provider", default_factory
        )

        self.assertEqual(result, mock_factory)

    @patch("sys.modules")
    def test_get_factory_function_returns_non_default(self, mock_modules):
        """Test that non-default factory is returned when different factory is found."""
        other_factory = Mock()
        mock_workflow = Mock()
        mock_workflow.create_transcription_provider = other_factory
        mock_modules.get.return_value = mock_workflow

        default_factory = Mock()
        result = orchestration._get_factory_function(
            "create_transcription_provider", default_factory
        )

        self.assertEqual(result, other_factory)
        self.assertNotEqual(result, default_factory)


@pytest.mark.unit
class TestCreateTranscriptionProvider(unittest.TestCase):
    """Tests for _create_transcription_provider helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=False,
        )

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_transcription_provider_success(self, mock_logger, mock_get_factory):
        """Test successful transcription provider creation."""
        mock_provider = Mock()
        mock_provider.initialize = Mock()
        mock_factory = Mock(return_value=mock_provider)
        mock_get_factory.return_value = mock_factory

        result = orchestration._create_transcription_provider(self.cfg)

        self.assertEqual(result, mock_provider)
        mock_provider.initialize.assert_called_once()

    def test_create_transcription_provider_disabled(self):
        """Test that None is returned when transcription is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
        )

        result = orchestration._create_transcription_provider(cfg)

        self.assertIsNone(result)

    def test_create_transcription_provider_dry_run(self):
        """Test that None is returned in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=True,
        )

        result = orchestration._create_transcription_provider(cfg)

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_transcription_provider_initialization_failure(
        self, mock_logger, mock_get_factory
    ):
        """Test that exception is raised when provider initialization fails."""
        mock_provider = Mock()
        mock_provider.initialize = Mock(side_effect=RuntimeError("Init failed"))
        mock_factory = Mock(return_value=mock_provider)
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(RuntimeError):
            orchestration._create_transcription_provider(self.cfg)


@pytest.mark.unit
class TestCreateSpeakerDetector(unittest.TestCase):
    """Tests for _create_speaker_detector helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            dry_run=False,
        )

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_speaker_detector_success(self, mock_logger, mock_get_factory):
        """Test successful speaker detector creation."""
        mock_detector = Mock()
        mock_detector.initialize = Mock()
        mock_factory = Mock(return_value=mock_detector)
        mock_get_factory.return_value = mock_factory

        result = orchestration._create_speaker_detector(self.cfg)

        self.assertEqual(result, mock_detector)
        mock_detector.initialize.assert_called_once()

    def test_create_speaker_detector_disabled(self):
        """Test that None is returned when auto_speakers is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=False,
        )

        result = orchestration._create_speaker_detector(cfg)

        self.assertIsNone(result)

    def test_create_speaker_detector_dry_run(self):
        """Test that None is returned in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            dry_run=True,
        )

        result = orchestration._create_speaker_detector(cfg)

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_speaker_detector_with_warmup(self, mock_logger, mock_get_factory):
        """Test speaker detector creation with warmup method."""
        mock_detector = Mock()
        mock_detector.initialize = Mock()
        mock_detector.warmup = Mock()
        mock_factory = Mock(return_value=mock_detector)
        mock_get_factory.return_value = mock_factory

        result = orchestration._create_speaker_detector(self.cfg)

        self.assertEqual(result, mock_detector)
        mock_detector.warmup.assert_called_once_with(timeout_s=600)


@pytest.mark.unit
class TestCreateSummarizationProvider(unittest.TestCase):
    """Tests for _create_summarization_provider helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
            dry_run=False,
        )

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_summarization_provider_success(self, mock_logger, mock_get_factory):
        """Test successful summarization provider creation."""
        mock_provider = Mock()
        mock_provider.initialize = Mock()
        mock_factory = Mock(return_value=mock_provider)
        mock_get_factory.return_value = mock_factory

        result = orchestration._create_summarization_provider(self.cfg)

        self.assertEqual(result, mock_provider)
        mock_provider.initialize.assert_called_once()

    def test_create_summarization_provider_disabled(self):
        """Test that None is returned when generate_summaries is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=False,
        )

        result = orchestration._create_summarization_provider(cfg)

        self.assertIsNone(result)

    def test_create_summarization_provider_dry_run(self):
        """Test that None is returned in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,
            dry_run=True,
        )

        result = orchestration._create_summarization_provider(cfg)

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_summarization_provider_import_error(self, mock_logger, mock_get_factory):
        """Test that RuntimeError is raised on ImportError."""
        mock_get_factory.side_effect = ImportError("Missing dependency")

        with self.assertRaises(RuntimeError) as context:
            orchestration._create_summarization_provider(self.cfg)

        self.assertIn("dependencies not available", str(context.exception))


@pytest.mark.unit
class TestCreateAllProviders(unittest.TestCase):
    """Tests for _create_all_providers helper function."""

    @patch("podcast_scraper.workflow.orchestration._create_summarization_provider")
    @patch("podcast_scraper.workflow.orchestration._create_speaker_detector")
    @patch("podcast_scraper.workflow.orchestration._create_transcription_provider")
    def test_create_all_providers_all_enabled(
        self, mock_create_trans, mock_create_speaker, mock_create_summary
    ):
        """Test creating all providers when all are enabled."""
        mock_trans = Mock()
        mock_speaker = Mock()
        mock_summary = Mock()
        mock_create_trans.return_value = mock_trans
        mock_create_speaker.return_value = mock_speaker
        mock_create_summary.return_value = mock_summary

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
            dry_run=False,
        )

        trans, speaker, summary = orchestration._create_all_providers(cfg)

        self.assertEqual(trans, mock_trans)
        self.assertEqual(speaker, mock_speaker)
        self.assertEqual(summary, mock_summary)

    @patch("podcast_scraper.workflow.orchestration._create_summarization_provider")
    @patch("podcast_scraper.workflow.orchestration._create_speaker_detector")
    @patch("podcast_scraper.workflow.orchestration._create_transcription_provider")
    def test_create_all_providers_all_disabled(
        self, mock_create_trans, mock_create_speaker, mock_create_summary
    ):
        """Test creating all providers when all are disabled."""
        mock_create_trans.return_value = None
        mock_create_speaker.return_value = None
        mock_create_summary.return_value = None

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
            auto_speakers=False,
            generate_summaries=False,
        )

        trans, speaker, summary = orchestration._create_all_providers(cfg)

        self.assertIsNone(trans)
        self.assertIsNone(speaker)
        self.assertIsNone(summary)


@pytest.mark.unit
class TestSetupJsonlEmitter(unittest.TestCase):
    """Tests for _setup_jsonl_emitter helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=True,
            run_id="test-run",
        )
        self.pipeline_metrics = metrics.Metrics()
        self.output_dir = "/tmp/test_output"

    def test_setup_jsonl_emitter_disabled(self):
        """Test that None is returned when JSONL metrics are disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=False,
        )

        result = orchestration._setup_jsonl_emitter(cfg, self.output_dir, self.pipeline_metrics)

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.jsonl_emitter.JSONLEmitter")
    def test_setup_jsonl_emitter_enabled(self, mock_emitter_class):
        """Test JSONL emitter setup when enabled."""
        mock_emitter = Mock()
        mock_emitter.__enter__ = Mock(return_value=mock_emitter)
        mock_emitter_class.return_value = mock_emitter

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=True,
        )

        result = orchestration._setup_jsonl_emitter(cfg, self.output_dir, self.pipeline_metrics)

        self.assertEqual(result, mock_emitter)
        mock_emitter.__enter__.assert_called_once()
        mock_emitter.emit_run_started.assert_called_once()

    @patch("podcast_scraper.workflow.jsonl_emitter.JSONLEmitter")
    def test_setup_jsonl_emitter_custom_path(self, mock_emitter_class):
        """Test JSONL emitter setup with custom path."""
        mock_emitter = Mock()
        mock_emitter.__enter__ = Mock(return_value=mock_emitter)
        mock_emitter_class.return_value = mock_emitter

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            jsonl_metrics_enabled=True,
            jsonl_metrics_path="/custom/path.jsonl",
            run_id="test-run",
        )

        result = orchestration._setup_jsonl_emitter(cfg, self.output_dir, self.pipeline_metrics)

        self.assertEqual(result, mock_emitter)
        mock_emitter_class.assert_called_once_with(self.pipeline_metrics, "/custom/path.jsonl")


@pytest.mark.unit
class TestSetupLoggingAndDevices(unittest.TestCase):
    """Tests for _setup_logging_and_devices helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_device="cpu",
            summary_device="cpu",
            generate_summaries=True,  # Required for summarization device to be recorded
            generate_metadata=True,  # Required when generate_summaries=True
        )
        self.pipeline_metrics = metrics.Metrics()

    @patch("podcast_scraper.workflow.orchestration._log_effective_parallelism")
    @patch("podcast_scraper.workflow.orchestration._log_provider_ownership")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_setup_logging_and_devices_with_providers(
        self, mock_logger, mock_log_ownership, mock_log_parallelism
    ):
        """Test setup logging with all providers."""
        mock_transcription = Mock()
        mock_transcription._detect_whisper_device = Mock(return_value="cpu")
        mock_speaker = Mock()
        mock_summary = Mock()
        mock_summary._map_model = Mock()
        mock_summary._map_model.device = "cpu"

        orchestration._setup_logging_and_devices(
            self.cfg,
            mock_transcription,
            mock_speaker,
            mock_summary,
            self.pipeline_metrics,
        )

        mock_log_parallelism.assert_called_once()
        mock_log_ownership.assert_called_once()
        # Verify that devices were recorded (single string values, not lists)
        self.assertEqual(self.pipeline_metrics.transcription_device, "cpu")
        self.assertEqual(self.pipeline_metrics.summarization_device, "cpu")

    @patch("podcast_scraper.workflow.orchestration._log_effective_parallelism")
    @patch("podcast_scraper.workflow.orchestration._log_provider_ownership")
    def test_setup_logging_and_devices_without_providers(
        self, mock_log_ownership, mock_log_parallelism
    ):
        """Test setup logging without providers."""
        orchestration._setup_logging_and_devices(self.cfg, None, None, None, self.pipeline_metrics)

        mock_log_parallelism.assert_called_once()
        mock_log_ownership.assert_called_once()


@pytest.mark.unit
class TestCreateRunManifest(unittest.TestCase):
    """Tests for _create_run_manifest helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=False,
            run_id="test-run",
        )
        self.output_dir = "/tmp/test_output"

    def test_create_run_manifest_dry_run(self):
        """Test that None is returned in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=True,
            run_id="test-run",
        )

        result = orchestration._create_run_manifest(cfg, self.output_dir)

        self.assertIsNone(result)

    @patch("podcast_scraper.workflow.run_manifest.create_run_manifest")
    @patch("podcast_scraper.workflow.orchestration.logger")
    @patch("os.path.join")
    @patch("builtins.open", create=True)
    def test_create_run_manifest_success(
        self, mock_open, mock_join, mock_logger, mock_create_manifest
    ):
        """Test successful run manifest creation."""
        mock_manifest = Mock()
        mock_manifest.save_to_file = Mock()
        mock_create_manifest.return_value = mock_manifest
        mock_join.return_value = "/tmp/test_output/run_manifest.json"

        result = orchestration._create_run_manifest(self.cfg, self.output_dir)

        self.assertEqual(result, mock_manifest)
        mock_create_manifest.assert_called_once()
        mock_manifest.save_to_file.assert_called_once()

    @patch("podcast_scraper.workflow.run_manifest.create_run_manifest")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_create_run_manifest_handles_exception(self, mock_logger, mock_create_manifest):
        """Test that exception during manifest creation is handled."""
        mock_create_manifest.side_effect = RuntimeError("Manifest creation failed")

        result = orchestration._create_run_manifest(self.cfg, self.output_dir)

        self.assertIsNone(result)
        mock_logger.warning.assert_called()


@pytest.mark.unit
class TestFetchAndPrepareEpisodes(unittest.TestCase):
    """Tests for _fetch_and_prepare_episodes helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.pipeline_metrics = metrics.Metrics()

    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.prepare_episodes_from_feed"  # noqa: E501
    )
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.extract_feed_metadata_for_generation"  # noqa: E501
    )
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.fetch_and_parse_feed"  # noqa: E501
    )
    @patch("podcast_scraper.workflow.orchestration.time.time")
    def test_fetch_and_prepare_episodes_success(
        self, mock_time, mock_fetch, mock_extract_metadata, mock_prepare
    ):
        """Test successful fetch and prepare episodes."""
        mock_feed = Mock()
        mock_rss_bytes = b"<rss>...</rss>"
        mock_feed_metadata = Mock()
        mock_episodes = [Mock(), Mock()]
        mock_episodes[0].idx = 1
        mock_episodes[0].title = "Episode 1"
        mock_episodes[0].item = None
        mock_episodes[1].idx = 2
        mock_episodes[1].title = "Episode 2"
        mock_episodes[1].item = None

        mock_fetch.return_value = (mock_feed, mock_rss_bytes)
        mock_extract_metadata.return_value = mock_feed_metadata
        mock_prepare.return_value = mock_episodes
        mock_time.side_effect = [
            0.0,
            1.0,
            1.0,
            2.0,
        ]  # scraping_start, scraping_end, parsing_start, parsing_end

        feed, rss_bytes, feed_metadata, episodes = orchestration._fetch_and_prepare_episodes(
            self.cfg, self.pipeline_metrics
        )

        self.assertEqual(feed, mock_feed)
        self.assertEqual(rss_bytes, mock_rss_bytes)
        self.assertEqual(feed_metadata, mock_feed_metadata)
        self.assertEqual(episodes, mock_episodes)
        self.assertEqual(self.pipeline_metrics.episodes_scraped_total, 2)

    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.prepare_episodes_from_feed"  # noqa: E501
    )
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.extract_feed_metadata_for_generation"  # noqa: E501
    )
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.scraping.fetch_and_parse_feed"  # noqa: E501
    )
    @patch("podcast_scraper.workflow.orchestration.time.time")
    @patch("podcast_scraper.workflow.metadata_generation.generate_episode_id")
    def test_fetch_and_prepare_episodes_initializes_statuses(
        self, mock_generate_id, mock_time, mock_fetch, mock_extract_metadata, mock_prepare
    ):
        """Test that episode statuses are initialized."""
        import xml.etree.ElementTree as ET

        mock_feed = Mock()
        mock_rss_bytes = b"<rss>...</rss>"
        mock_feed_metadata = Mock()
        item = ET.Element("item")
        ET.SubElement(item, "guid").text = "ep1"
        episode = Mock()
        episode.idx = 1
        episode.title = "Episode 1"
        episode.item = item
        episode.number = None

        mock_fetch.return_value = (mock_feed, mock_rss_bytes)
        mock_extract_metadata.return_value = mock_feed_metadata
        mock_prepare.return_value = [episode]
        mock_generate_id.return_value = "episode-id-1"
        mock_time.side_effect = [0.0, 1.0, 1.0, 2.0]

        orchestration._fetch_and_prepare_episodes(self.cfg, self.pipeline_metrics)

        # Verify episode status and metrics were initialized
        # episode_statuses is a list of EpisodeStatus objects
        episode_ids = [status.episode_id for status in self.pipeline_metrics.episode_statuses]
        self.assertIn("episode-id-1", episode_ids)


@pytest.mark.unit
class TestSetupPipelineResources(unittest.TestCase):
    """Tests for _setup_pipeline_resources helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = config.Config(rss_url="https://example.com/feed.xml")
        self.feed = Mock()
        self.episodes = [Mock()]
        self.output_dir = "/tmp/test_output"
        self.pipeline_metrics = metrics.Metrics()

    @patch("podcast_scraper.workflow.orchestration.wf_stages.processing.setup_processing_resources")
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.transcription.setup_transcription_resources"  # noqa: E501
    )
    @patch(
        "podcast_scraper.workflow.orchestration.wf_stages.processing.detect_feed_hosts_and_patterns"
    )
    @patch("podcast_scraper.workflow.orchestration.time.time")
    def test_setup_pipeline_resources_success(
        self, mock_time, mock_detect_hosts, mock_setup_transcription, mock_setup_processing
    ):
        """Test successful pipeline resources setup."""
        mock_time.return_value = 0.0
        mock_host_result = Mock()
        mock_transcription_resources = Mock()
        mock_processing_resources = Mock()

        mock_detect_hosts.return_value = mock_host_result
        mock_setup_transcription.return_value = mock_transcription_resources
        mock_setup_processing.return_value = mock_processing_resources

        normalizing_start, host_result, trans_resources, proc_resources = (
            orchestration._setup_pipeline_resources(
                self.cfg,
                self.feed,
                self.episodes,
                self.output_dir,
                None,
                None,
                self.pipeline_metrics,
            )
        )

        self.assertEqual(normalizing_start, 0.0)
        self.assertEqual(host_result, mock_host_result)
        self.assertEqual(trans_resources, mock_transcription_resources)
        self.assertEqual(proc_resources, mock_processing_resources)


@pytest.mark.unit
class TestCleanupProviders(unittest.TestCase):
    """Tests for _cleanup_providers helper function."""

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_cleanup_providers_with_transcription_provider(self, mock_logger):
        """Test cleanup with transcription provider."""
        mock_provider = Mock()
        mock_provider.cleanup = Mock()
        mock_resources = Mock()
        mock_resources.transcription_provider = mock_provider

        orchestration._cleanup_providers(mock_resources, None)

        mock_provider.cleanup.assert_called_once()
        mock_logger.debug.assert_called()

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_cleanup_providers_with_summary_provider(self, mock_logger):
        """Test cleanup with summary provider."""
        mock_provider = Mock()
        mock_provider.cleanup = Mock()

        orchestration._cleanup_providers(None, mock_provider)

        mock_provider.cleanup.assert_called_once()
        mock_logger.debug.assert_called()

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_cleanup_providers_handles_exception(self, mock_logger):
        """Test that exceptions during cleanup are handled."""
        mock_provider = Mock()
        mock_provider.cleanup = Mock(side_effect=RuntimeError("Cleanup failed"))
        mock_resources = Mock()
        mock_resources.transcription_provider = mock_provider

        # Should not raise
        orchestration._cleanup_providers(mock_resources, None)

        mock_logger.warning.assert_called()

    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_cleanup_providers_with_preloaded_ml_provider(self, mock_logger):
        """Test cleanup with preloaded ML provider."""
        mock_ml_provider = Mock()
        mock_ml_provider.cleanup = Mock()

        # Patch the global variable
        with patch(
            "podcast_scraper.workflow.orchestration._preloaded_ml_provider", mock_ml_provider
        ):
            orchestration._cleanup_providers(None, None)

            mock_ml_provider.cleanup.assert_called_once()


@pytest.mark.unit
class TestFinalizePipeline(unittest.TestCase):
    """Tests for _finalize_pipeline helper function."""

    def setUp(self):
        """Set up test fixtures."""
        from podcast_scraper.workflow import metrics

        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=False,
            run_id="test-run",
        )
        self.pipeline_metrics = metrics.Metrics()
        self.transcription_resources = Mock()
        self.transcription_resources.temp_dir = "/tmp/test"
        self.output_dir = "/tmp/test_output"
        self.run_suffix = "testrun"
        self.episodes = [Mock()]

    @patch("podcast_scraper.workflow.orchestration.wf_helpers.generate_pipeline_summary")
    @patch("podcast_scraper.workflow.orchestration.wf_helpers.cleanup_pipeline")
    @patch("podcast_scraper.workflow.orchestration._log_episode_results")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_finalize_pipeline_basic(
        self, mock_logger, mock_log_results, mock_cleanup, mock_generate_summary
    ):
        """Test basic pipeline finalization."""
        mock_generate_summary.return_value = (5, "Summary text")

        result = orchestration._finalize_pipeline(
            self.cfg,
            5,
            self.transcription_resources,
            self.output_dir,
            self.run_suffix,
            self.pipeline_metrics,
            self.episodes,
            None,
            None,
            None,
            None,
        )

        self.assertEqual(result, (5, "Summary text"))
        mock_cleanup.assert_called_once()
        mock_log_results.assert_called_once()

    @patch("podcast_scraper.workflow.orchestration.wf_helpers.generate_pipeline_summary")
    @patch("podcast_scraper.workflow.orchestration.wf_helpers.cleanup_pipeline")
    @patch("podcast_scraper.workflow.orchestration._log_episode_results")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_finalize_pipeline_saves_metrics(
        self, mock_logger, mock_log_results, mock_cleanup, mock_generate_summary
    ):
        """Test that metrics are saved when configured."""
        mock_generate_summary.return_value = (5, "Summary")

        with patch.object(self.pipeline_metrics, "save_to_file") as mock_save_metrics:
            orchestration._finalize_pipeline(
                self.cfg,
                5,
                self.transcription_resources,
                self.output_dir,
                self.run_suffix,
                self.pipeline_metrics,
                self.episodes,
                None,
                None,
                None,
                None,
            )

            # Should save to default location
            mock_save_metrics.assert_called()

    @patch("podcast_scraper.workflow.orchestration.wf_helpers.generate_pipeline_summary")
    @patch("podcast_scraper.workflow.orchestration.wf_helpers.cleanup_pipeline")
    @patch("podcast_scraper.workflow.orchestration._log_episode_results")
    @patch("podcast_scraper.workflow.orchestration.logger")
    @patch("podcast_scraper.workflow.run_index.create_run_index")
    def test_finalize_pipeline_creates_run_index(
        self, mock_create_index, mock_logger, mock_log_results, mock_cleanup, mock_generate_summary
    ):
        """Test that run index is created."""
        mock_index = Mock()
        mock_index.save_to_file = Mock()
        mock_create_index.return_value = mock_index
        mock_generate_summary.return_value = (5, "Summary")

        orchestration._finalize_pipeline(
            self.cfg,
            5,
            self.transcription_resources,
            self.output_dir,
            self.run_suffix,
            self.pipeline_metrics,
            self.episodes,
            None,
            None,
            None,
            None,
        )

        mock_create_index.assert_called_once()
        mock_index.save_to_file.assert_called_once()

    @patch("podcast_scraper.workflow.orchestration.wf_helpers.generate_pipeline_summary")
    @patch("podcast_scraper.workflow.orchestration.wf_helpers.cleanup_pipeline")
    @patch("podcast_scraper.workflow.orchestration._log_episode_results")
    @patch("podcast_scraper.workflow.orchestration.logger")
    def test_finalize_pipeline_dry_run_skips_index(
        self, mock_logger, mock_log_results, mock_cleanup, mock_generate_summary
    ):
        """Test that run index is skipped in dry-run mode."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            dry_run=True,
            run_id="test-run",
        )
        mock_generate_summary.return_value = (5, "Summary")

        with patch("podcast_scraper.workflow.run_index.create_run_index") as mock_create_index:
            orchestration._finalize_pipeline(
                cfg,
                5,
                self.transcription_resources,
                self.output_dir,
                self.run_suffix,
                self.pipeline_metrics,
                self.episodes,
                None,
                None,
                None,
                None,
            )

            mock_create_index.assert_not_called()


@pytest.mark.unit
class TestOrchestrationErrorHandling(unittest.TestCase):
    """Tests for error handling in orchestration functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
        )

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_transcription_provider_initialization_error(self, mock_get_factory):
        """Test that transcription provider initialization errors are raised."""
        mock_provider = Mock()
        mock_provider.initialize.side_effect = Exception("Initialization failed")
        mock_factory = Mock(return_value=mock_provider)
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_transcription_provider(self.cfg)

        self.assertIn("Initialization failed", str(context.exception))

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_speaker_detector_initialization_error(self, mock_get_factory):
        """Test that speaker detector initialization errors are raised."""
        mock_detector = Mock()
        mock_detector.initialize.side_effect = Exception("Initialization failed")
        mock_factory = Mock(return_value=mock_detector)
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_speaker_detector(self.cfg)

        self.assertIn("Initialization failed", str(context.exception))

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_summarization_provider_initialization_error(self, mock_get_factory):
        """Test that summarization provider initialization errors are raised."""
        mock_provider = Mock()
        mock_provider.initialize.side_effect = Exception("Initialization failed")
        mock_factory = Mock(return_value=mock_provider)
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_summarization_provider(self.cfg)

        self.assertIn("Initialization failed", str(context.exception))

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_transcription_provider_factory_error(self, mock_get_factory):
        """Test that transcription provider factory errors are raised."""
        mock_factory = Mock(side_effect=Exception("Factory error"))
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_transcription_provider(self.cfg)

        self.assertIn("Factory error", str(context.exception))

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_speaker_detector_factory_error(self, mock_get_factory):
        """Test that speaker detector factory errors are raised."""
        mock_factory = Mock(side_effect=Exception("Factory error"))
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_speaker_detector(self.cfg)

        self.assertIn("Factory error", str(context.exception))

    @patch("podcast_scraper.workflow.orchestration._get_factory_function")
    def test_create_summarization_provider_factory_error(self, mock_get_factory):
        """Test that summarization provider factory errors are raised."""
        mock_factory = Mock(side_effect=Exception("Factory error"))
        mock_get_factory.return_value = mock_factory

        with self.assertRaises(Exception) as context:
            orchestration._create_summarization_provider(self.cfg)

        self.assertIn("Factory error", str(context.exception))

    def test_create_transcription_provider_disabled(self):
        """Test that transcription provider returns None when disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=False,
        )

        provider = orchestration._create_transcription_provider(cfg)

        self.assertIsNone(provider)

    def test_create_transcription_provider_dry_run(self):
        """Test that transcription provider returns None in dry run."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcribe_missing=True,
            dry_run=True,
        )

        provider = orchestration._create_transcription_provider(cfg)

        self.assertIsNone(provider)

    def test_create_speaker_detector_disabled(self):
        """Test that speaker detector returns None when disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=False,
        )

        detector = orchestration._create_speaker_detector(cfg)

        self.assertIsNone(detector)

    def test_create_speaker_detector_dry_run(self):
        """Test that speaker detector returns None in dry run."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            auto_speakers=True,
            dry_run=True,
        )

        detector = orchestration._create_speaker_detector(cfg)

        self.assertIsNone(detector)

    def test_create_summarization_provider_disabled(self):
        """Test that summarization provider returns None when disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=False,
        )

        provider = orchestration._create_summarization_provider(cfg)

        self.assertIsNone(provider)

    def test_create_summarization_provider_dry_run(self):
        """Test that summarization provider returns None in dry run."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            generate_summaries=True,
            generate_metadata=True,  # Required when generate_summaries=True
            dry_run=True,
        )

        provider = orchestration._create_summarization_provider(cfg)

        self.assertIsNone(provider)
