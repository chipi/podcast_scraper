#!/usr/bin/env python3
"""Tests for summarization failure handling.

These tests verify that summarization failures are handled correctly
with fail-fast behavior when generate_summaries=True.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import config
from podcast_scraper.workflow import metrics


def create_test_config(**kwargs):
    """Create a test configuration with defaults."""
    defaults = {
        "rss_url": "https://example.com/feed.xml",
        "output_dir": tempfile.mkdtemp(),
        "max_episodes": 1,
        "generate_metadata": True,
        "auto_speakers": False,
    }
    defaults.update(kwargs)
    return config.Config(**defaults)


class TestSummarizationInitializationFailure(unittest.TestCase):
    """Test summarization provider initialization failure handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("podcast_scraper.providers.ml.ml_provider.MLProvider._initialize_transformers")
    @patch("podcast_scraper.workflow.stages.setup.initialize_ml_environment")
    @patch("podcast_scraper.workflow.stages.setup.setup_pipeline_environment")
    @patch("podcast_scraper.workflow.stages.setup.preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow.stages.scraping.fetch_and_parse_feed")
    @patch("podcast_scraper.workflow.stages.scraping.extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow.stages.scraping.prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow.stages.processing.detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow.stages.transcription.setup_transcription_resources")
    @patch("podcast_scraper.workflow.stages.processing.setup_processing_resources")
    @patch("podcast_scraper.workflow.orchestration._create_summarization_provider_factory")
    @patch("podcast_scraper.transcription.factory.create_transcription_provider")
    @patch("podcast_scraper.providers.ml.summarizer.transformers", create=True)
    @patch("importlib.util.find_spec")
    def test_summarization_provider_initialization_failure_raises_error(
        self,
        mock_find_spec,
        mock_transformers,
        mock_create_transcription_provider,
        mock_create_provider,
        mock_setup_processing,
        mock_setup_transcription,
        mock_detect_hosts,
        mock_prepare_episodes,
        mock_extract_metadata,
        mock_fetch_feed,
        mock_preload_models,
        mock_setup_env,
        mock_init_env,
        mock_init_transformers,
    ):
        """Test that summarization provider initialization failure raises RuntimeError."""
        # Mock importlib.util.find_spec to prevent spacy.__spec__ errors
        # Return None to indicate spacy is not available (prevents ValueError from spacy.__spec__)
        mock_find_spec.return_value = None
        # Mock transformers to prevent actual import
        import sys

        # Set up transformers mock to return expected values
        # Create a mock model with nested structure for encoder/decoder access
        mock_model_instance = Mock()
        mock_model_instance.model = Mock()

        # Set up encoder with embed_positions.weight.shape as a tuple
        # Create a simple object with shape attribute (not a Mock to avoid auto-creation)
        # Use a tuple directly for shape to ensure it's subscriptable
        class MockWeight:
            def __init__(self):
                # Make shape a tuple that's directly accessible
                self._shape = (1024, 768)

            @property
            def shape(self):
                return self._shape

        class MockEmbedPositions:
            def __init__(self):
                self.weight = MockWeight()

        mock_encoder_embed_pos = MockEmbedPositions()
        # Configure encoder Mock to return our embed_positions object
        # Use configure_mock to ensure the attribute is properly set
        mock_encoder = Mock()
        mock_encoder.configure_mock(embed_positions=mock_encoder_embed_pos)
        mock_encoder.config = Mock()
        mock_encoder.config.hidden_size = 768
        # Configure model.model to return our encoder
        mock_model_instance.model.configure_mock(encoder=mock_encoder)

        # Set up decoder with embed_positions.weight.shape as a tuple
        class MockDecoderWeight:
            def __init__(self):
                # Make shape a tuple that's directly accessible
                self._shape = (1024, 768)

            @property
            def shape(self):
                return self._shape

        class MockDecoderEmbedPositions:
            def __init__(self):
                self.weight = MockDecoderWeight()

        mock_decoder_embed_pos = MockDecoderEmbedPositions()
        # Configure decoder Mock to return our embed_positions object
        # Use configure_mock to ensure the attribute is properly set
        mock_decoder = Mock()
        mock_decoder.configure_mock(embed_positions=mock_decoder_embed_pos)
        mock_decoder.config = Mock()
        mock_decoder.config.hidden_size = 768
        # Configure model.model to return our decoder
        mock_model_instance.model.configure_mock(decoder=mock_decoder)
        mock_auto_model = Mock()
        mock_auto_model.from_pretrained.return_value = (
            mock_model_instance,
            {},
        )  # (model, loading_info)
        mock_auto_tokenizer = Mock()
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForSeq2SeqLM = mock_auto_model
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        sys.modules["transformers"] = mock_transformers
        # Create config with generate_summaries=True
        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Mock pipeline setup
        from podcast_scraper.workflow.types import (
            FeedMetadata,
            HostDetectionResult,
            ProcessingResources,
            TranscriptionResources,
        )

        mock_setup_env.return_value = (self.output_dir, None, None)
        mock_fetch_feed.return_value = (Mock(), b"<rss></rss>")
        mock_extract_metadata.return_value = FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = []
        mock_detect_hosts.return_value = HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        # Mock transcription provider to prevent actual initialization
        mock_transcription_provider = Mock()
        mock_transcription_provider.initialize = Mock()
        mock_create_transcription_provider.return_value = mock_transcription_provider
        # Mock ML provider initialization to prevent actual model loading
        mock_init_transformers.return_value = None
        mock_setup_transcription.return_value = TranscriptionResources(
            transcription_provider=mock_transcription_provider,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_processing.return_value = ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=None,
        )

        # Mock provider creation to raise ImportError when called
        # The code checks podcast_scraper.workflow.create_summarization_provider first
        # So we need to ensure the mock is recognized
        import sys

        workflow_module = sys.modules.get("podcast_scraper.workflow")
        if workflow_module:
            # Set the re-exported function to our mock so the code recognizes it as a Mock
            workflow_module.create_summarization_provider = mock_create_provider
        mock_create_provider.side_effect = ImportError("ML dependencies not available")

        # Should raise RuntimeError when generate_summaries=True
        from podcast_scraper import workflow

        with self.assertRaises(RuntimeError) as context:
            workflow.run_pipeline(cfg)

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("dependencies not available", str(context.exception))

    @patch("podcast_scraper.providers.ml.ml_provider.MLProvider._initialize_transformers")
    @patch("podcast_scraper.workflow.stages.setup.initialize_ml_environment")
    @patch("podcast_scraper.workflow.stages.setup.setup_pipeline_environment")
    @patch("podcast_scraper.workflow.stages.setup.preload_ml_models_if_needed")
    @patch("podcast_scraper.workflow.stages.scraping.fetch_and_parse_feed")
    @patch("podcast_scraper.workflow.stages.scraping.extract_feed_metadata_for_generation")
    @patch("podcast_scraper.workflow.stages.scraping.prepare_episodes_from_feed")
    @patch("podcast_scraper.workflow.stages.processing.detect_feed_hosts_and_patterns")
    @patch("podcast_scraper.workflow.stages.transcription.setup_transcription_resources")
    @patch("podcast_scraper.workflow.stages.processing.setup_processing_resources")
    @patch("podcast_scraper.workflow.orchestration._create_summarization_provider_factory")
    @patch("podcast_scraper.transcription.factory.create_transcription_provider")
    @patch("podcast_scraper.providers.ml.summarizer.transformers", create=True)
    @patch("importlib.util.find_spec")
    def test_summarization_provider_initialization_exception_raises_error(
        self,
        mock_find_spec,
        mock_transformers,
        mock_create_transcription_provider,
        mock_create_provider,
        mock_setup_processing,
        mock_setup_transcription,
        mock_detect_hosts,
        mock_prepare_episodes,
        mock_extract_metadata,
        mock_fetch_feed,
        mock_preload_models,
        mock_setup_env,
        mock_init_env,
        mock_init_transformers,
    ):
        """Test that summarization provider initialization exception raises RuntimeError."""
        # Mock importlib.util.find_spec to prevent spacy.__spec__ errors
        # Return None to indicate spacy is not available (prevents ValueError from spacy.__spec__)
        mock_find_spec.return_value = None
        # Mock transformers to prevent actual import
        import sys

        # Set up transformers mock to return expected values
        # Create a mock model with nested structure for encoder/decoder access
        mock_model_instance = Mock()
        mock_model_instance.model = Mock()
        # Set up encoder with embed_positions.weight.shape as a tuple
        mock_encoder_embed_pos = Mock()

        # Create a simple object with shape attribute
        class MockWeight:
            shape = (1024, 768)

        mock_encoder_embed_pos.weight = MockWeight()
        mock_model_instance.model.encoder = Mock()
        mock_model_instance.model.encoder.embed_positions = mock_encoder_embed_pos
        mock_model_instance.model.encoder.config = Mock()
        mock_model_instance.model.encoder.config.hidden_size = 768

        # Set up decoder with embed_positions.weight.shape as a tuple
        class MockDecoderWeight:
            def __init__(self):
                # Make shape a tuple that's directly accessible
                self._shape = (1024, 768)

            @property
            def shape(self):
                return self._shape

        class MockDecoderEmbedPositions:
            def __init__(self):
                self.weight = MockDecoderWeight()

        mock_decoder_embed_pos = MockDecoderEmbedPositions()
        # Configure decoder Mock to return our embed_positions object
        # Use configure_mock to ensure the attribute is properly set
        mock_decoder = Mock()
        mock_decoder.configure_mock(embed_positions=mock_decoder_embed_pos)
        mock_decoder.config = Mock()
        mock_decoder.config.hidden_size = 768
        # Configure model.model to return our decoder
        mock_model_instance.model.configure_mock(decoder=mock_decoder)
        mock_auto_model = Mock()
        mock_auto_model.from_pretrained.return_value = (
            mock_model_instance,
            {},
        )  # (model, loading_info)
        mock_auto_tokenizer = Mock()
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_transformers.AutoModelForSeq2SeqLM = mock_auto_model
        mock_transformers.AutoTokenizer = mock_auto_tokenizer
        sys.modules["transformers"] = mock_transformers
        # Create config with generate_summaries=True
        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Mock pipeline setup
        from podcast_scraper.workflow.types import (
            FeedMetadata,
            HostDetectionResult,
            ProcessingResources,
            TranscriptionResources,
        )

        mock_setup_env.return_value = (self.output_dir, None, None)
        mock_fetch_feed.return_value = (Mock(), b"<rss></rss>")
        mock_extract_metadata.return_value = FeedMetadata(None, None, None)
        mock_prepare_episodes.return_value = []
        mock_detect_hosts.return_value = HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )
        # Mock transcription provider to prevent actual initialization
        mock_transcription_provider = Mock()
        mock_transcription_provider.initialize = Mock()
        mock_create_transcription_provider.return_value = mock_transcription_provider
        # Mock ML provider initialization to prevent actual model loading
        mock_init_transformers.return_value = None
        mock_setup_transcription.return_value = TranscriptionResources(
            transcription_provider=mock_transcription_provider,
            temp_dir=None,
            transcription_jobs=[],
            transcription_jobs_lock=None,
            saved_counter_lock=None,
        )
        mock_setup_processing.return_value = ProcessingResources(
            processing_jobs=[],
            processing_jobs_lock=None,
            processing_complete_event=None,
        )

        # Mock provider that raises exception during initialize()
        # The code checks podcast_scraper.workflow.create_summarization_provider first
        # So we need to ensure the mock is recognized
        import sys

        workflow_module = sys.modules.get("podcast_scraper.workflow")
        if workflow_module:
            # Set the re-exported function to our mock so the code recognizes it as a Mock
            workflow_module.create_summarization_provider = mock_create_provider

        # Mock provider creation to return a provider that fails on initialize
        mock_provider = Mock()
        mock_provider.initialize.side_effect = RuntimeError("Model load failed")

        def return_mock_provider(cfg):
            return mock_provider

        mock_create_provider.side_effect = return_mock_provider

        # Should raise RuntimeError when generate_summaries=True
        from podcast_scraper import workflow

        with self.assertRaises(RuntimeError) as context:
            workflow.run_pipeline(cfg)

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("Failed to initialize summarization provider", str(context.exception))


class TestEpisodeSummarizationFailure(unittest.TestCase):
    """Test episode-level summarization failure handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()
        # Create a test transcript file
        self.transcript_path = Path(self.output_dir) / "transcripts" / "ep01_test.txt"
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.write_text("This is a test transcript with enough content. " * 20)
        # Create metadata subdirectory
        metadata_dir = Path(self.output_dir) / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_episode_summarization_failure_raises_error(self, mock_clean):
        """Test that episode summarization failure raises RuntimeError."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        # Create a mock summary provider that raises exception
        mock_provider = Mock()
        mock_provider.summarize.side_effect = Exception("Provider error")

        # Call _generate_episode_summary which should raise RuntimeError
        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("Failed to generate summary", str(context.exception))

    def test_episode_summarization_without_provider_raises_error(self):
        """Test that episode summarization without provider raises RuntimeError."""
        from podcast_scraper.workflow import metadata_generation as metadata

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise ValueError when summary_provider is None but generate_summaries=True
        with self.assertRaises(ValueError) as context:
            metadata._generate_episode_summary(
                transcript_file_path=str(self.transcript_path.relative_to(self.output_dir)),
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=None,  # No provider
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("summary_provider is required", str(context.exception))


class TestParallelSummarizationFailure(unittest.TestCase):
    """Test parallel summarization failure handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.output_dir = tempfile.mkdtemp()
        self.episodes = [Mock()]
        self.episodes[0].idx = 1
        self.episodes[0].title_safe = "test_episode"
        self.feed = Mock()
        self.feed_metadata = Mock()
        self.host_detection_result = Mock()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_parallel_summarization_with_none_provider_raises_error(self):
        """Test parallel summarization raises RuntimeError when provider is None and generate_summaries=True."""  # noqa: E501
        from podcast_scraper.utils import filesystem

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Create a transcript file so the function finds episodes to summarize
        from pathlib import Path

        transcript_path_str = filesystem.build_whisper_output_path(
            self.episodes[0].idx, self.episodes[0].title_safe, None, self.output_dir
        )
        transcript_path = Path(transcript_path_str)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("This is a test transcript. " * 20)

        # Create mock feed metadata and host detection result
        from podcast_scraper.workflow.types import FeedMetadata, HostDetectionResult

        feed_metadata = FeedMetadata(None, None, None)
        host_detection_result = HostDetectionResult(
            cached_hosts=set(), heuristics=None, speaker_detector=None
        )

        # Should raise RuntimeError when summary_provider is None but generate_summaries=True
        from podcast_scraper.workflow.stages import summarization_stage

        with self.assertRaises(RuntimeError) as context:
            summarization_stage.parallel_episode_summarization(
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.output_dir,
                run_suffix=None,
                feed_metadata=feed_metadata,
                host_detection_result=host_detection_result,
                summary_provider=None,  # No provider
                download_args=[],
                pipeline_metrics=None,
            )

        self.assertIn("generate_summaries=True", str(context.exception))
        self.assertIn("Summary provider not available", str(context.exception))

    def test_parallel_summarization_no_provider_when_generate_summaries_false(self):
        """Test parallel summarization logs warning when provider is None and generate_summaries=False."""  # noqa: E501
        from podcast_scraper.utils import filesystem
        from podcast_scraper.workflow.stages import summarization_stage

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=False,  # Summaries disabled
            generate_metadata=True,
        )

        # Create a transcript file so the function finds episodes to summarize
        from pathlib import Path

        transcript_path_str = filesystem.build_whisper_output_path(
            self.episodes[0].idx, self.episodes[0].title_safe, None, self.output_dir
        )
        transcript_path = Path(transcript_path_str)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("This is a test transcript with enough content. " * 20)

        # Should not raise, just log warning and return
        with patch("podcast_scraper.workflow.stages.summarization_stage.logger") as mock_logger:
            summarization_stage.parallel_episode_summarization(
                episodes=self.episodes,
                feed=self.feed,
                cfg=cfg,
                effective_output_dir=self.output_dir,
                run_suffix=None,
                feed_metadata=self.feed_metadata,
                host_detection_result=self.host_detection_result,
                summary_provider=None,  # No provider
                download_args=[],
                pipeline_metrics=metrics.Metrics(),
            )

        # Should log warning when generate_summaries=False
        mock_logger.warning.assert_called_once()
        self.assertIn("not available", str(mock_logger.warning.call_args[0][0]))

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_empty_summary_raises_error(self, mock_clean):
        """Test that empty summary raises RuntimeError when generate_summaries=True."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {"summary": "", "metadata": {}}  # Empty summary

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise RuntimeError when summary is empty
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("empty result", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))

    @patch("podcast_scraper.preprocessing.clean_transcript")
    def test_non_string_summary_raises_error(self, mock_clean):
        """Test that non-string summary raises RuntimeError when generate_summaries=True."""
        import os

        from podcast_scraper.workflow import metadata_generation as metadata

        # Create transcript file
        transcript_path = os.path.join(self.output_dir, "transcript.txt")
        with open(transcript_path, "w") as f:
            f.write("This is a long transcript that should be long enough for summarization. " * 10)

        mock_clean.return_value = "Cleaned transcript text"
        mock_provider = Mock()
        mock_provider.summarize.return_value = {
            "summary": 123,  # Non-string summary
            "metadata": {},
        }

        cfg = create_test_config(
            output_dir=self.output_dir,
            generate_summaries=True,
            generate_metadata=True,
        )

        # Should raise RuntimeError when summary is not a string
        with self.assertRaises(RuntimeError) as context:
            metadata._generate_episode_summary(
                transcript_file_path="transcript.txt",
                output_dir=self.output_dir,
                cfg=cfg,
                episode_idx=1,
                summary_provider=mock_provider,
            )

        self.assertIn("not a string", str(context.exception))
        self.assertIn("generate_summaries=True", str(context.exception))
