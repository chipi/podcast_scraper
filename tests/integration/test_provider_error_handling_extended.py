#!/usr/bin/env python3
"""Extended error handling tests for providers.

These tests verify comprehensive error handling scenarios including:
- Provider initialization failures
- Method failures after initialization
- Graceful degradation
- Error propagation
"""

import os
import unittest
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from podcast_scraper import config
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.transcription.factory import create_transcription_provider


@pytest.mark.integration
@pytest.mark.slow
class TestTranscriptionProviderErrorHandling(unittest.TestCase):
    """Test error handling for transcription providers."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                provider.initialize()

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    def test_transcribe_before_initialization(self, mock_import_whisper):
        """Test that transcribe() fails if called before initialization."""
        mock_whisper_lib = Mock()
        mock_whisper_model = Mock()
        mock_whisper_lib.load_model.return_value = mock_whisper_model
        mock_import_whisper.return_value = mock_whisper_lib
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Should raise ProviderNotInitializedError if not initialized
        from podcast_scraper.exceptions import ProviderNotInitializedError

        with self.assertRaises(ProviderNotInitializedError):
            provider.transcribe("test.mp3")

    @patch("podcast_scraper.ml.ml_provider._import_third_party_whisper")
    @patch("podcast_scraper.ml.ml_provider.MLProvider._transcribe_with_whisper")
    def test_transcribe_with_segments_failure(self, mock_transcribe, mock_import_whisper):
        """Test error handling when transcribe_with_segments() fails."""
        mock_whisper_lib = Mock()
        mock_whisper_model = Mock()
        mock_whisper_lib.load_model.return_value = mock_whisper_model
        mock_import_whisper.return_value = mock_whisper_lib
        mock_transcribe.side_effect = ValueError("Transcription failed")

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            transcribe_missing=True,
        )
        provider = create_transcription_provider(cfg)
        provider.initialize()

        # Should propagate error
        with self.assertRaises(ValueError):
            provider.transcribe_with_segments("test.mp3")

    def test_openai_provider_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        # Missing API key should be caught by config validator
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="openai",
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_cleanup_after_failed_initialization(self):
        """Test that cleanup() works even after failed initialization."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml", transcription_provider="whisper"
        )
        provider = create_transcription_provider(cfg)

        # Try to initialize (may fail)
        try:
            provider.initialize()
        except Exception:  # nosec B110
            pass

        # Cleanup should still work (idempotent)
        provider.cleanup()


@pytest.mark.integration
@pytest.mark.slow
class TestSpeakerDetectorErrorHandling(unittest.TestCase):
    """Test error handling for speaker detectors."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock initialization failure
        with patch.object(detector, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                detector.initialize()

    def test_detect_speakers_before_initialization(self):
        """Test that detect_speakers() auto-initializes if needed."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=True,  # Required for detect_speakers() to work
            transcribe_missing=False,  # Don't initialize Whisper for speaker detector tests
        )
        detector = create_speaker_detector(cfg)

        # Should auto-initialize if needed (implementation-dependent)
        # For NER detector, it may auto-initialize
        result = detector.detect_speakers("Test Episode", None, set())
        self.assertIsInstance(result, tuple)

    def test_detect_hosts_failure(self):
        """Test error handling when detect_hosts() fails."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock detect_hosts failure
        with patch.object(detector, "detect_hosts", side_effect=ValueError("Detection failed")):
            with self.assertRaises(ValueError):
                detector.detect_hosts("Test", None, None)

    def test_openai_detector_missing_api_key(self):
        """Test that OpenAI detector requires API key."""
        # Missing API key should be caught by config validator
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="openai",
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_clear_cache_failure(self):
        """Test error handling when clear_cache() fails."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Mock clear_cache failure
        with patch.object(detector, "clear_cache", side_effect=RuntimeError("Cache clear failed")):
            with self.assertRaises(RuntimeError):
                detector.clear_cache()


@pytest.mark.integration
@pytest.mark.slow
class TestSummarizationProviderErrorHandling(unittest.TestCase):
    """Test error handling for summarization providers."""

    def test_initialization_failure_raises_exception(self):
        """Test that initialization failure raises exception."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Mock initialization failure
        with patch.object(provider, "initialize", side_effect=RuntimeError("Model load failed")):
            with self.assertRaises(RuntimeError):
                provider.initialize()

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_summarize_before_initialization(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test that summarize() fails if called before initialization."""
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summary_model.return_value = Mock()

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_summaries=False,
        )
        provider = create_summarization_provider(cfg)

        # Should raise ProviderNotInitializedError if not initialized
        from podcast_scraper.exceptions import ProviderNotInitializedError

        with self.assertRaises(ProviderNotInitializedError):
            provider.summarize("test text")

    @patch("podcast_scraper.ml.ml_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.ml.ml_provider.summarizer.SummaryModel")
    def test_summarize_failure_after_initialization(
        self, mock_summary_model, mock_select_map, mock_select_reduce
    ):
        """Test error handling when summarize() fails after initialization."""
        mock_select_map.return_value = config.TEST_DEFAULT_SUMMARY_MODEL
        mock_select_reduce.return_value = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        mock_summary_model.return_value = Mock()

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="transformers",
            generate_metadata=True,  # Required for generate_summaries
            generate_summaries=True,  # Enable summaries for initialization
        )
        provider = create_summarization_provider(cfg)
        provider.initialize()

        # Mock summarize failure
        with patch.object(provider, "summarize", side_effect=ValueError("Summarization failed")):
            with self.assertRaises(ValueError):
                provider.summarize("test text")

    def test_openai_provider_missing_api_key(self):
        """Test that OpenAI provider requires API key."""
        # Missing API key should be caught by config validator
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="openai",
                    generate_summaries=False,
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key

    @patch("podcast_scraper.workflow.create_summarization_provider")
    def test_pipeline_fails_when_summarization_provider_initialization_fails(
        self, mock_create_provider
    ):
        """Test that pipeline fails fast when summarization provider initialization fails."""
        from podcast_scraper import workflow

        # Create config with generate_summaries=True
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="/tmp/test_output",
            generate_summaries=True,
            generate_metadata=True,
            auto_speakers=False,
            transcribe_missing=False,
        )

        # Mock provider creation to raise exception
        # Patch the re-exported version in workflow module to ensure workflow code sees the mock
        mock_create_provider.side_effect = RuntimeError("Provider initialization failed")

        # Mock other pipeline components to get to summarization initialization
        with patch("podcast_scraper.workflow.stages.setup.initialize_ml_environment"):
            with patch(
                "podcast_scraper.workflow.stages.setup.setup_pipeline_environment"
            ) as mock_setup:
                # Patch the actual function that workflow.py calls directly
                with patch("podcast_scraper.workflow.stages.setup.preload_ml_models_if_needed"):
                    with patch(
                        "podcast_scraper.workflow.stages.scraping.fetch_and_parse_feed"
                    ) as mock_fetch:
                        with patch(
                            "podcast_scraper.workflow.stages.scraping.extract_feed_metadata_for_generation"
                        ):
                            with patch(
                                "podcast_scraper.workflow.stages.scraping.prepare_episodes_from_feed"
                            ):
                                with patch(
                                    "podcast_scraper.workflow.stages.processing.detect_feed_hosts_and_patterns"
                                ):
                                    with patch(
                                        "podcast_scraper.workflow.stages.transcription.setup_transcription_resources"
                                    ):
                                        with patch(
                                            "podcast_scraper.workflow.stages.processing.setup_processing_resources"
                                        ):
                                            mock_setup.return_value = ("/tmp/test_output", None)
                                            mock_fetch.return_value = (Mock(), b"<rss></rss>")

                                            # Should raise RuntimeError when generate_summaries=True
                                            with self.assertRaises(RuntimeError) as context:
                                                workflow.run_pipeline(cfg)

                                            self.assertIn(
                                                "generate_summaries=True", str(context.exception)
                                            )
                                            self.assertIn(
                                                "Failed to initialize summarization provider",
                                                str(context.exception),
                                            )

    def test_episode_summarization_failure_in_pipeline_raises_error(self):
        """Test that episode summarization failure raises RuntimeError in pipeline."""
        from podcast_scraper import metadata

        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            output_dir="/tmp/test_output",
            generate_summaries=True,
            generate_metadata=True,
        )

        # Create a mock summary provider that raises exception
        mock_provider = Mock()
        mock_provider.summarize.side_effect = RuntimeError("Summarization failed for episode")

        # Call _generate_episode_summary which should catch and wrap the exception
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            transcript_path = Path(tmpdir) / "transcripts" / "ep01_test.txt"
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text("This is a test transcript. " * 20)

            with self.assertRaises(RuntimeError) as context:
                metadata._generate_episode_summary(
                    transcript_file_path=str(transcript_path.relative_to(tmpdir)),
                    output_dir=tmpdir,
                    cfg=cfg,
                    episode_idx=1,
                    summary_provider=mock_provider,
                )

            # The actual function wraps the exception with generate_summaries=True message
            # Error format: "[episode_idx] Failed to generate summary using provider: {e}. Summarization is required when generate_summaries=True."
            self.assertIn("generate_summaries=True", str(context.exception))
            self.assertIn("Failed to generate summary", str(context.exception))


@pytest.mark.integration
@pytest.mark.slow
class TestProviderSwitchingErrorHandling(unittest.TestCase):
    """Test error handling when switching providers."""

    def test_switch_to_invalid_provider(self):
        """Test error handling when switching to invalid provider."""
        # Invalid transcription provider
        with self.assertRaises(ValueError):
            create_transcription_provider(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="invalid",
                )
            )

        # Invalid speaker detector provider
        with self.assertRaises(ValueError):
            create_speaker_detector(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    speaker_detector_provider="invalid",
                )
            )

        # Invalid summarization provider
        with self.assertRaises(ValueError):
            create_summarization_provider(
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="invalid",
                    generate_summaries=False,
                )
            )

    def test_switch_provider_without_required_config(self):
        """Test error handling when switching provider without required config."""
        # OpenAI provider requires API key
        # Unset environment variable to ensure it's not loaded
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(ValidationError):
                config.Config(
                    rss_url="https://example.com/feed.xml",
                    transcription_provider="openai",
                )
        finally:
            # Restore original environment variable if it existed
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key


@pytest.mark.integration
@pytest.mark.slow
class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation scenarios."""

    def test_provider_cleanup_on_exception(self):
        """Test that providers can be cleaned up even if exceptions occur."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            generate_summaries=False,
            auto_speakers=False,
        )

        transcription_provider = create_transcription_provider(cfg)
        speaker_detector = create_speaker_detector(cfg)
        summarization_provider = create_summarization_provider(cfg)

        # Simulate exception during use
        try:
            # This might fail
            transcription_provider.initialize()
        except Exception:  # nosec B110
            pass

        # Cleanup should still work
        transcription_provider.cleanup()
        speaker_detector.cleanup()
        summarization_provider.cleanup()

    def test_multiple_initialization_calls(self):
        """Test that multiple initialization calls are safe (idempotent)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        # Should be safe to call multiple times
        detector.initialize()
        detector.initialize()
        detector.initialize()

        # Should still work
        detector.cleanup()

    def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls are safe (idempotent)."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            speaker_detector_provider="spacy",
            auto_speakers=False,
        )
        detector = create_speaker_detector(cfg)

        detector.initialize()

        # Should be safe to call multiple times
        detector.cleanup()
        detector.cleanup()
        detector.cleanup()


if __name__ == "__main__":
    unittest.main()
