#!/usr/bin/env python3
"""E2E tests for Real ML Models.

These tests verify ML models work end-to-end using real models:
- Real Transformers models for summarization
- Real spaCy models for speaker detection
- Complete workflows with all real models

All tests use real ML models from Hugging Face and spaCy.
Tests are marked as @pytest.mark.ml_models.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Import cache helpers from integration tests
import sys

from podcast_scraper import Config, config, run_pipeline
from podcast_scraper.speaker_detectors.factory import create_speaker_detector
from podcast_scraper.speaker_detectors.ner_detector import NERSpeakerDetector
from podcast_scraper.summarization.factory import create_summarization_provider
from podcast_scraper.summarization.local_provider import TransformersSummarizationProvider

integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    require_transformers_model_cached,
    require_whisper_model_cached,
)

# Check if ML dependencies are available
TRANSFORMERS_AVAILABLE = False
SPACY_AVAILABLE = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.critical_path
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers dependencies not available")
class TestTransformersSummarization:
    """Real Transformers summarization model E2E tests."""

    def test_transformers_provider_summarize(self, e2e_server):
        """Test Transformers summarization provider with real model."""
        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        # Get transcript text from fixtures
        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01.txt"

        if not transcript_file.exists():
            pytest.skip(f"Transcript file not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory():
            # Create config with smallest Transformers model for speed
            cfg = Config(
                generate_metadata=True,  # Required for summaries
                generate_summaries=True,  # Required for provider to initialize
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                language="en",
            )

            # Initialize provider via factory
            provider = create_summarization_provider(cfg)
            provider.initialize()

            # Summarize text
            result = provider.summarize(
                text=transcript_text[:2000],  # Use first 2000 chars for speed
                episode_title="Test Episode",
            )

            # Verify result structure
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "summary" in result, "Result should contain 'summary' key"
            assert isinstance(result["summary"], str), "Summary should be a string"
            assert len(result["summary"]) > 0, "Summary should not be empty"

            # Cleanup
            provider.cleanup()

    def test_transformers_provider_in_full_workflow(self, e2e_server):
        """Test Transformers summarization provider in full workflow."""
        # Require model to be cached (fail fast if not)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1")

        # Require Whisper model for transcription (podcast1 has no transcript URL)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                generate_metadata=True,  # Required for summaries
                generate_summaries=True,
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # Use test default (small, fast)
                transcribe_missing=True,  # Required: podcast1 has no transcript URL
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created (use rglob to search recursively)
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created (summaries are stored in metadata)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.critical_path
@pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy dependencies not available")
class TestSpacySpeakerDetection:
    """Real spaCy speaker detection model E2E tests."""

    def test_spacy_detector_detect_speakers(self, e2e_server):
        """Test spaCy speaker detector with real model.

        Note: spaCy model (en_core_web_sm) is installed as a dependency.
        """
        with tempfile.TemporaryDirectory():
            # Create config with smallest spaCy model for speed
            cfg = Config(
                speaker_detector_provider="spacy",
                ner_model=config.DEFAULT_NER_MODEL,  # Default: en_core_web_sm
                auto_speakers=True,
            )

            # Initialize detector
            detector = create_speaker_detector(cfg)
            detector.initialize()

            # Detect speakers
            episode_title = "Episode 1: Building Trails That Last (with Liam Verbeek)"
            episode_description = "Maya talks with trail builder Liam Verbeek about drainage."
            known_hosts = set()

            speakers, hosts, success = detector.detect_speakers(
                episode_title=episode_title,
                episode_description=episode_description,
                known_hosts=known_hosts,
            )

            # Verify result
            assert isinstance(speakers, list), "Speakers should be a list"
            assert isinstance(hosts, set), "Hosts should be a set"
            assert isinstance(success, bool), "Success should be a boolean"
            # Note: Detection may or may not succeed depending on text content
            # This test verifies the model loads and runs correctly

            # Cleanup
            detector.cleanup()

    def test_spacy_detector_in_full_workflow(self, e2e_server):
        """Test spaCy speaker detector in full workflow.

        Note: spaCy model (en_core_web_sm) is installed as a dependency.
        """
        # Use podcast1_with_transcript which has a transcript URL, so no
        # transcription is needed (spaCy detector only needs transcript text)
        rss_url = e2e_server.urls.feed("podcast1_with_transcript")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                auto_speakers=True,
                speaker_detector_provider="spacy",
                ner_model=config.DEFAULT_NER_MODEL,  # Default: en_core_web_sm
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).glob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.critical_path
@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE or not SPACY_AVAILABLE,
    reason="ML dependencies not available",
)
class TestAllMLModelsTogether:
    """E2E tests with all real ML models working together."""

    def test_all_models_in_full_workflow(self, e2e_server):
        """Test complete workflow with all real ML models.

        Note: spaCy model (en_core_web_sm) is installed as a dependency.
        """
        # Require transformers model to be cached (fail fast if not)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,  # Enable Whisper transcription
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                generate_metadata=True,
                metadata_format="json",
                generate_summaries=True,
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
                auto_speakers=True,
                speaker_detector_provider="spacy",
                ner_model=config.DEFAULT_NER_MODEL,  # Same for tests and production
            )

            # Run pipeline with all models
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count > 0, "Should process at least one episode"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created (use rglob to search recursively)
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "At least one transcript file should be created"

            # Verify metadata file was created (use rglob to search recursively)
            metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
            assert len(metadata_files) > 0, "At least one metadata file should be created"

    def test_model_loading_and_cleanup(self, e2e_server):
        """Test that models load and cleanup correctly.

        Note: spaCy model (en_core_web_sm) is installed as a dependency.
        """
        # Require transformers model to be cached (spaCy model is installed as dependency)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        # Get transcript text from fixtures
        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01.txt"

        if not transcript_file.exists():
            pytest.skip(f"Transcript file not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")[:2000]  # First 2000 chars

        with tempfile.TemporaryDirectory():
            # Test Transformers provider
            cfg_summary = Config(
                generate_metadata=True,  # Required for summaries
                generate_summaries=True,  # Required for provider to initialize
                summary_provider="transformers",
                summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
            )
            provider = TransformersSummarizationProvider(cfg_summary)
            provider.initialize()
            result = provider.summarize(text=transcript_text, episode_title="Test")
            assert "summary" in result
            provider.cleanup()

            # Test spaCy detector
            cfg_speaker = Config(
                speaker_detector_provider="spacy",
                ner_model=config.DEFAULT_NER_MODEL,  # Same for tests and production
                auto_speakers=True,
            )
            detector = NERSpeakerDetector(cfg_speaker)
            detector.initialize()
            speakers, hosts, success = detector.detect_speakers(
                episode_title="Test Episode",
                episode_description="Test description",
                known_hosts=set(),
            )
            assert isinstance(speakers, list)
            assert isinstance(hosts, set)
            detector.cleanup()
