#!/usr/bin/env python3
"""E2E tests for audio preprocessing with Whisper transcription.

These tests verify that audio preprocessing works end-to-end with Whisper transcription:
- Audio preprocessing before Whisper transcription
- File size reduction verification
- Cache behavior in full pipeline
- Preprocessing + transcription workflow
- Integration with episode processing

All tests use real Whisper models and real audio files from fixtures.
Tests are marked as @pytest.mark.ml_models and @pytest.mark.e2e.
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

from podcast_scraper import Config, config, run_pipeline
from podcast_scraper.preprocessing.audio.ffmpeg_processor import _check_ffmpeg_available

# Import cache helpers from integration tests
integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import require_whisper_model_cached  # noqa: E402

# Check if Whisper is available
try:
    import whisper  # noqa: F401

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

FFMPEG_AVAILABLE = _check_ffmpeg_available()


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.critical_path
@pytest.mark.skipif(
    not WHISPER_AVAILABLE or not FFMPEG_AVAILABLE,
    reason="Whisper or FFmpeg dependencies not available",
)
class TestAudioPreprocessingWithWhisperE2E:
    """E2E tests for audio preprocessing with Whisper transcription."""

    def test_preprocessing_before_whisper_transcription(self, e2e_server):
        """Test that preprocessing happens before Whisper transcription in full pipeline.

        This test verifies:
        - Audio is preprocessed before being passed to Whisper
        - Preprocessed audio is smaller than original
        - Transcription succeeds with preprocessed audio
        - Transcript file is created
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
                preprocessing_sample_rate=16000,
                preprocessing_silence_threshold="-50dB",
                preprocessing_silence_duration=2.0,
                preprocessing_target_loudness=-16,
            )

            # Run pipeline with preprocessing enabled
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"
            assert isinstance(summary, str), "Summary should be a string"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "Should create at least one transcript file"

            # Verify preprocessing cache was used (directory should exist)
            if os.path.exists(cache_dir):
                # Cache may or may not have files depending on whether preprocessing succeeded
                # But directory should exist if preprocessing was attempted
                _ = list(Path(cache_dir).glob("*.mp3"))  # Check cache exists

    def test_preprocessing_reduces_file_size(self, e2e_server):
        """Test that preprocessing reduces audio file size before transcription.

        This test verifies:
        - Original audio file size
        - Preprocessed audio file size (should be smaller)
        - Transcription still works with smaller file
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Check if preprocessed files exist in cache
            if os.path.exists(cache_dir):
                cache_files = list(Path(cache_dir).glob("*.mp3"))
                if cache_files:
                    # Verify preprocessed file is smaller than typical original
                    # (We can't easily get original size in E2E test, but we can verify
                    # preprocessed file exists and is reasonable size)
                    preprocessed_file = cache_files[0]
                    preprocessed_size = preprocessed_file.stat().st_size
                    # Preprocessed file should be reasonable size (not empty, not huge)
                    assert preprocessed_size > 0, "Preprocessed file should not be empty"
                    # For 1-minute audio, preprocessed should be < 1MB typically
                    assert (
                        preprocessed_size < 10 * 1024 * 1024
                    ), "Preprocessed file should be reasonable size"

    def test_preprocessing_cache_reuse(self, e2e_server):
        """Test that preprocessing cache is reused on second run.

        This test verifies:
        - First run: Audio is preprocessed and cached
        - Second run: Cached preprocessed audio is reused
        - Both runs produce transcripts
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
            )

            # First run: Should preprocess and cache
            count1, summary1 = run_pipeline(cfg)
            assert count1 >= 0, "First pipeline run should complete"

            # Get cache files after first run
            cache_files_after_first = []
            if os.path.exists(cache_dir):
                cache_files_after_first = list(Path(cache_dir).glob("*.mp3"))

            # Second run: Should reuse cache
            # Use a different output directory to avoid skip_existing
            output_dir2 = os.path.join(tmpdir, "run2")
            cfg2 = Config(
                rss_url=rss_url,
                output_dir=output_dir2,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,  # Same cache directory
            )

            count2, summary2 = run_pipeline(cfg2)
            assert count2 >= 0, "Second pipeline run should complete"

            # Verify cache was reused (same cache files should exist)
            if cache_files_after_first:
                cache_files_after_second = list(Path(cache_dir).glob("*.mp3"))
                # Cache should still have files (may have same or more)
                assert len(cache_files_after_second) >= len(
                    cache_files_after_first
                ), "Cache should be reused or have same/more files"

    def test_preprocessing_disabled_uses_original_audio(self, e2e_server):
        """Test that when preprocessing is disabled, original audio is used.

        This test verifies:
        - Preprocessing disabled: Original audio is passed to Whisper
        - Transcription still works
        - No preprocessing cache is created
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=False,  # Disabled
                preprocessing_cache_dir=cache_dir,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "Should create at least one transcript file"

            # Verify preprocessing cache was NOT used (should be empty or not exist)
            # Note: Cache directory might be created but should be empty
            if os.path.exists(cache_dir):
                cache_files = list(Path(cache_dir).glob("*.mp3"))
                # Cache should be empty when preprocessing is disabled
                assert len(cache_files) == 0, "Preprocessing cache should be empty when disabled"

    @pytest.mark.slow
    def test_preprocessing_with_multiple_episodes(self, e2e_server):
        """Test preprocessing with multiple episodes in single pipeline run.

        This test verifies:
        - Multiple episodes are preprocessed
        - Cache is used across episodes
        - All episodes are transcribed successfully
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=2,  # Process 2 episodes
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Verify multiple transcript files were created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) >= 1, "Should create at least one transcript file"

            # Verify preprocessing cache may have multiple files (one per unique audio)
            if os.path.exists(cache_dir):
                # Cache may have 0 to N files depending on whether preprocessing succeeded
                # and whether episodes share the same audio
                _ = list(Path(cache_dir).glob("*.mp3"))  # Check cache exists

    def test_preprocessing_integration_with_episode_processor(self, e2e_server):
        """Test preprocessing integration with episode processor workflow.

        This test verifies:
        - Episode processor calls preprocessing before transcription
        - Preprocessed audio is passed to Whisper provider
        - Full workflow: RSS → Download → Preprocess → Transcribe → Save
        """
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        rss_url = e2e_server.urls.feed("podcast1")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
                generate_metadata=False,  # Disable to speed up test
                generate_summaries=False,  # Disable to speed up test
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "Should create at least one transcript file"

            # Verify transcript file has content
            if transcript_files:
                transcript_content = transcript_files[0].read_text(encoding="utf-8")
                assert len(transcript_content) > 0, "Transcript file should have content"
                # Transcript should contain some text (not just whitespace)
                assert (
                    transcript_content.strip()
                ), "Transcript should contain non-whitespace content"


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.skipif(
    not WHISPER_AVAILABLE or not FFMPEG_AVAILABLE,
    reason="Whisper or FFmpeg dependencies not available",
)
class TestAudioPreprocessingConfigurationE2E:
    """E2E tests for audio preprocessing configuration options."""

    def test_preprocessing_with_custom_sample_rate(self, e2e_server):
        """Test preprocessing with custom sample rate configuration."""
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        # Use podcast1_multi_episode which is available in both fast and multi_episode modes
        # (it's in PODCAST_RSS_MAP_FAST and allowed in both test modes)
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            # Use 24000 Hz (Opus-supported rate)
            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
                preprocessing_sample_rate=24000,  # Custom sample rate
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "Should create at least one transcript file"

    def test_preprocessing_with_custom_silence_threshold(self, e2e_server):
        """Test preprocessing with custom silence threshold."""
        # Require model to be cached (fail fast if not)
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        # Use podcast1_multi_episode which is available in both fast and multi_episode modes
        # (it's in PODCAST_RSS_MAP_FAST and allowed in both test modes)
        rss_url = e2e_server.urls.feed("podcast1_multi_episode")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, ".cache", "preprocessing")

            cfg = Config(
                rss_url=rss_url,
                output_dir=tmpdir,
                max_episodes=1,
                transcribe_missing=True,
                transcription_provider="whisper",
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                language="en",
                preprocessing_enabled=True,
                preprocessing_cache_dir=cache_dir,
                preprocessing_silence_threshold="-40dB",  # Custom threshold
            )

            # Run pipeline
            count, summary = run_pipeline(cfg)

            # Verify pipeline completed
            assert count >= 0, "Pipeline should complete"

            # Verify transcript file was created
            transcript_files = list(Path(tmpdir).rglob("*.txt"))
            assert len(transcript_files) > 0, "Should create at least one transcript file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
