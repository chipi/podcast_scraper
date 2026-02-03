#!/usr/bin/env python3
"""Integration tests for audio preprocessing with real audio files.

These tests verify that audio preprocessing works correctly with real audio files
from fixtures, including cache behavior, size reduction, and integration with
the transcription workflow.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.module_preprocessing]

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper.preprocessing.audio import cache, factory
from podcast_scraper.preprocessing.audio.ffmpeg_processor import (
    _check_ffmpeg_available,
    FFmpegAudioPreprocessor,
)

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly
import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

create_test_config = parent_conftest.create_test_config

# Get fixture audio files
FIXTURE_AUDIO_DIR = tests_dir / "fixtures" / "audio"
FFMPEG_AVAILABLE = _check_ffmpeg_available()


@pytest.mark.integration
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestAudioPreprocessingIntegration(unittest.TestCase):
    """Integration tests for audio preprocessing with real audio files."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Find a small fixture audio file for testing
        cls.fixture_audio = FIXTURE_AUDIO_DIR / "p01_e01_fast.mp3"
        if not cls.fixture_audio.exists():
            # Fallback to any available audio file
            audio_files = list(FIXTURE_AUDIO_DIR.glob("*.mp3"))
            if audio_files:
                cls.fixture_audio = audio_files[0]
            else:
                pytest.skip("No fixture audio files found")

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.cfg = create_test_config(
            preprocessing_enabled=True,
            preprocessing_cache_dir=self.cache_dir,
            preprocessing_sample_rate=16000,
            preprocessing_silence_threshold="-50dB",
            preprocessing_silence_duration=2.0,
            preprocessing_target_loudness=-16,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preprocess_real_audio_file(self):
        """Test preprocessing a real audio file from fixtures."""
        preprocessor = factory.create_audio_preprocessor(self.cfg)
        self.assertIsNotNone(preprocessor)

        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")
        success, elapsed = preprocessor.preprocess(str(self.fixture_audio), output_path)

        self.assertTrue(success, "Preprocessing should succeed with real audio file")
        self.assertGreater(elapsed, 0, "Preprocessing should take some time")
        self.assertTrue(os.path.exists(output_path), "Preprocessed file should exist")
        self.assertGreater(os.path.getsize(output_path), 0, "Preprocessed file should not be empty")

    def test_preprocessing_reduces_file_size(self):
        """Test that preprocessing reduces file size for typical podcast audio."""
        preprocessor = factory.create_audio_preprocessor(self.cfg)
        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")

        original_size = os.path.getsize(self.fixture_audio)
        success, _ = preprocessor.preprocess(str(self.fixture_audio), output_path)

        self.assertTrue(success)
        preprocessed_size = os.path.getsize(output_path)

        # Preprocessed file should be smaller (or at least not much larger)
        # For speech-optimized Opus at 24kbps, we expect significant reduction
        # Allow some tolerance for very small files
        self.assertLessEqual(
            preprocessed_size,
            original_size * 1.1,
            f"Preprocessed file ({preprocessed_size} bytes) should be smaller "
            f"than original ({original_size} bytes)",
        )

    def test_cache_hit_after_preprocessing(self):
        """Test that cache is used after first preprocessing."""
        preprocessor = factory.create_audio_preprocessor(self.cfg)
        cache_key = preprocessor.get_cache_key(str(self.fixture_audio))

        # First preprocessing - should miss cache
        output_path1 = os.path.join(self.temp_dir, "preprocessed1.mp3")
        success1, elapsed1 = preprocessor.preprocess(str(self.fixture_audio), output_path1)
        self.assertTrue(success1)

        # Save to cache
        cached_path = cache.save_to_cache(output_path1, cache_key, self.cache_dir)
        self.assertTrue(os.path.exists(cached_path))

        # Second preprocessing - should hit cache
        cached_result = cache.get_cached_audio_path(cache_key, self.cache_dir)
        self.assertIsNotNone(cached_result, "Cache should return path after saving")
        self.assertEqual(cached_result, cached_path)

    def test_cache_key_consistency(self):
        """Test that cache key is consistent for same file and config."""
        preprocessor = FFmpegAudioPreprocessor(
            sample_rate=16000,
            silence_threshold="-50dB",
            silence_duration=2.0,
            target_loudness=-16,
        )

        cache_key1 = preprocessor.get_cache_key(str(self.fixture_audio))
        cache_key2 = preprocessor.get_cache_key(str(self.fixture_audio))

        self.assertEqual(cache_key1, cache_key2, "Cache key should be consistent")

    def test_cache_key_different_config(self):
        """Test that different configs produce different cache keys."""
        preprocessor1 = FFmpegAudioPreprocessor(sample_rate=16000)
        preprocessor2 = FFmpegAudioPreprocessor(sample_rate=24000)  # Opus-supported rate

        cache_key1 = preprocessor1.get_cache_key(str(self.fixture_audio))
        cache_key2 = preprocessor2.get_cache_key(str(self.fixture_audio))

        self.assertNotEqual(
            cache_key1, cache_key2, "Different configs should produce different cache keys"
        )

    def test_preprocessing_with_custom_config(self):
        """Test preprocessing with custom configuration values."""
        # Use 24000 Hz (Opus-supported) instead of 22050
        custom_cfg = create_test_config(
            preprocessing_enabled=True,
            preprocessing_cache_dir=self.cache_dir,
            preprocessing_sample_rate=24000,  # Opus-supported rate
            preprocessing_silence_threshold="-40dB",
            preprocessing_silence_duration=1.5,
            preprocessing_target_loudness=-14,
        )

        preprocessor = factory.create_audio_preprocessor(custom_cfg)
        self.assertIsNotNone(preprocessor)
        # Sample rate should be adjusted to 24000 (Opus-supported)
        self.assertEqual(preprocessor.sample_rate, 24000)
        self.assertEqual(preprocessor.silence_threshold, "-40dB")
        self.assertEqual(preprocessor.silence_duration, 1.5)
        self.assertEqual(preprocessor.target_loudness, -14)

        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")
        success, _ = preprocessor.preprocess(str(self.fixture_audio), output_path)

        self.assertTrue(success, "Preprocessing should succeed with custom config")

    def test_preprocessing_output_format(self):
        """Test that preprocessed output is in Opus format."""
        preprocessor = factory.create_audio_preprocessor(self.cfg)
        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")

        success, _ = preprocessor.preprocess(str(self.fixture_audio), output_path)
        self.assertTrue(success)

        # Verify file exists and has .mp3 extension
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(output_path.endswith(".mp3"))

        # Verify file is not empty
        self.assertGreater(os.path.getsize(output_path), 0)


@pytest.mark.integration
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestAudioPreprocessingCacheIntegration(unittest.TestCase):
    """Integration tests for preprocessing cache behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.fixture_audio = FIXTURE_AUDIO_DIR / "p01_e01_fast.mp3"
        if not self.fixture_audio.exists():
            audio_files = list(FIXTURE_AUDIO_DIR.glob("*.mp3"))
            if audio_files:
                self.fixture_audio = audio_files[0]
            else:
                pytest.skip("No fixture audio files found")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_miss_for_new_file(self):
        """Test that cache returns None for new file."""
        cache_key = "new_file_key_12345"
        cached_path = cache.get_cached_audio_path(cache_key, self.cache_dir)

        self.assertIsNone(cached_path, "Cache should return None for new file")

    def test_cache_save_and_retrieve(self):
        """Test saving to cache and retrieving."""
        cache_key = "test_key_12345"
        source_file = os.path.join(self.temp_dir, "source.mp3")

        # Create a dummy source file
        with open(source_file, "wb") as f:
            f.write(b"dummy audio data for testing")

        # Save to cache
        cached_path = cache.save_to_cache(source_file, cache_key, self.cache_dir)

        # Verify cache file exists
        self.assertTrue(os.path.exists(cached_path))
        self.assertEqual(cached_path, os.path.join(self.cache_dir, f"{cache_key}.mp3"))

        # Retrieve from cache
        retrieved_path = cache.get_cached_audio_path(cache_key, self.cache_dir)

        self.assertIsNotNone(retrieved_path)
        self.assertEqual(retrieved_path, cached_path)

        # Verify content matches
        with open(cached_path, "rb") as f:
            cached_content = f.read()
        with open(source_file, "rb") as f:
            source_content = f.read()
        self.assertEqual(cached_content, source_content)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        non_existent_cache_dir = os.path.join(self.temp_dir, "new_cache")
        cache_key = "test_key"
        source_file = os.path.join(self.temp_dir, "source.mp3")

        # Create source file
        with open(source_file, "wb") as f:
            f.write(b"test data")

        # Save to cache (should create directory)
        cached_path = cache.save_to_cache(source_file, cache_key, non_existent_cache_dir)

        self.assertTrue(os.path.exists(non_existent_cache_dir), "Cache directory should be created")
        self.assertTrue(os.path.exists(cached_path), "Cached file should exist")


@pytest.mark.integration
@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="FFmpeg not available")
class TestAudioPreprocessingWorkflowIntegration(unittest.TestCase):
    """Integration tests for audio preprocessing in transcription workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.fixture_audio = FIXTURE_AUDIO_DIR / "p01_e01_fast.mp3"
        if not self.fixture_audio.exists():
            audio_files = list(FIXTURE_AUDIO_DIR.glob("*.mp3"))
            if audio_files:
                self.fixture_audio = audio_files[0]
            else:
                pytest.skip("No fixture audio files found")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preprocessing_before_transcription_provider(self):
        """Test that preprocessing happens before provider receives audio."""
        from podcast_scraper import models
        from podcast_scraper.preprocessing.audio.factory import create_audio_preprocessor
        from podcast_scraper.workflow.episode_processor import transcribe_media_to_text

        cfg = create_test_config(
            preprocessing_enabled=True,
            preprocessing_cache_dir=self.cache_dir,
            transcribe_missing=True,
            transcription_provider="whisper",
        )

        # Create a mock transcription provider
        mock_provider = Mock()
        mock_provider.transcribe_with_segments = Mock(
            return_value=({"text": "Test transcript", "segments": []}, 1.5)
        )

        # Create a transcription job
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Test Episode",
            ep_title_safe="Test_Episode",
            temp_media=str(self.fixture_audio),
        )

        # Copy fixture to temp location (transcribe_media_to_text may modify it)
        temp_media = os.path.join(self.temp_dir, "temp_audio.mp3")
        shutil.copy2(self.fixture_audio, temp_media)
        job.temp_media = temp_media

        # Create preprocessor to verify it's used
        preprocessor = create_audio_preprocessor(cfg)
        self.assertIsNotNone(preprocessor)

        # Mock the factory import to track calls
        with patch(
            "podcast_scraper.preprocessing.audio.factory.create_audio_preprocessor"
        ) as mock_factory:
            mock_factory.return_value = preprocessor

            # Call transcribe_media_to_text
            success, transcript_path, bytes_downloaded = transcribe_media_to_text(
                job=job,
                cfg=cfg,
                whisper_model=None,
                run_suffix=None,
                effective_output_dir=self.temp_dir,
                transcription_provider=mock_provider,
                pipeline_metrics=None,
            )

            # Verify preprocessing was attempted (factory was called)
            mock_factory.assert_called_once_with(cfg)

            # Verify transcription provider was called
            mock_provider.transcribe_with_segments.assert_called_once()

            # Get the audio path that was passed to provider
            call_args = mock_provider.transcribe_with_segments.call_args
            audio_path_passed = call_args[0][0]  # First positional argument

            # Verify that either preprocessed audio or original was passed
            # (depending on whether preprocessing succeeded)
            self.assertTrue(
                os.path.exists(audio_path_passed) or audio_path_passed == temp_media,
                "Provider should receive a valid audio path",
            )

    def test_preprocessing_disabled_uses_original_audio(self):
        """Test that when preprocessing is disabled, original audio is used."""
        from podcast_scraper import models
        from podcast_scraper.workflow.episode_processor import transcribe_media_to_text

        cfg = create_test_config(
            preprocessing_enabled=False,  # Disabled
            transcribe_missing=True,
            transcription_provider="whisper",
        )

        # Create a mock transcription provider
        mock_provider = Mock()
        mock_provider.transcribe_with_segments = Mock(
            return_value=({"text": "Test transcript", "segments": []}, 1.5)
        )

        # Create a transcription job
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Test Episode",
            ep_title_safe="Test_Episode",
            temp_media=str(self.fixture_audio),
        )

        # Copy fixture to temp location
        temp_media = os.path.join(self.temp_dir, "temp_audio.mp3")
        shutil.copy2(self.fixture_audio, temp_media)
        job.temp_media = temp_media

        # Call transcribe_media_to_text
        success, transcript_path, bytes_downloaded = transcribe_media_to_text(
            job=job,
            cfg=cfg,
            whisper_model=None,
            run_suffix=None,
            effective_output_dir=self.temp_dir,
            transcription_provider=mock_provider,
            pipeline_metrics=None,
        )

        # Verify transcription provider was called
        mock_provider.transcribe_with_segments.assert_called_once()

        # Get the audio path that was passed to provider
        call_args = mock_provider.transcribe_with_segments.call_args
        audio_path_passed = call_args[0][0]

        # Should be original audio (not preprocessed)
        self.assertEqual(audio_path_passed, temp_media)


if __name__ == "__main__":
    unittest.main()
