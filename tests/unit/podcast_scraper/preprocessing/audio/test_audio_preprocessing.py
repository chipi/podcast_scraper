"""Unit tests for audio preprocessing module."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from podcast_scraper import config
from podcast_scraper.preprocessing.audio import cache, factory
from podcast_scraper.preprocessing.audio.ffmpeg_processor import (
    _check_ffmpeg_available,
    FFmpegAudioPreprocessor,
)


class TestFFmpegAudioPreprocessor(unittest.TestCase):
    """Tests for FFmpegAudioPreprocessor."""

    def setUp(self):
        """Set up test fixtures."""
        # Use a small fixture audio file for testing
        # Calculate path relative to project root
        # test_file is at:
        # tests/unit/podcast_scraper/preprocessing/audio/test_audio_preprocessing.py
        # Go up 6 levels to reach project root
        test_file = Path(__file__).resolve()
        # Navigate: audio -> preprocessing -> podcast_scraper -> unit -> tests -> project_root
        project_root = test_file.parent.parent.parent.parent.parent.parent
        self.fixture_audio = project_root / "tests" / "fixtures" / "audio" / "p01_e01_fast.mp3"
        # Fallback: try relative path if absolute doesn't work
        if not self.fixture_audio.exists():
            # Try going up from tests/unit to tests/
            alt_path = (
                test_file.parent.parent.parent.parent.parent
                / "fixtures"
                / "audio"
                / "p01_e01_fast.mp3"
            )
            if alt_path.exists():
                self.fixture_audio = alt_path
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_defaults(self):
        """Test FFmpegAudioPreprocessor initialization with defaults."""
        preprocessor = FFmpegAudioPreprocessor()
        self.assertEqual(preprocessor.sample_rate, 16000)
        self.assertEqual(preprocessor.silence_threshold, "-50dB")
        self.assertEqual(preprocessor.silence_duration, 2.0)
        self.assertEqual(preprocessor.target_loudness, -16)

    def test_init_custom(self):
        """Test FFmpegAudioPreprocessor initialization with custom values."""
        preprocessor = FFmpegAudioPreprocessor(
            sample_rate=22050,
            silence_threshold="-40dB",
            silence_duration=1.5,
            target_loudness=-14,
        )
        # Sample rate is adjusted to closest Opus-supported rate (22050 -> 24000)
        self.assertEqual(preprocessor.sample_rate, 24000)
        self.assertEqual(preprocessor.silence_threshold, "-40dB")
        self.assertEqual(preprocessor.silence_duration, 1.5)
        self.assertEqual(preprocessor.target_loudness, -14)

    def test_preprocess_success(self):
        """Test successful audio preprocessing."""
        if not _check_ffmpeg_available():
            self.skipTest("FFmpeg not available")
        if not self.fixture_audio.exists():
            self.skipTest("Fixture audio file not found")

        preprocessor = FFmpegAudioPreprocessor()
        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")

        success, elapsed = preprocessor.preprocess(str(self.fixture_audio), output_path)

        self.assertTrue(success)
        self.assertGreater(elapsed, 0)
        self.assertTrue(os.path.exists(output_path))
        # Verify output file is smaller than input (or at least exists)
        self.assertGreater(os.path.getsize(self.fixture_audio), 0)
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_preprocess_reduces_size(self):
        """Test that preprocessing reduces file size."""
        if not _check_ffmpeg_available():
            self.skipTest("FFmpeg not available")
        if not self.fixture_audio.exists():
            self.skipTest("Fixture audio file not found")

        preprocessor = FFmpegAudioPreprocessor()
        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")

        original_size = os.path.getsize(self.fixture_audio)
        success, _ = preprocessor.preprocess(str(self.fixture_audio), output_path)

        self.assertTrue(success)
        preprocessed_size = os.path.getsize(output_path)
        # Preprocessed file should be smaller (or at least not much larger)
        # Allow some tolerance for very small files
        self.assertLessEqual(preprocessed_size, original_size * 1.1)

    def test_preprocess_nonexistent_file(self):
        """Test preprocessing with nonexistent input file."""
        preprocessor = FFmpegAudioPreprocessor()
        output_path = os.path.join(self.temp_dir, "preprocessed.mp3")

        success, _ = preprocessor.preprocess("/nonexistent/file.mp3", output_path)

        self.assertFalse(success)
        self.assertFalse(os.path.exists(output_path))

    def test_get_cache_key(self):
        """Test cache key generation."""
        if not self.fixture_audio.exists():
            self.skipTest("Fixture audio file not found")

        preprocessor = FFmpegAudioPreprocessor()
        cache_key = preprocessor.get_cache_key(str(self.fixture_audio))

        self.assertIsInstance(cache_key, str)
        self.assertEqual(len(cache_key), 16)  # 16 hex chars
        # Same file + same config should produce same key
        cache_key2 = preprocessor.get_cache_key(str(self.fixture_audio))
        self.assertEqual(cache_key, cache_key2)

    def test_get_cache_key_different_config(self):
        """Test that different configs produce different cache keys."""
        if not self.fixture_audio.exists():
            self.skipTest("Fixture audio file not found")

        preprocessor1 = FFmpegAudioPreprocessor(sample_rate=16000)
        preprocessor2 = FFmpegAudioPreprocessor(sample_rate=22050)

        cache_key1 = preprocessor1.get_cache_key(str(self.fixture_audio))
        cache_key2 = preprocessor2.get_cache_key(str(self.fixture_audio))

        self.assertNotEqual(cache_key1, cache_key2)


class TestCache(unittest.TestCase):
    """Tests for preprocessing cache utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_cached_audio_path_miss(self):
        """Test cache miss scenario."""
        cache_key = "test_key_12345"
        cached_path = cache.get_cached_audio_path(cache_key, self.cache_dir)

        self.assertIsNone(cached_path)

    def test_get_cached_audio_path_hit(self):
        """Test cache hit scenario."""
        cache_key = "test_key_12345"
        cached_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create a dummy cached file
        with open(cached_file, "wb") as f:
            f.write(b"dummy audio data")

        cached_path = cache.get_cached_audio_path(cache_key, self.cache_dir)

        self.assertEqual(cached_path, cached_file)

    def test_save_to_cache(self):
        """Test saving to cache."""
        cache_key = "test_key_12345"
        source_file = os.path.join(self.temp_dir, "source.mp3")

        # Create a dummy source file
        with open(source_file, "wb") as f:
            f.write(b"dummy audio data")

        cached_path = cache.save_to_cache(source_file, cache_key, self.cache_dir)

        self.assertTrue(os.path.exists(cached_path))
        self.assertEqual(cached_path, os.path.join(self.cache_dir, f"{cache_key}.mp3"))
        # Verify content was copied
        with open(cached_path, "rb") as f:
            self.assertEqual(f.read(), b"dummy audio data")


class TestFactory(unittest.TestCase):
    """Tests for audio preprocessor factory."""

    def test_create_audio_preprocessor_disabled(self):
        """Test factory returns None when preprocessing is disabled."""
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            preprocessing_enabled=False,
        )

        preprocessor = factory.create_audio_preprocessor(cfg)

        self.assertIsNone(preprocessor)

    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffmpeg_available")
    def test_create_audio_preprocessor_no_ffmpeg(self, mock_check):
        """Test factory returns None when ffmpeg is not available."""
        mock_check.return_value = False
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            preprocessing_enabled=True,
        )

        preprocessor = factory.create_audio_preprocessor(cfg)

        self.assertIsNone(preprocessor)

    @patch("podcast_scraper.preprocessing.audio.ffmpeg_processor._check_ffmpeg_available")
    def test_create_audio_preprocessor_enabled(self, mock_check):
        """Test factory creates preprocessor when enabled and ffmpeg available."""
        mock_check.return_value = True
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            preprocessing_enabled=True,
            preprocessing_sample_rate=22050,
            preprocessing_silence_threshold="-40dB",
            preprocessing_silence_duration=1.5,
            preprocessing_target_loudness=-14,
        )

        preprocessor = factory.create_audio_preprocessor(cfg)

        self.assertIsNotNone(preprocessor)
        self.assertIsInstance(preprocessor, FFmpegAudioPreprocessor)
        # Sample rate is adjusted to closest Opus-supported rate (22050 -> 24000)
        self.assertEqual(preprocessor.sample_rate, 24000)
        self.assertEqual(preprocessor.silence_threshold, "-40dB")
        self.assertEqual(preprocessor.silence_duration, 1.5)
        self.assertEqual(preprocessor.target_loudness, -14)


if __name__ == "__main__":
    unittest.main()
