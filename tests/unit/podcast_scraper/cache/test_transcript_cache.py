#!/usr/bin/env python3
"""Unit tests for transcript caching functionality."""

import os
import tempfile
import unittest
from pathlib import Path

import pytest

from podcast_scraper.cache import transcript_cache

pytestmark = [pytest.mark.unit]


class TestTranscriptCache(unittest.TestCase):
    """Tests for transcript caching functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_audio_hash(self):
        """Test audio hash generation."""
        # Create a test audio file
        audio_path = os.path.join(self.temp_dir, "test.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"test audio content" * 100)

        hash_value = transcript_cache.get_audio_hash(audio_path)
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 16)  # First 16 hex chars (64 bits)

    def test_get_audio_hash_handles_missing_file(self):
        """Test audio hash generation handles missing file gracefully."""
        audio_path = os.path.join(self.temp_dir, "nonexistent.mp3")
        hash_value = transcript_cache.get_audio_hash(audio_path)
        # Should return hash of path as fallback
        self.assertIsInstance(hash_value, str)
        self.assertEqual(len(hash_value), 16)  # First 16 hex chars

    def test_get_cached_transcript_miss(self):
        """Test cache miss when transcript doesn't exist."""
        result = transcript_cache.get_cached_transcript(
            "nonexistent_hash", cache_dir=self.cache_dir
        )
        self.assertIsNone(result)

    def test_get_cached_transcript_hit(self):
        """Test cache hit when transcript exists."""
        import json

        audio_hash = "test_hash_12345"
        cache_path = Path(self.cache_dir) / f"{audio_hash}.json"

        transcript_text = "This is a cached transcript."
        cache_data = {"transcript": transcript_text}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

        result = transcript_cache.get_cached_transcript(audio_hash, cache_dir=self.cache_dir)
        self.assertEqual(result, transcript_text)

    def test_save_transcript_to_cache(self):
        """Test saving transcript to cache."""
        import json

        audio_hash = "test_hash_12345"
        transcript_text = "This is a test transcript."

        result_path = transcript_cache.save_transcript_to_cache(
            audio_hash, transcript_text, cache_dir=self.cache_dir
        )

        self.assertTrue(os.path.exists(result_path))
        # Verify JSON content
        with open(result_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        self.assertEqual(cache_data["transcript"], transcript_text)

    def test_save_transcript_to_cache_with_provider_model(self):
        """Test saving transcript with provider and model info."""
        import json

        audio_hash = "test_hash_12345"
        transcript_text = "This is a test transcript."

        result_path = transcript_cache.save_transcript_to_cache(
            audio_hash,
            transcript_text,
            provider_name="openai",
            model="whisper-1",
            cache_dir=self.cache_dir,
        )

        self.assertTrue(os.path.exists(result_path))
        # Check JSON contains provider and model
        with open(result_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        self.assertEqual(cache_data["transcript"], transcript_text)
        self.assertEqual(cache_data["provider"], "openai")
        self.assertEqual(cache_data["model"], "whisper-1")

    def test_cache_round_trip(self):
        """Test complete cache round trip: save then retrieve."""
        audio_hash = "test_hash_12345"
        transcript_text = "Round trip test transcript."

        # Save
        save_path = transcript_cache.save_transcript_to_cache(
            audio_hash, transcript_text, cache_dir=self.cache_dir
        )
        self.assertTrue(os.path.exists(save_path))

        # Retrieve
        retrieved = transcript_cache.get_cached_transcript(audio_hash, cache_dir=self.cache_dir)
        self.assertEqual(retrieved, transcript_text)

    def test_save_transcript_to_cache_with_non_string_model(self):
        """Test saving transcript with non-string model (defensive conversion)."""
        import json
        from unittest.mock import Mock

        audio_hash = "test_hash_non_string_model"
        transcript_text = "Test transcript with non-string model."

        # Create a mock model object (simulating Whisper model object)
        mock_model = Mock()
        mock_model.__str__ = Mock(return_value="<MockModel object>")

        result_path = transcript_cache.save_transcript_to_cache(
            audio_hash,
            transcript_text,
            provider_name="whisper",
            model=mock_model,  # Non-string model object
            cache_dir=self.cache_dir,
        )

        self.assertTrue(os.path.exists(result_path))
        # Check JSON contains converted model string
        with open(result_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        self.assertEqual(cache_data["transcript"], transcript_text)
        self.assertEqual(cache_data["provider"], "whisper")
        # Model should be converted to string representation
        self.assertEqual(cache_data["model"], "<MockModel object>")

    def test_get_cached_transcript_entry_without_segments(self):
        """Entry API returns transcript and None segments when key absent."""
        import json

        audio_hash = "entry_no_seg"
        cache_path = Path(self.cache_dir) / f"{audio_hash}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"transcript": "hello"}),
            encoding="utf-8",
        )
        entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir=self.cache_dir)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry[0], "hello")
        self.assertIsNone(entry[1])

    def test_save_and_load_segments_round_trip(self):
        """Optional segments persist and reload for GI timestamp sidecar parity."""
        audio_hash = "seg_roundtrip"
        text = "Hello world"
        segs = [{"start": 0.0, "end": 1.0, "text": "Hello world"}]
        transcript_cache.save_transcript_to_cache(
            audio_hash, text, cache_dir=self.cache_dir, segments=segs
        )
        entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir=self.cache_dir)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry[0], text)
        self.assertEqual(entry[1], segs)
        self.assertEqual(transcript_cache.get_cached_transcript(audio_hash, self.cache_dir), text)

    def test_malformed_segments_in_cache_file_ignored(self):
        """Invalid segments payload must not break transcript read."""
        import json

        audio_hash = "bad_seg"
        cache_path = Path(self.cache_dir) / f"{audio_hash}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"transcript": "ok", "segments": "not-a-list"}),
            encoding="utf-8",
        )
        entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir=self.cache_dir)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry[0], "ok")
        self.assertIsNone(entry[1])

    def test_segments_with_non_dict_item_ignored(self):
        import json

        audio_hash = "bad_seg_item"
        cache_path = Path(self.cache_dir) / f"{audio_hash}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"transcript": "ok", "segments": [{"start": 0, "end": 1, "text": "a"}, "x"]}
            ),
            encoding="utf-8",
        )
        entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir=self.cache_dir)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry[0], "ok")
        self.assertIsNone(entry[1])
