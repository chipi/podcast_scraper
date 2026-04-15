#!/usr/bin/env python3
"""Integration tests for transcript caching functionality."""

import os
import tempfile
import unittest

import pytest

from podcast_scraper.cache import transcript_cache

pytestmark = [pytest.mark.integration]


class TestTranscriptCacheIntegration(unittest.TestCase):
    """Integration tests for transcript caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_integration_with_transcription(self):
        """Test that transcript cache works with transcription pipeline."""
        # Create a test audio file
        audio_path = os.path.join(self.temp_dir, "test.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"fake audio content" * 1000)

        # Get hash
        audio_hash = transcript_cache.get_audio_hash(audio_path)

        # Save transcript to cache
        transcript_text = "This is a test transcript."
        cache_path = transcript_cache.save_transcript_to_cache(
            audio_hash, transcript_text, cache_dir=self.cache_dir
        )

        # Verify cache file exists
        self.assertTrue(os.path.exists(cache_path))

        # Retrieve from cache
        cached_transcript = transcript_cache.get_cached_transcript(
            audio_hash, cache_dir=self.cache_dir
        )
        self.assertEqual(cached_transcript, transcript_text)

    def test_cache_enables_fast_reprocessing(self):
        """Test that cache enables fast reprocessing with different providers."""
        # Create test audio
        audio_path = os.path.join(self.temp_dir, "test.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"fake audio content" * 1000)

        audio_hash = transcript_cache.get_audio_hash(audio_path)

        # Save transcript from first provider
        transcript_text = "Transcript from provider 1"
        transcript_cache.save_transcript_to_cache(
            audio_hash,
            transcript_text,
            provider_name="whisper",
            model="large-v3",
            cache_dir=self.cache_dir,
        )

        # Retrieve with same provider/model
        cached = transcript_cache.get_cached_transcript(audio_hash, cache_dir=self.cache_dir)
        self.assertEqual(cached, transcript_text)

    def test_cache_segments_round_trip_for_gi_sidecar(self):
        """Transcript cache stores optional segments for GI quote timing (issue #540)."""
        audio_path = os.path.join(self.temp_dir, "seg.mp3")
        with open(audio_path, "wb") as f:
            f.write(b"fake audio content" * 1000)

        audio_hash = transcript_cache.get_audio_hash(audio_path)
        transcript_text = "One two three."
        segs = [
            {"start": 0.0, "end": 1.0, "text": "One "},
            {"start": 1.0, "end": 2.0, "text": "two "},
            {"start": 2.0, "end": 3.0, "text": "three."},
        ]
        transcript_cache.save_transcript_to_cache(
            audio_hash,
            transcript_text,
            provider_name="whisper",
            model="base",
            cache_dir=self.cache_dir,
            segments=segs,
        )

        entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir=self.cache_dir)
        self.assertIsNotNone(entry)
        assert entry is not None
        t, loaded = entry
        self.assertEqual(t, transcript_text)
        self.assertEqual(loaded, segs)
