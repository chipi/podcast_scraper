"""Unit tests for GitHub #561 MP3 preprocessing bitrate policy."""

import unittest

import pytest

from podcast_scraper import config
from podcast_scraper.preprocessing.audio.factory import (
    mp3_bitrates_to_probe_for_cache,
    next_lower_mp3_bitrate_kbps,
    resolve_preprocessing_mp3_bitrate_kbps,
)


@pytest.mark.unit
class TestPreprocessMp3Policy(unittest.TestCase):
    def test_resolve_auto_openai(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            preprocessing_mp3_bitrate_kbps=None,
            transcription_provider="openai",
            openai_api_key="sk-test-key-for-unit-resolve",
        )
        self.assertEqual(resolve_preprocessing_mp3_bitrate_kbps(cfg), 48)

    def test_resolve_auto_gemini(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            preprocessing_mp3_bitrate_kbps=None,
            transcription_provider="gemini",
            gemini_api_key="test-gemini-key-for-unit-resolve",
        )
        self.assertEqual(resolve_preprocessing_mp3_bitrate_kbps(cfg), 48)

    def test_resolve_auto_whisper(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            preprocessing_mp3_bitrate_kbps=None,
            transcription_provider="whisper",
        )
        self.assertEqual(resolve_preprocessing_mp3_bitrate_kbps(cfg), 64)

    def test_resolve_explicit_override(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            preprocessing_mp3_bitrate_kbps=56,
            transcription_provider="openai",
            openai_api_key="sk-test-key-for-unit-resolve",
        )
        self.assertEqual(resolve_preprocessing_mp3_bitrate_kbps(cfg), 56)

    def test_mp3_bitrates_to_probe_descending(self) -> None:
        self.assertEqual(mp3_bitrates_to_probe_for_cache(48), [48, 40, 32, 24])

    def test_next_lower_rung(self) -> None:
        self.assertEqual(next_lower_mp3_bitrate_kbps(48), 40)
        self.assertEqual(next_lower_mp3_bitrate_kbps(40), 32)
        self.assertIsNone(next_lower_mp3_bitrate_kbps(24))
