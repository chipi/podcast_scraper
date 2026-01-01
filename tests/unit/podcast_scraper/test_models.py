#!/usr/bin/env python3
"""Unit tests for models.py dataclasses.

Tests cover:
- RssFeed: Feed metadata and episode items
- Episode: Episode metadata and content URLs
- TranscriptionJob: Whisper transcription job data
"""

import os
import sys
import unittest
import xml.etree.ElementTree as ET

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper import models


class TestRssFeed(unittest.TestCase):
    """Tests for RssFeed dataclass."""

    def test_rss_feed_basic_instantiation(self):
        """Test basic RssFeed instantiation with required fields."""
        item1 = ET.Element("item")
        item2 = ET.Element("item")
        feed = models.RssFeed(
            title="Test Podcast",
            items=[item1, item2],
            base_url="https://example.com/feed.xml",
        )

        self.assertEqual(feed.title, "Test Podcast")
        self.assertEqual(len(feed.items), 2)
        self.assertEqual(feed.items[0], item1)
        self.assertEqual(feed.items[1], item2)
        self.assertEqual(feed.base_url, "https://example.com/feed.xml")
        self.assertEqual(feed.authors, [])

    def test_rss_feed_with_authors(self):
        """Test RssFeed instantiation with authors."""
        item = ET.Element("item")
        feed = models.RssFeed(
            title="Test Podcast",
            items=[item],
            base_url="https://example.com/feed.xml",
            authors=["John Doe", "Jane Smith"],
        )

        self.assertEqual(feed.title, "Test Podcast")
        self.assertEqual(len(feed.authors), 2)
        self.assertIn("John Doe", feed.authors)
        self.assertIn("Jane Smith", feed.authors)

    def test_rss_feed_empty_items(self):
        """Test RssFeed with empty items list."""
        feed = models.RssFeed(
            title="Empty Feed",
            items=[],
            base_url="https://example.com/feed.xml",
        )

        self.assertEqual(feed.title, "Empty Feed")
        self.assertEqual(len(feed.items), 0)
        self.assertEqual(feed.authors, [])

    def test_rss_feed_authors_default_factory(self):
        """Test that authors defaults to empty list via default_factory."""
        item = ET.Element("item")
        feed1 = models.RssFeed(
            title="Feed 1",
            items=[item],
            base_url="https://example.com/feed1.xml",
        )
        feed2 = models.RssFeed(
            title="Feed 2",
            items=[item],
            base_url="https://example.com/feed2.xml",
        )

        # Each feed should have its own empty list, not shared reference
        self.assertEqual(feed1.authors, [])
        self.assertEqual(feed2.authors, [])
        feed1.authors.append("Author 1")
        self.assertEqual(len(feed1.authors), 1)
        self.assertEqual(len(feed2.authors), 0)  # Should not affect feed2

    def test_rss_feed_string_representation(self):
        """Test RssFeed string representation."""
        item = ET.Element("item")
        feed = models.RssFeed(
            title="Test Podcast",
            items=[item],
            base_url="https://example.com/feed.xml",
            authors=["John Doe"],
        )

        # Dataclass should have __repr__ that includes all fields
        repr_str = repr(feed)
        self.assertIn("Test Podcast", repr_str)
        self.assertIn("https://example.com/feed.xml", repr_str)


class TestEpisode(unittest.TestCase):
    """Tests for Episode dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.item = ET.Element("item")
        title_el = ET.SubElement(self.item, "title")
        title_el.text = "Test Episode"

    def test_episode_basic_instantiation(self):
        """Test basic Episode instantiation with required fields."""
        episode = models.Episode(
            idx=0,
            title="Episode 1: Introduction",
            title_safe="episode-1-introduction",
            item=self.item,
            transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
        )

        self.assertEqual(episode.idx, 0)
        self.assertEqual(episode.title, "Episode 1: Introduction")
        self.assertEqual(episode.title_safe, "episode-1-introduction")
        self.assertEqual(episode.item, self.item)
        self.assertEqual(len(episode.transcript_urls), 1)
        self.assertEqual(episode.transcript_urls[0][0], "https://example.com/transcript.vtt")
        self.assertEqual(episode.transcript_urls[0][1], "text/vtt")
        self.assertIsNone(episode.media_url)
        self.assertIsNone(episode.media_type)

    def test_episode_with_media(self):
        """Test Episode instantiation with media URL and type."""
        episode = models.Episode(
            idx=1,
            title="Episode 2",
            title_safe="episode-2",
            item=self.item,
            transcript_urls=[],
            media_url="https://example.com/episode.mp3",
            media_type="audio/mpeg",
        )

        self.assertEqual(episode.media_url, "https://example.com/episode.mp3")
        self.assertEqual(episode.media_type, "audio/mpeg")

    def test_episode_multiple_transcript_urls(self):
        """Test Episode with multiple transcript URLs."""
        episode = models.Episode(
            idx=2,
            title="Episode 3",
            title_safe="episode-3",
            item=self.item,
            transcript_urls=[
                ("https://example.com/transcript.vtt", "text/vtt"),
                ("https://example.com/transcript.srt", "text/srt"),
                ("https://example.com/transcript.txt", None),
            ],
        )

        self.assertEqual(len(episode.transcript_urls), 3)
        self.assertEqual(episode.transcript_urls[0][1], "text/vtt")
        self.assertEqual(episode.transcript_urls[1][1], "text/srt")
        self.assertIsNone(episode.transcript_urls[2][1])

    def test_episode_empty_transcript_urls(self):
        """Test Episode with empty transcript URLs list."""
        episode = models.Episode(
            idx=3,
            title="Episode 4",
            title_safe="episode-4",
            item=self.item,
            transcript_urls=[],
        )

        self.assertEqual(len(episode.transcript_urls), 0)

    def test_episode_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        episode = models.Episode(
            idx=4,
            title="Episode 5",
            title_safe="episode-5",
            item=self.item,
            transcript_urls=[],
        )

        self.assertIsNone(episode.media_url)
        self.assertIsNone(episode.media_type)

    def test_episode_negative_index(self):
        """Test Episode with negative index (edge case)."""
        episode = models.Episode(
            idx=-1,
            title="Episode -1",
            title_safe="episode--1",
            item=self.item,
            transcript_urls=[],
        )

        self.assertEqual(episode.idx, -1)

    def test_episode_string_representation(self):
        """Test Episode string representation."""
        episode = models.Episode(
            idx=0,
            title="Test Episode",
            title_safe="test-episode",
            item=self.item,
            transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
            media_url="https://example.com/episode.mp3",
            media_type="audio/mpeg",
        )

        # Dataclass should have __repr__ that includes all fields
        repr_str = repr(episode)
        self.assertIn("Test Episode", repr_str)
        self.assertIn("test-episode", repr_str)


class TestTranscriptionJob(unittest.TestCase):
    """Tests for TranscriptionJob dataclass."""

    def test_transcription_job_basic_instantiation(self):
        """Test basic TranscriptionJob instantiation with required fields."""
        job = models.TranscriptionJob(
            idx=0,
            ep_title="Episode 1: Introduction",
            ep_title_safe="episode-1-introduction",
            temp_media="/tmp/episode-1.mp3",
        )

        self.assertEqual(job.idx, 0)
        self.assertEqual(job.ep_title, "Episode 1: Introduction")
        self.assertEqual(job.ep_title_safe, "episode-1-introduction")
        self.assertEqual(job.temp_media, "/tmp/episode-1.mp3")
        self.assertIsNone(job.detected_speaker_names)

    def test_transcription_job_with_speaker_names(self):
        """Test TranscriptionJob with detected speaker names."""
        job = models.TranscriptionJob(
            idx=1,
            ep_title="Episode 2",
            ep_title_safe="episode-2",
            temp_media="/tmp/episode-2.mp3",
            detected_speaker_names=["Alice", "Bob", "Charlie"],
        )

        self.assertEqual(len(job.detected_speaker_names), 3)
        self.assertIn("Alice", job.detected_speaker_names)
        self.assertIn("Bob", job.detected_speaker_names)
        self.assertIn("Charlie", job.detected_speaker_names)

    def test_transcription_job_empty_speaker_names(self):
        """Test TranscriptionJob with empty speaker names list."""
        job = models.TranscriptionJob(
            idx=2,
            ep_title="Episode 3",
            ep_title_safe="episode-3",
            temp_media="/tmp/episode-3.mp3",
            detected_speaker_names=[],
        )

        self.assertEqual(job.detected_speaker_names, [])

    def test_transcription_job_speaker_names_default_none(self):
        """Test that detected_speaker_names defaults to None."""
        job = models.TranscriptionJob(
            idx=3,
            ep_title="Episode 4",
            ep_title_safe="episode-4",
            temp_media="/tmp/episode-4.mp3",
        )

        self.assertIsNone(job.detected_speaker_names)

    def test_transcription_job_negative_index(self):
        """Test TranscriptionJob with negative index (edge case)."""
        job = models.TranscriptionJob(
            idx=-1,
            ep_title="Episode -1",
            ep_title_safe="episode--1",
            temp_media="/tmp/episode--1.mp3",
        )

        self.assertEqual(job.idx, -1)

    def test_transcription_job_string_representation(self):
        """Test TranscriptionJob string representation."""
        job = models.TranscriptionJob(
            idx=0,
            ep_title="Test Episode",
            ep_title_safe="test-episode",
            temp_media="/tmp/test-episode.mp3",
            detected_speaker_names=["Alice", "Bob"],
        )

        # Dataclass should have __repr__ that includes all fields
        repr_str = repr(job)
        self.assertIn("Test Episode", repr_str)
        self.assertIn("test-episode", repr_str)
        self.assertIn("/tmp/test-episode.mp3", repr_str)


class TestModelsEquality(unittest.TestCase):
    """Tests for dataclass equality and comparison."""

    def test_rss_feed_equality(self):
        """Test RssFeed equality comparison."""
        item1 = ET.Element("item")
        item2 = ET.Element("item")
        feed1 = models.RssFeed(
            title="Test Podcast",
            items=[item1, item2],
            base_url="https://example.com/feed.xml",
            authors=["John Doe"],
        )
        feed2 = models.RssFeed(
            title="Test Podcast",
            items=[item1, item2],
            base_url="https://example.com/feed.xml",
            authors=["John Doe"],
        )

        # Dataclasses compare by value
        self.assertEqual(feed1, feed2)

    def test_episode_equality(self):
        """Test Episode equality comparison."""
        item = ET.Element("item")
        episode1 = models.Episode(
            idx=0,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
        )
        episode2 = models.Episode(
            idx=0,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[("https://example.com/transcript.vtt", "text/vtt")],
        )

        self.assertEqual(episode1, episode2)

    def test_transcription_job_equality(self):
        """Test TranscriptionJob equality comparison."""
        job1 = models.TranscriptionJob(
            idx=0,
            ep_title="Episode 1",
            ep_title_safe="episode-1",
            temp_media="/tmp/episode-1.mp3",
            detected_speaker_names=["Alice"],
        )
        job2 = models.TranscriptionJob(
            idx=0,
            ep_title="Episode 1",
            ep_title_safe="episode-1",
            temp_media="/tmp/episode-1.mp3",
            detected_speaker_names=["Alice"],
        )

        self.assertEqual(job1, job2)


if __name__ == "__main__":
    unittest.main()
