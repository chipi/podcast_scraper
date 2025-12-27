#!/usr/bin/env python3
"""Integration tests for component workflows.

These tests verify that components work together in realistic workflows:
- RSS parsing → Episode creation → Provider usage → File output
- Config → Factory → Provider → Actual method call → Real output
- Multiple components working together

These tests use:
- Real internal implementations (Config, factories, providers, RSS parser, metadata)
- Real filesystem I/O
- Mocked HTTP responses (to avoid external network)
- Mocked ML model loading (for speed, will be fixed in Stage 3)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import downloader, metadata, rss_parser

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: E402
    build_rss_xml_with_transcript,
    create_rss_response,
    create_test_config,
    create_transcript_response,
)


@pytest.mark.integration
class TestRSSParsingToEpisodeWorkflow(unittest.TestCase):
    """Test RSS parsing → Episode creation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(output_dir=self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_rss_parsing_to_episode_creation(self):
        """Test that RSS parsing creates episodes correctly."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Step 1: Fetch and parse RSS (real RSS parser)
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Verify feed was parsed correctly
            self.assertIsNotNone(feed)
            self.assertEqual(feed.title, "Test Feed")
            self.assertEqual(len(feed.items), 1)

            # Step 2: Create episodes from RSS items (real episode creation)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]

            # Verify episodes were created correctly
            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertEqual(episode.idx, 1)
            self.assertEqual(episode.title, "Episode 1")
            self.assertEqual(len(episode.transcript_urls), 1)
            self.assertEqual(episode.transcript_urls[0][0], transcript_url)

    def test_rss_parsing_with_multiple_episodes(self):
        """Test RSS parsing with multiple episodes."""
        rss_url = "https://example.com/feed.xml"
        rss_xml = """<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="https://example.com/ep1.txt" type="text/plain" />
    </item>
    <item>
      <title>Episode 2</title>
      <podcast:transcript url="https://example.com/ep2.txt" type="text/plain" />
    </item>
    <item>
      <title>Episode 3</title>
      <podcast:transcript url="https://example.com/ep3.txt" type="text/plain" />
    </item>
  </channel>
</rss>""".strip()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Parse RSS
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Create episodes
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]

            # Verify all episodes were created
            self.assertEqual(len(episodes), 3)
            self.assertEqual(episodes[0].title, "Episode 1")
            self.assertEqual(episodes[1].title, "Episode 2")
            self.assertEqual(episodes[2].title, "Episode 3")


@pytest.mark.integration
class TestConfigToProviderWorkflow(unittest.TestCase):
    """Test Config → Factory → Provider workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.cfg = create_test_config(
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,  # Disable to avoid loading models
            auto_speakers=False,  # Disable to avoid loading spaCy
        )

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_config_to_provider_creation(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_load_whisper,
    ):
        """Test that Config → Factory creates providers correctly."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Mock model loading
        mock_load_whisper.return_value = Mock()
        mock_get_ner.return_value = Mock()
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summary_model.return_value = Mock()

        # Step 1: Create providers from config (real factories)
        transcription_provider = create_transcription_provider(self.cfg)
        speaker_detector = create_speaker_detector(self.cfg)
        summarization_provider = create_summarization_provider(self.cfg)

        # Verify providers were created
        self.assertIsNotNone(transcription_provider)
        self.assertIsNotNone(speaker_detector)
        self.assertIsNotNone(summarization_provider)

        # Step 2: Verify providers have required methods (real protocol compliance)
        self.assertTrue(hasattr(transcription_provider, "transcribe"))
        self.assertTrue(hasattr(speaker_detector, "detect_speakers"))
        self.assertTrue(hasattr(summarization_provider, "summarize"))

        # Step 3: Verify providers can be initialized (real initialization)
        if hasattr(transcription_provider, "initialize"):
            transcription_provider.initialize()  # type: ignore[attr-defined]
        if hasattr(speaker_detector, "initialize"):
            speaker_detector.initialize()  # type: ignore[attr-defined]
        if hasattr(summarization_provider, "initialize"):
            summarization_provider.initialize()  # type: ignore[attr-defined]

        # All should be initialized successfully
        # (Some providers may not have is_initialized property, which is fine)


@pytest.mark.integration
class TestRSSToMetadataWorkflow(unittest.TestCase):
    """Test RSS → Parse → Episode → Metadata generation workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_rss_to_metadata_generation(self):
        """Test complete workflow: RSS → Parse → Episode → Metadata → File output."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Step 1: Fetch and parse RSS (real RSS parser)
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Step 2: Create episodes from RSS items (real episode creation)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]

            # Step 3: Generate metadata (real metadata generation)
            episode = episodes[0]
            metadata_path = metadata.generate_episode_metadata(
                feed=feed,
                episode=episode,
                feed_url=rss_url,
                cfg=self.cfg,
                output_dir=self.temp_dir,
                run_suffix=None,
                transcript_file_path="0001 - Episode_1.vtt",
                transcript_source="direct_download",
                whisper_model=None,
                detected_hosts=feed.authors,
                detected_guests=[],
            )

            # Step 4: Verify metadata file was created (real filesystem I/O)
            self.assertIsNotNone(metadata_path)
            self.assertTrue(os.path.exists(metadata_path))
            self.assertTrue(metadata_path.endswith(".metadata.json"))

            # Step 5: Verify metadata content (real file reading)
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Verify metadata structure
            self.assertIn("feed", data)
            self.assertIn("episode", data)
            self.assertIn("content", data)
            self.assertIn("processing", data)

            # Verify feed metadata
            self.assertEqual(data["feed"]["title"], "Test Feed")
            self.assertEqual(data["feed"]["url"], rss_url)

            # Verify episode metadata
            self.assertEqual(data["episode"]["title"], "Episode 1")

            # Verify content metadata
            self.assertEqual(data["content"]["transcript_source"], "direct_download")
            self.assertEqual(data["content"]["detected_hosts"], feed.authors)


@pytest.mark.integration
class TestMultipleComponentsWorkflow(unittest.TestCase):
    """Test multiple components working together."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
            transcription_provider="whisper",
            speaker_detector_provider="ner",
            summary_provider="local",
            generate_summaries=False,
            auto_speakers=False,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function."""

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    @patch("podcast_scraper.transcription.whisper_provider.whisper_integration.load_whisper_model")
    @patch("podcast_scraper.speaker_detectors.ner_detector.speaker_detection.get_ner_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_reduce_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.select_summary_model")
    @patch("podcast_scraper.summarization.local_provider.summarizer.SummaryModel")
    def test_full_component_workflow(
        self,
        mock_summary_model,
        mock_select_map,
        mock_select_reduce,
        mock_get_ner,
        mock_load_whisper,
    ):
        """Test full workflow: RSS → Parse → Episode → Providers → Metadata → File output."""
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider

        # Mock model loading
        mock_load_whisper.return_value = Mock()
        mock_get_ner.return_value = Mock()
        mock_select_map.return_value = "facebook/bart-base"
        mock_select_reduce.return_value = "facebook/bart-base"
        mock_summary_model.return_value = Mock()

        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Step 1: Fetch and parse RSS (real RSS parser)
            feed = rss_parser.fetch_and_parse_rss(self.cfg)

            # Step 2: Create episodes (real episode creation)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]

            # Step 3: Create providers (real factories)
            transcription_provider = create_transcription_provider(self.cfg)
            speaker_detector = create_speaker_detector(self.cfg)
            summarization_provider = create_summarization_provider(self.cfg)

            # Step 4: Initialize providers (real initialization)
            if hasattr(transcription_provider, "initialize"):
                transcription_provider.initialize()  # type: ignore[attr-defined]
            if hasattr(speaker_detector, "initialize"):
                speaker_detector.initialize()  # type: ignore[attr-defined]
            if hasattr(summarization_provider, "initialize"):
                summarization_provider.initialize()  # type: ignore[attr-defined]

            # Step 5: Use providers (real method calls, but mocked models)
            episode = episodes[0]
            # Note: We're not actually calling transcribe/detect_speakers/summarize
            # because that would require real models. This test verifies the workflow
            # up to the point where providers are ready to use.

            # Step 6: Generate metadata (real metadata generation)
            metadata_path = metadata.generate_episode_metadata(
                feed=feed,
                episode=episode,
                feed_url=rss_url,
                cfg=self.cfg,
                output_dir=self.temp_dir,
                run_suffix=None,
                transcript_file_path="0001 - Episode_1.vtt",
                transcript_source="direct_download",
                whisper_model=None,
                detected_hosts=feed.authors,
                detected_guests=[],
            )

            # Step 7: Verify output (real filesystem I/O)
            self.assertIsNotNone(metadata_path)
            self.assertTrue(os.path.exists(metadata_path))

            # Verify all components worked together
            self.assertIsNotNone(feed)
            self.assertEqual(len(episodes), 1)
            self.assertIsNotNone(transcription_provider)
            self.assertIsNotNone(speaker_detector)
            self.assertIsNotNone(summarization_provider)
            self.assertIsNotNone(metadata_path)


if __name__ == "__main__":
    unittest.main()
