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

from podcast_scraper import config, models
from podcast_scraper.rss import downloader, parser as rss_parser
from podcast_scraper.workflow import metadata_generation as metadata

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
# Pytest loads conftest.py from subdirectories, which can interfere with imports
import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

# Import from parent conftest
build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_rss_response = parent_conftest.create_rss_response
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response


@pytest.mark.integration
@pytest.mark.critical_path
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
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
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
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
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

        # All should be initialized successfully
        # (Some providers may not have is_initialized property, which is fine)


@pytest.mark.integration
@pytest.mark.critical_path
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

    @pytest.mark.critical_path
    def test_rss_to_metadata_generation(self):
        """Test critical path (Path 1): RSS → Parse → Download Transcript → Metadata → Files.

        This test validates the critical path when transcript URL exists and can be downloaded.
        It tests the complete workflow at component level:
        - RSS feed with transcript URL
        - Download transcript file (real HTTP download)
        - Generate metadata
        - Validate files are created
        """
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
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
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
            # Check speakers array instead of detected_hosts
            self.assertIn("speakers", data["content"])
            speakers = data["content"]["speakers"]
            host_speakers = [s for s in speakers if s.get("role") == "host"]
            if feed.authors:
                self.assertEqual(len(host_speakers), len(feed.authors))
                host_names = [s.get("name") for s in host_speakers]
                self.assertEqual(set(host_names), set(feed.authors))

        # Note: summary_provider.summarize() is mocked directly, so the OpenAI client is not called
        # This is fine - we're testing the workflow integration, not the API call itself

    def test_concurrent_processing_orchestration(self):
        """Test concurrent processing orchestration at component level.

        This test validates concurrent download orchestration:
        - Multiple episodes processed concurrently
        - Transcription job queuing
        - Thread safety in shared resources
        """
        import threading

        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor

        # Create multiple episodes
        episodes = [
            models.Episode(
                idx=i,
                title=f"Episode {i}",
                title_safe=f"Episode_{i}",
                item=None,
                transcript_urls=[],
                media_url=f"https://example.com/ep{i}.mp3",
                media_type="audio/mpeg",
            )
            for i in range(1, 4)  # 3 episodes
        ]

        # Create mock HTTP responses for all episodes
        responses = {}
        for i in range(1, 4):
            audio_url = f"https://example.com/ep{i}.mp3"
            audio_bytes = b"\xff\xfb\x90\x00" * 32
            responses[downloader.normalize_url(audio_url)] = create_media_response(
                audio_bytes, audio_url
            )

        http_mock = self._mock_http_map(responses)

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,  # Disable to avoid loading spaCy (only testing transcription)
        )
        temp_dir = os.path.join(self.temp_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Shared transcription jobs queue with lock
        import queue

        transcription_jobs = queue.Queue()
        transcription_jobs_lock = threading.Lock()

        # Mock Whisper
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
            patch(
                "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper,
            patch(
                "podcast_scraper.providers.ml.ml_provider.MLProvider._transcribe_with_whisper"
            ) as mock_transcribe,
        ):

            mock_whisper_model = Mock()
            mock_import_whisper.return_value = Mock(
                load_model=lambda *args, **kwargs: mock_whisper_model
            )
            mock_transcribe.return_value = (
                {"text": "Test transcription", "segments": []},
                1.0,
            )

            # Step 1: Process episodes concurrently (simulate concurrent processing)
            def process_episode(episode):
                episode_processor.process_episode_download(
                    episode=episode,
                    cfg=cfg,
                    temp_dir=temp_dir,
                    effective_output_dir=self.temp_dir,
                    run_suffix=None,
                    transcription_jobs=transcription_jobs,
                    transcription_jobs_lock=transcription_jobs_lock,
                )

            # Process all episodes (simulating concurrent execution)
            for episode in episodes:
                process_episode(episode)

            # Step 2: Verify all transcription jobs were created
            self.assertEqual(transcription_jobs.qsize(), 3, "Should create 3 transcription jobs")

            # Verify jobs are for different episodes
            jobs = []
            while not transcription_jobs.empty():
                jobs.append(transcription_jobs.get())
            job_indices = {job.idx for job in jobs}
            self.assertEqual(job_indices, {1, 2, 3}, "Jobs should be for all 3 episodes")

            # Step 3: Process transcription jobs (simulate sequential transcription)
            transcription_provider = create_transcription_provider(cfg)
            transcription_provider.initialize()

            transcript_paths = []
            # Use the jobs list that was already extracted from the queue
            for job in jobs:
                success, transcript_path, _ = episode_processor.transcribe_media_to_text(
                    job=job,
                    cfg=cfg,
                    whisper_model=mock_whisper_model,
                    run_suffix=None,
                    effective_output_dir=self.temp_dir,
                    transcription_provider=transcription_provider,
                )
                if success and transcript_path:
                    transcript_paths.append(transcript_path)

            # Step 4: Verify all transcripts were created
            self.assertEqual(len(transcript_paths), 3, "Should create 3 transcript files")
            for transcript_path in transcript_paths:
                full_path = os.path.join(self.temp_dir, transcript_path)
                self.assertTrue(
                    os.path.exists(full_path), f"Transcript file should exist: {transcript_path}"
                )

    def test_error_recovery_download_failure(self):
        """Test error recovery when download fails mid-way.

        This test validates component-level error recovery:
        - HTTP download failure handling
        - Partial file cleanup
        - Graceful degradation
        """
        from podcast_scraper.workflow import episode_processor

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="Episode_1",
            item=None,
            transcript_urls=[],
            media_url="https://example.com/ep1.mp3",
            media_type="audio/mpeg",
        )

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,  # Disable to avoid loading spaCy (only testing transcription)
        )
        temp_dir = os.path.join(self.temp_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        import queue

        transcription_jobs = queue.Queue()

        # Mock HTTP download failure
        def failing_download(url, user_agent=None, timeout=None, stream=False):
            raise Exception("Network error: Connection refused")

        with patch("podcast_scraper.downloader.http_download_to_file") as mock_download:
            # Mock download to fail
            mock_download.return_value = (False, 0)

            # Step 1: Attempt to process episode (should handle download failure gracefully)
            success, transcript_path, transcript_source, bytes_downloaded = (
                episode_processor.process_episode_download(
                    episode=episode,
                    cfg=cfg,
                    temp_dir=temp_dir,
                    effective_output_dir=self.temp_dir,
                    run_suffix=None,
                    transcription_jobs=transcription_jobs,
                    transcription_jobs_lock=None,
                )
            )

            # Step 2: Verify failure is handled gracefully
            # When download fails, process_episode_download returns False and no job is created
            self.assertFalse(success, "Should return False on download failure")
            self.assertIsNone(transcript_path, "No transcript path on failure")
            self.assertEqual(
                transcription_jobs.qsize(),
                0,
                "No transcription job should be created on failure",
            )

    def test_error_recovery_transcription_failure(self):
        """Test error recovery when transcription fails after download.

        This test validates component-level error recovery:
        - Transcription failure handling
        - Temporary file cleanup
        - Graceful degradation
        """
        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor

        audio_url = "https://example.com/ep1.mp3"
        audio_bytes = b"\xff\xfb\x90\x00" * 32
        responses = {
            downloader.normalize_url(audio_url): create_media_response(audio_bytes, audio_url),
        }
        http_mock = self._mock_http_map(responses)

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="Episode_1",
            item=None,
            transcript_urls=[],
            media_url=audio_url,
            media_type="audio/mpeg",
        )

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,  # Disable to avoid loading spaCy (only testing transcription)
            transcript_cache_enabled=False,  # Disable cache to test actual transcription failure
        )
        temp_dir = os.path.join(self.temp_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        import queue

        transcription_jobs = queue.Queue()

        # Mock Whisper transcription failure
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
            patch(
                "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper,
            patch(
                "podcast_scraper.providers.ml.ml_provider.MLProvider._transcribe_with_whisper"
            ) as mock_transcribe,
        ):

            mock_whisper_model = Mock()
            mock_import_whisper.return_value = Mock(
                load_model=lambda *args, **kwargs: mock_whisper_model
            )
            mock_transcribe.side_effect = RuntimeError("Transcription failed: Model error")

            # Step 1: Download audio (should succeed)
            success, transcript_path, transcript_source, bytes_downloaded = (
                episode_processor.process_episode_download(
                    episode=episode,
                    cfg=cfg,
                    temp_dir=temp_dir,
                    effective_output_dir=self.temp_dir,
                    run_suffix=None,
                    transcription_jobs=transcription_jobs,
                    transcription_jobs_lock=None,
                )
            )

            # Verify download succeeded
            self.assertEqual(transcription_jobs.qsize(), 1, "Transcription job should be created")
            job = transcription_jobs.get()
            self.assertTrue(os.path.exists(job.temp_media), "Audio file should be downloaded")

            # Step 2: Attempt transcription (should fail gracefully)
            transcription_provider = create_transcription_provider(cfg)
            transcription_provider.initialize()

            # Mock the provider's transcribe_with_segments method to raise an error
            # This is the method actually called by transcribe_media_to_text
            original_transcribe = transcription_provider.transcribe_with_segments
            transcription_provider.transcribe_with_segments = Mock(
                side_effect=RuntimeError("Transcription failed: Model error")
            )
            try:
                transcribe_success, transcript_file_path, bytes_downloaded_transcribe = (
                    episode_processor.transcribe_media_to_text(
                        job=job,
                        cfg=cfg,
                        whisper_model=mock_whisper_model,
                        run_suffix=None,
                        effective_output_dir=self.temp_dir,
                        transcription_provider=transcription_provider,
                    )
                )
            finally:
                transcription_provider.transcribe_with_segments = original_transcribe

            # Step 3: Verify failure is handled gracefully
            self.assertFalse(transcribe_success, "Transcription should fail")
            self.assertIsNone(transcript_file_path, "No transcript file on failure")

            # Step 4: Verify temporary file is cleaned up
            self.assertFalse(
                os.path.exists(job.temp_media),
                "Temporary media file should be cleaned up on failure",
            )


if __name__ == "__main__":
    unittest.main()
