#!/usr/bin/env python3
"""E2E tests for component workflows with real ML models.

Extracted from tests/integration/workflow/test_component_workflows.py —
these tests use real Whisper, spaCy, and/or Transformers models, so they
belong in E2E per the 3-tier ML/AI testing policy.

Tests:
- RSS → Parse → Download audio → Transcribe (real Whisper) → Metadata
- Episode processor audio download and transcription (real Whisper)
- Speaker detection in transcription workflow (real spaCy + Whisper)
- Full workflow with NER and summarization (real spaCy + Whisper + Transformers)
- Summarization workflow (real Transformers)
- Full component workflow: RSS → NER → Summarization → Metadata (real spaCy + Transformers)
"""

import json
import os
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config, models
from podcast_scraper.rss import downloader, parser as rss_parser
from podcast_scraper.workflow import metadata_generation as metadata

# Import from parent conftest
tests_dir = Path(__file__).parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_rss_response = parent_conftest.create_rss_response
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response

# Import ML model cache helpers from integration directory
integration_dir = tests_dir / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
from ml_model_cache_helpers import (  # noqa: E402
    require_spacy_model_cached,
    require_transformers_model_cached,
    require_whisper_model_cached,
)


class _ComponentWorkflowBase(unittest.TestCase):
    """Shared setUp/tearDown and helpers for component workflow E2E tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.critical_path
class TestRSSToMetadataWorkflowML(_ComponentWorkflowBase):
    """E2E tests for RSS-to-metadata workflows requiring real ML models.

    Extracted from TestRSSToMetadataWorkflow in integration tests.
    """

    def test_rss_to_transcription_workflow(self):
        """RSS -> Parse -> Download audio -> Transcribe (real Whisper) -> Metadata -> Files."""
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor

        rss_url = "https://example.com/feed.xml"
        audio_url = "https://example.com/ep1.mp3"
        rss_xml = f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Episode 1</title>
      <enclosure url="{audio_url}" type="audio/mpeg" />
    </item>
  </channel>
</rss>""".strip()

        fixture_audio_path = (
            Path(__file__).parent.parent / "fixtures" / "audio" / "p01_e01_fast.mp3"
        )
        with open(fixture_audio_path, "rb") as f:
            audio_bytes = f.read()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(audio_url): create_media_response(audio_bytes, audio_url),
        }

        http_mock = self._mock_http_map(responses)

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            feed = rss_parser.fetch_and_parse_rss(self.cfg)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]

            episode = episodes[0]
            self.assertEqual(len(episode.transcript_urls), 0)
            self.assertIsNotNone(episode.media_url)
            self.assertEqual(episode.media_url, audio_url)

            cfg = create_test_config(
                output_dir=self.temp_dir,
                transcribe_missing=True,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                generate_metadata=True,
                auto_speakers=False,
            )
            temp_dir = os.path.join(self.temp_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            transcription_jobs = queue.Queue()

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

            self.assertEqual(transcription_jobs.qsize(), 1)
            job = transcription_jobs.get()
            self.assertIsNotNone(job.temp_media)
            self.assertTrue(os.path.exists(job.temp_media))

            transcription_provider = create_transcription_provider(cfg)
            transcription_provider.initialize()

            try:
                transcribe_success, transcript_file_path, bytes_downloaded_transcribe = (
                    episode_processor.transcribe_media_to_text(
                        job=job,
                        cfg=cfg,
                        whisper_model=None,
                        run_suffix=None,
                        effective_output_dir=self.temp_dir,
                        transcription_provider=transcription_provider,
                    )
                )

                self.assertTrue(transcribe_success)
                self.assertIsNotNone(transcript_file_path)

                transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
                self.assertTrue(os.path.exists(transcript_full_path))

                metadata_path = metadata.generate_episode_metadata(
                    feed=feed,
                    episode=episode,
                    feed_url=rss_url,
                    cfg=cfg,
                    output_dir=self.temp_dir,
                    run_suffix=None,
                    transcript_file_path=transcript_file_path,
                    transcript_source="whisper_transcription",
                    whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                    detected_hosts=feed.authors if feed.authors else [],
                    detected_guests=[],
                )

                self.assertIsNotNone(metadata_path)
                self.assertTrue(os.path.exists(metadata_path))
                self.assertTrue(metadata_path.endswith(".metadata.json"))

                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertEqual(data["content"]["transcript_source"], "whisper_transcription")
                self.assertEqual(
                    data["content"]["whisper_model"],
                    config.TEST_DEFAULT_WHISPER_MODEL,
                )
            finally:
                if hasattr(transcription_provider, "cleanup"):
                    transcription_provider.cleanup()

    def test_episode_processor_audio_download_and_transcription(self):
        """Episode processor: download audio -> transcribe with real Whisper."""
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor

        audio_url = "https://example.com/ep1.mp3"
        episode = models.Episode(
            idx=1,
            title="Episode 1: Test",
            title_safe="Episode_1_Test",
            item=None,
            transcript_urls=[],
            media_url=audio_url,
            media_type="audio/mpeg",
        )

        fixture_audio_path = (
            Path(__file__).parent.parent / "fixtures" / "audio" / "p01_e01_fast.mp3"
        )
        with open(fixture_audio_path, "rb") as f:
            audio_bytes = f.read()

        responses = {
            downloader.normalize_url(audio_url): create_media_response(audio_bytes, audio_url),
        }

        http_mock = self._mock_http_map(responses)

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=False,
        )
        temp_dir = os.path.join(self.temp_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        transcription_jobs = queue.Queue()

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
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

            self.assertEqual(transcription_jobs.qsize(), 1)
            job = transcription_jobs.get()
            self.assertEqual(job.idx, 1)
            self.assertEqual(job.ep_title, "Episode 1: Test")
            self.assertIsNotNone(job.temp_media)
            self.assertTrue(os.path.exists(job.temp_media))
            file_size = os.path.getsize(job.temp_media)
            self.assertGreater(file_size, 0)

            transcription_provider = create_transcription_provider(cfg)
            transcription_provider.initialize()

            try:
                transcribe_success, transcript_file_path, bytes_downloaded_transcribe = (
                    episode_processor.transcribe_media_to_text(
                        job=job,
                        cfg=cfg,
                        whisper_model=None,
                        run_suffix=None,
                        effective_output_dir=self.temp_dir,
                        transcription_provider=transcription_provider,
                    )
                )

                self.assertTrue(transcribe_success)
                self.assertIsNotNone(transcript_file_path)

                transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
                self.assertTrue(os.path.exists(transcript_full_path))
                self.assertTrue(transcript_file_path.endswith(".txt"))
                self.assertIn("Episode_1_Test", transcript_file_path)

                with open(transcript_full_path, "r", encoding="utf-8") as f:
                    transcript_content = f.read()
                self.assertGreater(len(transcript_content), 0)
            finally:
                if hasattr(transcription_provider, "cleanup"):
                    transcription_provider.cleanup()

            self.assertTrue(transcribe_success)

    def test_speaker_detection_in_transcription_workflow(self):
        """Speaker detection with real spaCy + Whisper transcription."""
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
        require_spacy_model_cached(config.DEFAULT_NER_MODEL)

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor, metadata_generation as metadata

        rss_url = "https://example.com/feed.xml"
        audio_url = "https://example.com/ep1.mp3"
        ep_desc_speaker = "In this episode, we talk with Bob Guest about technology."
        rss_xml = f"""<?xml version='1.0'?>
<rss
  xmlns:podcast="https://podcastindex.org/namespace/1.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <author>John Host</author>
    <itunes:author>Jane Host</itunes:author>
    <item>
      <title>Episode 1: Interview with Bob Guest</title>
      <description>{ep_desc_speaker}</description>
      <enclosure url="{audio_url}" type="audio/mpeg" />
    </item>
  </channel>
</rss>""".strip()

        fixture_audio_path = (
            Path(__file__).parent.parent / "fixtures" / "audio" / "p01_e01_fast.mp3"
        )
        with open(fixture_audio_path, "rb") as f:
            audio_bytes = f.read()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(audio_url): create_media_response(audio_bytes, audio_url),
        }

        http_mock = self._mock_http_map(responses)

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=True,
            screenplay=True,
            screenplay_num_speakers=3,
            generate_metadata=True,
        )

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            feed = rss_parser.fetch_and_parse_rss(cfg)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]
            episode = episodes[0]

            speaker_detector = create_speaker_detector(cfg)
            speaker_detector.initialize()

            try:
                detected_hosts = speaker_detector.detect_hosts(
                    feed_title=feed.title,
                    feed_description=None,
                    feed_authors=feed.authors,
                )

                self.assertGreater(len(detected_hosts), 0)
                self.assertTrue(
                    any("John" in name or "Jane" in name for name in detected_hosts),
                    f"At least one host should be detected. Got: {detected_hosts}",
                )

                episode_description = rss_parser.extract_episode_description(episode.item)
                detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                    speaker_detector.detect_speakers(
                        episode_title=episode.title,
                        episode_description=episode_description,
                        known_hosts=detected_hosts,
                    )
                )

                self.assertTrue(detection_succeeded)
                self.assertGreater(len(detected_speakers), 0)

                temp_dir = os.path.join(self.temp_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)

                transcription_jobs = queue.Queue()

                success, transcript_path, transcript_source, bytes_downloaded = (
                    episode_processor.process_episode_download(
                        episode=episode,
                        cfg=cfg,
                        temp_dir=temp_dir,
                        effective_output_dir=self.temp_dir,
                        run_suffix=None,
                        transcription_jobs=transcription_jobs,
                        transcription_jobs_lock=None,
                        detected_speaker_names=detected_speakers,
                    )
                )

                self.assertEqual(transcription_jobs.qsize(), 1)
                job = transcription_jobs.get()
                self.assertIsNotNone(job.detected_speaker_names)
                self.assertEqual(job.detected_speaker_names, detected_speakers)

                transcription_provider = create_transcription_provider(cfg)
                transcription_provider.initialize()

                try:
                    transcribe_success, transcript_file_path, _ = (
                        episode_processor.transcribe_media_to_text(
                            job=job,
                            cfg=cfg,
                            whisper_model=None,
                            run_suffix=None,
                            effective_output_dir=self.temp_dir,
                            transcription_provider=transcription_provider,
                        )
                    )

                    self.assertTrue(transcribe_success)
                    transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
                    self.assertTrue(os.path.exists(transcript_full_path))

                    with open(transcript_full_path, "r", encoding="utf-8") as f:
                        transcript_content = f.read()

                    if detected_speakers:
                        self.assertGreater(len(transcript_content), 0)

                    metadata_path = metadata.generate_episode_metadata(
                        feed=feed,
                        episode=episode,
                        feed_url=rss_url,
                        cfg=cfg,
                        output_dir=self.temp_dir,
                        run_suffix=None,
                        transcript_file_path=transcript_file_path,
                        transcript_source="whisper_transcription",
                        whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                        detected_hosts=list(detected_hosts),
                        detected_guests=[s for s in detected_speakers if s not in detected_hosts],
                    )

                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    self.assertIn("speakers", data["content"])
                    speakers = data["content"]["speakers"]
                    self.assertIsInstance(speakers, list)
                    host_speakers = [s for s in speakers if s.get("role") == "host"]
                    guest_speakers = [s for s in speakers if s.get("role") == "guest"]
                    self.assertEqual(len(host_speakers), len(detected_hosts))
                    self.assertIsInstance(guest_speakers, list)
                finally:
                    if hasattr(transcription_provider, "cleanup"):
                        transcription_provider.cleanup()
            finally:
                if hasattr(speaker_detector, "cleanup"):
                    speaker_detector.cleanup()

    def test_full_workflow_with_ner_and_summarization(self):
        """Full workflow: RSS -> Whisper -> spaCy NER -> Transformers summarization -> Metadata."""
        require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
        require_spacy_model_cached(config.DEFAULT_NER_MODEL)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import episode_processor, metadata_generation as metadata

        rss_url = "https://example.com/feed.xml"
        audio_url = "https://example.com/ep1.mp3"
        ep_desc_ner = (
            "In this episode, we talk with Bob Guest about technology " "and software development."
        )
        rss_xml = f"""<?xml version='1.0'?>
<rss
  xmlns:podcast="https://podcastindex.org/namespace/1.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <author>John Host</author>
    <itunes:author>Jane Host</itunes:author>
    <item>
      <title>Episode 1: Interview with Bob Guest</title>
      <description>{ep_desc_ner}</description>
      <enclosure url="{audio_url}" type="audio/mpeg" />
    </item>
  </channel>
</rss>""".strip()

        fixture_audio_path = (
            Path(__file__).parent.parent / "fixtures" / "audio" / "p01_e01_fast.mp3"
        )
        with open(fixture_audio_path, "rb") as f:
            audio_bytes = f.read()

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(audio_url): create_media_response(audio_bytes, audio_url),
        }

        http_mock = self._mock_http_map(responses)

        cfg = create_test_config(
            output_dir=self.temp_dir,
            transcribe_missing=True,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            auto_speakers=True,
            generate_summaries=True,
            generate_metadata=True,
            metadata_format="json",
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
            summary_device="cpu",
        )

        from podcast_scraper.providers.ml import summarizer
        from tests.integration.ml_model_cache_helpers import _is_transformers_model_cached

        map_model_name = summarizer.select_summary_model(cfg)
        reduce_model_name = summarizer.select_reduce_model(cfg, map_model_name)
        if reduce_model_name != map_model_name:
            if not _is_transformers_model_cached(reduce_model_name, None):
                pytest.skip(
                    f"REDUCE model '{reduce_model_name}' is not cached. "
                    f"Run 'make preload-ml-models' to pre-cache all required models."
                )

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            feed = rss_parser.fetch_and_parse_rss(cfg)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]
            episode = episodes[0]

            speaker_detector = create_speaker_detector(cfg)
            speaker_detector.initialize()

            try:
                detected_hosts = speaker_detector.detect_hosts(
                    feed_title=feed.title,
                    feed_description=None,
                    feed_authors=feed.authors,
                )

                self.assertGreater(len(detected_hosts), 0)
                self.assertTrue(
                    any("John" in name or "Jane" in name for name in detected_hosts),
                    f"At least one host should be detected. Got: {detected_hosts}",
                )

                episode_description = rss_parser.extract_episode_description(episode.item)
                detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                    speaker_detector.detect_speakers(
                        episode_title=episode.title,
                        episode_description=episode_description,
                        known_hosts=detected_hosts,
                    )
                )

                self.assertTrue(detection_succeeded)
                self.assertGreater(len(detected_speakers), 0)

                temp_dir = os.path.join(self.temp_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)

                transcription_jobs = queue.Queue()

                success, transcript_path, transcript_source, bytes_downloaded = (
                    episode_processor.process_episode_download(
                        episode=episode,
                        cfg=cfg,
                        temp_dir=temp_dir,
                        effective_output_dir=self.temp_dir,
                        run_suffix=None,
                        transcription_jobs=transcription_jobs,
                        transcription_jobs_lock=None,
                        detected_speaker_names=detected_speakers,
                    )
                )

                self.assertEqual(transcription_jobs.qsize(), 1)
                job = transcription_jobs.get()

                transcription_provider = create_transcription_provider(cfg)
                transcription_provider.initialize()

                try:
                    transcribe_success, transcript_file_path, _ = (
                        episode_processor.transcribe_media_to_text(
                            job=job,
                            cfg=cfg,
                            whisper_model=None,
                            run_suffix=None,
                            effective_output_dir=self.temp_dir,
                            transcription_provider=transcription_provider,
                        )
                    )

                    self.assertTrue(transcribe_success)
                    transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
                    self.assertTrue(os.path.exists(transcript_full_path))

                    summary_provider = create_summarization_provider(cfg)
                    summary_provider.initialize()

                    try:
                        metadata_path = metadata.generate_episode_metadata(
                            feed=feed,
                            episode=episode,
                            feed_url=rss_url,
                            cfg=cfg,
                            output_dir=self.temp_dir,
                            run_suffix=None,
                            transcript_file_path=transcript_file_path,
                            transcript_source="whisper_transcription",
                            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                            detected_hosts=list(detected_hosts),
                            detected_guests=[
                                s for s in detected_speakers if s not in detected_hosts
                            ],
                            summary_provider=summary_provider,
                        )

                        self.assertIsNotNone(metadata_path)
                        self.assertTrue(os.path.exists(metadata_path))

                        with open(metadata_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        self.assertIn("speakers", data["content"])
                        speakers = data["content"]["speakers"]
                        self.assertIsInstance(speakers, list)
                        host_speakers = [s for s in speakers if s.get("role") == "host"]
                        guest_speakers = [s for s in speakers if s.get("role") == "guest"]
                        self.assertEqual(len(host_speakers), len(detected_hosts))
                        self.assertIsInstance(guest_speakers, list)

                        self.assertIn("summary", data)
                        summary = data["summary"]
                        self.assertIsInstance(summary, dict)
                        self.assertIn("bullets", summary)
                        self.assertIsInstance(summary["bullets"], list)
                        self.assertGreater(len(summary["bullets"]), 0)
                        self.assertIn("short_summary", summary)
                        self.assertIsInstance(summary["short_summary"], str)
                        self.assertGreater(len(summary["short_summary"]), 0)

                        self.assertEqual(
                            data["content"]["transcript_source"],
                            "whisper_transcription",
                        )
                    finally:
                        if hasattr(summary_provider, "cleanup"):
                            summary_provider.cleanup()
                finally:
                    if hasattr(transcription_provider, "cleanup"):
                        transcription_provider.cleanup()
            finally:
                if hasattr(speaker_detector, "cleanup"):
                    speaker_detector.cleanup()

    def test_summarization_workflow(self):
        """Summarization workflow with real Transformers model."""
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.workflow import metadata_generation as metadata
        from tests.integration.ml_model_cache_helpers import (
            _is_transformers_model_cached,
            require_transformers_model_cached,
        )

        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        transcript_text = """This is a test transcript for summarization.
        It contains multiple sentences and paragraphs.
        The content should be long enough to require summarization.
        We need at least 50 characters for the summarization to work.
        This transcript discusses various topics related to technology
        and software development.
        It covers best practices, design patterns, and implementation
        details.
        The goal is to create a comprehensive summary of this content.
        We want to ensure the summarization model can process this text
        and generate a meaningful summary.
        The transcript should be substantial enough to test the full
        summarization workflow."""

        transcript_file_path = "0001 - Test_Episode.txt"
        transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
        os.makedirs(os.path.dirname(transcript_full_path) or ".", exist_ok=True)
        with open(transcript_full_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_summaries=True,
            generate_metadata=True,
            metadata_format="json",
            summary_provider="transformers",
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
            summary_device="cpu",
        )

        from podcast_scraper.providers.ml import summarizer

        map_model_name = summarizer.select_summary_model(cfg)
        reduce_model_name = summarizer.select_reduce_model(cfg, map_model_name)
        if reduce_model_name != map_model_name:
            if not _is_transformers_model_cached(reduce_model_name, None):
                pytest.skip(
                    f"REDUCE model '{reduce_model_name}' is not cached. "
                    f"Run 'make preload-ml-models' to pre-cache all required models."
                )

        summary_provider = create_summarization_provider(cfg)
        summary_provider.initialize()

        feed = create_test_feed()
        episode = create_test_episode()

        metadata_path = metadata.generate_episode_metadata(
            feed=feed,
            episode=episode,
            feed_url="https://example.com/feed.xml",
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path=transcript_file_path,
            transcript_source="direct_download",
            summary_provider=summary_provider,
        )

        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("summary", data)
        self.assertIsNotNone(data["summary"])
        self.assertIsInstance(data["summary"], dict)
        self.assertIn("bullets", data["summary"])
        self.assertIsInstance(data["summary"]["bullets"], list)
        self.assertGreater(len(data["summary"]["bullets"]), 0)
        self.assertIn("schema_status", data["summary"])
        self.assertIn("short_summary", data["summary"])
        self.assertIsInstance(data["summary"]["short_summary"], str)
        self.assertGreater(len(data["summary"]["short_summary"]), 0)

        if hasattr(summary_provider, "cleanup"):
            summary_provider.cleanup()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.ml_models
class TestMultipleComponentsWorkflowML(_ComponentWorkflowBase):
    """E2E test for full component workflow with real spaCy + Transformers.

    Extracted from TestMultipleComponentsWorkflow in integration tests.
    """

    def setUp(self):
        super().setUp()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            generate_summaries=False,
            auto_speakers=False,
        )

    def test_full_component_workflow(self):
        """Full workflow: RSS -> NER (real spaCy) -> Summarization
        (real Transformers) -> Metadata."""
        require_spacy_model_cached(config.DEFAULT_NER_MODEL)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

        from podcast_scraper.speaker_detectors.factory import create_speaker_detector
        from podcast_scraper.summarization.factory import create_summarization_provider
        from podcast_scraper.transcription.factory import create_transcription_provider
        from podcast_scraper.workflow import metadata_generation as metadata

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
            transcription_provider="whisper",
            speaker_detector_provider="spacy",
            summary_provider="transformers",
            generate_summaries=True,
            auto_speakers=True,
            summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
            summary_device="cpu",
        )

        from podcast_scraper.providers.ml import summarizer
        from tests.integration.ml_model_cache_helpers import _is_transformers_model_cached

        map_model_name = summarizer.select_summary_model(cfg)
        reduce_model_name = summarizer.select_reduce_model(cfg, map_model_name)
        if reduce_model_name != map_model_name:
            if not _is_transformers_model_cached(reduce_model_name, None):
                pytest.skip(
                    f"REDUCE model '{reduce_model_name}' is not cached. "
                    f"Run 'make preload-ml-models' to pre-cache all required models."
                )

        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        ep_desc_component = (
            "In this episode, we talk with Bob Guest about technology " "and software development."
        )
        rss_xml = f"""<?xml version='1.0'?>
<rss
  xmlns:podcast="https://podcastindex.org/namespace/1.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Test Feed</title>
    <author>John Host</author>
    <itunes:author>Jane Host</itunes:author>
    <item>
      <title>Episode 1: Interview with Bob Guest</title>
      <description>{ep_desc_component}</description>
      <podcast:transcript url="{transcript_url}" type="text/vtt" />
    </item>
  </channel>
</rss>""".strip()
        transcript_text = (
            "Episode 1 transcript with content about technology and software "
            "development. Bob Guest discusses various topics."
        )

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
            feed = rss_parser.fetch_and_parse_rss(self.cfg)
            episodes = [
                rss_parser.create_episode_from_item(item, idx, feed.base_url)
                for idx, item in enumerate(feed.items, start=1)
            ]
            episode = episodes[0]

            transcription_provider = create_transcription_provider(cfg)
            speaker_detector = create_speaker_detector(cfg)
            summarization_provider = create_summarization_provider(cfg)

            if hasattr(transcription_provider, "initialize"):
                transcription_provider.initialize()
            if hasattr(speaker_detector, "initialize"):
                speaker_detector.initialize()
            if hasattr(summarization_provider, "initialize"):
                summarization_provider.initialize()

            try:
                detected_hosts = speaker_detector.detect_hosts(
                    feed_title=feed.title,
                    feed_description=None,
                    feed_authors=feed.authors,
                )

                self.assertGreater(len(detected_hosts), 0)
                self.assertTrue(
                    any("John" in name or "Jane" in name for name in detected_hosts),
                    f"At least one host should be detected. Got: {detected_hosts}",
                )

                episode_description = rss_parser.extract_episode_description(episode.item)
                detected_speakers, detected_hosts_set, detection_succeeded, _ = (
                    speaker_detector.detect_speakers(
                        episode_title=episode.title,
                        episode_description=episode_description,
                        known_hosts=detected_hosts,
                    )
                )

                self.assertTrue(detection_succeeded)
                self.assertGreater(len(detected_speakers), 0)
                self.assertTrue(
                    any(host in detected_speakers for host in detected_hosts),
                    f"At least hosts should be in detected_speakers. Got: {detected_speakers}",
                )

                transcript_file_path = "0001 - Episode_1.txt"
                transcript_full_path = os.path.join(self.temp_dir, transcript_file_path)
                os.makedirs(os.path.dirname(transcript_full_path) or ".", exist_ok=True)
                with open(transcript_full_path, "w", encoding="utf-8") as f:
                    f.write(transcript_text)

                metadata_path = metadata.generate_episode_metadata(
                    feed=feed,
                    episode=episode,
                    feed_url=rss_url,
                    cfg=cfg,
                    output_dir=self.temp_dir,
                    run_suffix=None,
                    transcript_file_path=transcript_file_path,
                    transcript_source="direct_download",
                    whisper_model=None,
                    detected_hosts=list(detected_hosts),
                    detected_guests=[s for s in detected_speakers if s not in detected_hosts],
                    summary_provider=summarization_provider,
                )

                self.assertIsNotNone(metadata_path)
                self.assertTrue(os.path.exists(metadata_path))

                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertIn("speakers", data["content"])
                speakers = data["content"]["speakers"]
                self.assertIsInstance(speakers, list)
                host_speakers = [s for s in speakers if s.get("role") == "host"]
                guest_speakers = [s for s in speakers if s.get("role") == "guest"]
                self.assertEqual(len(host_speakers), len(detected_hosts))
                self.assertIsInstance(guest_speakers, list)

                self.assertIn("summary", data)
                summary = data["summary"]
                self.assertIsInstance(summary, dict)
                self.assertIn("bullets", summary)
                self.assertIsInstance(summary["bullets"], list)
                self.assertGreater(len(summary["bullets"]), 0)
                self.assertIn("short_summary", summary)
                self.assertIsInstance(summary["short_summary"], str)
                self.assertGreater(len(summary["short_summary"]), 0)

                self.assertEqual(data["content"]["transcript_source"], "direct_download")
            finally:
                if hasattr(transcription_provider, "cleanup"):
                    transcription_provider.cleanup()
                if hasattr(speaker_detector, "cleanup"):
                    speaker_detector.cleanup()
                if hasattr(summarization_provider, "cleanup"):
                    summarization_provider.cleanup()


if __name__ == "__main__":
    unittest.main()
