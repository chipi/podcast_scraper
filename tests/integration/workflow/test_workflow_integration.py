#!/usr/bin/env python3
"""Integration tests for podcast_scraper workflow.

These tests verify component interactions using mocked HTTP responses and mocked Whisper.
Moved from tests/e2e/test_workflow_e2e.py as part of Phase 3 test pyramid refactoring.
"""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

import podcast_scraper
import podcast_scraper.cli as cli
from podcast_scraper import config, downloader

pytestmark = [pytest.mark.integration, pytest.mark.module_workflow]


# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent.parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

# Import from parent conftest explicitly to avoid conflicts with infrastructure conftest
import importlib.util

parent_conftest_path = tests_dir / "conftest.py"
spec = importlib.util.spec_from_file_location("parent_conftest", parent_conftest_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load conftest from {parent_conftest_path}")
parent_conftest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_conftest)

build_rss_xml_with_media = parent_conftest.build_rss_xml_with_media
build_rss_xml_with_speakers = parent_conftest.build_rss_xml_with_speakers
build_rss_xml_with_transcript = parent_conftest.build_rss_xml_with_transcript
create_media_response = parent_conftest.create_media_response
create_mock_spacy_model = parent_conftest.create_mock_spacy_model
create_rss_response = parent_conftest.create_rss_response
create_test_args = parent_conftest.create_test_args
create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
create_transcript_response = parent_conftest.create_transcript_response
MockHTTPResponse = parent_conftest.MockHTTPResponse
TEST_BASE_URL = parent_conftest.TEST_BASE_URL
TEST_CONTENT_TYPE_SRT = parent_conftest.TEST_CONTENT_TYPE_SRT
TEST_CONTENT_TYPE_VTT = parent_conftest.TEST_CONTENT_TYPE_VTT
TEST_CUSTOM_OUTPUT_DIR = parent_conftest.TEST_CUSTOM_OUTPUT_DIR
TEST_EPISODE_TITLE = parent_conftest.TEST_EPISODE_TITLE
TEST_EPISODE_TITLE_SPECIAL = parent_conftest.TEST_EPISODE_TITLE_SPECIAL
TEST_FEED_TITLE = parent_conftest.TEST_FEED_TITLE
TEST_FEED_URL = parent_conftest.TEST_FEED_URL
TEST_FULL_URL = parent_conftest.TEST_FULL_URL
TEST_MEDIA_TYPE_M4A = parent_conftest.TEST_MEDIA_TYPE_M4A
TEST_MEDIA_TYPE_MP3 = parent_conftest.TEST_MEDIA_TYPE_MP3
TEST_MEDIA_URL = parent_conftest.TEST_MEDIA_URL
TEST_OUTPUT_DIR = parent_conftest.TEST_OUTPUT_DIR
TEST_PATH = parent_conftest.TEST_PATH
TEST_RELATIVE_MEDIA = parent_conftest.TEST_RELATIVE_MEDIA
TEST_RELATIVE_TRANSCRIPT = parent_conftest.TEST_RELATIVE_TRANSCRIPT
TEST_RUN_ID = parent_conftest.TEST_RUN_ID
TEST_TRANSCRIPT_TYPE_SRT = parent_conftest.TEST_TRANSCRIPT_TYPE_SRT
TEST_TRANSCRIPT_TYPE_VTT = parent_conftest.TEST_TRANSCRIPT_TYPE_VTT
TEST_TRANSCRIPT_URL = parent_conftest.TEST_TRANSCRIPT_URL
TEST_TRANSCRIPT_URL_SRT = parent_conftest.TEST_TRANSCRIPT_URL_SRT


@pytest.mark.integration
class TestIntegrationMain(unittest.TestCase):
    """Integration tests using mocked HTTP responses and mocked Whisper."""

    def _mock_http_map(self, mapping):
        """Return side effect function for fetch_url using mapping dict."""

        def _side_effect(url, user_agent, timeout, stream=False):
            normalized = downloader.normalize_url(url)
            resp = mapping.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_integration_main_downloads_transcript(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Integration Feed", transcript_url)
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
            with tempfile.TemporaryDirectory() as tmpdir:
                exit_code = cli.main([rss_url, "--output-dir", tmpdir, "--no-auto-speakers"])
                self.assertEqual(exit_code, 0)
                # Files are now saved in run_<suffix>/transcripts/ with run suffix in filename
                # Find the run directory and transcript file
                import glob

                run_dirs = glob.glob(os.path.join(tmpdir, "run_*"))
                self.assertGreater(len(run_dirs), 0, "Should have at least one run directory")
                run_dir = run_dirs[0]
                transcripts_dir = os.path.join(run_dir, "transcripts")
                # Find transcript file (may have run suffix in filename)
                transcript_files = glob.glob(os.path.join(transcripts_dir, "0001 - Episode 1*.txt"))
                self.assertGreater(
                    len(transcript_files), 0, f"Should find transcript file in {transcripts_dir}"
                )
                expected_path = transcript_files[0]
                self.assertTrue(os.path.exists(expected_path))
                with open(expected_path, "r", encoding="utf-8") as fh:
                    self.assertEqual(fh.read().strip(), transcript_text)

    def test_integration_main_multi_feed_corpus_layout_440(self):
        """GitHub #440: two feeds under corpus_parent/feeds/<stable>/ (mocked HTTP)."""
        rss_a = "https://alpha.example/feed.xml"
        rss_b = "https://beta.example/feed.xml"
        tr_a = "https://alpha.example/ep1.txt"
        tr_b = "https://beta.example/ep1.txt"
        rss_xml_a = build_rss_xml_with_transcript("Feed A", tr_a)
        rss_xml_b = build_rss_xml_with_transcript("Feed B", tr_b)
        responses = {
            downloader.normalize_url(rss_a): create_rss_response(rss_xml_a, rss_a),
            downloader.normalize_url(tr_a): create_transcript_response("Alpha transcript", tr_a),
            downloader.normalize_url(rss_b): create_rss_response(rss_xml_b, rss_b),
            downloader.normalize_url(tr_b): create_transcript_response("Beta transcript", tr_b),
        }

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            with tempfile.TemporaryDirectory() as corpus:
                exit_code = cli.main(
                    [
                        rss_a,
                        "--rss",
                        rss_b,
                        "--output-dir",
                        corpus,
                        "--no-auto-speakers",
                        "--max-episodes",
                        "1",
                    ]
                )
                self.assertEqual(exit_code, 0)
                feeds_root = os.path.join(corpus, "feeds")
                self.assertTrue(os.path.isdir(feeds_root), f"missing {feeds_root}")
                subdirs = [
                    d for d in os.listdir(feeds_root) if os.path.isdir(os.path.join(feeds_root, d))
                ]
                self.assertEqual(
                    len(subdirs),
                    2,
                    f"expected two feed workspace dirs under feeds/, got {subdirs!r}",
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(corpus, "corpus_manifest.json")),
                    "multi-feed finalize should write corpus_manifest.json (#506)",
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(corpus, "corpus_run_summary.json")),
                    "multi-feed finalize should write corpus_run_summary.json (#506)",
                )

    @pytest.mark.critical_path
    def test_integration_main_multi_feed_partial_failure_still_writes_corpus_artifacts_506(self):
        """One failing feed still runs finalize: manifest, summary, overall_ok false (#506)."""
        rss_a = "https://alpha.example/feed.xml"
        rss_b = "https://beta.example/feed.xml"

        def fake_run(cfg):
            if cfg.rss_url and "beta.example" in cfg.rss_url:
                raise ValueError("simulated feed B failure")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            exit_code = cli.main(
                [
                    rss_a,
                    "--rss",
                    rss_b,
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                ],
                run_pipeline_fn=fake_run,
            )
            self.assertEqual(exit_code, 1)
            summary_path = os.path.join(corpus, "corpus_run_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            self.assertTrue(os.path.isfile(os.path.join(corpus, "corpus_manifest.json")))
            with open(summary_path, encoding="utf-8") as fh:
                blob = json.load(fh)
            self.assertEqual(blob.get("schema_version"), "1.2.0")
            self.assertIs(blob.get("overall_ok"), False)
            feeds = blob.get("feeds") or []
            self.assertEqual(len(feeds), 2)
            by_url = {row["feed_url"]: row for row in feeds}
            self.assertTrue(by_url[rss_a].get("ok"))
            self.assertFalse(by_url[rss_b].get("ok"))
            bi = blob.get("batch_incidents") or {}
            self.assertGreaterEqual(bi.get("lines_in_window", 0), 1)
            self.assertEqual((bi.get("feed_incidents_unique") or {}).get("hard"), 1)

    def test_integration_service_run_multi_feed_corpus_layout_440(self):
        """GitHub #440: ``service.run`` with ``rss_urls`` matches CLI corpus layout (mocked HTTP)."""
        from podcast_scraper import service

        rss_a = "https://alpha.example/feed.xml"
        rss_b = "https://beta.example/feed.xml"
        tr_a = "https://alpha.example/ep1.txt"
        tr_b = "https://beta.example/ep1.txt"
        rss_xml_a = build_rss_xml_with_transcript("Feed A", tr_a)
        rss_xml_b = build_rss_xml_with_transcript("Feed B", tr_b)
        responses = {
            downloader.normalize_url(rss_a): create_rss_response(rss_xml_a, rss_a),
            downloader.normalize_url(tr_a): create_transcript_response("Alpha transcript", tr_a),
            downloader.normalize_url(rss_b): create_rss_response(rss_xml_b, rss_b),
            downloader.normalize_url(tr_b): create_transcript_response("Beta transcript", tr_b),
        }

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            with tempfile.TemporaryDirectory() as corpus:
                cfg = config.Config(
                    rss_urls=[rss_a, rss_b],
                    output_dir=corpus,
                    max_episodes=1,
                    auto_speakers=False,
                    user_agent="test",
                    timeout=30,
                )
                result = service.run(cfg)
                self.assertTrue(result.success, result.error)
                self.assertGreaterEqual(result.episodes_processed, 2)
                feeds_root = os.path.join(corpus, "feeds")
                self.assertTrue(os.path.isdir(feeds_root))
                subdirs = [
                    d for d in os.listdir(feeds_root) if os.path.isdir(os.path.join(feeds_root, d))
                ]
                self.assertEqual(len(subdirs), 2)
                self.assertTrue(
                    os.path.isfile(os.path.join(corpus, "corpus_manifest.json")),
                    "service multi-feed should write corpus_manifest.json (#506)",
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(corpus, "corpus_run_summary.json")),
                    "service multi-feed should write corpus_run_summary.json (#506)",
                )

    @pytest.mark.critical_path
    def test_integration_service_multi_feed_partial_failure_writes_artifacts_506(self):
        """``service.run`` multi-feed: one failing feed still finalizes corpus JSON (#506)."""
        from podcast_scraper import service

        rss_a = "https://alpha.example/feed.xml"
        rss_b = "https://beta.example/feed.xml"

        def _side_effect(cfg):
            if cfg.rss_url and "beta.example" in cfg.rss_url:
                raise ValueError("simulated feed B failure")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            with patch("podcast_scraper.service.workflow.run_pipeline", side_effect=_side_effect):
                cfg = config.Config(
                    rss_urls=[rss_a, rss_b],
                    output_dir=corpus,
                    max_episodes=1,
                    auto_speakers=False,
                    user_agent="test",
                    timeout=30,
                )
                result = service.run(cfg)

            self.assertFalse(result.success)
            self.assertIsNotNone(result.multi_feed_summary)
            self.assertFalse(result.multi_feed_summary.get("overall_ok"))
            summary_path = os.path.join(corpus, "corpus_run_summary.json")
            self.assertTrue(os.path.isfile(summary_path))
            self.assertTrue(os.path.isfile(os.path.join(corpus, "corpus_manifest.json")))
            with open(summary_path, encoding="utf-8") as fh:
                blob = json.load(fh)
            self.assertIs(blob.get("overall_ok"), False)
            self.assertEqual(blob.get("schema_version"), "1.2.0")
            bi = blob.get("batch_incidents") or {}
            self.assertGreaterEqual(bi.get("lines_in_window", 0), 1)

    def test_integration_main_whisper_fallback(self):
        rss_url = "https://example.com/feed.xml"
        media_url = "https://example.com/ep1.mp3"
        rss_xml = build_rss_xml_with_media("Integration Feed", media_url)
        media_bytes = b"FAKE AUDIO DATA"
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(media_url): create_media_response(media_bytes, media_url),
        }

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model._is_cpu_device = False
        transcribed_text = "Hello from Whisper"

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            with patch(
                "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper:
                mock_whisper_lib = Mock()
                mock_whisper_lib.load_model.return_value = mock_model
                mock_import_whisper.return_value = mock_whisper_lib
                with patch(
                    "podcast_scraper.providers.ml.ml_provider.MLProvider._transcribe_with_whisper",
                    return_value=({"text": transcribed_text, "segments": []}, 1.0),
                ) as mock_transcribe:
                    with patch(
                        "podcast_scraper.cache.transcript_cache.get_cached_transcript_entry",
                        return_value=None,
                    ):  # Disable cache to ensure transcription is called
                        with tempfile.TemporaryDirectory() as tmpdir:
                            exit_code = cli.main(
                                [
                                    rss_url,
                                    "--output-dir",
                                    tmpdir,
                                    "--transcribe-missing",
                                    "--whisper-model",
                                    config.TEST_DEFAULT_WHISPER_MODEL,
                                    "--run-id",
                                    "testrun",
                                ]
                            )
                            self.assertEqual(exit_code, 0)
                            # Note: Episodes may be skipped if they already have transcripts
                            # Check if transcription was called (only if episode wasn't skipped)
                            if mock_import_whisper.called:
                                mock_transcribe.assert_called()
                                # Files are now saved in run_<suffix>/transcripts/ with run suffix in filename
                                # Find the run directory and transcript file
                                import glob

                                run_dirs = glob.glob(os.path.join(tmpdir, "run_*"))
                                self.assertGreater(
                                    len(run_dirs), 0, "Should have at least one run directory"
                                )
                                run_dir = run_dirs[0]
                                transcripts_dir = os.path.join(run_dir, "transcripts")
                                # Find transcript file (may have run suffix in filename)
                                transcript_files = glob.glob(
                                    os.path.join(transcripts_dir, "0001 - Episode 1*.txt")
                                )
                                self.assertGreater(
                                    len(transcript_files),
                                    0,
                                    f"Should find transcript file in {transcripts_dir}",
                                )
                                out_path = Path(transcript_files[0])
                                self.assertTrue(out_path.exists())
                                self.assertEqual(
                                    out_path.read_text(encoding="utf-8").strip(), transcribed_text
                                )

    def test_path_traversal_attempt_normalized(self):
        """Test that path traversal attempts are normalized correctly.

        This test validates that malicious paths with '..' components are properly
        normalized by validate_and_normalize_output_dir, and that files are created
        in the normalized location (not the malicious location).

        The test is platform-agnostic by:
        1. Using validate_and_normalize_output_dir to get the expected normalized path
        2. Constructing the expected file path from the normalized directory
        3. Verifying the file exists at the expected location
        """
        from podcast_scraper.utils import filesystem

        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Integration Feed", transcript_url)
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
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a path with traversal attempts
                malicious = os.path.join(tmpdir, "..", "danger", "..", "final")

                # Get the normalized path that will actually be used
                # This is what validate_and_normalize_output_dir returns
                normalized_dir = filesystem.validate_and_normalize_output_dir(malicious)

                # Run the CLI with the malicious path
                exit_code = cli.main([rss_url, "--output-dir", malicious, "--no-auto-speakers"])
                self.assertEqual(exit_code, 0)

                # Files are now saved in run_<suffix>/transcripts/ with run suffix in filename
                # Find the run directory and transcript file (similar to test_integration_main_downloads_transcript)
                import glob

                run_dirs = glob.glob(os.path.join(normalized_dir, "run_*"))
                self.assertGreater(
                    len(run_dirs), 0, f"Should have at least one run directory in {normalized_dir}"
                )
                # Use the most recently modified run dir (current run); normalized_dir can be
                # reused across runs (e.g. .../T/final) so run_dirs may contain older runs
                run_dir = max(run_dirs, key=lambda p: os.path.getmtime(p))
                transcripts_dir = os.path.join(run_dir, "transcripts")
                # Find transcript file (may have run suffix in filename; use broad glob for resilience)
                transcript_files = glob.glob(os.path.join(transcripts_dir, "0001*.txt"))
                if not transcript_files:
                    transcript_files = glob.glob(os.path.join(transcripts_dir, "*.txt"))
                self.assertGreater(
                    len(transcript_files),
                    0,
                    f"Should find transcript file in {transcripts_dir} (listing: {os.listdir(transcripts_dir) if os.path.isdir(transcripts_dir) else 'not a dir'})",
                )
                expected_transcript_path = Path(transcript_files[0])

                # Verify the file exists at the normalized location (not the malicious location)
                self.assertTrue(
                    expected_transcript_path.exists(),
                    f"Transcript file should exist at normalized path: {expected_transcript_path}",
                )

                # Verify the file content
                with open(expected_transcript_path, "r", encoding="utf-8") as fh:
                    self.assertEqual(fh.read().strip(), transcript_text)

                # Verify the malicious path resolves to the normalized path
                # (path normalization should collapse the '..' components)
                malicious_path = Path(malicious)
                self.assertEqual(
                    malicious_path.resolve(),
                    Path(normalized_dir),
                    f"Malicious path '{malicious}' should resolve to normalized path '{normalized_dir}'",
                )

                # Verify the final path does not contain '..' components
                self.assertNotIn("..", str(expected_transcript_path))

    def test_config_override_precedence_integration(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Integration Feed", transcript_url)
        transcript_text = "Episode 1 transcript"
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": rss_url,
                "output_dir": tmpdir,  # Add output_dir to config
                "timeout": 60,
                "log_level": "WARNING",
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            observed_timeouts = []

            def tracking_open(url, user_agent, timeout, stream=False):
                observed_timeouts.append(timeout)
                return self._mock_http_map(responses)(url, user_agent, timeout, stream)

            with (
                patch("podcast_scraper.downloader.fetch_url", side_effect=tracking_open),
                patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=tracking_open),
            ):
                with patch("podcast_scraper.workflow.orchestration.apply_log_level") as mock_apply:
                    exit_code = cli.main(
                        [
                            "--config",
                            cfg_path,
                            "--timeout",
                            "10",
                            "--log-level",
                            "DEBUG",
                        ]
                    )
                    self.assertEqual(exit_code, 0)
                    self.assertTrue(observed_timeouts)
                    self.assertTrue(all(timeout == 10 for timeout in observed_timeouts))
                    # apply_log_level is called with (level, log_file) where log_file can be None
                    # Check that it was called with DEBUG and any log_file value (including None)
                    mock_apply.assert_called()
                    call_args = mock_apply.call_args
                    self.assertEqual(call_args[0][0], "DEBUG")
                    # log_file can be None or not provided
                    if len(call_args[0]) > 1:
                        self.assertIsNone(call_args[0][1])

    def test_dry_run_skips_downloads(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Integration Feed", transcript_url)
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
            with tempfile.TemporaryDirectory() as tmpdir:
                expected_path = os.path.join(tmpdir, "transcripts", "0001 - Episode 1.txt")
                import logging

                with self.assertLogs(logging.getLogger("podcast_scraper"), level="INFO") as log_ctx:
                    exit_code = cli.main(
                        [
                            rss_url,
                            "--output-dir",
                            tmpdir,
                            "--dry-run",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                self.assertFalse(os.path.exists(expected_path))
                log_text = "\n".join(log_ctx.output)
                self.assertIn("Dry run complete. transcripts_planned=1", log_text)
                self.assertIn("Direct downloads planned: 1", log_text)
                self.assertIn("Whisper transcriptions planned: 0", log_text)
                self.assertIn("would save as", log_text)


@pytest.mark.integration
class TestLibraryAPIIntegration(unittest.TestCase):
    """Integration tests for library API (podcast_scraper.run_pipeline()) with mocked Whisper."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _mock_http_map(self, responses):
        """Create HTTP mock side effect function.

        Args:
            responses: Dict mapping normalized URLs to MockHTTPResponse objects

        Returns:
            Side effect function for fetch_url
        """

        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            normalized = downloader.normalize_url(url)
            resp = responses.get(normalized)
            if resp is None:
                raise AssertionError(f"Unexpected HTTP request: {normalized}")
            return resp

        return _side_effect

    def test_e2e_library_basic_transcript_download(self):
        """E2E: Use library API to download transcript."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Library Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript from library API"

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
            cfg = podcast_scraper.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                max_episodes=1,
                transcribe_missing=False,  # Don't use Whisper (downloading transcripts)
                auto_speakers=False,  # Disable to avoid run suffix
                generate_summaries=False,  # Disable to avoid run suffix
            )

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)
            self.assertIsInstance(summary, str)
            self.assertIn("transcripts", summary.lower())

            # Verify file was created (now in run_<suffix>/transcripts/ subdirectory)
            # Files are saved with run suffix even when no ML features are enabled
            import glob

            run_dirs = glob.glob(os.path.join(self.temp_dir, "run_*"))
            self.assertGreater(len(run_dirs), 0, "Should have at least one run directory")
            run_dir = run_dirs[0]
            transcripts_dir = os.path.join(run_dir, "transcripts")
            # Find transcript file (may have run suffix in filename)
            transcript_files = glob.glob(os.path.join(transcripts_dir, "0001 - Episode 1*.txt"))
            self.assertGreater(
                len(transcript_files), 0, f"Should find transcript file in {transcripts_dir}"
            )
            expected_path = transcript_files[0]
            self.assertTrue(os.path.exists(expected_path))
            with open(expected_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read().strip(), transcript_text)

    def test_e2e_library_with_config_file(self):
        """E2E: Load config from file and run pipeline."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Config File Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript from config file"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        # Create config file
        cfg_path = os.path.join(self.temp_dir, "test_config.json")
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
            "timeout": 30,
            "transcribe_missing": False,  # Don't use Whisper (downloading transcripts)
            "auto_speakers": False,  # Disable to avoid run suffix
            "generate_summaries": False,  # Disable to avoid run suffix
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            # Load config from file
            config_dict = podcast_scraper.load_config_file(cfg_path)
            cfg = podcast_scraper.Config(**config_dict)

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
            self.assertIsInstance(summary, str)

            # Verify file was created (now in run_<suffix>/transcripts/ subdirectory)
            import glob

            run_dirs = glob.glob(os.path.join(self.temp_dir, "run_*"))
            self.assertGreater(len(run_dirs), 0, "Should have at least one run directory")
            run_dir = run_dirs[0]
            transcripts_dir = os.path.join(run_dir, "transcripts")
            # Find transcript file (may have run suffix in filename)
            transcript_files = glob.glob(os.path.join(transcripts_dir, "0001 - Episode 1*.txt"))
            self.assertGreater(
                len(transcript_files), 0, f"Should find transcript file in {transcripts_dir}"
            )
            expected_path = transcript_files[0]
            self.assertTrue(os.path.exists(expected_path))

    def test_e2e_library_whisper_fallback(self):
        """E2E: Library API with Whisper transcription."""
        rss_url = "https://example.com/feed.xml"
        media_url = "https://example.com/ep1.mp3"
        rss_xml = build_rss_xml_with_media("Whisper Library Test Feed", media_url)
        media_bytes = b"FAKE AUDIO DATA"
        transcribed_text = "Hello from Whisper library API"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(media_url): create_media_response(media_bytes, media_url),
        }

        mock_model = Mock()
        mock_model.device.type = "cpu"
        mock_model._is_cpu_device = False

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            with patch(
                "podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper:
                mock_whisper_lib = Mock()
                mock_whisper_lib.load_model.return_value = mock_model
                mock_import_whisper.return_value = mock_whisper_lib
                with patch(
                    "podcast_scraper.providers.ml.ml_provider.MLProvider._transcribe_with_whisper",
                    return_value=({"text": transcribed_text, "segments": []}, 1.0),
                ) as mock_transcribe:
                    cfg = podcast_scraper.Config(
                        rss_url=rss_url,
                        output_dir=self.temp_dir,
                        transcribe_missing=True,
                        whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                        max_episodes=1,
                        transcript_cache_enabled=False,  # Disable cache to ensure transcription is called
                    )

                    count, summary = podcast_scraper.run_pipeline(cfg)

                    # Verify Whisper was called (only if episode wasn't skipped)
                    # Note: Episodes may be skipped if they already have transcripts
                    if mock_import_whisper.called:
                        mock_transcribe.assert_called()

                    # Verify return values
                    self.assertIsInstance(count, int)
                    self.assertGreater(count, 0)
                    self.assertIsInstance(summary, str)
                    self.assertIn("transcripts", summary.lower())

                    # Verify transcript file was created
                    effective_dir = Path(self.temp_dir).resolve()
                    # Find the transcript file (name may vary based on run_id)
                    transcript_files = list(effective_dir.glob("**/*.txt"))
                    self.assertGreater(len(transcript_files), 0)
                    self.assertIn(transcribed_text, transcript_files[0].read_text(encoding="utf-8"))

    def test_e2e_library_error_handling_invalid_feed(self):
        """E2E: Library API handles invalid feed URL gracefully."""
        invalid_url = "https://invalid-feed.example.com/feed.xml"

        # Mock HTTP error
        def _side_effect(url, user_agent=None, timeout=None, stream=False):
            raise requests.RequestException("Connection failed")

        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=_side_effect),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=_side_effect),
        ):
            cfg = podcast_scraper.Config(
                rss_url=invalid_url,
                output_dir=self.temp_dir,
            )

            # Should raise ValueError, RuntimeError, or RequestException
            # (RequestException may propagate through retry logic)
            with self.assertRaises((ValueError, RuntimeError, requests.RequestException)):
                podcast_scraper.run_pipeline(cfg)

    def test_e2e_library_error_handling_empty_feed(self):
        """E2E: Library API handles empty feed gracefully."""
        rss_url = "https://example.com/empty_feed.xml"
        rss_xml = """<?xml version='1.0'?>
<rss>
  <channel>
    <title>Empty Feed</title>
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
            cfg = podcast_scraper.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
            )

            # Should handle empty feed gracefully
            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values indicate no episodes
            self.assertIsInstance(count, int)
            self.assertEqual(count, 0)
            self.assertIsInstance(summary, str)

    def test_e2e_library_dry_run(self):
        """E2E: Library API with dry-run mode."""
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("Dry Run Test Feed", transcript_url)
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
            cfg = podcast_scraper.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                max_episodes=1,
                dry_run=True,
            )

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)
            self.assertIsInstance(summary, str)
            self.assertIn("dry run", summary.lower())

            # Verify no files were created in dry-run mode
            expected_path = os.path.join(self.temp_dir, "0001 - Episode 1.txt")
            self.assertFalse(os.path.exists(expected_path))

    def test_e2e_library_with_yaml_config(self):
        """E2E: Load YAML config file and run pipeline."""
        import yaml

        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = build_rss_xml_with_transcript("YAML Config Test Feed", transcript_url)
        transcript_text = "Episode 1 transcript from YAML config"

        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
            downloader.normalize_url(transcript_url): create_transcript_response(
                transcript_text, transcript_url
            ),
        }

        # Create YAML config file
        cfg_path = os.path.join(self.temp_dir, "test_config.yaml")
        config_data = {
            "rss": rss_url,
            "output_dir": self.temp_dir,
            "max_episodes": 1,
            "timeout": 30,
            "log_level": "INFO",
            "transcribe_missing": False,  # Don't use Whisper (downloading transcripts)
            "auto_speakers": False,  # Disable to avoid run suffix
            "generate_summaries": False,  # Disable to avoid run suffix
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with (
            patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock),
            patch("podcast_scraper.downloader.fetch_rss_feed_url", side_effect=http_mock),
        ):
            # Load config from YAML file
            config_dict = podcast_scraper.load_config_file(cfg_path)
            cfg = podcast_scraper.Config(**config_dict)

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
            self.assertIsInstance(summary, str)

            # Verify file was created (now in run_<suffix>/transcripts/ subdirectory)
            import glob

            run_dirs = glob.glob(os.path.join(self.temp_dir, "run_*"))
            self.assertGreater(len(run_dirs), 0, "Should have at least one run directory")
            run_dir = run_dirs[0]
            transcripts_dir = os.path.join(run_dir, "transcripts")
            # Find transcript file (may have run suffix in filename)
            transcript_files = glob.glob(os.path.join(transcripts_dir, "0001 - Episode 1*.txt"))
            self.assertGreater(
                len(transcript_files), 0, f"Should find transcript file in {transcripts_dir}"
            )
            expected_path = transcript_files[0]
            self.assertTrue(os.path.exists(expected_path))
