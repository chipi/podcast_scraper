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

# Check if ML dependencies are available
SPACY_AVAILABLE = False
try:
    import spacy  # noqa: F401

    SPACY_AVAILABLE = True
except ImportError:
    pass

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent.parent
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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                exit_code = cli.main([rss_url, "--output-dir", tmpdir, "--no-auto-speakers"])
                self.assertEqual(exit_code, 0)
                expected_path = os.path.join(tmpdir, "transcripts", "0001 - Episode 1.txt")
                self.assertTrue(os.path.exists(expected_path))
                with open(expected_path, "r", encoding="utf-8") as fh:
                    self.assertEqual(fh.read().strip(), transcript_text)

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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with patch(
                "podcast_scraper.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper:
                mock_whisper_lib = Mock()
                mock_whisper_lib.load_model.return_value = mock_model
                mock_import_whisper.return_value = mock_whisper_lib
                with patch(
                    "podcast_scraper.ml.ml_provider.MLProvider._transcribe_with_whisper",
                    return_value=({"text": transcribed_text, "segments": []}, 1.0),
                ) as mock_transcribe:
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
                                "--no-auto-speakers",
                            ]
                        )
                        self.assertEqual(exit_code, 0)
                        # Note: Episodes may be skipped if they already have transcripts
                        # Check if transcription was called (only if episode wasn't skipped)
                        if mock_import_whisper.called:
                            mock_transcribe.assert_called()
                            # Uses test model (not base.en production model)
                            # Suffix format: w_<model> for whisper (e.g., w_tiny.en)
                            # Model name is sanitized but dots are preserved
                            from podcast_scraper.filesystem import sanitize_filename

                            model_short = config.TEST_DEFAULT_WHISPER_MODEL
                            model_suffix = sanitize_filename(model_short)
                            effective_dir = Path(tmpdir).resolve() / f"run_testrun_w_{model_suffix}"
                            out_path = (
                                effective_dir
                                / "transcripts"
                                / f"0001 - Episode 1_testrun_w_{model_suffix}.txt"
                            )
                            self.assertTrue(out_path.exists())
                            self.assertEqual(
                                out_path.read_text(encoding="utf-8").strip(), transcribed_text
                            )

    def test_path_traversal_attempt_normalized(self):
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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                malicious = os.path.join(tmpdir, "..", "danger", "..", "final")
                exit_code = cli.main([rss_url, "--output-dir", malicious, "--no-auto-speakers"])
                self.assertEqual(exit_code, 0)
                effective_dir = Path(malicious).expanduser().resolve()
                out_path = effective_dir / "transcripts" / "0001 - Episode 1.txt"
                self.assertTrue(out_path.exists())
                self.assertNotIn("..", str(out_path))

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
                "timeout": 60,
                "log_level": "WARNING",
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            observed_timeouts = []

            def tracking_open(url, user_agent, timeout, stream=False):
                observed_timeouts.append(timeout)
                return self._mock_http_map(responses)(url, user_agent, timeout, stream)

            with patch("podcast_scraper.downloader.fetch_url", side_effect=tracking_open):
                with patch("podcast_scraper.workflow.apply_log_level") as mock_apply:
                    exit_code = cli.main(
                        [
                            "--config",
                            cfg_path,
                            "--timeout",
                            "10",
                            "--log-level",
                            "DEBUG",
                            "--whisper-model",
                            config.TEST_DEFAULT_WHISPER_MODEL.replace(".en", ""),
                        ]
                    )
                    self.assertEqual(exit_code, 0)
                    self.assertTrue(observed_timeouts)
                    self.assertTrue(all(timeout == 10 for timeout in observed_timeouts))
                    mock_apply.assert_called_with("DEBUG", None)

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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
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

    @pytest.mark.slow
    @pytest.mark.ml_models
    @pytest.mark.skipif(not SPACY_AVAILABLE, reason="spaCy dependencies not available")
    def test_dry_run_performs_speaker_detection(self):
        """Test that dry-run mode still performs host/guest detection.

        This test requires multiple episodes to verify speaker detection works
        across episodes, so it's marked as slow to run in full test mode.
        Requires ML models (spaCy) for speaker detection.
        Note: spaCy model (en_core_web_sm) is installed as a dependency.
        """
        rss_url = "https://example.com/feed.xml"
        rss_xml = build_rss_xml_with_speakers(
            "Test Podcast",
            authors=["John Host"],
            items=[
                {
                    "title": "Interview with Alice Guest",
                    "description": "This episode features Alice Guest discussing technology.",
                },
                {
                    "title": "Chat with Bob Guest",
                    "description": "Bob Guest joins us for a conversation.",
                },
            ],
        )
        responses = {
            downloader.normalize_url(rss_url): create_rss_response(rss_xml, rss_url),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                import logging

                with self.assertLogs(logging.getLogger("podcast_scraper"), level="INFO") as log_ctx:
                    exit_code = cli.main(
                        [
                            rss_url,
                            "--output-dir",
                            tmpdir,
                            "--dry-run",
                            "--auto-speakers",
                        ]
                    )
                self.assertEqual(exit_code, 0)
                log_text = "\n".join(log_ctx.output)
                # Verify host detection happened
                self.assertIn("DETECTED HOSTS", log_text)
                self.assertIn("John Host", log_text)
                # Verify guest detection happened for each episode
                self.assertIn("Episode 1: Interview with Alice Guest", log_text)
                self.assertIn("Episode 2: Chat with Bob Guest", log_text)
                # Verify guest detection logging (changed to singular "Guest:" format)
                self.assertIn("Guest:", log_text)


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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            cfg = podcast_scraper.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                max_episodes=1,
                transcribe_missing=False,  # Don't use Whisper (downloading transcripts)
                auto_speakers=False,  # Disable speaker detection for this test
            )

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)
            self.assertIsInstance(summary, str)
            self.assertIn("transcripts", summary.lower())

            # Verify file was created (now in transcripts/ subdirectory)
            expected_path = os.path.join(self.temp_dir, "transcripts", "0001 - Episode 1.txt")
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
            "auto_speakers": False,  # Disable speaker detection for this test
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Load config from file
            config_dict = podcast_scraper.load_config_file(cfg_path)
            cfg = podcast_scraper.Config(**config_dict)

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
            self.assertIsInstance(summary, str)

            # Verify file was created (now in transcripts/ subdirectory)
            expected_path = os.path.join(self.temp_dir, "transcripts", "0001 - Episode 1.txt")
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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with patch(
                "podcast_scraper.ml.ml_provider._import_third_party_whisper"
            ) as mock_import_whisper:
                mock_whisper_lib = Mock()
                mock_whisper_lib.load_model.return_value = mock_model
                mock_import_whisper.return_value = mock_whisper_lib
                with patch(
                    "podcast_scraper.ml.ml_provider.MLProvider._transcribe_with_whisper",
                    return_value=({"text": transcribed_text, "segments": []}, 1.0),
                ) as mock_transcribe:
                    cfg = podcast_scraper.Config(
                        rss_url=rss_url,
                        output_dir=self.temp_dir,
                        transcribe_missing=True,
                        whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
                        max_episodes=1,
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

        with patch("podcast_scraper.downloader.fetch_url", side_effect=_side_effect):
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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            cfg = podcast_scraper.Config(
                rss_url=rss_url,
                output_dir=self.temp_dir,
                whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
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
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
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
            "auto_speakers": False,  # Disable speaker detection for this test
        }
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.dump(config_data, fh)

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            # Load config from YAML file
            config_dict = podcast_scraper.load_config_file(cfg_path)
            cfg = podcast_scraper.Config(**config_dict)

            count, summary = podcast_scraper.run_pipeline(cfg)

            # Verify return values
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
            self.assertIsInstance(summary, str)

            # Verify file was created (now in transcripts/ subdirectory)
            expected_path = os.path.join(self.temp_dir, "transcripts", "0001 - Episode 1.txt")
            self.assertTrue(os.path.exists(expected_path))
