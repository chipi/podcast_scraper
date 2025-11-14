#!/usr/bin/env python3
"""Tests for command-line interface."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import shared test utilities from conftest
# Note: pytest automatically loads conftest.py, but we need explicit imports for unittest
import json
import sys
import tempfile
import unittest
from pathlib import Path

import podcast_scraper.cli as cli

# Add tests directory to path for conftest import
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from conftest import (  # noqa: F401, E402
    TEST_BASE_URL,
    TEST_CONTENT_TYPE_SRT,
    TEST_CONTENT_TYPE_VTT,
    TEST_CUSTOM_OUTPUT_DIR,
    TEST_EPISODE_TITLE,
    TEST_EPISODE_TITLE_SPECIAL,
    TEST_FEED_TITLE,
    TEST_FEED_URL,
    TEST_FULL_URL,
    TEST_MEDIA_TYPE_M4A,
    TEST_MEDIA_TYPE_MP3,
    TEST_MEDIA_URL,
    TEST_OUTPUT_DIR,
    TEST_PATH,
    TEST_RELATIVE_MEDIA,
    TEST_RELATIVE_TRANSCRIPT,
    TEST_RUN_ID,
    TEST_TRANSCRIPT_TYPE_SRT,
    TEST_TRANSCRIPT_TYPE_VTT,
    TEST_TRANSCRIPT_URL,
    TEST_TRANSCRIPT_URL_SRT,
    MockHTTPResponse,
    build_rss_xml_with_media,
    build_rss_xml_with_speakers,
    build_rss_xml_with_transcript,
    create_media_response,
    create_mock_spacy_model,
    create_rss_response,
    create_test_args,
    create_test_config,
    create_test_episode,
    create_test_feed,
    create_transcript_response,
)


class TestConfigFileSupport(unittest.TestCase):
    """Tests for configuration file loading."""

    def test_json_config_applied(self):
        """Config values should populate defaults when CLI flags are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": TEST_FEED_URL,
                "timeout": 45,
                "transcribe_missing": True,
                "prefer_type": ["text/vtt", ".srt"],
                "skip_existing": True,
                "dry_run": True,
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            args = cli.parse_args(["--config", cfg_path])
            self.assertEqual(args.timeout, 45)
            self.assertTrue(args.transcribe_missing)
            self.assertEqual(args.prefer_type, ["text/vtt", ".srt"])
            self.assertTrue(args.skip_existing)
            self.assertTrue(args.dry_run)
            self.assertEqual(args.rss, TEST_FEED_URL)

    def test_cli_overrides_config(self):
        """Command line arguments should override config defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": TEST_FEED_URL,
                "timeout": 99,
                "run_id": "from-config",
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            args = cli.parse_args(
                [
                    "--config",
                    cfg_path,
                    "--timeout",
                    "10",
                    TEST_FEED_URL,
                ]
            )
            self.assertEqual(args.timeout, 10)
            self.assertEqual(args.run_id, "from-config")

    def test_unknown_config_key_raises(self):
        """Unknown config entries should raise a ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "not_a_real_option": True,
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            with self.assertRaises(ValueError):
                cli.parse_args(["--config", cfg_path, TEST_FEED_URL])

    def test_log_level_from_config(self):
        """Log level should be applied from configuration files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "rss": TEST_FEED_URL,
                "log_level": "debug",
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(config_data, fh)

            args = cli.parse_args(["--config", cfg_path])
            self.assertEqual(args.log_level, "DEBUG")

    def test_yaml_config_applied(self):
        """YAML config files are parsed via PyYAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.yaml")
            config_text = """
rss: https://example.com/feed.yaml
timeout: 55
prefer_type:
  - text/plain
speaker_names:
  - Host
  - Guest
skip_existing: true
dry_run: true
""".strip()
            with open(cfg_path, "w", encoding="utf-8") as fh:
                fh.write(config_text)

            args = cli.parse_args(["--config", cfg_path])
            self.assertEqual(args.timeout, 55)
            self.assertEqual(args.prefer_type, ["text/plain"])
            self.assertEqual(args.speaker_names, "Host,Guest")
            self.assertEqual(args.rss, "https://example.com/feed.yaml")
            self.assertTrue(args.skip_existing)
            self.assertTrue(args.dry_run)

    def test_skip_existing_and_clean_output_flags(self):
        """CLI flags are parsed for resume/reset behaviour."""
        args = cli.parse_args([TEST_FEED_URL, "--skip-existing", "--clean-output", "--dry-run"])
        self.assertTrue(args.skip_existing)
        self.assertTrue(args.clean_output)
        self.assertTrue(args.dry_run)
