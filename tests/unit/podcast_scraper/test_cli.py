#!/usr/bin/env python3
"""Tests for CLI argument parsing and validation.

These tests verify the CLI argument parsing, validation, and configuration
building functions.
"""

import argparse
import copy
import logging
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from podcast_scraper import cli, config

pytestmark = [pytest.mark.unit, pytest.mark.module_cli]


class TestValidateRssUrl(unittest.TestCase):
    """Test _validate_rss_url function."""

    def test_valid_http_url(self):
        """Test that valid HTTP URL passes validation."""
        errors = []
        cli._validate_rss_url("http://example.com/feed.xml", errors)
        self.assertEqual(errors, [])

    def test_valid_https_url(self):
        """Test that valid HTTPS URL passes validation."""
        errors = []
        cli._validate_rss_url("https://example.com/feed.xml", errors)
        self.assertEqual(errors, [])

    def test_empty_url(self):
        """Test that empty URL adds error."""
        errors = []
        cli._validate_rss_url("", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL is required", errors[0])

    def test_url_without_scheme(self):
        """Test that URL without http/https scheme adds error."""
        errors = []
        cli._validate_rss_url("example.com/feed.xml", errors)
        # URL without scheme may not have netloc, so may trigger hostname error instead
        self.assertGreater(len(errors), 0)
        # Should have either scheme or hostname error
        error_text = " ".join(errors)
        self.assertTrue(
            "RSS URL must be http or https" in error_text
            or "RSS URL must have a valid hostname" in error_text
        )

    def test_url_with_invalid_scheme(self):
        """Test that URL with invalid scheme adds error."""
        errors = []
        cli._validate_rss_url("ftp://example.com/feed.xml", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL must be http or https", errors[0])

    def test_url_without_hostname(self):
        """Test that URL without hostname adds error."""
        errors = []
        cli._validate_rss_url("http:///feed.xml", errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("RSS URL must have a valid hostname", errors[0])

    def test_url_with_path_and_query(self):
        """Test that URL with path and query parameters is valid."""
        errors = []
        cli._validate_rss_url("https://example.com/feed.xml?param=value", errors)
        self.assertEqual(errors, [])


class TestValidateWhisperConfig(unittest.TestCase):
    """Test _validate_whisper_config function."""

    def test_valid_model(self):
        """Test that valid Whisper model passes validation."""
        # Tests should use test default (tiny.en), not production default (base)
        args = Namespace(transcribe_missing=True, whisper_model=config.TEST_DEFAULT_WHISPER_MODEL)
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(errors, [])

    def test_valid_model_when_transcribe_disabled(self):
        """Test that model validation is skipped when transcribe_missing is False."""
        args = Namespace(transcribe_missing=False, whisper_model="invalid")
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(errors, [])

    def test_invalid_model(self):
        """Test that invalid Whisper model adds error."""
        args = Namespace(transcribe_missing=True, whisper_model="invalid")
        errors = []
        cli._validate_whisper_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--whisper-model must be one of", errors[0])

    def test_all_valid_models(self):
        """Test that all valid models pass validation."""
        valid_models = (
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
            "tiny.en",
            "base.en",
            "small.en",
            "medium.en",
            "large.en",
        )
        for model in valid_models:
            with self.subTest(model=model):
                args = Namespace(transcribe_missing=True, whisper_model=model)
                errors = []
                cli._validate_whisper_config(args, errors)
                self.assertEqual(errors, [], f"Model {model} should be valid")


class TestValidateSpeakerConfig(unittest.TestCase):
    """Test _validate_speaker_config function."""

    def test_valid_num_speakers(self):
        """Test that valid num_speakers passes validation."""
        args = Namespace(screenplay=True, num_speakers=2, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_num_speakers_below_minimum(self):
        """Test that num_speakers below minimum adds error."""
        args = Namespace(screenplay=True, num_speakers=1, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        # Note: Validation only checks if screenplay is True
        # If screenplay is True and num_speakers < MIN_NUM_SPEAKERS, should add error
        if args.screenplay and args.num_speakers < config.MIN_NUM_SPEAKERS:
            self.assertEqual(len(errors), 1)
            self.assertIn("--num-speakers must be at least", errors[0])
        else:
            # If validation doesn't trigger, that's also valid behavior
            self.assertGreaterEqual(len(errors), 0)

    def test_num_speakers_validation_when_screenplay_false(self):
        """Test that num_speakers validation is skipped when screenplay is False."""
        args = Namespace(screenplay=False, num_speakers=1, speaker_names=None)
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_valid_speaker_names(self):
        """Test that valid speaker names pass validation."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice, Bob")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_speaker_names_with_whitespace(self):
        """Test that speaker names with whitespace are handled."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="  Alice  ,  Bob  ")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])

    def test_speaker_names_single_name(self):
        """Test that single speaker name adds error."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice")
        errors = []
        cli._validate_speaker_config(args, errors)
        # Validation only checks if speaker_names is provided
        if args.speaker_names:
            names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
            if len(names) < config.MIN_NUM_SPEAKERS:
                self.assertEqual(len(errors), 1)
                self.assertIn("At least two speaker names required", errors[0])
            else:
                self.assertEqual(errors, [])
        else:
            self.assertEqual(errors, [])

    def test_speaker_names_empty(self):
        """Test that empty speaker names string adds error."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="")
        errors = []
        cli._validate_speaker_config(args, errors)
        # Empty string splits to empty list after filtering, which has length 0
        # Validation only checks if speaker_names is truthy and has < 2 names
        if args.speaker_names:
            names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
            if len(names) < config.MIN_NUM_SPEAKERS:
                self.assertEqual(len(errors), 1)
                self.assertIn("At least two speaker names required", errors[0])
            else:
                self.assertEqual(errors, [])
        else:
            # Empty string might be falsy, so no validation
            self.assertEqual(errors, [])

    def test_speaker_names_multiple_names(self):
        """Test that multiple speaker names pass validation."""
        args = Namespace(screenplay=False, num_speakers=2, speaker_names="Alice, Bob, Charlie")
        errors = []
        cli._validate_speaker_config(args, errors)
        self.assertEqual(errors, [])


class TestValidateWorkersConfig(unittest.TestCase):
    """Test _validate_workers_config function."""

    def test_valid_workers(self):
        """Test that valid workers count passes validation."""
        args = Namespace(workers=4)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(errors, [])

    def test_workers_zero(self):
        """Test that workers=0 adds error."""
        args = Namespace(workers=0)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--workers must be at least 1", errors[0])

    def test_workers_negative(self):
        """Test that negative workers adds error."""
        args = Namespace(workers=-1)
        errors = []
        cli._validate_workers_config(args, errors)
        self.assertEqual(len(errors), 1)
        self.assertIn("--workers must be at least 1", errors[0])


class TestValidateArgs(unittest.TestCase):
    """Test validate_args function."""

    def test_valid_args(self):
        """Test that valid arguments pass validation."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are collected."""
        args = Namespace(
            rss="",  # Empty RSS
            max_episodes=-1,  # Invalid
            timeout=0,  # Invalid
            delay_ms=-1,  # Invalid
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=0,  # Invalid
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        error_msg = str(cm.exception)
        self.assertIn("Invalid input parameters", error_msg)
        # Should contain multiple errors
        self.assertIn("At least one RSS URL is required", error_msg)
        self.assertIn("--max-episodes must be positive", error_msg)
        self.assertIn("--timeout must be positive", error_msg)
        self.assertIn("--delay-ms must be non-negative", error_msg)
        self.assertIn("--workers must be at least 1", error_msg)

    def test_invalid_output_dir(self):
        """Test that invalid output directory adds error."""
        # Empty output_dir might be handled differently - validate_args only checks
        # if output_dir is provided. If it's empty string, it might not trigger validation.
        # Let's test with a clearly invalid path instead
        args2 = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir="\x00invalid",  # Invalid path with null byte
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args2)
        error_msg = str(cm.exception)
        self.assertIn("Invalid input parameters", error_msg)

    def test_max_episodes_none(self):
        """Test that max_episodes=None is valid."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=None,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)

    def test_max_episodes_zero(self):
        """Test that max_episodes=0 adds error."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=0,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        error_msg = str(cm.exception)
        self.assertIn("--max-episodes must be positive", error_msg)

    def test_episode_offset_negative(self):
        """Test that negative --episode-offset adds error (GitHub #521)."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            episode_offset=-1,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--episode-offset must be non-negative", str(cm.exception))

    def test_delay_ms_zero(self):
        """Test that delay_ms=0 is valid."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        # Should not raise
        cli.validate_args(args)

    def test_multi_feed_requires_output_dir(self):
        """Two or more feeds require --output-dir as corpus parent (GitHub #440)."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            rss_extra=["https://other.example/feed.xml"],
            rss_file=None,
            rss_urls=None,
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("Multi-feed mode", str(cm.exception))

    def test_multi_feed_ok_with_output_dir(self):
        """Two feeds with output_dir pass validation."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            rss_extra=["https://other.example/feed.xml"],
            rss_file=None,
            rss_urls=None,
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir="./corpus_out",
        )
        cli.validate_args(args)

    def test_validate_args_download_resilience_flags_out_of_range(self):
        """CLI download-resilience flags must stay within Config bounds."""
        base = dict(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
        )
        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, http_retry_total=21))
        self.assertIn("--http-retry-total must be between 0 and 20", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, http_backoff_factor=10.01))
        self.assertIn("--http-backoff-factor must be between 0.0 and 10.0", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, episode_retry_max=11))
        self.assertIn("--episode-retry-max must be between 0 and 10", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, episode_retry_delay_sec=121.0))
        self.assertIn("--episode-retry-delay-sec must be between 0.0 and 120.0", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, rss_retry_total=21))
        self.assertIn("--rss-retry-total must be between 0 and 20", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, rss_backoff_factor=10.01))
        self.assertIn("--rss-backoff-factor must be between 0.0 and 10.0", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, host_request_interval_ms=600_001))
        self.assertIn("--host-request-interval-ms must be between 0 and 600000", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(Namespace(**base, circuit_breaker_scope="podcast"))
        self.assertIn("--circuit-breaker-scope must be feed or host", str(cm.exception))


class TestCollectFeedUrls(unittest.TestCase):
    """Tests for collect_feed_urls (GitHub #440)."""

    def test_dedupes_urls(self):
        """Duplicate URLs are merged preserving order."""
        args = Namespace(
            rss="https://example.com/a.xml",
            rss_extra=["https://example.com/b.xml", "https://example.com/a.xml"],
            rss_urls=None,
            rss_file=None,
        )
        got = cli.collect_feed_urls(args)
        self.assertEqual(
            got,
            ["https://example.com/a.xml", "https://example.com/b.xml"],
        )

    def test_merges_rss_urls_from_config_defaults(self):
        """Config merge can supply rss_urls; collect merges with CLI positional."""
        args = Namespace(
            rss="https://example.com/a.xml",
            rss_extra=[],
            rss_urls=["https://example.com/b.xml"],
            rss_file=None,
        )
        got = cli.collect_feed_urls(args)
        self.assertEqual(
            got,
            ["https://example.com/a.xml", "https://example.com/b.xml"],
        )


class TestLoadRssUrlsFromFile(unittest.TestCase):
    """Tests for _load_rss_urls_from_file (GitHub #440)."""

    def test_skips_comments_and_blank_lines(self):
        """Blank lines and # comments are ignored."""
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("# first line comment\n\nhttps://a.example/feed.xml\n")
            path = fh.name
        try:
            got = cli._load_rss_urls_from_file(path)
            self.assertEqual(got, ["https://a.example/feed.xml"])
        finally:
            os.unlink(path)

    def test_missing_file_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            cli._load_rss_urls_from_file("/nonexistent/rss_file_440.txt")
        self.assertIn("not a readable file", str(ctx.exception).lower())


class TestParseArgsMultiFeed440(unittest.TestCase):
    """parse_args + validation for multi-feed CLI (GitHub #440)."""

    def test_parse_args_collects_two_flags_and_requires_output_dir(self):
        with tempfile.TemporaryDirectory() as corpus:
            args = cli.parse_args(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                ]
            )
            urls = cli.collect_feed_urls(args)
            self.assertEqual(len(urls), 2)

    def test_parse_args_rejects_two_feeds_without_output_dir(self):
        with self.assertRaises(ValueError) as ctx:
            cli.parse_args(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                ]
            )
        self.assertIn("Multi-feed mode", str(ctx.exception))

    def test_parse_args_rss_file_merges_with_positional(self):
        """``--rss-file`` lines merge with positional URL (GitHub #440)."""
        with tempfile.TemporaryDirectory() as corpus:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".txt", delete=False, encoding="utf-8"
            ) as fh:
                fh.write("# comment\n\nhttps://b.example/feed.xml\n")
                fpath = fh.name
            try:
                args = cli.parse_args(
                    [
                        "https://a.example/feed.xml",
                        "--rss-file",
                        fpath,
                        "--output-dir",
                        corpus,
                    ]
                )
                urls = cli.collect_feed_urls(args)
                self.assertEqual(
                    urls,
                    ["https://a.example/feed.xml", "https://b.example/feed.xml"],
                )
            finally:
                os.unlink(fpath)


class TestMainMultiFeed440(unittest.TestCase):
    """cli.main multi-feed orchestration with injected pipeline (GitHub #440)."""

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_invokes_pipeline_once_per_feed_distinct_output_dirs(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        recorded: list[tuple[str, str]] = []

        def fake_run(cfg: config.Config) -> tuple[int, str]:
            assert cfg.rss_url is not None
            assert cfg.output_dir is not None
            recorded.append((cfg.rss_url, cfg.output_dir))
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            code = cli.main(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                ],
                run_pipeline_fn=fake_run,
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(recorded), 2)
        urls = {r[0] for r in recorded}
        self.assertEqual(
            urls,
            {"https://a.example/feed.xml", "https://b.example/feed.xml"},
        )
        for _url, out in recorded:
            self.assertIn(f"{os.sep}feeds{os.sep}", out)

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_multi_feed_returns_1_when_one_feed_pipeline_raises(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        def fake_run(cfg: config.Config) -> tuple[int, str]:
            if "b.example" in (cfg.rss_url or ""):
                raise RuntimeError("simulated feed failure")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            code = cli.main(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                ],
                run_pipeline_fn=fake_run,
            )
        self.assertEqual(code, 1)

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_multi_feed_default_exit_zero_when_only_soft_failures(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        def fake_run(cfg: config.Config) -> tuple[int, str]:
            if cfg.rss_url and "b.example" in cfg.rss_url:
                raise ValueError("Failed to fetch RSS feed.")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            code = cli.main(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                ],
                run_pipeline_fn=fake_run,
            )
        self.assertEqual(code, 0)

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_multi_feed_strict_exit_1_when_only_soft_failures(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        def fake_run(cfg: config.Config) -> tuple[int, str]:
            if cfg.rss_url and "b.example" in cfg.rss_url:
                raise ValueError("Failed to fetch RSS feed.")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            code = cli.main(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                    "--multi-feed-strict",
                ],
                run_pipeline_fn=fake_run,
            )
        self.assertEqual(code, 1)

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_multi_feed_lenient_still_fails_on_hard_failure(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        def fake_run(cfg: config.Config) -> tuple[int, str]:
            if cfg.rss_url and "b.example" in cfg.rss_url:
                raise RuntimeError("hard failure")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as corpus:
            code = cli.main(
                [
                    "https://a.example/feed.xml",
                    "--rss",
                    "https://b.example/feed.xml",
                    "--output-dir",
                    corpus,
                    "--max-episodes",
                    "1",
                ],
                run_pipeline_fn=fake_run,
            )
        self.assertEqual(code, 1)


class TestCorpusStatus506(unittest.TestCase):
    """corpus-status subcommand (GitHub #506)."""

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_corpus_status_text(self, _mock_py: object, _mock_ff: object) -> None:
        with tempfile.TemporaryDirectory() as corpus:
            feeds = os.path.join(corpus, "feeds", "f1", "metadata")
            os.makedirs(feeds)
            with open(os.path.join(feeds, "x.metadata.json"), "w", encoding="utf-8") as fh:
                fh.write("{}")
            code = cli.main(["corpus-status", "--output-dir", corpus])
        self.assertEqual(code, 0)

    def test_parse_corpus_status_json_format(self) -> None:
        args = cli.parse_args(["corpus-status", "--output-dir", "/tmp/corpus", "--format", "json"])
        self.assertEqual(args.command, "corpus-status")
        self.assertEqual(args.corpus_status_format, "json")

    @patch.object(cli, "_validate_ffmpeg")
    @patch.object(cli, "_validate_python_version")
    def test_main_corpus_status_json_prints_payload(
        self, _mock_py: object, _mock_ff: object
    ) -> None:
        import json
        from io import StringIO

        with tempfile.TemporaryDirectory() as corpus:
            feeds = os.path.join(corpus, "feeds", "f1", "metadata")
            os.makedirs(feeds)
            with patch("sys.stdout", new=StringIO()) as buf:
                code = cli.main(
                    ["corpus-status", "--output-dir", corpus, "--format", "json"],
                )
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertIn("corpus_parent", payload)
            self.assertIn("feeds_subdirs", payload)


class TestBuildConfig(unittest.TestCase):
    """Test _build_config function."""

    def test_build_config_from_args(self):
        """Test that Config is built correctly from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir="./output",
            max_episodes=10,
            user_agent="test-agent",
            timeout=30,
            delay_ms=100,
            prefer_type=["vtt", "srt"],
            transcribe_missing=True,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=True,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names="Alice, Bob",
            run_id="test_run",
            skip_existing=False,  # Cannot be True with clean_output
            reuse_media=False,
            clean_output=True,
            dry_run=False,
            generate_metadata=True,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=True,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level="DEBUG",
            log_file=None,
            language="en",
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=4,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
            gi_qa_score_min=0.11,
            gi_nli_entailment_min=0.41,
            gi_embedding_model="minilm-l6",
        )
        cfg = cli._build_config(args)
        self.assertIsInstance(cfg, config.Config)
        self.assertEqual(cfg.rss_url, "https://example.com/feed.xml")
        # output_dir is derived and normalized to absolute path
        self.assertIsNotNone(cfg.output_dir)
        self.assertIn("output", cfg.output_dir)  # Should contain "output" somewhere
        self.assertEqual(cfg.max_episodes, 10)
        self.assertEqual(cfg.user_agent, "test-agent")
        self.assertEqual(cfg.timeout, 30)
        self.assertEqual(cfg.delay_ms, 100)
        # prefer_type from args becomes prefer_types in Config
        # The actual value might be normalized, so just check it's a list
        self.assertIsInstance(cfg.prefer_types, list)
        self.assertIn("vtt", cfg.prefer_types)
        self.assertIn("srt", cfg.prefer_types)
        self.assertTrue(cfg.transcribe_missing)
        self.assertEqual(cfg.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)
        self.assertTrue(cfg.screenplay)
        self.assertEqual(cfg.screenplay_gap_s, 2.0)
        self.assertEqual(cfg.screenplay_num_speakers, 2)
        # speaker_names is split and stripped
        self.assertEqual(len(cfg.screenplay_speaker_names), 2)
        self.assertIn("Alice", cfg.screenplay_speaker_names)
        self.assertIn("Bob", cfg.screenplay_speaker_names)
        self.assertEqual(cfg.run_id, "test_run")
        # skip_existing and clean_output are mutually exclusive
        self.assertFalse(cfg.skip_existing)  # Set to False because clean_output=True
        self.assertTrue(cfg.clean_output)
        self.assertFalse(cfg.dry_run)
        self.assertTrue(cfg.generate_metadata)
        self.assertTrue(cfg.generate_summaries)
        self.assertEqual(cfg.log_level, "DEBUG")
        self.assertIsNone(cfg.log_file)
        self.assertEqual(cfg.language, "en")
        self.assertEqual(cfg.workers, 4)
        self.assertAlmostEqual(cfg.gi_qa_score_min, 0.11)
        self.assertAlmostEqual(cfg.gi_nli_entailment_min, 0.41)
        self.assertEqual(cfg.gi_embedding_model, "minilm-l6")

    def test_build_config_hybrid_internal_preprocessing_after_pattern(self):
        """Issue #419: CLI flag maps into Config.hybrid_internal_preprocessing_after_pattern."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--summary-provider",
                "hybrid_ml",
                "--transcript-cleaning-strategy",
                "pattern",
                "--hybrid-internal-preprocessing-after-pattern",
                "cleaning_none",
            ]
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.summary_provider, "hybrid_ml")
        self.assertEqual(cfg.transcript_cleaning_strategy, "pattern")
        self.assertEqual(cfg.hybrid_internal_preprocessing_after_pattern, "cleaning_none")

    def test_build_config_download_resilience_cli_overrides(self):
        """Six download-resilience CLI flags map into Config when set."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--http-retry-total",
                "5",
                "--http-backoff-factor",
                "1.25",
                "--rss-retry-total",
                "6",
                "--rss-backoff-factor",
                "0.5",
                "--episode-retry-max",
                "2",
                "--episode-retry-delay-sec",
                "7.5",
            ]
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.http_retry_total, 5)
        self.assertAlmostEqual(cfg.http_backoff_factor, 1.25)
        self.assertEqual(cfg.rss_retry_total, 6)
        self.assertAlmostEqual(cfg.rss_backoff_factor, 0.5)
        self.assertEqual(cfg.episode_retry_max, 2)
        self.assertAlmostEqual(cfg.episode_retry_delay_sec, 7.5)

    def test_build_config_issue522_cli_overrides(self) -> None:
        """Issue #522: fair HTTP flags map into Config."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--host-request-interval-ms",
                "50",
                "--host-max-concurrent",
                "3",
                "--circuit-breaker",
                "--circuit-breaker-failure-threshold",
                "4",
                "--circuit-breaker-scope",
                "host",
                "--rss-conditional-get",
                "--rss-cache-dir",
                "/tmp/rss_cond_test",
            ]
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.host_request_interval_ms, 50)
        self.assertEqual(cfg.host_max_concurrent, 3)
        self.assertTrue(cfg.circuit_breaker_enabled)
        self.assertEqual(cfg.circuit_breaker_failure_threshold, 4)
        self.assertEqual(cfg.circuit_breaker_scope, "host")
        self.assertTrue(cfg.rss_conditional_get)
        self.assertEqual(cfg.rss_cache_dir, "/tmp/rss_cond_test")

    def test_build_config_no_circuit_breaker_flag(self) -> None:
        """--no-circuit-breaker forces circuit_breaker_enabled False."""
        args = cli.parse_args(["https://example.com/feed.xml", "--no-circuit-breaker"])
        cfg = cli._build_config(args)
        self.assertFalse(cfg.circuit_breaker_enabled)

    def test_build_config_no_rss_conditional_flag(self) -> None:
        """--no-rss-conditional-get forces rss_conditional_get False."""
        args = cli.parse_args(["https://example.com/feed.xml", "--no-rss-conditional-get"])
        cfg = cli._build_config(args)
        self.assertFalse(cfg.rss_conditional_get)

    def test_parse_args_circuit_breaker_mutex(self) -> None:
        """--circuit-breaker and --no-circuit-breaker are mutually exclusive."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--circuit-breaker",
                    "--no-circuit-breaker",
                ]
            )

    def test_parse_args_rss_conditional_mutex(self) -> None:
        """--rss-conditional-get and --no-rss-conditional-get are mutually exclusive."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--rss-conditional-get",
                    "--no-rss-conditional-get",
                ]
            )

    def test_build_config_with_defaults(self):
        """Test that Config uses defaults when args are not provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertIsInstance(cfg, config.Config)
        # Check that defaults are used
        # output_dir is derived from rss_url when None, so it won't be None
        self.assertIsNotNone(cfg.output_dir)  # Derived from RSS URL
        self.assertIsNone(cfg.max_episodes)
        self.assertFalse(cfg.transcribe_missing)
        self.assertFalse(cfg.screenplay)
        # GIL tuning keys omitted from Namespace → Config field defaults
        self.assertAlmostEqual(cfg.gi_qa_score_min, 0.3)
        self.assertAlmostEqual(cfg.gi_nli_entailment_min, 0.5)
        self.assertFalse(cfg.append)

    def test_build_config_append_true_when_flag_set(self):
        """CLI --append maps to Config.append (GitHub #444)."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            append=True,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertTrue(cfg.append)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_transcription_provider(self):
        """Test that _build_config includes transcription_provider from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="openai",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.transcription_provider, "openai")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_speaker_detector_provider(self):
        """Test that _build_config includes speaker_detector_provider from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="openai",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.speaker_detector_provider, "openai")

    def test_build_config_with_openai_api_base(self):
        """Test that _build_config includes openai_api_base from args."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base="http://localhost:8000/v1",
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_api_base, "http://localhost:8000/v1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_loads_openai_api_key_from_env(self):
        """Test that _build_config loads openai_api_key from environment."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        # The Config object should load the key from environment via field validator
        self.assertEqual(cfg.openai_api_key, "sk-test123")

    @patch.dict(os.environ, {}, clear=True)
    def test_build_config_openai_api_key_none_when_not_in_env(self):
        """Test that _build_config sets openai_api_key to None when not in environment."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        # The Config object should have None when env var is not set
        self.assertIsNone(cfg.openai_api_key)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_transcription_model(self):
        """Test that _build_config includes openai_transcription_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="openai",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model="whisper-1",
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_transcription_model, "whisper-1")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_speaker_model(self):
        """Test that _build_config includes openai_speaker_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="openai",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model="gpt-4o",
            openai_summary_model=None,
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_speaker_model, "gpt-4o")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_summary_model(self):
        """Test that _build_config includes openai_summary_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=True,  # Required when generate_summaries=True
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=True,
            metrics_output=None,
            summary_provider="openai",
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model="gpt-4o-mini",
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_summary_model, "gpt-4o-mini")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_insight_model(self):
        """Test that _build_config includes openai_insight_model when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=True,  # Required when generate_summaries=True
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=True,
            metrics_output=None,
            summary_provider="openai",
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model="gpt-4o-mini",
            openai_insight_model="gpt-4o",
            openai_temperature=None,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_insight_model, "gpt-4o")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"})
    def test_build_config_with_openai_temperature(self):
        """Test that _build_config includes openai_temperature when provided."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=0.7,
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.openai_temperature, 0.7)

    def test_build_config_vector_flags_from_cli_namespace(self):
        """Vector / semantic corpus fields from argparse reach Config."""
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
            vector_index_path="runs/search_idx",
            vector_embedding_model="minilm-l6",
            vector_chunk_size_tokens=400,
            vector_chunk_overlap_tokens=40,
            vector_backend="faiss",
            vector_faiss_index_mode="ivf_flat",
            vector_search=True,
            vector_index_types="insight, kg_entity",
        )
        cfg = cli._build_config(args)
        self.assertTrue(cfg.vector_search)
        self.assertEqual(cfg.vector_backend, "faiss")
        self.assertEqual(cfg.vector_faiss_index_mode, "ivf_flat")
        self.assertEqual(cfg.vector_chunk_size_tokens, 400)
        self.assertEqual(cfg.vector_chunk_overlap_tokens, 40)
        self.assertIn("search_idx", (cfg.vector_index_path or ""))
        self.assertEqual(cfg.vector_index_types, ["insight", "kg_entity"])

    def test_build_config_vector_index_types_as_list_and_empty_comma_string(self):
        args = Namespace(
            rss="https://example.com/feed.xml",
            output_dir=None,
            max_episodes=None,
            user_agent=None,
            timeout=30,
            delay_ms=0,
            prefer_type=[],
            transcribe_missing=False,
            transcription_provider="whisper",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            whisper_device=None,
            screenplay=False,
            screenplay_gap=2.0,
            num_speakers=2,
            speaker_names=None,
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level=None,
            log_file=None,
            language=None,
            ner_model=None,
            speaker_detector_provider="spacy",
            auto_speakers=False,
            cache_detected_hosts=False,
            workers=1,
            openai_api_base=None,
            openai_transcription_model=None,
            openai_speaker_model=None,
            openai_summary_model=None,
            openai_temperature=None,
            vector_search=False,
            vector_index_types=["quote", "summary"],
        )
        cfg = cli._build_config(args)
        self.assertEqual(cfg.vector_index_types, ["quote", "summary"])

        args2 = copy.copy(args)
        args2.vector_index_types = "  , ,  "
        cfg2 = cli._build_config(args2)
        self.assertIsNone(cfg2.vector_index_types)


class TestParseArgs(unittest.TestCase):
    """Test parse_args function."""

    def test_parse_args_basic(self):
        """Test parsing basic arguments."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.rss, "https://example.com/feed.xml")
        self.assertIsNotNone(args)

    def test_parse_args_with_options(self):
        """Test parsing arguments with various options."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--max-episodes",
                "10",
                "--timeout",
                "60",
                "--transcribe-missing",
                "--whisper-model",
                config.TEST_DEFAULT_WHISPER_MODEL,
            ]
        )
        self.assertEqual(args.rss, "https://example.com/feed.xml")
        self.assertEqual(args.max_episodes, 10)
        self.assertEqual(args.timeout, 60)
        self.assertTrue(args.transcribe_missing)
        # CLI should preserve the value passed by user, not override with test default
        self.assertEqual(args.whisper_model, config.TEST_DEFAULT_WHISPER_MODEL)

    def test_parse_args_with_transcription_provider(self):
        """Test parsing --transcription-provider argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--transcription-provider",
                "openai",
            ]
        )
        self.assertEqual(args.transcription_provider, "openai")

    def test_parse_args_with_speaker_detector_provider(self):
        """Test parsing --speaker-detector-provider argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--speaker-detector-provider",
                "openai",
            ]
        )
        self.assertEqual(args.speaker_detector_provider, "openai")

    def test_parse_args_with_openai_api_base(self):
        """Test parsing --openai-api-base argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-api-base",
                "http://localhost:8000/v1",
            ]
        )
        self.assertEqual(args.openai_api_base, "http://localhost:8000/v1")

    def test_parse_args_transcription_provider_default(self):
        """Test that --transcription-provider defaults to 'whisper'."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.transcription_provider, "whisper")

    def test_parse_args_speaker_detector_provider_default(self):
        """Test that --speaker-detector-provider defaults to 'spacy'."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertEqual(args.speaker_detector_provider, "spacy")

    def test_parse_args_openai_api_base_default(self):
        """Test that --openai-api-base defaults to None."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertIsNone(args.openai_api_base)

    def test_parse_args_hybrid_internal_preprocessing_after_pattern(self):
        """Issue #419: --hybrid-internal-preprocessing-after-pattern is parsed."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--summary-provider",
                "hybrid_ml",
                "--transcript-cleaning-strategy",
                "pattern",
                "--hybrid-internal-preprocessing-after-pattern",
                "cleaning_none",
            ]
        )
        self.assertEqual(args.summary_provider, "hybrid_ml")
        self.assertEqual(args.transcript_cleaning_strategy, "pattern")
        self.assertEqual(args.hybrid_internal_preprocessing_after_pattern, "cleaning_none")

    def test_parse_args_with_openai_transcription_model(self):
        """Test parsing --openai-transcription-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-transcription-model",
                "whisper-1",
            ]
        )
        self.assertEqual(args.openai_transcription_model, "whisper-1")

    def test_parse_args_with_openai_speaker_model(self):
        """Test parsing --openai-speaker-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-speaker-model",
                "gpt-4o",
            ]
        )
        self.assertEqual(args.openai_speaker_model, "gpt-4o")

    def test_parse_args_with_openai_summary_model(self):
        """Test parsing --openai-summary-model argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-summary-model",
                "gpt-4o-mini",
            ]
        )
        self.assertEqual(args.openai_summary_model, "gpt-4o-mini")

    def test_parse_args_with_openai_temperature(self):
        """Test parsing --openai-temperature argument."""
        args = cli.parse_args(
            [
                "https://example.com/feed.xml",
                "--openai-temperature",
                "0.7",
            ]
        )
        self.assertEqual(args.openai_temperature, 0.7)

    def test_parse_args_openai_model_defaults(self):
        """Test that OpenAI model arguments default to None."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertIsNone(args.openai_transcription_model)
        self.assertIsNone(args.openai_speaker_model)
        self.assertIsNone(args.openai_summary_model)
        self.assertIsNone(args.openai_insight_model)
        self.assertIsNone(args.openai_temperature)

    def test_parse_args_invalid_transcription_provider(self):
        """Test that invalid --transcription-provider raises error."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--transcription-provider",
                    "invalid",
                ]
            )

    def test_parse_args_invalid_speaker_detector_provider(self):
        """Test that invalid --speaker-detector-provider raises error."""
        with self.assertRaises(SystemExit):
            cli.parse_args(
                [
                    "https://example.com/feed.xml",
                    "--speaker-detector-provider",
                    "invalid",
                ]
            )

    @patch("builtins.print")
    def test_parse_args_version(self, mock_print):
        """Test that --version flag prints version and exits."""
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)
        mock_print.assert_called_once()

    def test_parse_args_validates_args(self):
        """Test that parse_args validates arguments."""
        with self.assertRaises(ValueError):
            cli.parse_args(["", "--workers", "0"])  # Invalid RSS URL and workers

    @patch("podcast_scraper.cli._load_and_merge_config")
    def test_parse_args_with_config_file(self, mock_load_config):
        """Test parsing arguments with config file."""
        mock_args = Namespace(
            rss="https://example.com/feed.xml",
            max_episodes=10,
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Test default: tiny.en
            whisper_device=None,
            screenplay=False,
            screenplay_gap=1.25,
            num_speakers=2,
            speaker_names=None,
            workers=4,
            output_dir=None,
            prefer_type=[],
            user_agent="test",
            run_id=None,
            skip_existing=False,
            reuse_media=False,
            clean_output=False,
            dry_run=False,
            generate_metadata=False,
            metadata_format="json",
            metadata_subdirectory=None,
            generate_summaries=False,
            metrics_output=None,
            summary_provider=None,
            summary_model=None,
            summary_reduce_model=None,
            summary_device=None,
            summary_chunk_size=None,
            summary_prompt=None,
            save_cleaned_transcript=False,
            log_level="INFO",
            log_file=None,
            language="en",
            ner_model=None,
            auto_speakers=False,
            cache_detected_hosts=False,
            version=False,
            config="config.yaml",
        )
        mock_load_config.return_value = mock_args

        args = cli.parse_args(["--config", "config.yaml", "https://example.com/feed.xml"])

        mock_load_config.assert_called_once()
        self.assertIsNotNone(args)


class TestAddArgumentGroups(unittest.TestCase):
    """Test argument group addition functions."""

    def test_add_common_arguments(self):
        """Test that _add_common_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        # Check that key arguments are present
        self.assertTrue(parser._actions[0].dest in ["help", "rss", "config"])
        # Find rss argument
        rss_action = next((a for a in parser._actions if a.dest == "rss"), None)
        self.assertIsNotNone(rss_action)

    def test_add_transcription_arguments(self):
        """Test that _add_transcription_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_transcription_arguments(parser)

        # Check that transcription arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("transcribe_missing", action_dests)
        self.assertIn("transcription_provider", action_dests)
        self.assertIn("whisper_model", action_dests)
        self.assertIn("screenplay", action_dests)

        # Check that transcription_provider has correct choices
        transcription_action = next(
            (a for a in parser._actions if a.dest == "transcription_provider"), None
        )
        self.assertIsNotNone(transcription_action)
        self.assertEqual(transcription_action.choices, ["whisper", "openai", "gemini", "mistral"])
        self.assertEqual(transcription_action.default, "whisper")

    def test_add_mistral_arguments(self):
        """Test that _add_mistral_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_mistral_arguments(parser)

        # Check that Mistral arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("mistral_api_key", action_dests)
        self.assertIn("mistral_api_base", action_dests)
        self.assertIn("mistral_transcription_model", action_dests)
        self.assertIn("mistral_speaker_model", action_dests)
        self.assertIn("mistral_summary_model", action_dests)
        self.assertIn("mistral_temperature", action_dests)
        self.assertIn("mistral_max_tokens", action_dests)
        self.assertIn("mistral_cleaning_model", action_dests)
        self.assertIn("mistral_cleaning_temperature", action_dests)

    def test_add_deepseek_arguments(self):
        """Test that _add_deepseek_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_deepseek_arguments(parser)

        # Check that DeepSeek arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("deepseek_api_key", action_dests)
        self.assertIn("deepseek_api_base", action_dests)
        self.assertIn("deepseek_speaker_model", action_dests)
        self.assertIn("deepseek_summary_model", action_dests)
        self.assertIn("deepseek_temperature", action_dests)
        self.assertIn("deepseek_max_tokens", action_dests)
        self.assertIn("deepseek_cleaning_model", action_dests)
        self.assertIn("deepseek_cleaning_temperature", action_dests)

    def test_add_grok_arguments(self):
        """Test that _add_grok_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_grok_arguments(parser)

        # Check that Grok arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("grok_api_key", action_dests)
        self.assertIn("grok_api_base", action_dests)
        self.assertIn("grok_speaker_model", action_dests)
        self.assertIn("grok_summary_model", action_dests)
        self.assertIn("grok_temperature", action_dests)
        self.assertIn("grok_max_tokens", action_dests)
        self.assertIn("grok_cleaning_model", action_dests)
        self.assertIn("grok_cleaning_temperature", action_dests)

    def test_add_ollama_arguments(self):
        """Test that _add_ollama_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_ollama_arguments(parser)

        # Check that Ollama arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("ollama_api_base", action_dests)
        self.assertIn("ollama_speaker_model", action_dests)
        self.assertIn("ollama_summary_model", action_dests)
        self.assertIn("ollama_temperature", action_dests)
        self.assertIn("ollama_max_tokens", action_dests)
        self.assertIn("ollama_timeout", action_dests)
        self.assertIn("ollama_cleaning_model", action_dests)
        self.assertIn("ollama_cleaning_temperature", action_dests)
        # Ollama doesn't have API key (local service)
        self.assertNotIn("ollama_api_key", action_dests)

    def test_add_metadata_arguments(self):
        """Test that _add_metadata_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_metadata_arguments(parser)

        # Check that metadata arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("generate_metadata", action_dests)
        self.assertIn("download_podcast_artwork", action_dests)
        self.assertIn("metadata_format", action_dests)

    def test_add_speaker_detection_arguments(self):
        """Test that _add_speaker_detection_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_speaker_detection_arguments(parser)

        # Check that speaker detection arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("language", action_dests)
        self.assertIn("speaker_detector_provider", action_dests)
        self.assertIn("auto_speakers", action_dests)

        # Check that speaker_detector_provider has correct choices
        speaker_detector_action = next(
            (a for a in parser._actions if a.dest == "speaker_detector_provider"), None
        )
        self.assertIsNotNone(speaker_detector_action)
        self.assertEqual(
            speaker_detector_action.choices,
            ["spacy", "openai", "gemini", "anthropic", "mistral", "grok", "deepseek", "ollama"],
        )
        self.assertEqual(speaker_detector_action.default, "spacy")

    def test_add_summarization_arguments(self):
        """Test that _add_summarization_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_summarization_arguments(parser)

        # Check that summarization arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("generate_summaries", action_dests)
        self.assertIn("summary_provider", action_dests)

    def test_add_cache_arguments(self):
        """Test that _add_cache_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_cache_arguments(parser)

        # Check that cache arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("cache_info", action_dests)
        self.assertIn("prune_cache", action_dests)
        self.assertIn("cache_dir", action_dests)

    def test_add_openai_arguments(self):
        """Test that _add_openai_arguments adds expected arguments."""
        parser = argparse.ArgumentParser()
        cli._add_openai_arguments(parser)

        # Check that OpenAI arguments are present
        action_dests = [a.dest for a in parser._actions]
        self.assertIn("openai_api_key", action_dests)
        self.assertIn("openai_api_base", action_dests)
        self.assertIn("openai_transcription_model", action_dests)
        self.assertIn("openai_speaker_model", action_dests)
        self.assertIn("openai_summary_model", action_dests)
        self.assertIn("openai_insight_model", action_dests)
        self.assertIn("openai_temperature", action_dests)
        self.assertIn("openai_max_tokens", action_dests)
        self.assertIn("openai_cleaning_model", action_dests)
        self.assertIn("openai_cleaning_temperature", action_dests)

        # Check that openai_api_base has correct default
        openai_action = next((a for a in parser._actions if a.dest == "openai_api_base"), None)
        self.assertIsNotNone(openai_action)
        self.assertIsNone(openai_action.default)

        # Check that model arguments have correct defaults (None, uses config defaults)
        for model_arg in [
            "openai_transcription_model",
            "openai_speaker_model",
            "openai_summary_model",
            "openai_insight_model",
            "openai_max_tokens",
            "openai_cleaning_model",
            "openai_cleaning_temperature",
        ]:
            action = next((a for a in parser._actions if a.dest == model_arg), None)
            self.assertIsNotNone(action, f"{model_arg} should be present")
            self.assertIsNone(action.default, f"{model_arg} default should be None")


class TestCacheSubcommand(unittest.TestCase):
    """Test cache subcommand parsing and execution."""

    def test_parse_cache_args_status(self):
        """Test parsing cache --status command."""
        args = cli._parse_cache_args(["--status"])
        self.assertTrue(args.status)
        self.assertFalse(args.clean)

    def test_parse_cache_args_clean_all(self):
        """Test parsing cache --clean all command."""
        args = cli._parse_cache_args(["--clean", "all"])
        self.assertFalse(args.status)
        self.assertEqual(args.clean, "all")

    def test_parse_cache_args_clean_whisper(self):
        """Test parsing cache --clean whisper command."""
        args = cli._parse_cache_args(["--clean", "whisper"])
        self.assertEqual(args.clean, "whisper")

    def test_parse_cache_args_clean_with_yes(self):
        """Test parsing cache --clean with --yes flag."""
        args = cli._parse_cache_args(["--clean", "all", "--yes"])
        self.assertEqual(args.clean, "all")
        self.assertTrue(args.yes)

    def test_parse_args_detects_cache_subcommand(self):
        """Test that parse_args detects cache subcommand."""
        args = cli.parse_args(["cache", "--status"])
        self.assertTrue(hasattr(args, "command"))
        self.assertEqual(args.command, "cache")
        self.assertTrue(args.status)

    def test_parse_args_normal_command_not_cache(self):
        """Test that normal command is not treated as cache."""
        args = cli.parse_args(["https://example.com/feed.xml"])
        self.assertFalse(hasattr(args, "command") and args.command == "cache")

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status(self, mock_get_info, mock_format_size):
        """Test main() with cache --status command."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {"dir": Path("/whisper"), "size": 100, "count": 1, "models": []},
            "transformers": {"dir": Path("/transformers"), "size": 200, "count": 2, "models": []},
            "spacy": {"dir": Path("/spacy"), "size": 50, "count": 1, "models": []},
            "total_size": 350,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)
        mock_get_info.assert_called_once()

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_all_caches")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_clean_all(self, mock_get_info, mock_clean, mock_format_size):
        """Test main() with cache --clean all --yes command."""
        mock_get_info.return_value = {"total_size": 350}
        mock_clean.return_value = {"whisper": (1, 100), "transformers": (2, 200), "spacy": (1, 50)}
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "all", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_whisper_cache")
    def test_main_cache_clean_whisper(self, mock_clean, mock_format_size):
        """Test main() with cache --clean whisper --yes command."""
        mock_clean.return_value = (1, 100)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "whisper", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_transformers_cache")
    def test_main_cache_clean_transformers(self, mock_clean, mock_format_size):
        """Test main() with cache --clean transformers --yes command."""
        mock_clean.return_value = (2, 200)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "transformers", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.clean_spacy_cache")
    def test_main_cache_clean_spacy(self, mock_clean, mock_format_size):
        """Test main() with cache --clean spacy --yes command."""
        mock_clean.return_value = (1, 50)
        mock_format_size.side_effect = lambda x: f"{x} B"  # Simple formatter

        exit_code = cli.main(["cache", "--clean", "spacy", "--yes"])

        self.assertEqual(exit_code, 0)
        mock_clean.assert_called_once_with(confirm=False)

    def test_parse_cache_args_requires_status_or_clean(self):
        """Test that _parse_cache_args requires either --status or --clean."""
        with self.assertRaises(SystemExit):
            cli._parse_cache_args([])

    def test_parse_cache_args_clean_defaults_to_all(self):
        """Test that --clean without argument defaults to 'all'."""
        args = cli._parse_cache_args(["--clean"])
        self.assertEqual(args.clean, "all")

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_with_models(self, mock_get_info, mock_format_size):
        """Test main() with cache --status when models exist."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {
                "dir": Path("/whisper"),
                "size": 100,
                "count": 1,
                "models": [{"name": "base.en.pt", "size": 100}],
            },
            "transformers": {
                "dir": Path("/transformers"),
                "size": 200,
                "count": 1,
                "models": [{"name": "facebook/bart-base", "size": 200}],
            },
            "spacy": {
                "dir": Path("/spacy"),
                "size": 50,
                "count": 1,
                "models": [{"name": "en_core_web_sm", "size": 50}],
            },
            "total_size": 350,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)
        mock_get_info.assert_called_once()

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_spacy_no_dir(self, mock_get_info, mock_format_size):
        """Test main() with cache --status when spacy has no cache dir."""
        from pathlib import Path

        mock_get_info.return_value = {
            "whisper": {"dir": Path("/whisper"), "size": 100, "count": 0, "models": []},
            "transformers": {"dir": Path("/transformers"), "size": 0, "count": 0, "models": []},
            "spacy": {"dir": None, "size": 0, "count": 0, "models": []},
            "total_size": 100,
        }
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])

        self.assertEqual(exit_code, 0)

    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_exception_handling(self, mock_get_info):
        """Test main() handles exceptions from cache operations."""
        # Test exception handling path
        mock_get_info.side_effect = Exception("Cache operation failed")

        exit_code = cli.main(["cache", "--status"])
        self.assertEqual(exit_code, 1)

    @patch("podcast_scraper.cache_manager.format_size")
    @patch("podcast_scraper.cache_manager.get_all_cache_info")
    def test_main_cache_status_exception(self, mock_get_info, mock_format_size):
        """Test main() handles exceptions from cache operations."""
        mock_get_info.side_effect = Exception("Cache operation failed")
        mock_format_size.side_effect = lambda x: f"{x} B"

        exit_code = cli.main(["cache", "--status"])
        self.assertEqual(exit_code, 1)


class TestGiSubcommand(unittest.TestCase):
    """Test gi subcommand parsing and execution (inspect, show-insight)."""

    def test_parse_args_detects_gi_subcommand(self):
        """Test that parse_args detects gi subcommand."""
        args = cli.parse_args(["gi", "inspect", "--episode-path", "/path/to/ep.gi.json"])
        self.assertTrue(hasattr(args, "command"))
        self.assertEqual(args.command, "gi")
        self.assertEqual(args.gi_subcommand, "inspect")
        self.assertEqual(args.episode_path, "/path/to/ep.gi.json")

    def test_parse_args_gi_show_insight(self):
        """Test that parse_args parses gi show-insight."""
        args = cli.parse_args(
            ["gi", "show-insight", "--id", "insight:ep:0", "--output-dir", "/out"]
        )
        self.assertEqual(args.command, "gi")
        self.assertEqual(args.gi_subcommand, "show-insight")
        self.assertEqual(getattr(args, "id"), "insight:ep:0")
        self.assertEqual(args.output_dir, "/out")

    def test_parse_gi_args_validate(self):
        """Test _parse_gi_args for validate subcommand (parity with kg validate)."""
        args = cli._parse_gi_args(["validate", "/tmp/sample.gi.json", "--strict"])
        self.assertEqual(args.gi_subcommand, "validate")
        self.assertEqual(args.paths, ["/tmp/sample.gi.json"])
        self.assertTrue(args.strict)

    def test_parse_gi_args_inspect(self):
        """Test _parse_gi_args for inspect subcommand."""
        args = cli._parse_gi_args(["inspect", "--episode-path", "x.gi.json"])
        self.assertEqual(args.gi_subcommand, "inspect")
        self.assertEqual(args.episode_path, "x.gi.json")
        self.assertTrue(args.stats)

    def test_parse_gi_args_inspect_feed_id(self):
        """inspect accepts --feed-id for multi-feed corpus parents."""
        args = cli._parse_gi_args(
            [
                "inspect",
                "--output-dir",
                "/corpus",
                "--episode-id",
                "ep:x",
                "--feed-id",
                "feed_a",
            ]
        )
        self.assertEqual(args.gi_subcommand, "inspect")
        self.assertEqual(args.feed_id, "feed_a")

    def test_parse_gi_inspect_strict_and_no_stats(self):
        """inspect accepts --strict and --no-stats."""
        args = cli.parse_args(
            ["gi", "inspect", "--episode-path", "/x.gi.json", "--strict", "--no-stats"]
        )
        self.assertTrue(args.strict)
        self.assertFalse(args.stats)

    def test_parse_gi_show_insight_missing_id_exits(self):
        """show-insight requires --id (argparse exit 2)."""
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(["gi", "show-insight", "--output-dir", "/tmp"])
        self.assertEqual(cm.exception.code, 2)

    def test_parse_gi_args_show_insight(self):
        """Test _parse_gi_args for show-insight subcommand."""
        args = cli._parse_gi_args(["show-insight", "--id", "insight:1:0"])
        self.assertEqual(args.gi_subcommand, "show-insight")
        self.assertEqual(getattr(args, "id"), "insight:1:0")

    def test_main_gi_inspect_with_fixture_artifact(self):
        """Test main() with gi inspect on a real artifact file returns 0."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        metadata_dir = tmp / "metadata"
        metadata_dir.mkdir(parents=True)
        artifact = build_artifact("ep:1", "Transcript here.", prompt_version="v1")
        gi_path = metadata_dir / "ep1.gi.json"
        write_artifact(gi_path, artifact, validate=True)
        exit_code = cli.main(["gi", "inspect", "--episode-path", str(gi_path), "--format", "json"])
        self.assertEqual(exit_code, 0)

    def get_tmp_dir(self):
        """Return a temporary directory path (created via tempfile)."""
        import tempfile

        return tempfile.mkdtemp()

    def test_main_gi_show_insight_no_artifact_exits_nonzero(self):
        """Test main() with gi show-insight when artifact not found exits with 1."""
        import tempfile

        # --id required; --output-dir with no matching artifact -> error message, exit 1
        tmp = tempfile.mkdtemp()
        exit_code = cli.main(
            ["gi", "show-insight", "--id", "insight:nonexistent:0", "--output-dir", tmp]
        )
        self.assertEqual(exit_code, 1)

    def test_parse_args_gi_explore(self):
        """Test that parse_args parses gi explore."""
        args = cli.parse_args(["gi", "explore", "--topic", "AI", "--output-dir", "/out"])
        self.assertEqual(args.command, "gi")
        self.assertEqual(args.gi_subcommand, "explore")
        self.assertEqual(args.topic, "AI")
        self.assertEqual(args.output_dir, "/out")

    def test_parse_args_gi_query(self):
        """Test that parse_args parses gi query."""
        args = cli.parse_args(
            [
                "gi",
                "query",
                "--output-dir",
                "/out",
                "--question",
                "What insights about inflation?",
            ]
        )
        self.assertEqual(args.command, "gi")
        self.assertEqual(args.gi_subcommand, "query")
        self.assertEqual(args.output_dir, "/out")
        self.assertEqual(args.question, "What insights about inflation?")
        self.assertEqual(args.query_limit, 20)

    def test_main_gi_query_no_pattern_exits_2(self):
        """gi query with unmatched question exits 2 (invalid args)."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(
            [
                "gi",
                "query",
                "--output-dir",
                str(tmp),
                "--question",
                "Random unmatched phrase xyz.",
            ]
        )
        self.assertEqual(exit_code, 2)

    def test_main_gi_query_topic_pattern_returns_0(self):
        """gi query with UC4 topic pattern runs and exits 0."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(
            [
                "gi",
                "query",
                "--output-dir",
                str(tmp),
                "--question",
                "What insights about stub?",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_main_gi_query_top_topics_returns_0(self):
        """gi query with topic-leaderboard pattern runs and exits 0."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(
            [
                "gi",
                "query",
                "--output-dir",
                str(tmp),
                "--question",
                "Top topics?",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_main_gi_validate_strict_passes(self):
        """gi validate --strict on a valid artifact exits 0."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        gi_path = tmp / "metadata" / "ep1.gi.json"
        write_artifact(gi_path, build_artifact("ep:1", "Transcript.", prompt_version="v1"))
        exit_code = cli.main(["gi", "validate", "--strict", str(tmp)])
        self.assertEqual(exit_code, 0)

    def test_parse_gi_args_export(self):
        """gi export parses --output-dir and --format merged."""
        args = cli._parse_gi_args(
            ["export", "--output-dir", "/out", "--format", "merged", "--out", "/tmp/b.json"]
        )
        self.assertEqual(args.gi_subcommand, "export")
        self.assertEqual(args.output_dir, "/out")
        self.assertEqual(args.format, "merged")
        self.assertEqual(args.out, "/tmp/b.json")

    def test_main_gi_export_merged_writes_bundle(self):
        """gi export --format merged writes gi_corpus_bundle JSON."""
        import json
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        write_artifact(
            tmp / "metadata" / "a.gi.json",
            build_artifact("ep:1", "Hello.", prompt_version="v1"),
            validate=True,
        )
        out_path = tmp / "bundle.json"
        exit_code = cli.main(
            [
                "gi",
                "export",
                "--output-dir",
                str(tmp),
                "--format",
                "merged",
                "--out",
                str(out_path),
            ]
        )
        self.assertEqual(exit_code, 0)
        data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(data.get("export_kind"), "gi_corpus_bundle")
        self.assertEqual(data.get("artifact_count"), 1)

    def test_main_gi_explore_no_artifacts_exits_3(self):
        """Test main() with gi explore when no .gi.json found exits 3."""
        import tempfile

        tmp = tempfile.mkdtemp()
        exit_code = cli.main(["gi", "explore", "--output-dir", tmp])
        self.assertEqual(exit_code, 3)

    def test_main_gi_explore_with_artifact_returns_0(self):
        """Test main() with gi explore on dir with one artifact returns 0."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(["gi", "explore", "--output-dir", str(tmp), "--format", "json"])
        self.assertEqual(exit_code, 0)

    def test_main_gi_explore_topic_no_match_returns_4(self):
        """Test main() with gi explore when --topic matches no insights returns 4."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(
            [
                "gi",
                "explore",
                "--topic",
                "xyznonexistent_topic_123",
                "--output-dir",
                str(tmp),
                "--format",
                "json",
            ]
        )
        self.assertEqual(exit_code, 4)

    def test_main_gi_explore_strict_invalid_artifact_exits_5(self):
        """Test gi explore --strict exits 5 when an artifact fails validation."""
        from pathlib import Path

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        (tmp / "metadata" / "bad.gi.json").write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
        exit_code = cli.main(
            ["gi", "explore", "--output-dir", str(tmp), "--strict", "--format", "json"]
        )
        self.assertEqual(exit_code, 5)

    def test_main_gi_show_insight_wrong_id_returns_1(self):
        """Test main() with gi show-insight when --id not in artifact returns 1."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        (tmp / "transcripts").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        (tmp / "transcripts" / "ep1.txt").write_text("Transcript.", encoding="utf-8")
        exit_code = cli.main(
            [
                "gi",
                "show-insight",
                "--id",
                "insight:nonexistent:0",
                "--output-dir",
                str(tmp),
            ]
        )
        self.assertEqual(exit_code, 1)

    def test_main_gi_inspect_output_dir_without_episode_id_returns_1(self):
        """Test main() with gi inspect --output-dir but no --episode-id returns 1."""
        import tempfile

        tmp = tempfile.mkdtemp()
        exit_code = cli.main(["gi", "inspect", "--output-dir", tmp])
        self.assertEqual(exit_code, 1)

    def test_main_gi_inspect_artifact_not_found_for_episode_returns_1(self):
        """Test main() with gi inspect --output-dir + --episode-id when no artifact matches."""
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        tmp = Path(self.get_tmp_dir())
        (tmp / "metadata").mkdir(parents=True)
        art = build_artifact("ep:1", "Transcript.", prompt_version="v1")
        write_artifact(tmp / "metadata" / "ep1.gi.json", art, validate=True)
        exit_code = cli.main(
            ["gi", "inspect", "--output-dir", str(tmp), "--episode-id", "ep:nonexistent"]
        )
        self.assertEqual(exit_code, 1)

    def test_main_gi_inspect_multi_feed_with_feed_id_returns_0(self):
        """gi inspect resolves duplicate episode_id across feeds when --feed-id is set."""
        import json
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        corpus = Path(self.get_tmp_dir()) / "corpus"
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            mdir.mkdir(parents=True)
            meta_doc = {"feed": {"feed_id": fid}, "episode": {"episode_id": "dup"}}
            (mdir / "ep.metadata.json").write_text(json.dumps(meta_doc), encoding="utf-8")
            art = build_artifact("dup", "Transcript.", prompt_version="v1")
            write_artifact(mdir / "ep.gi.json", art, validate=True)
        exit_code = cli.main(
            [
                "gi",
                "inspect",
                "--output-dir",
                str(corpus),
                "--episode-id",
                "dup",
                "--feed-id",
                "feed_a",
                "--format",
                "json",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_main_gi_inspect_multi_feed_ambiguous_returns_1(self):
        """Same episode_id in two feeds without --feed-id logs error and exits 1."""
        import json
        from pathlib import Path

        from podcast_scraper.gi import build_artifact, write_artifact

        corpus = Path(self.get_tmp_dir()) / "corpus"
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            mdir.mkdir(parents=True)
            meta_doc = {"feed": {"feed_id": fid}, "episode": {"episode_id": "dup"}}
            (mdir / "ep.metadata.json").write_text(json.dumps(meta_doc), encoding="utf-8")
            art = build_artifact("dup", "Transcript.", prompt_version="v1")
            write_artifact(mdir / "ep.gi.json", art, validate=True)
        with self.assertLogs("podcast_scraper.cli", level="ERROR") as cm:
            exit_code = cli.main(
                ["gi", "inspect", "--output-dir", str(corpus), "--episode-id", "dup"]
            )
        self.assertEqual(exit_code, 1)
        joined = "\n".join(cm.output)
        self.assertIn("Multiple GI artifacts", joined)
        self.assertIn("--feed-id", joined)

    def test_main_gi_explore_output_dir_is_file_returns_3(self):
        """Test main() with gi explore when --output-dir is a file (not dir) returns 3."""
        import tempfile

        fd, path = tempfile.mkstemp()
        try:
            exit_code = cli.main(["gi", "explore", "--output-dir", path])
            self.assertEqual(exit_code, 3)
        finally:
            import os

            os.close(fd)
            os.unlink(path)

    def test_main_gi_no_subcommand_exits_2(self):
        """Test main() with 'gi' and no subcommand exits with 2 (argparse)."""
        exit_code = cli.main(["gi"])
        self.assertEqual(exit_code, 2)


class TestKgSubcommandMultiFeed(unittest.TestCase):
    """``kg inspect`` + ``--feed-id`` for multi-feed corpus.

    Class name avoids clashing with ``tests/.../kg/test_kg_cli.py`` under xdist.
    """

    def get_tmp_dir(self) -> str:
        import tempfile

        return tempfile.mkdtemp()

    def test_parse_kg_args_inspect_feed_id(self):
        args = cli._parse_kg_args(
            [
                "inspect",
                "--output-dir",
                "/corpus",
                "--episode-id",
                "ep:x",
                "--feed-id",
                "feed_a",
            ]
        )
        self.assertEqual(args.kg_subcommand, "inspect")
        self.assertEqual(args.feed_id, "feed_a")

    def test_main_kg_inspect_multi_feed_with_feed_id_returns_0(self):
        import json
        import shutil
        from pathlib import Path

        corpus = Path(self.get_tmp_dir()) / "corpus"
        minimal = (
            Path(__file__).resolve().parent.parent.parent / "fixtures" / "kg" / "minimal.kg.json"
        )
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            mdir.mkdir(parents=True)
            meta_doc = {"feed": {"feed_id": fid}, "episode": {"episode_id": "dup"}}
            (mdir / "x.metadata.json").write_text(json.dumps(meta_doc), encoding="utf-8")
            dest = mdir / "x.kg.json"
            shutil.copy(minimal, dest)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            payload["episode_id"] = "dup"
            dest.write_text(json.dumps(payload), encoding="utf-8")
        exit_code = cli.main(
            [
                "kg",
                "inspect",
                "--output-dir",
                str(corpus),
                "--episode-id",
                "dup",
                "--feed-id",
                "feed_a",
                "--format",
                "json",
            ]
        )
        self.assertEqual(exit_code, 0)

    def test_main_kg_inspect_multi_feed_ambiguous_returns_1(self):
        import json
        import shutil
        from pathlib import Path

        corpus = Path(self.get_tmp_dir()) / "corpus"
        minimal = (
            Path(__file__).resolve().parent.parent.parent / "fixtures" / "kg" / "minimal.kg.json"
        )
        for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
            mdir = corpus / "feeds" / slug / "run" / "metadata"
            mdir.mkdir(parents=True)
            meta_doc = {"feed": {"feed_id": fid}, "episode": {"episode_id": "dup"}}
            (mdir / "x.metadata.json").write_text(json.dumps(meta_doc), encoding="utf-8")
            dest = mdir / "x.kg.json"
            shutil.copy(minimal, dest)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            payload["episode_id"] = "dup"
            dest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertLogs("podcast_scraper.cli", level="ERROR") as cm:
            exit_code = cli.main(
                ["kg", "inspect", "--output-dir", str(corpus), "--episode-id", "dup"]
            )
        self.assertEqual(exit_code, 1)
        joined = "\n".join(cm.output)
        self.assertIn("Multiple KG artifacts", joined)
        self.assertIn("--feed-id", joined)


class TestLoadAndMergeConfig(unittest.TestCase):
    """Test _load_and_merge_config function."""

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_validation_error(self, mock_load):
        """Test that _load_and_merge_config handles validation errors."""
        # Config file with unknown keys should raise ValueError
        mock_load.return_value = {"unknown_key": "value"}
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        with self.assertRaises(ValueError) as cm:
            cli._load_and_merge_config(parser, "config.yaml", None)
        self.assertIn("Unknown config option", str(cm.exception))

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_file_not_found(self, mock_load):
        """Test handling of missing config file."""
        mock_load.side_effect = FileNotFoundError("File not found")
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        with self.assertRaises(FileNotFoundError):
            cli._load_and_merge_config(parser, "missing.yaml", None)

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_validation_error_in_config(self, mock_load):
        """Test that _load_and_merge_config handles ValidationError from Config."""
        # Use an invalid enum value that will trigger ValidationError
        # log_level must be one of the valid log levels
        mock_load.return_value = {
            "rss": "https://example.com/feed.xml",
            "log_level": "INVALID_LEVEL",  # Invalid: not a valid log level
        }
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        # Pass empty argv to avoid argparse parsing sys.argv
        # Config validation will happen when Config is created, raising ValidationError
        # which gets converted to ValueError
        with self.assertRaises(ValueError) as cm:
            cli._load_and_merge_config(parser, "config.yaml", [])
        # Should raise ValidationError which gets converted to ValueError
        # Error message contains "invalid configuration" (lowercase)
        self.assertIn("invalid", str(cm.exception).lower())

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_missing_rss(self, mock_load):
        """Test that _load_and_merge_config raises error when RSS is missing."""
        mock_load.return_value = {"max_episodes": 5}  # No RSS URL
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        # Pass empty argv to avoid argparse parsing sys.argv
        with self.assertRaises(ValueError) as cm:
            cli._load_and_merge_config(parser, "config.yaml", [])
        self.assertIn("RSS URL is required", str(cm.exception))

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_accepts_feeds_validation_alias(self, mock_load):
        """YAML ``feeds`` (validation alias for rss_urls) is not an unknown key."""
        mock_load.return_value = {
            "feeds": [
                "https://a.example/feed.xml",
                "https://b.example/feed.xml",
            ],
            "output_dir": "/tmp/corpus_multi",
            "max_episodes": 1,
            "user_agent": "test",
            "timeout": 30,
        }
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        args = cli._load_and_merge_config(parser, "config.yaml", [])
        self.assertEqual(
            args.rss_urls,
            ["https://a.example/feed.xml", "https://b.example/feed.xml"],
        )
        self.assertEqual(args.output_dir, "/tmp/corpus_multi")

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_accepts_deprecated_multi_feed_soft_fail_key(
        self, mock_load
    ) -> None:
        """Deprecated YAML key is allowed in merge and maps to multi_feed_strict."""
        mock_load.return_value = {
            "feeds": [
                "https://a.example/feed.xml",
                "https://b.example/feed.xml",
            ],
            "output_dir": "/tmp/corpus_dep",
            "max_episodes": 1,
            "user_agent": "test",
            "timeout": 30,
            "multi_feed_soft_fail_exit_zero": True,
        }
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        args = cli._load_and_merge_config(parser, "config.yaml", [])
        self.assertFalse(getattr(args, "multi_feed_strict", True))

    @patch("podcast_scraper.cli.config.load_config_file")
    def test_load_and_merge_config_speaker_names_list(self, mock_load):
        """Test that _load_and_merge_config converts speaker_names list to comma-separated."""
        mock_load.return_value = {
            "rss": "https://example.com/feed.xml",  # Use 'rss' not 'rss_url' for CLI
            "speaker_names": ["Host 1", "Host 2"],
        }
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)

        # Pass empty argv to avoid argparse parsing sys.argv
        args = cli._load_and_merge_config(parser, "config.yaml", [])
        self.assertEqual(args.speaker_names, "Host 1,Host 2")


class TestCLIErrorHandling(unittest.TestCase):
    """Test error handling in CLI functions."""

    def test_validate_args_append_and_clean_output_conflict(self):
        """--append and --clean-output are rejected together (GitHub #444)."""
        out = tempfile.mkdtemp()
        try:
            args = argparse.Namespace(
                rss="https://example.com/feed.xml",
                rss_extra=[],
                rss_file=None,
                timeout=30,
                delay_ms=0,
                transcribe_missing=False,
                auto_speakers=False,
                screenplay=False,
                num_speakers=2,
                speaker_names=None,
                workers=1,
                output_dir=out,
                max_episodes=None,
                whisper_model="base.en",
                clean_output=True,
                append=True,
            )

            with self.assertRaises(ValueError) as cm:
                cli.validate_args(args)
            self.assertIn("append", str(cm.exception).lower())
        finally:
            os.rmdir(out)

    def test_validate_args_timeout_negative(self):
        """Test that negative timeout raises error."""
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=-1,
            delay_ms=0,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir=None,
            max_episodes=None,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--timeout must be positive", str(cm.exception))

    def test_validate_args_timeout_zero(self):
        """Test that zero timeout raises error."""
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=0,
            delay_ms=0,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir=None,
            max_episodes=None,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--timeout must be positive", str(cm.exception))

    def test_validate_args_delay_ms_negative(self):
        """Test that negative delay_ms raises error."""
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=30,
            delay_ms=-1,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir=None,
            max_episodes=None,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--delay-ms must be non-negative", str(cm.exception))

    def test_validate_args_max_episodes_negative(self):
        """Test that negative max_episodes raises error."""
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir=None,
            max_episodes=-1,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--max-episodes must be positive", str(cm.exception))

    def test_validate_args_max_episodes_zero(self):
        """Test that zero max_episodes raises error."""
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir=None,
            max_episodes=0,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("--max-episodes must be positive", str(cm.exception))

    @patch("podcast_scraper.cli.filesystem.validate_and_normalize_output_dir")
    def test_validate_args_output_dir_validation_error(self, mock_validate):
        """Test that output_dir validation errors are caught."""
        mock_validate.side_effect = ValueError("Invalid output directory")
        args = argparse.Namespace(
            rss="https://example.com/feed.xml",
            timeout=30,
            delay_ms=0,
            transcribe_missing=False,
            auto_speakers=False,
            screenplay=False,
            num_speakers=2,
            speaker_names=None,
            workers=1,
            output_dir="/invalid/path",
            max_episodes=None,
            whisper_model="base.en",
        )

        with self.assertRaises(ValueError) as cm:
            cli.validate_args(args)
        self.assertIn("Invalid output directory", str(cm.exception))

    @patch("podcast_scraper.cli._validate_ffmpeg")
    @patch("podcast_scraper.cli.parse_args")
    def test_main_handles_value_error(self, mock_parse, mock_validate_ffmpeg):
        """Test that main() handles ValueError from parse_args."""
        mock_parse.side_effect = ValueError("Invalid arguments")

        exit_code = cli.main([])

        self.assertEqual(exit_code, 1)

    @patch("podcast_scraper.cli._validate_ffmpeg")
    @patch("podcast_scraper.cli.parse_args")
    def test_main_handles_system_exit(self, mock_parse, mock_validate_ffmpeg):
        """Test that main() handles SystemExit from parse_args (e.g., --help)."""
        mock_parse.side_effect = SystemExit(0)

        exit_code = cli.main([])

        self.assertEqual(exit_code, 0)

    @patch("podcast_scraper.cli._validate_ffmpeg")
    @patch("podcast_scraper.cli.parse_args")
    def test_main_handles_system_exit_non_zero(self, mock_parse, mock_validate_ffmpeg):
        """Test that main() handles SystemExit with non-zero code."""
        mock_parse.side_effect = SystemExit(2)

        exit_code = cli.main([])

        self.assertEqual(exit_code, 2)

    @patch("podcast_scraper.cli._validate_ffmpeg")
    @patch("podcast_scraper.cli.parse_args")
    def test_main_handles_system_exit_no_code(self, mock_parse, mock_validate_ffmpeg):
        """Test that main() handles SystemExit without code."""
        mock_parse.side_effect = SystemExit(None)

        exit_code = cli.main([])

        self.assertEqual(exit_code, 0)

    def test_parse_args_version_exits(self):
        """Test that --version causes SystemExit."""
        with self.assertRaises(SystemExit) as cm:
            cli.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)


class TestLogConfigurationGilHybridWarning(unittest.TestCase):
    """Runtime warning when API summary stack pairs with local GIL evidence (WIP hybrid note)."""

    def test_warns_when_api_summary_and_local_evidence(self):
        log = logging.getLogger("test_gil_hybrid_warn")
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            with self.assertLogs(log, level="WARNING") as cm:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    generate_metadata=True,
                    generate_gi=True,
                    gi_insight_source="summary_bullets",
                    gi_require_grounding=True,
                    summary_provider="openai",
                    openai_api_key="sk-test-key-for-unit-tests",
                    quote_extraction_provider="openai",
                    entailment_provider="transformers",
                    gil_evidence_match_summary_provider=False,
                )
                cli._log_configuration(cfg, log)
        messages = " ".join(r.getMessage() for r in cm.records)
        self.assertIn("GIL:", messages)
        self.assertIn("sentence-transformers", messages)

    def test_warns_when_hybrid_ml_summary_and_local_evidence(self):
        """Same GIL hybrid warning when summary stack is hybrid_ml (in API-align set)."""
        log = logging.getLogger("test_gil_hybrid_hybrid_ml")
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            with self.assertLogs(log, level="WARNING") as cm:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    generate_metadata=True,
                    generate_gi=True,
                    gi_insight_source="summary_bullets",
                    gi_require_grounding=True,
                    summary_provider="hybrid_ml",
                    quote_extraction_provider="transformers",
                    entailment_provider="transformers",
                    gil_evidence_match_summary_provider=False,
                )
                cli._log_configuration(cfg, log)
        messages = " ".join(r.getMessage() for r in cm.records)
        self.assertIn("GIL:", messages)
        self.assertIn("sentence-transformers", messages)

    def test_no_hybrid_warning_when_evidence_matches_openai(self):
        log = logging.getLogger("test_gil_hybrid_aligned")
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            with patch.object(log, "warning") as mock_warn:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    generate_metadata=True,
                    generate_gi=True,
                    gi_insight_source="summary_bullets",
                    gi_require_grounding=True,
                    summary_provider="openai",
                    openai_api_key="sk-test-key-for-unit-tests",
                )
                cli._log_configuration(cfg, log)
                hybrid_calls = [
                    c
                    for c in mock_warn.call_args_list
                    if c.args and "sentence-transformers" in str(c.args[0])
                ]
                self.assertEqual(hybrid_calls, [])


class TestLogConfigurationGiStubWarning(unittest.TestCase):
    """_log_configuration warns when GIL uses stub insights outside test env (Issue #460)."""

    def test_warns_when_generate_gi_stub_and_not_test_env(self):
        log = logging.getLogger("test_gil_stub_warn")
        with patch("podcast_scraper.config._is_test_environment", return_value=False):
            with self.assertLogs(log, level="WARNING") as cm:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    generate_metadata=True,
                    generate_gi=True,
                    gi_insight_source="stub",
                )
                cli._log_configuration(cfg, log)
        messages = " ".join(r.getMessage() for r in cm.records)
        self.assertIn("gi_insight_source", messages)
        self.assertIn("stub", messages)

    def test_no_stub_warning_when_test_environment(self):
        log = logging.getLogger("test_gil_stub_no_warn")
        with patch("podcast_scraper.config._is_test_environment", return_value=True):
            with patch.object(log, "warning") as mock_warn:
                cfg = config.Config(
                    rss_url="https://example.com/feed.xml",
                    generate_metadata=True,
                    generate_gi=True,
                    gi_insight_source="stub",
                )
                cli._log_configuration(cfg, log)
                mock_warn.assert_not_called()


class TestGiValidateAndExportCli(unittest.TestCase):
    """GI subcommands: error paths use format_exception_for_log."""

    def test_run_gi_validate_collect_paths_error_returns_invalid_args(self):
        from argparse import Namespace

        from podcast_scraper.gi.explore import EXIT_INVALID_ARGS

        log = logging.getLogger("test_gi_validate_collect")
        args = Namespace(paths=["/unlikely/path/only"], strict=False, quiet=True)
        with patch(
            "podcast_scraper.gi.io.collect_gi_paths_from_inputs",
            side_effect=ValueError("synthetic path error"),
        ):
            rc = cli._run_gi_validate(args, log)
        self.assertEqual(rc, EXIT_INVALID_ARGS)

    def test_run_gi_export_load_artifacts_failure_returns_one(self):
        import tempfile
        from argparse import Namespace
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root / "gi_run"
            out.mkdir()
            log = logging.getLogger("test_gi_export_load")
            args = Namespace(output_dir=str(out), strict=False, format="ndjson", out=None)
            fake_path = root / "one.gi.json"
            fake_path.write_text("{}", encoding="utf-8")
            with (
                patch(
                    "podcast_scraper.gi.explore.scan_artifact_paths",
                    return_value=[fake_path],
                ),
                patch(
                    "podcast_scraper.gi.corpus.load_gi_artifacts",
                    side_effect=RuntimeError("synthetic load failure"),
                ),
            ):
                rc = cli._run_gi_export(args, log)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
