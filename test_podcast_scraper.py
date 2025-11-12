#!/usr/bin/env python3
"""Tests for podcast_scraper package."""

import os
import sys

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET  # nosec B405 - tests construct safe XML elements
from pathlib import Path
from unittest.mock import patch

import requests
from platformdirs import user_cache_dir, user_data_dir

import podcast_scraper
import podcast_scraper.cli as cli
from podcast_scraper import (
    downloader,
    episode_processor,
    filesystem,
    models,
    rss_parser,
)

# Test constants
TEST_BASE_URL = "https://example.com"
TEST_FEED_URL = "https://example.com/feed.xml"
TEST_PATH = "/path"
TEST_FULL_URL = f"{TEST_BASE_URL}{TEST_PATH}"
TEST_TRANSCRIPT_URL = f"{TEST_BASE_URL}/transcript.vtt"
TEST_TRANSCRIPT_URL_SRT = f"{TEST_BASE_URL}/transcript.srt"
TEST_MEDIA_URL = f"{TEST_BASE_URL}/episode.mp3"
TEST_RELATIVE_TRANSCRIPT = "transcripts/ep1.vtt"
TEST_RELATIVE_MEDIA = "episodes/ep1.mp3"
TEST_EPISODE_TITLE = "Episode Title"
TEST_EPISODE_TITLE_SPECIAL = "Episode: Title/With\\Special*Chars?"
TEST_FEED_TITLE = "Test Feed"
TEST_OUTPUT_DIR = "output"
TEST_CUSTOM_OUTPUT_DIR = "my_output"
TEST_RUN_ID = "test_run"
TEST_MEDIA_TYPE_MP3 = "audio/mpeg"
TEST_MEDIA_TYPE_M4A = "audio/m4a"
TEST_TRANSCRIPT_TYPE_VTT = "text/vtt"
TEST_TRANSCRIPT_TYPE_SRT = "text/srt"
TEST_CONTENT_TYPE_VTT = "text/vtt"
TEST_CONTENT_TYPE_SRT = "text/srt"


class TestHTTPSessionConfiguration(unittest.TestCase):
    """Tests for HTTP session retry configuration."""

    def test_configure_http_session_mounts_retry_adapters(self):
        session = requests.Session()
        try:
            podcast_scraper.downloader._configure_http_session(session)
            https_adapter = session.get_adapter("https://")
            http_adapter = session.get_adapter("http://")

            self.assertIsInstance(https_adapter, requests.adapters.HTTPAdapter)
            self.assertIsInstance(http_adapter, requests.adapters.HTTPAdapter)

            https_retry = https_adapter.max_retries
            http_retry = http_adapter.max_retries

            self.assertEqual(https_retry.total, downloader.DEFAULT_HTTP_RETRY_TOTAL)
            self.assertEqual(http_retry.total, downloader.DEFAULT_HTTP_RETRY_TOTAL)
            self.assertEqual(https_retry.backoff_factor, downloader.DEFAULT_HTTP_BACKOFF_FACTOR)
            self.assertEqual(http_retry.backoff_factor, downloader.DEFAULT_HTTP_BACKOFF_FACTOR)
            self.assertEqual(https_retry.allowed_methods, downloader.HTTP_RETRY_ALLOWED_METHODS)
            self.assertEqual(http_retry.allowed_methods, downloader.HTTP_RETRY_ALLOWED_METHODS)
            self.assertEqual(
                set(https_retry.status_forcelist), set(downloader.HTTP_RETRY_STATUS_CODES)
            )
            self.assertEqual(
                set(http_retry.status_forcelist), set(downloader.HTTP_RETRY_STATUS_CODES)
            )
        finally:
            session.close()


class TestNormalizeURL(unittest.TestCase):
    """Tests for normalize_url function."""

    def test_simple_url(self):
        """Test normalizing a simple URL."""
        url = TEST_FULL_URL
        result = downloader.normalize_url(url)
        self.assertEqual(result, url)

    def test_url_with_non_ascii(self):
        """Test normalizing URL with non-ASCII characters."""
        url = f"{TEST_FULL_URL}/тест"
        result = downloader.normalize_url(url)
        self.assertIn("%D1%82%D0%B5%D1%81%D1%82", result)

    def test_url_with_query(self):
        """Test normalizing URL with query parameters."""
        url = f"{TEST_FULL_URL}?param=value&other=тест"
        result = downloader.normalize_url(url)
        self.assertIn("param=value", result)
        self.assertIn("%D1%82%D0%B5%D1%81%D1%82", result)

    def test_relative_url(self):
        """Test normalizing a relative URL."""
        url = "/path/to/resource"
        result = downloader.normalize_url(url)
        self.assertEqual(result, url)


class TestSanitizeFilename(unittest.TestCase):
    """Tests for sanitize_filename function."""

    def test_simple_filename(self):
        """Test sanitizing a simple filename."""
        name = "episode_title"
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, "episode_title")

    def test_filename_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        name = TEST_EPISODE_TITLE_SPECIAL
        result = filesystem.sanitize_filename(name)
        self.assertNotIn(":", result)
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)
        self.assertNotIn("*", result)
        self.assertNotIn("?", result)

    def test_filename_with_whitespace(self):
        """Test sanitizing filename with whitespace."""
        name = f"  {TEST_EPISODE_TITLE}   "
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, TEST_EPISODE_TITLE)

    def test_empty_filename(self):
        """Test sanitizing empty filename."""
        name = ""
        result = filesystem.sanitize_filename(name)
        self.assertEqual(result, "untitled")


class TestValidateAndNormalizeOutputDir(unittest.TestCase):
    """Tests for validate_and_normalize_output_dir function."""

    def test_valid_relative_path(self):
        """Test validating a valid relative path."""
        path = TEST_OUTPUT_DIR
        result = filesystem.validate_and_normalize_output_dir(path)
        self.assertIsInstance(result, str)
        self.assertIn(TEST_OUTPUT_DIR, result)

    def test_path_traversal_attempt(self):
        """Test that path traversal attempts are handled."""
        path = "../../etc/passwd"
        result = filesystem.validate_and_normalize_output_dir(path)
        # Should normalize but not allow traversal
        self.assertIsInstance(result, str)

    def test_absolute_path(self):
        """Test validating an absolute path."""
        # Use a path in the current working directory to avoid system directory restrictions
        test_path = os.path.join(os.getcwd(), "test_output")
        try:
            result = filesystem.validate_and_normalize_output_dir(test_path)
            self.assertIsInstance(result, str)
            self.assertIn("test_output", result)
        finally:
            # Clean up if directory was created
            if os.path.exists(test_path) and os.path.isdir(test_path):
                try:
                    os.rmdir(test_path)
                except OSError:
                    pass

    def test_platformdirs_user_locations_do_not_warn(self):
        """Paths under platformdirs user data/cache locations should not warn."""
        candidates = {
            user_data_dir("podcast_scraper"),
            user_data_dir("podcast-scraper"),
            user_cache_dir("podcast_scraper"),
            user_cache_dir("podcast-scraper"),
        }
        for candidate in {c for c in candidates if c}:
            with self.subTest(candidate=candidate):
                with patch.object(filesystem.logger, "warning") as mock_warning:
                    result = filesystem.validate_and_normalize_output_dir(candidate)
                self.assertIsInstance(result, str)
                resolved_result = Path(result).resolve()
                resolved_candidate = Path(candidate).expanduser().resolve()
                self.assertTrue(
                    resolved_result == resolved_candidate
                    or resolved_result.is_relative_to(resolved_candidate)
                )
                mock_warning.assert_not_called()


class TestDeriveOutputDir(unittest.TestCase):
    """Tests for derive_output_dir function."""

    def test_default_output_dir(self):
        """Test deriving default output directory from RSS URL."""
        rss_url = TEST_FEED_URL
        result = filesystem.derive_output_dir(rss_url, None)
        self.assertIn("output_rss", result)
        self.assertIn("example.com", result)
        self.assertEqual(result, "output_rss_example.com_6904f1c4")

    def test_custom_output_dir(self):
        """Test using custom output directory."""
        rss_url = TEST_FEED_URL
        custom = TEST_CUSTOM_OUTPUT_DIR
        result = filesystem.derive_output_dir(rss_url, custom)
        # Result is normalized to absolute path
        self.assertIn(TEST_CUSTOM_OUTPUT_DIR, result)


class TestParseRSSItems(unittest.TestCase):
    """Tests for parse_rss_items function."""

    def test_parse_simple_rss(self):
        """Test parsing a simple RSS feed."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <item>
                    <title>Episode 1</title>
                </item>
                <item>
                    <title>Episode 2</title>
                </item>
            </channel>
        </rss>""".encode()
        title, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(items), 2)

    def test_parse_rss_with_namespace(self):
        """Test parsing RSS feed with namespace."""
        xml_bytes = f"""<?xml version="1.0"?>
        <rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
            <channel>
                <title>{TEST_FEED_TITLE}</title>
                <item>
                    <title>Episode 1</title>
                </item>
            </channel>
        </rss>""".encode()
        title, items = rss_parser.parse_rss_items(xml_bytes)
        self.assertEqual(title, TEST_FEED_TITLE)
        self.assertEqual(len(items), 1)


class TestFindTranscriptURLs(unittest.TestCase):
    """Tests for find_transcript_urls function."""

    def test_find_podcast_transcript(self):
        """Test finding podcast:transcript URLs."""
        item = ET.Element("item")
        transcript = ET.SubElement(item, "podcast:transcript")
        transcript.set("url", TEST_TRANSCRIPT_URL)
        transcript.set("type", TEST_TRANSCRIPT_TYPE_VTT)

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], TEST_TRANSCRIPT_URL)
        self.assertEqual(result[0][1], TEST_TRANSCRIPT_TYPE_VTT)

    def test_find_relative_transcript_url(self):
        """Test finding relative transcript URLs."""
        item = ET.Element("item")
        transcript = ET.SubElement(item, "transcript")
        transcript.text = TEST_RELATIVE_TRANSCRIPT

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], f"{TEST_BASE_URL}/{TEST_RELATIVE_TRANSCRIPT}")

    def test_find_multiple_transcripts(self):
        """Test finding multiple transcript URLs."""
        item = ET.Element("item")
        transcript1 = ET.SubElement(item, "podcast:transcript")
        transcript1.set("url", TEST_TRANSCRIPT_URL)
        transcript2 = ET.SubElement(item, "podcast:transcript")
        transcript2.set("url", TEST_TRANSCRIPT_URL_SRT)

        base_url = TEST_FEED_URL
        result = rss_parser.find_transcript_urls(item, base_url)
        self.assertEqual(len(result), 2)


class TestFindEnclosureMedia(unittest.TestCase):
    """Tests for find_enclosure_media function."""

    def test_find_enclosure(self):
        """Test finding enclosure media."""
        item = ET.Element("item")
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", TEST_MEDIA_URL)
        enclosure.set("type", TEST_MEDIA_TYPE_MP3)

        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], TEST_MEDIA_URL)
        self.assertEqual(result[1], TEST_MEDIA_TYPE_MP3)

    def test_find_relative_enclosure_url(self):
        """Test finding relative enclosure URLs."""
        item = ET.Element("item")
        enclosure = ET.SubElement(item, "enclosure")
        enclosure.set("url", TEST_RELATIVE_MEDIA)
        enclosure.set("type", TEST_MEDIA_TYPE_MP3)

        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], f"{TEST_BASE_URL}/{TEST_RELATIVE_MEDIA}")

    def test_no_enclosure(self):
        """Test when no enclosure is found."""
        item = ET.Element("item")
        base_url = TEST_FEED_URL
        result = rss_parser.find_enclosure_media(item, base_url)
        self.assertIsNone(result)


class TestChooseTranscriptURL(unittest.TestCase):
    """Tests for choose_transcript_url function."""

    def test_choose_first_when_no_preference(self):
        """Test choosing first URL when no preference specified."""
        candidates = [
            (TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT),
            (TEST_TRANSCRIPT_URL_SRT, TEST_TRANSCRIPT_TYPE_SRT),
        ]
        result = rss_parser.choose_transcript_url(candidates, [])
        self.assertEqual(result, candidates[0])

    def test_choose_by_preference(self):
        """Test choosing URL by preferred type."""
        candidates = [
            (TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT),
            (TEST_TRANSCRIPT_URL_SRT, TEST_TRANSCRIPT_TYPE_SRT),
        ]
        result = rss_parser.choose_transcript_url(candidates, [TEST_TRANSCRIPT_TYPE_SRT])
        self.assertEqual(result, candidates[1])

    def test_empty_candidates(self):
        """Test with empty candidates list."""
        result = rss_parser.choose_transcript_url([], [])
        self.assertIsNone(result)


class TestExtractEpisodeTitle(unittest.TestCase):
    """Tests for extract_episode_title function."""

    def test_extract_title(self):
        """Test extracting episode title."""
        item = ET.Element("item")
        title = ET.SubElement(item, "title")
        title.text = TEST_EPISODE_TITLE

        result_title, result_safe = rss_parser.extract_episode_title(item, 1)
        self.assertEqual(result_title, TEST_EPISODE_TITLE)
        # sanitize_filename preserves spaces, only replaces special chars
        self.assertEqual(result_safe, TEST_EPISODE_TITLE)

    def test_fallback_to_episode_number(self):
        """Test fallback to episode number when no title."""
        item = ET.Element("item")
        result_title, result_safe = rss_parser.extract_episode_title(item, 5)
        self.assertEqual(result_title, "episode_5")
        self.assertEqual(result_safe, "episode_5")


class TestDeriveMediaExtension(unittest.TestCase):
    """Tests for derive_media_extension function."""

    def test_derive_from_media_type(self):
        """Test deriving extension from media type."""
        result = episode_processor.derive_media_extension(
            TEST_MEDIA_TYPE_MP3, f"{TEST_BASE_URL}/audio"
        )
        self.assertEqual(result, ".mp3")

    def test_derive_from_url(self):
        """Test deriving extension from URL."""
        result = episode_processor.derive_media_extension(None, TEST_MEDIA_URL)
        self.assertEqual(result, ".mp3")

    def test_default_extension(self):
        """Test default extension when type and URL don't match."""
        result = episode_processor.derive_media_extension(None, f"{TEST_BASE_URL}/audio")
        self.assertEqual(result, ".bin")


class TestDeriveTranscriptExtension(unittest.TestCase):
    """Tests for derive_transcript_extension function."""

    def test_derive_from_transcript_type(self):
        """Test deriving extension from transcript type."""
        result = episode_processor.derive_transcript_extension(
            TEST_TRANSCRIPT_TYPE_VTT, None, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_content_type(self):
        """Test deriving extension from content type."""
        result = episode_processor.derive_transcript_extension(
            None, TEST_CONTENT_TYPE_VTT, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".vtt")

    def test_derive_from_url(self):
        """Test deriving extension from URL."""
        result = episode_processor.derive_transcript_extension(None, None, TEST_TRANSCRIPT_URL_SRT)
        self.assertEqual(result, ".srt")

    def test_url_with_query_keeps_extension(self):
        """URL query parameters should not override file extension."""
        url = f"{TEST_TRANSCRIPT_URL}?alt=json"
        result = episode_processor.derive_transcript_extension(None, None, url)
        self.assertEqual(result, ".vtt")

    def test_default_extension(self):
        """Test default extension."""
        result = episode_processor.derive_transcript_extension(
            None, None, f"{TEST_BASE_URL}/transcript"
        )
        self.assertEqual(result, ".txt")


class TestSetupOutputDirectory(unittest.TestCase):
    """Tests for setup_output_directory function."""

    def test_with_run_id(self):
        """Test setting up output directory with run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=TEST_OUTPUT_DIR,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=False,
            whisper_model="base",
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=TEST_RUN_ID,
            skip_existing=False,
            clean_output=False,
        )
        output_dir, run_suffix = filesystem.setup_output_directory(cfg)
        self.assertIn(f"run_{TEST_RUN_ID}", output_dir)
        self.assertEqual(run_suffix, TEST_RUN_ID)

    def test_with_whisper_auto_run_id(self):
        """Test setting up output directory with Whisper auto run ID."""
        cfg = podcast_scraper.Config(
            rss_url=TEST_FEED_URL,
            output_dir=TEST_OUTPUT_DIR,
            max_episodes=None,
            user_agent="test",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=True,
            whisper_model="base",
            screenplay=False,
            screenplay_gap_s=2.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=None,
            skip_existing=False,
            clean_output=False,
        )
        output_dir, run_suffix = filesystem.setup_output_directory(cfg)
        self.assertIn("whisper_base", run_suffix or "")


class TestWriteFile(unittest.TestCase):
    """Tests for write_file function."""

    def test_write_file(self):
        """Test writing a file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            data = b"test content"
            filesystem.write_file(tmp_path, data)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, "rb") as f:
                self.assertEqual(f.read(), data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestSkipExisting(unittest.TestCase):
    """Tests for skip-existing resume behaviour."""

    def _make_config(self, **overrides):
        defaults = dict(
            rss_url=TEST_FEED_URL,
            output_dir=".",
            max_episodes=None,
            user_agent="test-agent",
            timeout=30,
            delay_ms=0,
            prefer_types=[],
            transcribe_missing=True,
            whisper_model="base",
            screenplay=False,
            screenplay_gap_s=1.0,
            screenplay_num_speakers=2,
            screenplay_speaker_names=[],
            run_id=None,
            log_level="INFO",
            workers=1,
            skip_existing=True,
            clean_output=False,
        )
        defaults.update(overrides)
        return podcast_scraper.Config(**defaults)

    def test_process_transcript_download_skips_existing_file(self):
        """Existing transcript files are not re-downloaded when --skip-existing is in effect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ep_title = "Episode 1"
            ep_title_safe = filesystem.sanitize_filename(ep_title)
            existing_path = os.path.join(tmpdir, f"0001 - {ep_title_safe}.txt")
            with open(existing_path, "wb") as fh:
                fh.write(b"original")

            cfg = self._make_config(output_dir=tmpdir, transcribe_missing=False)

            # Create Episode object
            item = ET.Element("item")
            episode = models.Episode(
                idx=1,
                title=ep_title,
                title_safe=ep_title_safe,
                item=item,
                transcript_urls=[],
                media_url=None,
                media_type=None,
            )

            with patch("podcast_scraper.downloader.http_get") as mock_http_get:
                result = episode_processor.process_transcript_download(
                    episode,
                    f"{TEST_BASE_URL}/ep1.txt",
                    "text/plain",
                    cfg,
                    tmpdir,
                    None,
                )

            self.assertFalse(result)
            mock_http_get.assert_not_called()

    def test_download_media_skips_when_whisper_output_exists(self):
        """Whisper download step is skipped if final transcript already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = os.path.join(tmpdir, ".tmp_media")
            os.makedirs(temp_dir, exist_ok=True)
            ep_title = "Episode 2"
            ep_title_safe = filesystem.sanitize_filename(ep_title)

            final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, tmpdir)
            with open(final_path, "wb") as fh:
                fh.write(b"existing transcript")

            cfg = self._make_config(output_dir=tmpdir)

            item = ET.Element("item")
            ET.SubElement(
                item,
                "enclosure",
                attrib={"url": TEST_MEDIA_URL, "type": TEST_MEDIA_TYPE_MP3},
            )

            # Create Episode object
            episode = models.Episode(
                idx=1,
                title=ep_title,
                title_safe=ep_title_safe,
                item=item,
                transcript_urls=[],
                media_url=TEST_MEDIA_URL,
                media_type=TEST_MEDIA_TYPE_MP3,
            )

            with patch("podcast_scraper.downloader.http_download_to_file") as mock_download:
                job = episode_processor.download_media_for_transcription(
                    episode,
                    cfg,
                    temp_dir,
                    tmpdir,
                    None,
                )

            self.assertIsNone(job)
            mock_download.assert_not_called()


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


class TestFormatScreenplay(unittest.TestCase):
    """Tests for format_screenplay_from_segments function."""

    def test_format_screenplay(self):
        """Test formatting screenplay from segments."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Hello, this is speaker one."},
            {"start": 6.0, "end": 10.0, "text": "And this is speaker two."},
            {"start": 15.0, "end": 20.0, "text": "Speaker one again."},
        ]
        result = podcast_scraper.whisper.format_screenplay_from_segments(
            segments, 2, ["Speaker1", "Speaker2"], 2.0
        )
        self.assertIn("Speaker1", result)
        self.assertIn("Speaker2", result)
        self.assertIn("Hello", result)


class MockHTTPResponse:
    """Simple mock for HTTP responses used in integration-style tests."""

    def __init__(self, *, content=b"", url="", headers=None, chunks=None):
        self.content = content
        self.url = url
        self.headers = headers or {}
        self._chunks = chunks if chunks is not None else [content]

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=1):
        for chunk in self._chunks:
            yield chunk

    def close(self):
        return None


class TestIntegrationMain(unittest.TestCase):
    """Higher-level integration tests using mocked HTTP responses."""

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
        rss_xml = f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Integration Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="{transcript_url}" type="text/plain" />
    </item>
  </channel>
</rss>
""".strip()
        transcript_text = "Episode 1 transcript"
        responses = {
            downloader.normalize_url(rss_url): MockHTTPResponse(
                content=rss_xml.encode("utf-8"),
                url=rss_url,
                headers={"Content-Type": "application/rss+xml"},
            ),
            downloader.normalize_url(transcript_url): MockHTTPResponse(
                url=transcript_url,
                headers={
                    "Content-Type": "text/plain",
                    "Content-Length": str(len(transcript_text.encode("utf-8"))),
                },
                chunks=[transcript_text.encode("utf-8")],
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                exit_code = cli.main([rss_url, "--output-dir", tmpdir])
                self.assertEqual(exit_code, 0)
                expected_path = os.path.join(tmpdir, "0001 - Episode 1.txt")
                self.assertTrue(os.path.exists(expected_path))
                with open(expected_path, "r", encoding="utf-8") as fh:
                    self.assertEqual(fh.read().strip(), transcript_text)

    def test_integration_main_whisper_fallback(self):
        rss_url = "https://example.com/feed.xml"
        media_url = "https://example.com/ep1.mp3"
        rss_xml = f"""<?xml version='1.0'?>
<rss>
  <channel>
    <title>Integration Feed</title>
    <item>
      <title>Episode 1</title>
      <enclosure url="{media_url}" type="audio/mpeg" />
    </item>
  </channel>
</rss>
""".strip()
        media_bytes = b"FAKE AUDIO DATA"
        responses = {
            downloader.normalize_url(rss_url): MockHTTPResponse(
                content=rss_xml.encode("utf-8"),
                url=rss_url,
                headers={"Content-Type": "application/rss+xml"},
            ),
            downloader.normalize_url(media_url): MockHTTPResponse(
                url=media_url,
                headers={"Content-Type": "audio/mpeg", "Content-Length": str(len(media_bytes))},
                chunks=[media_bytes],
            ),
        }

        mock_model = object()
        transcribed_text = "Hello from Whisper"

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with patch(
                "podcast_scraper.whisper.load_whisper_model", return_value=mock_model
            ) as mock_load:
                with patch(
                    "podcast_scraper.whisper.transcribe_with_whisper",
                    return_value=({"text": transcribed_text}, 1.0),
                ) as mock_transcribe:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        exit_code = cli.main(
                            [
                                rss_url,
                                "--output-dir",
                                tmpdir,
                                "--transcribe-missing",
                                "--run-id",
                                "testrun",
                            ]
                        )
                        self.assertEqual(exit_code, 0)
                        mock_load.assert_called_once()
                        mock_transcribe.assert_called_once()
                        effective_dir = Path(tmpdir).resolve() / "run_testrun_whisper_base"
                        out_path = effective_dir / "0001 - Episode 1_testrun_whisper_base.txt"
                        self.assertTrue(out_path.exists())
                        self.assertEqual(
                            out_path.read_text(encoding="utf-8").strip(), transcribed_text
                        )

    def test_path_traversal_attempt_normalized(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Integration Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="{transcript_url}" type="text/plain" />
    </item>
  </channel>
</rss>
""".strip()
        transcript_text = "Episode 1 transcript"
        responses = {
            downloader.normalize_url(rss_url): MockHTTPResponse(
                content=rss_xml.encode("utf-8"),
                url=rss_url,
            ),
            downloader.normalize_url(transcript_url): MockHTTPResponse(
                url=transcript_url,
                headers={"Content-Type": "text/plain"},
                chunks=[transcript_text.encode("utf-8")],
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                malicious = os.path.join(tmpdir, "..", "danger", "..", "final")
                exit_code = cli.main([rss_url, "--output-dir", malicious])
                self.assertEqual(exit_code, 0)
                effective_dir = Path(malicious).expanduser().resolve()
                out_path = effective_dir / "0001 - Episode 1.txt"
                self.assertTrue(out_path.exists())
                self.assertNotIn("..", str(out_path))

    def test_config_override_precedence_integration(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Integration Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="{transcript_url}" type="text/plain" />
    </item>
  </channel>
</rss>
""".strip()
        transcript_text = "Episode 1 transcript"
        responses = {
            downloader.normalize_url(rss_url): MockHTTPResponse(
                content=rss_xml.encode("utf-8"),
                url=rss_url,
            ),
            downloader.normalize_url(transcript_url): MockHTTPResponse(
                url=transcript_url,
                headers={"Content-Type": "text/plain"},
                chunks=[transcript_text.encode("utf-8")],
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
                        ]
                    )
                    self.assertEqual(exit_code, 0)
                    self.assertTrue(observed_timeouts)
                    self.assertTrue(all(timeout == 10 for timeout in observed_timeouts))
                    mock_apply.assert_called_with("DEBUG")

    def test_dry_run_skips_downloads(self):
        rss_url = "https://example.com/feed.xml"
        transcript_url = "https://example.com/ep1.txt"
        rss_xml = f"""<?xml version='1.0'?>
<rss xmlns:podcast="https://podcastindex.org/namespace/1.0">
  <channel>
    <title>Integration Feed</title>
    <item>
      <title>Episode 1</title>
      <podcast:transcript url="{transcript_url}" type="text/plain" />
    </item>
  </channel>
</rss>
""".strip()
        transcript_text = "Episode 1 transcript"
        responses = {
            downloader.normalize_url(rss_url): MockHTTPResponse(
                content=rss_xml.encode("utf-8"),
                url=rss_url,
                headers={"Content-Type": "application/rss+xml"},
            ),
            downloader.normalize_url(transcript_url): MockHTTPResponse(
                url=transcript_url,
                headers={
                    "Content-Type": "text/plain",
                    "Content-Length": str(len(transcript_text.encode("utf-8"))),
                },
                chunks=[transcript_text.encode("utf-8")],
            ),
        }

        http_mock = self._mock_http_map(responses)
        with patch("podcast_scraper.downloader.fetch_url", side_effect=http_mock):
            with tempfile.TemporaryDirectory() as tmpdir:
                expected_path = os.path.join(tmpdir, "0001 - Episode 1.txt")
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
                self.assertIn(
                    "Dry run complete. transcripts_planned=1 (direct=1, whisper=0)", log_text
                )
                self.assertIn("would save as", log_text)


if __name__ == "__main__":
    unittest.main()
