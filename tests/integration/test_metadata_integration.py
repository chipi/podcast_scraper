#!/usr/bin/env python3
"""Integration tests for metadata generation.

These tests perform real file I/O operations and test the full metadata generation workflow.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from podcast_scraper import config, metadata

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

create_test_config = parent_conftest.create_test_config
create_test_episode = parent_conftest.create_test_episode
create_test_feed = parent_conftest.create_test_feed
TEST_FEED_URL = parent_conftest.TEST_FEED_URL
TEST_FEED_TITLE = parent_conftest.TEST_FEED_TITLE
TEST_EPISODE_TITLE = parent_conftest.TEST_EPISODE_TITLE
TEST_TRANSCRIPT_TYPE_SRT = parent_conftest.TEST_TRANSCRIPT_TYPE_SRT
TEST_TRANSCRIPT_TYPE_VTT = parent_conftest.TEST_TRANSCRIPT_TYPE_VTT
TEST_TRANSCRIPT_URL = parent_conftest.TEST_TRANSCRIPT_URL
TEST_TRANSCRIPT_URL_SRT = parent_conftest.TEST_TRANSCRIPT_URL_SRT


@pytest.mark.integration
@pytest.mark.critical_path
class TestMetadataGenerationIntegration(unittest.TestCase):
    """Integration tests for metadata generation that perform real file I/O."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.feed = create_test_feed()
        self.episode = create_test_episode()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_metadata_json(self):
        """Test metadata generation in JSON format."""
        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_Title.vtt",
            transcript_source="direct_download",
            whisper_model=None,
            detected_hosts=["Test Host"],
            detected_guests=["Test Guest"],
        )

        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))
        self.assertTrue(metadata_path.endswith(".metadata.json"))

        # Verify JSON is valid and contains expected fields
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("feed", data)
        self.assertIn("episode", data)
        self.assertIn("content", data)
        self.assertIn("processing", data)

        # Check feed metadata
        self.assertEqual(data["feed"]["title"], TEST_FEED_TITLE)
        self.assertEqual(data["feed"]["url"], TEST_FEED_URL)
        self.assertIn("feed_id", data["feed"])
        self.assertEqual(data["feed"]["authors"], ["Test Host"])

        # Check episode metadata
        self.assertEqual(data["episode"]["title"], TEST_EPISODE_TITLE)
        self.assertIn("episode_id", data["episode"])

        # Check content metadata
        self.assertEqual(data["content"]["transcript_source"], "direct_download")
        self.assertEqual(data["content"]["detected_hosts"], ["Test Host"])
        self.assertEqual(data["content"]["detected_guests"], ["Test Guest"])

        # Check processing metadata
        self.assertEqual(data["processing"]["schema_version"], "1.0.0")
        self.assertIn("processing_timestamp", data["processing"])

    def test_metadata_generation_with_transcription(self):
        """Test metadata generation with Whisper transcription source (critical path).

        This test validates that metadata correctly includes transcription metadata
        when transcript is created via Whisper transcription. This is part of the
        critical path integration test coverage.
        """
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        # Generate metadata with transcription source
        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_Title.txt",
            transcript_source="whisper_transcription",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            detected_hosts=["Test Host"],
            detected_guests=["Test Guest"],
        )

        # Verify metadata file was created
        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))
        self.assertTrue(metadata_path.endswith(".metadata.json"))

        # Verify metadata content
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify transcript source is whisper_transcription
        self.assertEqual(data["content"]["transcript_source"], "whisper_transcription")

        # Verify transcription metadata fields are present
        self.assertEqual(
            data["content"]["whisper_model"], config.TEST_DEFAULT_WHISPER_MODEL
        )  # Metadata stores model name as-is (with .en suffix if provided)

        # Verify transcript file path is included
        self.assertIn("transcript_file_path", data["content"])
        self.assertEqual(data["content"]["transcript_file_path"], "0001 - Episode_Title.txt")

        # Verify other content fields
        self.assertEqual(data["content"]["detected_hosts"], ["Test Host"])
        self.assertEqual(data["content"]["detected_guests"], ["Test Guest"])

        # Verify processing metadata
        self.assertIn("processing", data)
        self.assertIn("processing_timestamp", data["processing"])


@pytest.mark.integration
@pytest.mark.slow
class TestMetadataGenerationComprehensive(unittest.TestCase):
    """Comprehensive tests for metadata generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.feed = create_test_feed()
        self.episode = create_test_episode()
        self.cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_metadata_json_structure(self):
        """Test that JSON metadata has correct structure."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check top-level structure
        self.assertIn("feed", data)
        self.assertIn("episode", data)
        self.assertIn("content", data)
        self.assertIn("processing", data)

        # Check feed structure
        self.assertIn("feed_id", data["feed"])
        self.assertIn("title", data["feed"])
        self.assertIn("url", data["feed"])

        # Check episode structure
        self.assertIn("episode_id", data["episode"])
        self.assertIn("title", data["episode"])

        # Check content structure
        self.assertIn("transcript_urls", data["content"])

        # Check processing structure
        self.assertIn("schema_version", data["processing"])
        self.assertIn("processing_timestamp", data["processing"])

    def test_metadata_yaml_structure(self):
        """Test that YAML metadata has correct structure."""
        import yaml

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="yaml",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Check top-level structure
        self.assertIn("feed", data)
        self.assertIn("episode", data)
        self.assertIn("content", data)
        self.assertIn("processing", data)

    def test_metadata_with_whisper_transcription(self):
        """Test metadata generation with Whisper transcription."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            transcript_file_path="0001 - Episode_Title.txt",
            transcript_source="whisper_transcription",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["content"]["transcript_source"], "whisper_transcription")
        self.assertEqual(data["content"]["whisper_model"], config.TEST_DEFAULT_WHISPER_MODEL)

    def test_metadata_with_subdirectory(self):
        """Test metadata generation with subdirectory."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
            metadata_subdirectory="metadata",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        # Should be in subdirectory
        self.assertIn("metadata", metadata_path)
        self.assertTrue(os.path.exists(metadata_path))

    def test_metadata_with_run_suffix(self):
        """Test metadata generation with run suffix."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix="test_run",
        )

        # Should include run suffix in filename
        self.assertIn("test_run", metadata_path)

    def test_metadata_iso8601_dates(self):
        """Test that dates are in ISO 8601 format."""
        from datetime import datetime

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        feed = create_test_feed()

        metadata_path = metadata.generate_episode_metadata(
            feed=feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            feed_last_updated=datetime(2024, 1, 15, 10, 30, 0),
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check that dates are ISO 8601 strings
        if "last_updated" in data["feed"]:
            date_str = data["feed"]["last_updated"]
            # Should be ISO 8601 format
            self.assertIn("T", date_str or "")
            self.assertIn(":", date_str or "")

        # Processing timestamp should be ISO 8601
        proc_timestamp = data["processing"]["processing_timestamp"]
        self.assertIn("T", proc_timestamp)
        self.assertIn(":", proc_timestamp)

    def test_metadata_multiple_transcript_urls(self):
        """Test metadata with multiple transcript URLs."""
        episode = create_test_episode(
            transcript_urls=[
                (TEST_TRANSCRIPT_URL, TEST_TRANSCRIPT_TYPE_VTT),
                (TEST_TRANSCRIPT_URL_SRT, TEST_TRANSCRIPT_TYPE_SRT),
            ],
        )

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
        )

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Should have multiple transcript URLs
        self.assertEqual(len(data["content"]["transcript_urls"]), 2)
        self.assertEqual(data["content"]["transcript_urls"][0]["url"], TEST_TRANSCRIPT_URL)
        self.assertEqual(data["content"]["transcript_urls"][1]["url"], TEST_TRANSCRIPT_URL_SRT)

    def test_generate_metadata_yaml(self):
        """Test metadata generation in YAML format."""
        import yaml

        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="yaml",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_Title.vtt",
            transcript_source="direct_download",
        )

        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))
        self.assertTrue(metadata_path.endswith(".metadata.yaml"))

        # Verify YAML is valid
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("feed", data)
        self.assertIn("episode", data)

    def test_generate_metadata_with_subdirectory(self):
        """Test metadata generation with subdirectory."""
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_subdirectory="metadata",
        )

        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
        )

        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))
        self.assertIn("metadata", metadata_path)

    def test_generate_metadata_with_whisper_transcription(self):
        """Test metadata generation for Whisper transcription."""
        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=self.cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_Title.txt",
            transcript_source="whisper_transcription",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,  # Use test default (tiny.en), not base
        )

        self.assertIsNotNone(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["content"]["transcript_source"], "whisper_transcription")
        self.assertEqual(data["content"]["whisper_model"], config.TEST_DEFAULT_WHISPER_MODEL)

    def test_metadata_generation_with_transcription(self):
        """Test metadata generation with Whisper transcription source (critical path).

        This test validates that metadata correctly includes transcription metadata
        when transcript is created via Whisper transcription. This is part of the
        critical path integration test coverage.
        """
        cfg = create_test_config(
            output_dir=self.temp_dir,
            generate_metadata=True,
            metadata_format="json",
        )

        # Generate metadata with transcription source
        metadata_path = metadata.generate_episode_metadata(
            feed=self.feed,
            episode=self.episode,
            feed_url=TEST_FEED_URL,
            cfg=cfg,
            output_dir=self.temp_dir,
            run_suffix=None,
            transcript_file_path="0001 - Episode_Title.txt",
            transcript_source="whisper_transcription",
            whisper_model=config.TEST_DEFAULT_WHISPER_MODEL,
            detected_hosts=["Test Host"],
            detected_guests=["Test Guest"],
        )

        # Verify metadata file was created
        self.assertIsNotNone(metadata_path)
        self.assertTrue(os.path.exists(metadata_path))
        self.assertTrue(metadata_path.endswith(".metadata.json"))

        # Verify metadata content
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify transcript source is whisper_transcription
        self.assertEqual(data["content"]["transcript_source"], "whisper_transcription")

        # Verify transcription metadata fields are present
        self.assertEqual(
            data["content"]["whisper_model"], config.TEST_DEFAULT_WHISPER_MODEL
        )  # Metadata stores model name as-is (with .en suffix if provided)

        # Verify transcript file path is included
        self.assertIn("transcript_file_path", data["content"])
        self.assertEqual(data["content"]["transcript_file_path"], "0001 - Episode_Title.txt")

        # Verify other content fields
        self.assertEqual(data["content"]["detected_hosts"], ["Test Host"])
        self.assertEqual(data["content"]["detected_guests"], ["Test Guest"])

        # Verify processing metadata
        self.assertIn("processing", data)
        self.assertIn("processing_timestamp", data["processing"])
