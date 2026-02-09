"""Unit tests for podcast_scraper.workflow.run_index module.

Tests for run index generation and helper functions.
"""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import pytest

from podcast_scraper import models
from podcast_scraper.workflow import run_index


@pytest.mark.unit
class TestBuildStatusMap(unittest.TestCase):
    """Tests for _build_status_map helper function."""

    def test_build_status_map_with_statuses(self):
        """Test building status map from episode statuses."""
        status1 = Mock()
        status1.episode_id = "ep1"
        status1.status = "ok"
        status1.error_type = None
        status1.error_message = None
        status1.stage = "transcribed"

        status2 = Mock()
        status2.episode_id = "ep2"
        status2.status = "failed"
        status2.error_type = "ValueError"
        status2.error_message = "Test error"
        status2.stage = "transcription"

        statuses = [status1, status2]
        result = run_index._build_status_map(statuses)

        self.assertEqual(len(result), 2)
        self.assertEqual(result["ep1"]["status"], "ok")
        self.assertEqual(result["ep1"]["stage"], "transcribed")
        self.assertIsNone(result["ep1"]["error_type"])
        self.assertEqual(result["ep2"]["status"], "failed")
        self.assertEqual(result["ep2"]["error_type"], "ValueError")
        self.assertEqual(result["ep2"]["error_message"], "Test error")
        self.assertEqual(result["ep2"]["stage"], "transcription")

    def test_build_status_map_with_none(self):
        """Test building status map with None input."""
        result = run_index._build_status_map(None)
        self.assertEqual(result, {})

    def test_build_status_map_with_empty_list(self):
        """Test building status map with empty list."""
        result = run_index._build_status_map([])
        self.assertEqual(result, {})

    def test_build_status_map_with_missing_episode_id(self):
        """Test building status map when episode_id is missing."""
        status = Mock()
        status.episode_id = None
        status.status = "ok"

        result = run_index._build_status_map([status])
        self.assertEqual(result, {})

    def test_build_status_map_with_default_values(self):
        """Test building status map with status objects missing some attributes."""
        status = Mock(spec=["episode_id"])  # Only episode_id attribute exists
        status.episode_id = "ep1"
        # Missing status, error_type, error_message, stage attributes

        result = run_index._build_status_map([status])
        self.assertEqual(len(result), 1)
        self.assertEqual(result["ep1"]["status"], "ok")  # Default value
        self.assertIsNone(result["ep1"]["error_type"])
        self.assertIsNone(result["ep1"]["error_message"])
        self.assertIsNone(result["ep1"]["stage"])


@pytest.mark.unit
class TestFindTranscriptFile(unittest.TestCase):
    """Tests for _find_transcript_file helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.transcripts_dir = os.path.join(self.temp_dir, "transcripts")
        os.makedirs(self.transcripts_dir, exist_ok=True)

        self.episode = Mock()
        self.episode.idx = 1
        self.episode_title_safe = "Test_Episode"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_transcript_file_with_run_suffix(self):
        """Test finding transcript file with run suffix."""
        run_suffix = "testrun"
        transcript_path = os.path.join(
            self.transcripts_dir, f"0001 - {self.episode_title_safe}_{run_suffix}.txt"
        )
        with open(transcript_path, "w") as f:
            f.write("Transcript content")

        result = run_index._find_transcript_file(
            self.episode,
            self.episode_title_safe,
            self.transcripts_dir,
            self.temp_dir,
            run_suffix,
        )

        self.assertIsNotNone(result)
        self.assertIn("transcripts", result)
        self.assertIn(self.episode_title_safe, result)

    def test_find_transcript_file_without_run_suffix(self):
        """Test finding transcript file without run suffix."""
        transcript_path = os.path.join(
            self.transcripts_dir, f"0001 - {self.episode_title_safe}.txt"
        )
        with open(transcript_path, "w") as f:
            f.write("Transcript content")

        result = run_index._find_transcript_file(
            self.episode,
            self.episode_title_safe,
            self.transcripts_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNotNone(result)
        self.assertIn("transcripts", result)

    def test_find_transcript_file_different_extensions(self):
        """Test finding transcript file with different extensions."""
        # Test each extension separately to avoid finding previous files
        for ext in [".md", ".html", ".vtt", ".srt"]:
            # Clean up any existing files first
            import glob

            existing = glob.glob(
                os.path.join(self.transcripts_dir, f"0001 - {self.episode_title_safe}*")
            )
            for f in existing:
                os.remove(f)

            transcript_path = os.path.join(
                self.transcripts_dir, f"0001 - {self.episode_title_safe}{ext}"
            )
            with open(transcript_path, "w") as f:
                f.write("Content")

            result = run_index._find_transcript_file(
                self.episode,
                self.episode_title_safe,
                self.transcripts_dir,
                self.temp_dir,
                None,
            )

            self.assertIsNotNone(result, f"Should find file with {ext} extension")
            # Result is relative path, check it contains the extension
            self.assertIn(ext, result or "")

    def test_find_transcript_file_not_found(self):
        """Test finding transcript file when it doesn't exist."""
        result = run_index._find_transcript_file(
            self.episode,
            self.episode_title_safe,
            self.transcripts_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNone(result)

    def test_find_transcript_file_directory_not_exists(self):
        """Test finding transcript file when directory doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, "nonexistent")
        result = run_index._find_transcript_file(
            self.episode,
            self.episode_title_safe,
            non_existent_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNone(result)

    def test_find_transcript_file_glob_fallback(self):
        """Test finding transcript file using glob fallback."""
        # Create file with slightly different pattern
        transcript_path = os.path.join(
            self.transcripts_dir, f"0001 - {self.episode_title_safe}_other_suffix.txt"
        )
        with open(transcript_path, "w") as f:
            f.write("Content")

        result = run_index._find_transcript_file(
            self.episode,
            self.episode_title_safe,
            self.transcripts_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNotNone(result)
        self.assertIn("transcripts", result)


@pytest.mark.unit
class TestFindMetadataFile(unittest.TestCase):
    """Tests for _find_metadata_file helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.episode = Mock()
        self.episode.idx = 1
        self.episode_title_safe = "Test_Episode"

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_metadata_file_with_run_suffix(self):
        """Test finding metadata file with run suffix."""
        run_suffix = "testrun"
        metadata_path = os.path.join(
            self.metadata_dir,
            f"0001 - {self.episode_title_safe}_{run_suffix}.metadata.json",
        )
        with open(metadata_path, "w") as f:
            f.write('{"test": "data"}')

        result = run_index._find_metadata_file(
            self.episode,
            self.episode_title_safe,
            self.metadata_dir,
            self.temp_dir,
            run_suffix,
        )

        self.assertIsNotNone(result)
        self.assertIn("metadata", result)
        self.assertIn(".metadata.json", result)

    def test_find_metadata_file_without_run_suffix(self):
        """Test finding metadata file without run suffix."""
        metadata_path = os.path.join(
            self.metadata_dir, f"0001 - {self.episode_title_safe}.metadata.json"
        )
        with open(metadata_path, "w") as f:
            f.write('{"test": "data"}')

        result = run_index._find_metadata_file(
            self.episode,
            self.episode_title_safe,
            self.metadata_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNotNone(result)
        self.assertIn("metadata", result)

    def test_find_metadata_file_different_extensions(self):
        """Test finding metadata file with different extensions."""
        # Test each extension separately to avoid finding previous files
        for ext in [".yaml", ".yml"]:
            # Clean up any existing files first
            import glob

            existing = glob.glob(
                os.path.join(self.metadata_dir, f"0001 - {self.episode_title_safe}.metadata*")
            )
            for f in existing:
                os.remove(f)

            metadata_path = os.path.join(
                self.metadata_dir, f"0001 - {self.episode_title_safe}.metadata{ext}"
            )
            with open(metadata_path, "w") as f:
                f.write("test: data")

            result = run_index._find_metadata_file(
                self.episode,
                self.episode_title_safe,
                self.metadata_dir,
                self.temp_dir,
                None,
            )

            self.assertIsNotNone(result, f"Should find file with {ext} extension")
            self.assertIn(f".metadata{ext}", result)

    def test_find_metadata_file_not_found(self):
        """Test finding metadata file when it doesn't exist."""
        result = run_index._find_metadata_file(
            self.episode,
            self.episode_title_safe,
            self.metadata_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNone(result)

    def test_find_metadata_file_directory_not_exists(self):
        """Test finding metadata file when directory doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, "nonexistent")
        result = run_index._find_metadata_file(
            self.episode,
            self.episode_title_safe,
            non_existent_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNone(result)

    def test_find_metadata_file_glob_fallback(self):
        """Test finding metadata file using glob fallback."""
        # Create file with slightly different pattern
        metadata_path = os.path.join(
            self.metadata_dir,
            f"0001 - {self.episode_title_safe}_other.metadata.json",
        )
        with open(metadata_path, "w") as f:
            f.write('{"test": "data"}')

        result = run_index._find_metadata_file(
            self.episode,
            self.episode_title_safe,
            self.metadata_dir,
            self.temp_dir,
            None,
        )

        self.assertIsNotNone(result)
        self.assertIn("metadata", result)


@pytest.mark.unit
class TestDetermineEpisodeStatus(unittest.TestCase):
    """Tests for _determine_episode_status helper function."""

    def setUp(self):
        """Set up test fixtures."""
        self.episode = Mock()

    def test_determine_episode_status_with_metadata(self):
        """Test status determination when metadata exists."""
        status = run_index._determine_episode_status(
            metadata_path="metadata.json",
            transcript_path=None,
            status_from_map=None,
            episode=self.episode,
        )
        self.assertEqual(status, "ok")

    def test_determine_episode_status_with_transcript_only(self):
        """Test status determination when only transcript exists."""
        status = run_index._determine_episode_status(
            metadata_path=None,
            transcript_path="transcript.txt",
            status_from_map=None,
            episode=self.episode,
        )
        self.assertEqual(status, "ok")

    def test_determine_episode_status_with_status_from_map(self):
        """Test status determination using status from map."""
        status = run_index._determine_episode_status(
            metadata_path=None,
            transcript_path=None,
            status_from_map="failed",
            episode=self.episode,
        )
        self.assertEqual(status, "failed")

    def test_determine_episode_status_skipped_no_transcript_url(self):
        """Test status determination when episode has no transcript URL."""
        self.episode.transcript_url = None
        status = run_index._determine_episode_status(
            metadata_path=None,
            transcript_path=None,
            status_from_map=None,
            episode=self.episode,
        )
        self.assertEqual(status, "skipped")

    def test_determine_episode_status_skipped_no_transcript_url_attr(self):
        """Test status determination when episode has no transcript_url attribute."""

        # Create episode without transcript_url attribute
        # Use a simple object that doesn't have transcript_url
        class EpisodeWithoutTranscriptUrl:
            pass

        episode = EpisodeWithoutTranscriptUrl()

        status = run_index._determine_episode_status(
            metadata_path=None,
            transcript_path=None,
            status_from_map=None,
            episode=episode,
        )
        # hasattr(episode, "transcript_url") will be False, so it should return "skipped"
        self.assertEqual(status, "skipped")

    def test_determine_episode_status_failed_with_transcript_url(self):
        """Test status determination when episode has transcript URL but no files."""
        self.episode.transcript_url = "https://example.com/transcript.txt"
        status = run_index._determine_episode_status(
            metadata_path=None,
            transcript_path=None,
            status_from_map=None,
            episode=self.episode,
        )
        self.assertEqual(status, "failed")


@pytest.mark.unit
class TestExtractEpisodeMetadataForId(unittest.TestCase):
    """Tests for _extract_episode_metadata_for_id helper function."""

    def test_extract_episode_metadata_with_item(self):
        """Test extracting metadata from episode with RSS item."""
        item = ET.Element("item")
        guid_elem = ET.SubElement(item, "guid")
        guid_elem.text = "episode-guid-123"
        link_elem = ET.SubElement(item, "link")
        link_elem.text = "https://example.com/episode"
        pub_date_elem = ET.SubElement(item, "pubDate")
        pub_date_elem.text = "Mon, 01 Jan 2024 12:00:00 GMT"

        episode = Mock()
        episode.item = item
        episode.number = 1

        guid, link, published_date, number = run_index._extract_episode_metadata_for_id(episode)

        self.assertEqual(guid, "episode-guid-123")
        self.assertEqual(link, "https://example.com/episode")
        self.assertIsNotNone(published_date)
        self.assertEqual(number, 1)

    def test_extract_episode_metadata_without_item(self):
        """Test extracting metadata from episode without RSS item."""
        episode = Mock()
        episode.item = None
        episode.number = None

        guid, link, published_date, number = run_index._extract_episode_metadata_for_id(episode)

        self.assertIsNone(guid)
        self.assertIsNone(link)
        self.assertIsNone(published_date)
        self.assertIsNone(number)

    def test_extract_episode_metadata_with_missing_elements(self):
        """Test extracting metadata when RSS item has missing elements."""
        item = ET.Element("item")
        # No guid, link, or pubDate elements

        episode = Mock()
        episode.item = item
        episode.number = 2

        guid, link, published_date, number = run_index._extract_episode_metadata_for_id(episode)

        self.assertIsNone(guid)
        self.assertIsNone(link)
        self.assertIsNone(published_date)
        self.assertEqual(number, 2)

    def test_extract_episode_metadata_with_empty_text(self):
        """Test extracting metadata when RSS elements have empty text."""
        item = ET.Element("item")
        guid_elem = ET.SubElement(item, "guid")
        guid_elem.text = ""  # Empty text
        link_elem = ET.SubElement(item, "link")
        link_elem.text = "   "  # Whitespace only

        episode = Mock()
        episode.item = item
        episode.number = None

        guid, link, published_date, number = run_index._extract_episode_metadata_for_id(episode)

        self.assertIsNone(guid)  # Empty text should result in None
        # Whitespace-only text gets stripped to empty string, which is truthy but empty
        # The function returns stripped text, so check it's empty string
        self.assertEqual(link, "")


@pytest.mark.unit
class TestCreateRunIndex(unittest.TestCase):
    """Tests for create_run_index main function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.transcripts_dir = os.path.join(self.temp_dir, "transcripts")
        self.metadata_dir = os.path.join(self.temp_dir, "metadata")
        os.makedirs(self.transcripts_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_run_index_basic(self):
        """Test creating run index with basic episode."""
        item = ET.Element("item")
        ET.SubElement(item, "title").text = "Episode 1"
        ET.SubElement(item, "guid").text = "ep1"

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[],
        )

        # Create transcript file
        transcript_path = os.path.join(self.transcripts_dir, "0001 - episode-1.txt")
        with open(transcript_path, "w") as f:
            f.write("Transcript")

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[episode],
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=None,
        )

        self.assertEqual(run_idx.run_id, "test-run")
        self.assertEqual(run_idx.feed_url, "https://example.com/feed.xml")
        self.assertEqual(len(run_idx.episodes), 1)
        self.assertEqual(run_idx.episodes_processed, 1)
        self.assertEqual(run_idx.episodes_failed, 0)
        self.assertEqual(run_idx.episodes_skipped, 0)
        self.assertEqual(run_idx.episodes[0].status, "ok")
        self.assertIsNotNone(run_idx.episodes[0].transcript_path)

    def test_create_run_index_with_metadata(self):
        """Test creating run index with metadata file."""
        item = ET.Element("item")
        ET.SubElement(item, "title").text = "Episode 1"
        ET.SubElement(item, "guid").text = "ep1"

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[],
        )

        # Create metadata file
        metadata_path = os.path.join(self.metadata_dir, "0001 - episode-1.metadata.json")
        with open(metadata_path, "w") as f:
            f.write('{"test": "data"}')

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[episode],
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=None,
        )

        self.assertEqual(run_idx.episodes[0].status, "ok")
        self.assertIsNotNone(run_idx.episodes[0].metadata_path)

    def test_create_run_index_with_statuses(self):
        """Test creating run index with episode statuses."""
        item = ET.Element("item")
        ET.SubElement(item, "title").text = "Episode 1"
        ET.SubElement(item, "guid").text = "ep1"

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[],
        )

        # Create status object
        status = Mock()
        status.episode_id = "test-episode-id"  # Will be generated from episode
        status.status = "failed"
        status.error_type = "ValueError"
        status.error_message = "Test error"
        status.stage = "transcription"

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[episode],
            effective_output_dir=self.temp_dir,
            episode_statuses=[status],
            run_suffix=None,
        )

        # Status from filesystem takes precedence (no files = failed/skipped)
        # But error info from status_map should be included
        self.assertEqual(len(run_idx.episodes), 1)

    def test_create_run_index_with_multiple_episodes(self):
        """Test creating run index with multiple episodes."""
        episodes = []
        for i in range(3):
            item = ET.Element("item")
            ET.SubElement(item, "title").text = f"Episode {i+1}"
            ET.SubElement(item, "guid").text = f"ep{i+1}"

            episode = models.Episode(
                idx=i + 1,
                title=f"Episode {i+1}",
                title_safe=f"episode-{i+1}",
                item=item,
                transcript_urls=[],
            )
            episodes.append(episode)

            # Create transcript for first episode only
            if i == 0:
                transcript_path = os.path.join(
                    self.transcripts_dir, f"000{i+1} - episode-{i+1}.txt"
                )
                with open(transcript_path, "w") as f:
                    f.write("Transcript")

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=episodes,
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=None,
        )

        self.assertEqual(len(run_idx.episodes), 3)
        self.assertEqual(run_idx.episodes_processed, 1)
        # Episodes 2 and 3 have no files and no transcript_urls, so they're "skipped"
        self.assertEqual(run_idx.episodes_skipped, 2)
        self.assertEqual(run_idx.episodes_failed, 0)

    def test_create_run_index_with_run_suffix(self):
        """Test creating run index with run suffix."""
        item = ET.Element("item")
        ET.SubElement(item, "title").text = "Episode 1"
        ET.SubElement(item, "guid").text = "ep1"

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[],
        )

        run_suffix = "testrun"
        # Create transcript file with run suffix
        transcript_path = os.path.join(self.transcripts_dir, f"0001 - episode-1_{run_suffix}.txt")
        with open(transcript_path, "w") as f:
            f.write("Transcript")

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[episode],
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=run_suffix,
        )

        self.assertEqual(run_idx.episodes[0].status, "ok")
        self.assertIsNotNone(run_idx.episodes[0].transcript_path)
        self.assertIn(run_suffix, run_idx.episodes[0].transcript_path)

    def test_create_run_index_empty_episodes(self):
        """Test creating run index with empty episodes list."""
        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[],
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=None,
        )

        self.assertEqual(len(run_idx.episodes), 0)
        self.assertEqual(run_idx.episodes_processed, 0)
        self.assertEqual(run_idx.episodes_failed, 0)
        self.assertEqual(run_idx.episodes_skipped, 0)

    def test_create_run_index_skipped_episode(self):
        """Test creating run index with skipped episode (no transcript URL)."""
        item = ET.Element("item")
        ET.SubElement(item, "title").text = "Episode 1"
        ET.SubElement(item, "guid").text = "ep1"

        episode = models.Episode(
            idx=1,
            title="Episode 1",
            title_safe="episode-1",
            item=item,
            transcript_urls=[],  # No transcript URL
        )

        run_idx = run_index.create_run_index(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[episode],
            effective_output_dir=self.temp_dir,
            episode_statuses=None,
            run_suffix=None,
        )

        self.assertEqual(run_idx.episodes[0].status, "skipped")
        self.assertEqual(run_idx.episodes_skipped, 1)


@pytest.mark.unit
class TestRunIndexDataclass(unittest.TestCase):
    """Tests for RunIndex and EpisodeIndexEntry dataclasses."""

    def test_run_index_to_dict(self):
        """Test RunIndex.to_dict() method."""
        entry = run_index.EpisodeIndexEntry(
            episode_id="ep1",
            status="ok",
            transcript_path="transcript.txt",
            metadata_path="metadata.json",
        )

        run_idx = run_index.RunIndex(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[entry],
        )

        result = run_idx.to_dict()

        self.assertEqual(result["run_id"], "test-run")
        self.assertEqual(result["feed_url"], "https://example.com/feed.xml")
        self.assertEqual(len(result["episodes"]), 1)
        self.assertEqual(result["episodes"][0]["episode_id"], "ep1")
        self.assertEqual(result["episodes"][0]["status"], "ok")

    def test_run_index_to_json(self):
        """Test RunIndex.to_json() method."""
        entry = run_index.EpisodeIndexEntry(
            episode_id="ep1", status="ok", transcript_path="transcript.txt"
        )

        run_idx = run_index.RunIndex(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[entry],
        )

        json_str = run_idx.to_json()
        self.assertIn("test-run", json_str)
        self.assertIn("ep1", json_str)

    def test_run_index_save_to_file(self):
        """Test RunIndex.save_to_file() method."""
        import tempfile

        entry = run_index.EpisodeIndexEntry(
            episode_id="ep1", status="ok", transcript_path="transcript.txt"
        )

        run_idx = run_index.RunIndex(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=[entry],
        )

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            run_idx.save_to_file(temp_path)

            # Verify file was created and contains expected content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path, "r") as f:
                content = f.read()
                self.assertIn("test-run", content)
                self.assertIn("ep1", content)
        finally:
            os.unlink(temp_path)

    def test_run_index_post_init(self):
        """Test RunIndex.__post_init__ initializes episodes list."""
        run_idx = run_index.RunIndex(
            run_id="test-run",
            feed_url="https://example.com/feed.xml",
            episodes=None,  # None should be converted to empty list
        )

        self.assertIsNotNone(run_idx.episodes)
        self.assertEqual(len(run_idx.episodes), 0)
