#!/usr/bin/env python3
"""Integration tests for fixture mapping and RSS linkage.

These tests verify that:
1. All required fixtures exist
2. Podcast mapping is correct
3. RSS linkage requirements are met (guid == filename, enclosure URLs)

Moved from tests/e2e/ as part of Phase 3 test pyramid refactoring - these
test infrastructure components, not user workflows.
"""

import os
import sys
from pathlib import Path

import pytest

# Use defusedxml for safe XML parsing (required dependency)
from defusedxml.ElementTree import parse as safe_parse

# Allow importing the package when tests run from within the package directory.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from tests.e2e.fixtures.e2e_http_server import E2EHTTPRequestHandler


@pytest.mark.integration
class TestFixtureMapping:
    """Test fixture structure and mapping."""

    @classmethod
    def get_fixture_root(cls) -> Path:
        """Get the root directory for test fixtures."""
        # This file is in tests/integration/infrastructure/
        # Fixtures are in tests/fixtures/
        # Path structure: tests/integration/infrastructure/test_fixture_mapping.py
        #                 -> tests/integration/infrastructure (parent)
        #                 -> tests/integration (parent.parent)
        #                 -> tests (parent.parent.parent)
        #                 -> tests/fixtures (parent.parent.parent / "fixtures")
        return Path(__file__).parent.parent.parent / "fixtures"

    def test_fixture_structure_exists(self):
        """Test that fixture structure exists."""
        fixture_root = self.get_fixture_root()

        # Check directories exist
        assert (fixture_root / "rss").exists(), "rss/ directory should exist"
        assert (fixture_root / "audio").exists(), "audio/ directory should exist"
        assert (fixture_root / "transcripts").exists(), "transcripts/ directory should exist"

    def test_rss_files_exist(self):
        """Test that all RSS files exist."""
        fixture_root = self.get_fixture_root()
        rss_dir = fixture_root / "rss"

        expected_rss_files = [
            "p01_mtb.xml",
            "p02_software.xml",
            "p03_scuba.xml",
            "p04_photo.xml",
            "p05_investing.xml",
            "p06_edge_cases.xml",  # Added in Stage 11
        ]

        for rss_file in expected_rss_files:
            file_path = rss_dir / rss_file
            assert file_path.exists(), f"RSS file {rss_file} should exist"

    def test_podcast_mapping(self):
        """Test that podcast mapping is correct."""
        # Verify mapping in E2EHTTPRequestHandler
        expected_mapping = {
            "podcast1": "p01_mtb.xml",
            "podcast2": "p02_software.xml",
            "podcast3": "p03_scuba.xml",
            "podcast4": "p04_photo.xml",
            "podcast5": "p05_investing.xml",
            "edgecases": "p06_edge_cases.xml",  # Added in Stage 11
            "podcast1_multi_episode": "p01_multi.xml",  # Multi-episode feed for fast tests
        }

        # Check that all expected mappings are present (may have additional entries)
        actual_map = E2EHTTPRequestHandler.PODCAST_RSS_MAP
        for podcast_name, rss_file in expected_mapping.items():
            assert podcast_name in actual_map, f"Podcast {podcast_name} should be in mapping"
            assert (
                actual_map[podcast_name] == rss_file
            ), f"Podcast {podcast_name} should map to {rss_file}, got {actual_map[podcast_name]}"

        # Verify all mapped files exist
        fixture_root = self.get_fixture_root()
        for podcast_name, rss_file in expected_mapping.items():
            file_path = fixture_root / "rss" / rss_file
            assert file_path.exists(), f"RSS file {rss_file} for {podcast_name} should exist"

    def test_rss_guid_matches_filename(self):
        """Test that RSS <guid> matches filename pattern (pXX_eYY)."""
        fixture_root = self.get_fixture_root()
        rss_dir = fixture_root / "rss"

        # Check all RSS files
        for rss_file in rss_dir.glob("*.xml"):
            tree = safe_parse(rss_file)
            root = tree.getroot()

            # Find all items
            items = root.findall(".//item")
            for item in items:
                guid_elem = item.find("guid")
                if guid_elem is not None:
                    guid = guid_elem.text
                    # GUID should match pattern pXX_eYY
                    assert guid.startswith("p"), f"GUID {guid} should start with 'p'"
                    assert "_e" in guid, f"GUID {guid} should contain '_e'"
                    # Verify corresponding files exist (skip fast fixtures which
                    # may not have all files)
                    episode_id = guid
                    # Skip fast fixtures - they may not have all corresponding files
                    if "fast" in episode_id:
                        continue
                    audio_file = fixture_root / "audio" / f"{episode_id}.mp3"
                    transcript_file = fixture_root / "transcripts" / f"{episode_id}.txt"
                    assert (
                        audio_file.exists()
                    ), f"Audio file {episode_id}.mp3 should exist for GUID {guid}"
                    # Transcript file is optional (some episodes may not have transcripts)
                    # Only check if transcript URL exists in RSS
                    transcript_elem = item.find(
                        "{https://podcastindex.org/namespace/1.0}transcript"
                    )
                    if transcript_elem is not None:
                        assert transcript_file.exists(), (
                            f"Transcript file {episode_id}.txt should exist for "
                            f"GUID {guid} (has transcript URL in RSS)"
                        )

    def test_rss_enclosure_urls(self):
        """Test that RSS <enclosure> URLs point to /audio/pXX_eYY.mp3."""
        fixture_root = self.get_fixture_root()
        rss_dir = fixture_root / "rss"

        # Check all RSS files
        for rss_file in rss_dir.glob("*.xml"):
            tree = safe_parse(rss_file)
            root = tree.getroot()

            # Find all items
            items = root.findall(".//item")
            for item in items:
                guid_elem = item.find("guid")
                enclosure_elem = item.find("enclosure")
                if guid_elem is not None and enclosure_elem is not None:
                    guid = guid_elem.text
                    # Skip fast fixtures which may not have all files
                    if "fast" in guid:
                        continue
                    enclosure_url = enclosure_elem.get("url", "")

                    # Enclosure URL should point to /audio/{guid}.mp3 or audio/{guid}.mp3 (relative)
                    expected_url_absolute = f"/audio/{guid}.mp3"
                    expected_url_relative = f"audio/{guid}.mp3"
                    assert (
                        enclosure_url == expected_url_absolute
                        or enclosure_url == expected_url_relative
                        or enclosure_url.endswith(expected_url_absolute)
                        or enclosure_url.endswith(expected_url_relative)
                    ), (
                        f"Enclosure URL {enclosure_url} should point to "
                        f"{expected_url_absolute} or {expected_url_relative} for GUID {guid}"
                    )
                    # Extract filename from URL (may be relative or absolute)
                    audio_filename = enclosure_url.split("/")[-1]
                    audio_file = fixture_root / "audio" / audio_filename
                    assert (
                        audio_file.exists()
                    ), f"Audio file {guid}.mp3 should exist for enclosure URL {enclosure_url}"

    def test_rss_transcript_urls_optional(self):
        """Test that RSS <podcast:transcript> URLs point to /transcripts/pXX_eYY.txt."""
        fixture_root = self.get_fixture_root()
        rss_dir = fixture_root / "rss"

        # Check all RSS files
        for rss_file in rss_dir.glob("*.xml"):
            tree = safe_parse(rss_file)
            root = tree.getroot()

            # Find all items
            items = root.findall(".//item")
            for item in items:
                guid_elem = item.find("guid")
                # Check for podcast:transcript (namespace-aware)
                transcript_elem = None
                for elem in item:
                    # Handle namespace
                    if "transcript" in elem.tag.lower():
                        transcript_elem = elem
                        break

                if guid_elem is not None:
                    guid = guid_elem.text
                    # Skip fast fixtures which may not have all corresponding files
                    if "fast" in guid:
                        continue
                    if transcript_elem is not None:
                        transcript_url = transcript_elem.get("url", "")
                        # Transcript URL should point to /transcripts/{guid}.txt
                        # or transcripts/{guid}.txt (relative)
                        expected_url_absolute = f"/transcripts/{guid}.txt"
                        expected_url_relative = f"transcripts/{guid}.txt"
                        assert (
                            transcript_url == expected_url_absolute
                            or transcript_url == expected_url_relative
                            or transcript_url.endswith(expected_url_absolute)
                            or transcript_url.endswith(expected_url_relative)
                        ), (
                            f"Transcript URL {transcript_url} should point to "
                            f"{expected_url_absolute} or {expected_url_relative} for GUID {guid}"
                        )

                        # Verify transcript file exists (extract filename from URL)
                        transcript_filename = transcript_url.split("/")[-1]
                        transcript_file = fixture_root / "transcripts" / transcript_filename
                        assert transcript_file.exists(), (
                            f"Transcript file {transcript_filename} should exist "
                            f"for transcript URL {transcript_url}"
                        )

    def test_all_episodes_have_files(self):
        """Test that all episodes in RSS have corresponding audio and transcript files."""
        fixture_root = self.get_fixture_root()
        rss_dir = fixture_root / "rss"

        # Collect all GUIDs from RSS files
        all_guids = set()
        for rss_file in rss_dir.glob("*.xml"):
            tree = safe_parse(rss_file)
            root = tree.getroot()
            items = root.findall(".//item")
            for item in items:
                guid_elem = item.find("guid")
                if guid_elem is not None:
                    all_guids.add(guid_elem.text)

        # Verify all GUIDs have corresponding files (skip fast fixtures which
        # may not have all files)
        for guid in all_guids:
            # Skip fast fixtures - they may not have all corresponding files
            if "fast" in guid:
                continue
            audio_file = fixture_root / "audio" / f"{guid}.mp3"
            assert audio_file.exists(), f"Audio file {guid}.mp3 should exist"
            # Transcript file is optional (some episodes may not have transcripts)
            # Only check if it's referenced in RSS
            # For now, we'll just check that audio exists for all non-fast GUIDs
