"""Tests for GIL I/O helpers."""

import pytest

from podcast_scraper.gi.io import collect_gi_paths_from_inputs


@pytest.mark.unit
class TestCollectGiPaths:
    """collect_gi_paths_from_inputs."""

    def test_file_must_be_gi_json(self, tmp_path):
        """Reject non-.gi.json files."""
        p = tmp_path / "x.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Not a .gi.json"):
            collect_gi_paths_from_inputs([p])

    def test_directory_rglob(self, tmp_path):
        """Directory scan finds nested .gi.json."""
        (tmp_path / "metadata").mkdir()
        g = tmp_path / "metadata" / "a.gi.json"
        g.write_text("{}", encoding="utf-8")
        found = collect_gi_paths_from_inputs([tmp_path])
        assert g in found
