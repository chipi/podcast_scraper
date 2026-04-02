"""Tests for GIL I/O helpers."""

import pytest

from podcast_scraper.gi.io import (
    collect_gi_paths_from_inputs,
    read_artifact,
    write_artifact,
)


@pytest.mark.unit
class TestCollectGiPaths:
    """collect_gi_paths_from_inputs."""

    def test_file_must_be_gi_json(self, tmp_path):
        """Reject non-.gi.json files."""
        p = tmp_path / "x.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="Not a .gi.json"):
            collect_gi_paths_from_inputs([p])

    def test_missing_path_raises(self, tmp_path):
        missing = tmp_path / "nope.gi.json"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            collect_gi_paths_from_inputs([missing])

    def test_directory_rglob(self, tmp_path):
        """Directory scan finds nested .gi.json."""
        (tmp_path / "metadata").mkdir()
        g = tmp_path / "metadata" / "a.gi.json"
        g.write_text("{}", encoding="utf-8")
        found = collect_gi_paths_from_inputs([tmp_path])
        assert g in found


@pytest.mark.unit
class TestReadWriteArtifact:
    """read_artifact / write_artifact round-trip."""

    def test_round_trip_minimal_payload(self, tmp_path):
        payload = {
            "schema_version": "1.0",
            "model_version": "m",
            "prompt_version": "p",
            "episode_id": "e1",
            "nodes": [],
            "edges": [],
        }
        path = tmp_path / "out.gi.json"
        write_artifact(path, payload, validate=True)
        loaded = read_artifact(path)
        assert loaded["episode_id"] == "e1"
        assert loaded["nodes"] == []
