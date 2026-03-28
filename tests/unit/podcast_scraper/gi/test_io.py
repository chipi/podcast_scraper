#!/usr/bin/env python3
"""Tests for GIL artifact I/O."""

import pytest

from podcast_scraper.gi import read_artifact, write_artifact
from podcast_scraper.gi.pipeline import build_artifact


@pytest.mark.unit
class TestGILIO:
    """Read/write gi.json artifact."""

    def test_write_and_read_roundtrip(self, tmp_path):
        """Writing then reading returns same content."""
        payload = build_artifact("episode:1", "Hello world.", prompt_version="v1")
        path = tmp_path / "ep1.gi.json"
        write_artifact(path, payload, validate=True)
        assert path.exists()
        read_back = read_artifact(path)
        assert read_back["episode_id"] == payload["episode_id"]
        assert read_back["schema_version"] == payload["schema_version"]
        assert read_back["nodes"] == payload["nodes"]
        assert read_back["edges"] == payload["edges"]

    def test_write_creates_parent_dirs(self, tmp_path):
        """write_artifact creates parent directories."""
        path = tmp_path / "sub" / "dir" / "ep.gi.json"
        payload = build_artifact("ep:1", "")
        write_artifact(path, payload, validate=True)
        assert path.exists()

    def test_write_invalid_raises(self, tmp_path):
        """Writing invalid payload with validate=True raises."""
        path = tmp_path / "bad.gi.json"
        bad = {"schema_version": "2.0", "episode_id": "x", "nodes": [], "edges": []}
        with pytest.raises(ValueError, match="required key|1.0"):
            write_artifact(path, bad, validate=True)

    def test_read_artifact_missing_file_raises(self, tmp_path):
        """read_artifact raises FileNotFoundError when path does not exist."""
        path = tmp_path / "nonexistent.gi.json"
        assert not path.exists()
        with pytest.raises(FileNotFoundError):
            read_artifact(path)
