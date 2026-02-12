"""Tests for evaluation metadata validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from podcast_scraper.evaluation.metadata_validator import (
    validate_and_load_metadata,
    validate_episode_metadata,
)


@pytest.mark.unit
class TestValidateEpisodeMetadata:
    """Test validate_episode_metadata."""

    def test_valid_minimal(self):
        """Accepts metadata with episode_id and metadata_version."""
        meta = {"episode_id": "ep1", "metadata_version": "1.0"}
        validate_episode_metadata(meta, "ep1")

    def test_valid_with_source_episode_id(self):
        """Accepts source_episode_id instead of episode_id."""
        meta = {"source_episode_id": "ep1", "metadata_version": "1.0"}
        validate_episode_metadata(meta, "ep1")

    def test_valid_with_speakers_list(self):
        """Accepts speakers as list."""
        meta = {
            "episode_id": "ep1",
            "metadata_version": "1.0",
            "speakers": [{"name": "Host"}, {"name": "Guest"}],
        }
        validate_episode_metadata(meta, "ep1")

    def test_missing_episode_id_and_source(self):
        """Raises AssertionError when both episode_id and source_episode_id missing."""
        meta = {"metadata_version": "1.0"}
        with pytest.raises(AssertionError, match="Missing .episode_id. or .source_episode_id"):
            validate_episode_metadata(meta, "ep1")

    def test_missing_metadata_version(self):
        """Raises AssertionError when metadata_version missing."""
        meta = {"episode_id": "ep1"}
        with pytest.raises(AssertionError, match="Missing .metadata_version"):
            validate_episode_metadata(meta, "ep1")

    def test_speakers_must_be_list(self):
        """Raises AssertionError when speakers is not a list."""
        meta = {"episode_id": "ep1", "metadata_version": "1.0", "speakers": "not-a-list"}
        with pytest.raises(AssertionError, match="must be a list"):
            validate_episode_metadata(meta, "ep1")


@pytest.mark.unit
class TestValidateAndLoadMetadata:
    """Test validate_and_load_metadata."""

    def test_none_when_file_missing(self):
        """Returns None when path does not exist."""
        result = validate_and_load_metadata("/nonexistent/path.json", "ep1")
        assert result is None

    def test_load_and_validate_success(self):
        """Loads and returns validated metadata when file is valid."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"episode_id": "ep1", "metadata_version": "1.0"}, f)
            path = f.name
        try:
            result = validate_and_load_metadata(path, "ep1")
            assert result is not None
            assert result["episode_id"] == "ep1"
            assert result["metadata_version"] == "1.0"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_returns_none_on_invalid_json(self):
        """Returns None when file is not valid JSON."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("not json {")
            path = f.name
        try:
            result = validate_and_load_metadata(path, "ep1")
            assert result is None
        finally:
            Path(path).unlink(missing_ok=True)

    def test_raises_on_validation_failure(self):
        """Raises AssertionError when JSON valid but metadata invalid."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"metadata_version": "1.0"}, f)
            path = f.name
        try:
            with pytest.raises(AssertionError, match="Missing .episode_id"):
                validate_and_load_metadata(path, "ep1")
        finally:
            Path(path).unlink(missing_ok=True)
