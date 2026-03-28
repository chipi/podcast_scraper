#!/usr/bin/env python3
"""Tests for GIL schema validation."""

import pytest

from podcast_scraper.gi.schema import get_schema_path, load_schema, validate_artifact


@pytest.mark.unit
class TestGILSchema:
    """Minimal and strict validation of GIL artifacts."""

    def test_validate_artifact_minimal_valid(self):
        """Valid minimal artifact passes."""
        data = {
            "schema_version": "1.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        validate_artifact(data, strict=False)

    def test_validate_artifact_missing_required_key(self):
        """Missing required key raises ValueError."""
        data = {
            "schema_version": "1.0",
            "model_version": "stub",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        with pytest.raises(ValueError, match="required key"):
            validate_artifact(data, strict=False)

    def test_validate_artifact_bad_schema_version(self):
        """schema_version other than 1.0 raises."""
        data = {
            "schema_version": "2.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        with pytest.raises(ValueError, match="1.0"):
            validate_artifact(data, strict=False)

    def test_validate_artifact_nodes_not_array(self):
        """nodes must be an array."""
        data = {
            "schema_version": "1.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": {},
            "edges": [],
        }
        with pytest.raises(ValueError, match="nodes"):
            validate_artifact(data, strict=False)

    def test_validate_artifact_edges_not_array(self):
        """edges must be an array."""
        data = {
            "schema_version": "1.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": {},
        }
        with pytest.raises(ValueError, match="edges"):
            validate_artifact(data, strict=False)

    def test_get_schema_path_returns_path_or_none(self):
        """get_schema_path returns Path or None."""
        path = get_schema_path()
        if path is not None:
            assert path.name == "gi.schema.json"
            assert path.suffix == ".json"

    def test_load_schema_returns_dict_or_none(self):
        """load_schema returns schema dict or None."""
        schema = load_schema()
        if schema is not None:
            assert "$schema" in schema or "schema_version" in schema or "properties" in schema

    def test_validate_artifact_strict_rejects_extra_top_level_key(self):
        """Minimal valid artifact passes non-strict; with extra key fails strict."""
        data = {
            "schema_version": "1.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        validate_artifact(data, strict=False)
        data["extra_key"] = "not_allowed"
        with pytest.raises(
            ValueError, match="schema validation failed|additionalProperties|GIL artifact"
        ):
            validate_artifact(data, strict=True)
