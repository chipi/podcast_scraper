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
        """schema_version other than 1.0/2.0 raises."""
        data = {
            "schema_version": "9.9",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        with pytest.raises(ValueError, match="1.0"):
            validate_artifact(data, strict=False)

    def test_validate_artifact_minimal_2_0_valid(self):
        """schema_version 2.0 passes minimal validation."""
        data = {
            "schema_version": "2.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
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

    def test_validate_artifact_minimal_3_0_valid(self):
        """RFC-097 v3.0 passes minimal validation."""
        data = {
            "schema_version": "3.0",
            "model_version": "stub",
            "prompt_version": "v1",
            "episode_id": "episode:1",
            "nodes": [],
            "edges": [],
        }
        validate_artifact(data, strict=False)

    def test_validate_artifact_strict_accepts_v3_organization_and_mentions(self):
        """RFC-097 v3.0: Organization node + MENTIONS_PERSON/MENTIONS_ORG edges validate strict."""
        data = {
            "schema_version": "3.0",
            "model_version": "stub",
            "prompt_version": "v3",
            "episode_id": "episode:1",
            "nodes": [
                {
                    "id": "episode:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "podcast:p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "person:pat",
                    "type": "Person",
                    "properties": {"name": "Pat"},
                },
                {
                    "id": "org:openai",
                    "type": "Organization",
                    "properties": {"name": "OpenAI"},
                },
                {
                    "id": "topic:ai",
                    "type": "Topic",
                    "properties": {"label": "AI"},
                },
                {
                    "id": "insight:e1:abc",
                    "type": "Insight",
                    "properties": {
                        "text": "AI will be regulated.",
                        "episode_id": "episode:1",
                        "grounded": True,
                        "insight_type": "claim",
                        "position_hint": 0.42,
                    },
                },
            ],
            "edges": [
                {
                    "type": "MENTIONS_PERSON",
                    "from": "insight:e1:abc",
                    "to": "person:pat",
                    "properties": {"confidence": 0.85},
                },
                {
                    "type": "MENTIONS_ORG",
                    "from": "insight:e1:abc",
                    "to": "org:openai",
                    "properties": {"confidence": 0.71},
                },
                {
                    "type": "ABOUT",
                    "from": "insight:e1:abc",
                    "to": "topic:ai",
                    "properties": {"confidence": 0.79},
                },
            ],
        }
        validate_artifact(data, strict=True)

    def test_validate_artifact_strict_grounding_contract_preserved(self):
        """RFC-097 invariant: descriptive edges (ABOUT/MENTIONS_*) don't promote grounded.

        The schema cannot enforce the grounded ⇔ ≥1 SUPPORTED_BY invariant directly
        (that's pipeline-level), but we verify it accepts an ungrounded Insight that
        has descriptive edges — proving descriptive edges don't auto-promote.
        """
        data = {
            "schema_version": "3.0",
            "model_version": "stub",
            "prompt_version": "v3",
            "episode_id": "episode:1",
            "nodes": [
                {
                    "id": "episode:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "podcast:p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "topic:ai",
                    "type": "Topic",
                    "properties": {"label": "AI"},
                },
                {
                    "id": "insight:e1:abc",
                    "type": "Insight",
                    "properties": {
                        "text": "AI will be regulated.",
                        "episode_id": "episode:1",
                        "grounded": False,
                    },
                },
            ],
            "edges": [
                {
                    "type": "ABOUT",
                    "from": "insight:e1:abc",
                    "to": "topic:ai",
                    "properties": {"confidence": 0.79},
                },
            ],
        }
        validate_artifact(data, strict=True)
