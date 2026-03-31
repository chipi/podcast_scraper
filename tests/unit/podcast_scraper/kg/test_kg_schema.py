"""Unit tests for KG schema validation."""

import json
import unittest
from pathlib import Path

import pytest

from podcast_scraper.kg.schema import validate_artifact

pytestmark = [pytest.mark.unit]

_FIXTURE_MINIMAL = Path(__file__).resolve().parents[3] / "fixtures" / "kg" / "minimal.kg.json"


class TestKgSchema(unittest.TestCase):
    """Tests for validate_artifact."""

    def test_minimal_validate_missing_key(self) -> None:
        """Minimal validation rejects missing top-level keys."""
        with self.assertRaises(ValueError) as ctx:
            validate_artifact({"schema_version": "1.0"}, strict=False)
        self.assertIn("missing required key", str(ctx.exception).lower())

    def test_minimal_validate_bad_schema_version(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(
                {
                    "schema_version": "2.0",
                    "episode_id": "e:1",
                    "extraction": {
                        "model_version": "stub",
                        "extracted_at": "2024-01-01T00:00:00Z",
                        "transcript_ref": "t.txt",
                    },
                    "nodes": [],
                    "edges": [],
                },
                strict=False,
            )
        self.assertIn("schema_version", str(ctx.exception))

    def test_minimal_fixture_passes_strict_json_schema(self) -> None:
        """Checked-in minimal.kg.json matches v1 schema (#464)."""
        data = json.loads(_FIXTURE_MINIMAL.read_text(encoding="utf-8"))
        validate_artifact(data, strict=True)

    def test_strict_rejects_noncanonical_model_version(self) -> None:
        """extraction.model_version must be stub, summary_bullets, or provider:*."""
        base = {
            "schema_version": "1.0",
            "episode_id": "e:1",
            "extraction": {
                "model_version": "legacy-tier",
                "extracted_at": "2024-01-01T00:00:00Z",
                "transcript_ref": "t.txt",
            },
            "nodes": [
                {
                    "id": "episode:e:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                }
            ],
            "edges": [],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(base, strict=True)
        self.assertIn("validation failed", str(ctx.exception).lower())

    def test_strict_rejects_invalid_entity_role(self) -> None:
        """Entity.properties.role is host | guest | mentioned when present."""
        art = {
            "schema_version": "1.0",
            "episode_id": "e:1",
            "extraction": {
                "model_version": "stub",
                "extracted_at": "2024-01-01T00:00:00Z",
                "transcript_ref": "t.txt",
            },
            "nodes": [
                {
                    "id": "episode:e:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "entity:person:pat",
                    "type": "Entity",
                    "properties": {
                        "name": "Pat",
                        "entity_kind": "person",
                        "role": "moderator",
                    },
                },
            ],
            "edges": [],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_artifact(art, strict=True)
        self.assertIn("validation failed", str(ctx.exception).lower())

    def test_strict_accepts_reserved_related_to_edge(self) -> None:
        """RELATED_TO is allowed by schema though v1 builder does not emit it."""
        art = {
            "schema_version": "1.0",
            "episode_id": "e:1",
            "extraction": {
                "model_version": "stub",
                "extracted_at": "2024-01-01T00:00:00Z",
                "transcript_ref": "t.txt",
            },
            "nodes": [
                {
                    "id": "episode:e:1",
                    "type": "Episode",
                    "properties": {
                        "podcast_id": "p",
                        "title": "T",
                        "publish_date": "2024-01-01T00:00:00Z",
                    },
                },
                {
                    "id": "topic:a",
                    "type": "Topic",
                    "properties": {"label": "A", "slug": "a"},
                },
                {
                    "id": "topic:b",
                    "type": "Topic",
                    "properties": {"label": "B", "slug": "b"},
                },
            ],
            "edges": [
                {
                    "type": "RELATED_TO",
                    "from": "topic:a",
                    "to": "topic:b",
                    "properties": {},
                }
            ],
        }
        validate_artifact(art, strict=True)
