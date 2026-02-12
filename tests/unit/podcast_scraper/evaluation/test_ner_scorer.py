"""Unit tests for podcast_scraper.evaluation.ner_scorer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.evaluation.ner_scorer import (
    hash_text,
    match_exact,
    match_overlap,
    normalize_entity_text,
    validate_gold_episode,
    validate_gold_index,
)


@pytest.mark.unit
class TestHashText:
    """Tests for hash_text."""

    def test_deterministic(self):
        """Same input produces same hash."""
        assert hash_text("hello") == hash_text("hello")

    def test_different_inputs_different_hashes(self):
        """Different inputs produce different hashes."""
        assert hash_text("a") != hash_text("b")

    def test_hex_digest_length(self):
        """SHA256 hex digest is 64 chars."""
        assert len(hash_text("x")) == 64
        assert all(c in "0123456789abcdef" for c in hash_text("x"))


@pytest.mark.unit
class TestNormalizeEntityText:
    """Tests for normalize_entity_text."""

    def test_strips_whitespace(self):
        """Whitespace is stripped."""
        assert normalize_entity_text("  Alice  ") == "alice"

    def test_strips_trailing_punctuation(self):
        """Trailing punctuation like colon/comma is stripped."""
        assert normalize_entity_text("Maya:") == "maya"
        assert normalize_entity_text("Liam,") == "liam"

    def test_lowercase(self):
        """Output is lowercased."""
        assert normalize_entity_text("Alice") == "alice"


@pytest.mark.unit
class TestValidateGoldIndex:
    """Tests for validate_gold_index."""

    def test_valid_index_passes(self):
        """Valid index with required fields does not raise."""
        schema_path = Path("/nonexistent/schema.json")
        index_data = {
            "schema": "ner_entities_gold_index_v1",
            "dataset_id": "ds1",
            "episodes": [],
        }
        validate_gold_index(index_data, schema_path)

    def test_missing_required_field_raises(self):
        """Missing required field raises ValueError."""
        schema_path = Path("/nonexistent/schema.json")
        with pytest.raises(ValueError, match="required field"):
            validate_gold_index({"schema": "ner_entities_gold_index_v1"}, schema_path)

    def test_invalid_schema_raises(self):
        """Invalid schema value raises ValueError."""
        schema_path = Path("/nonexistent/schema.json")
        with pytest.raises(ValueError, match="Invalid schema"):
            validate_gold_index(
                {"schema": "wrong", "dataset_id": "ds1", "episodes": []},
                schema_path,
            )


@pytest.mark.unit
class TestValidateGoldEpisode:
    """Tests for validate_gold_episode."""

    def test_valid_episode_passes(self):
        """Valid episode with required fields does not raise."""
        schema_path = Path("/nonexistent/schema.json")
        episode_data = {
            "schema": "ner_entities_gold_v1",
            "dataset_id": "ds1",
            "episode_id": "e1",
            "text_fingerprint": "abc",
            "entities": [],
        }
        validate_gold_episode(episode_data, schema_path)

    def test_missing_required_field_raises(self):
        """Missing required field raises ValueError."""
        schema_path = Path("/nonexistent/schema.json")
        with pytest.raises(ValueError, match="required field"):
            validate_gold_episode({"schema": "ner_entities_gold_v1"}, schema_path)


@pytest.mark.unit
class TestMatchExact:
    """Tests for match_exact."""

    def test_exact_match_returns_true(self):
        """Same label, start, end returns True."""
        pred = {"label": "PERSON", "start": 0, "end": 5, "text": "Alice"}
        gold = {"label": "PERSON", "start": 0, "end": 5, "text": "Alice"}
        assert match_exact(pred, gold) is True

    def test_different_label_returns_false(self):
        """Different label returns False."""
        pred = {"label": "PERSON", "start": 0, "end": 5}
        gold = {"label": "ORG", "start": 0, "end": 5}
        assert match_exact(pred, gold) is False

    def test_different_span_returns_false(self):
        """Different start/end returns False."""
        pred = {"label": "PERSON", "start": 0, "end": 5}
        gold = {"label": "PERSON", "start": 1, "end": 5}
        assert match_exact(pred, gold) is False


@pytest.mark.unit
class TestMatchOverlap:
    """Tests for match_overlap."""

    def test_same_label_overlapping_span_returns_true(self):
        """Same label and overlapping spans returns True."""
        pred = {"label": "PERSON", "start": 0, "end": 10}
        gold = {"label": "PERSON", "start": 5, "end": 15}
        assert match_overlap(pred, gold) is True

    def test_different_label_returns_false(self):
        """Different label returns False."""
        pred = {"label": "PERSON", "start": 0, "end": 10}
        gold = {"label": "ORG", "start": 5, "end": 15}
        assert match_overlap(pred, gold) is False

    def test_non_overlapping_span_returns_false(self):
        """Non-overlapping spans returns False."""
        pred = {"label": "PERSON", "start": 0, "end": 5}
        gold = {"label": "PERSON", "start": 10, "end": 15}
        assert match_overlap(pred, gold) is False
