"""Unit tests for ``episode_duration_ms_map`` (RFC-097 v3.0 chunk-5).

The eval harness uses this helper to thread dataset-declared
``duration_minutes`` into ``gi.build_artifact`` so the ``position_hint``
waterfall's step 1 (RSS / dataset-declared duration) actually fires.
Without it, every eval-built insight has ``position_hint: None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.evaluation.experiment_config import episode_duration_ms_map

pytestmark = pytest.mark.unit


def _write_dataset(tmp_path: Path, dataset_id: str, episodes: list) -> Path:
    """Write a minimal dataset JSON under ``data/eval/datasets/`` shape.

    Returns the dataset path. Caller patches ``load_dataset_json`` to point
    here, since the helper looks up by id under the canonical path.
    """
    d = tmp_path / f"{dataset_id}.json"
    d.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "version": "1.0",
                "description": "test fixture",
                "created_at": "2026-06-22T00:00:00Z",
                "content_regime": "test",
                "num_episodes": len(episodes),
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )
    return d


def _patch_loader(dataset_dict: dict):
    """Patch ``load_dataset_json`` to return *dataset_dict* regardless of id."""
    return patch(
        "podcast_scraper.evaluation.experiment_config.load_dataset_json",
        return_value=dataset_dict,
    )


class TestEpisodeDurationMsMap:
    """Behavioural coverage for the dataset → duration_ms helper."""

    def test_empty_dataset_id_returns_empty_map(self):
        assert episode_duration_ms_map(None) == {}
        assert episode_duration_ms_map("") == {}

    def test_missing_dataset_file_returns_empty_map(self):
        """Helper swallows ``FileNotFoundError`` — callers no-op cleanly."""
        # No patch; the id is genuinely not on disk.
        assert episode_duration_ms_map("definitely-not-a-real-dataset-id") == {}

    def test_converts_minutes_to_ms_correctly(self):
        ds = {
            "dataset_id": "x",
            "episodes": [
                {"episode_id": "ep1", "duration_minutes": 10.5},
                {"episode_id": "ep2", "duration_minutes": 2.0},
            ],
        }
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        # 10.5 min = 630 sec = 630000 ms
        assert out["ep1"] == 630000
        assert out["ep2"] == 120000

    def test_integer_minutes_handled(self):
        """Integer ``duration_minutes`` (no decimal) converts the same way."""
        ds = {"episodes": [{"episode_id": "ep1", "duration_minutes": 5}]}
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {"ep1": 300000}

    def test_string_minutes_handled(self):
        """Stringified ``duration_minutes`` (e.g. ``"10.5"``) is coerced via float."""
        ds = {"episodes": [{"episode_id": "ep1", "duration_minutes": "10.5"}]}
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {"ep1": 630000}

    def test_missing_duration_skipped(self):
        """Episodes without ``duration_minutes`` are absent from the map."""
        ds = {
            "episodes": [
                {"episode_id": "ep1", "duration_minutes": 5.0},
                {"episode_id": "ep2"},  # no duration_minutes
            ],
        }
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert "ep1" in out
        assert "ep2" not in out

    def test_zero_or_negative_duration_skipped(self):
        """Non-positive values are skipped — they'd violate the waterfall contract."""
        ds = {
            "episodes": [
                {"episode_id": "ep1", "duration_minutes": 0},
                {"episode_id": "ep2", "duration_minutes": -5.0},
                {"episode_id": "ep3", "duration_minutes": 1.0},
            ],
        }
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {"ep3": 60000}

    def test_unparseable_duration_skipped(self):
        """Non-numeric ``duration_minutes`` doesn't crash; episode is just dropped."""
        ds = {
            "episodes": [
                {"episode_id": "ep1", "duration_minutes": "about ten"},
                {"episode_id": "ep2", "duration_minutes": 5.0},
            ],
        }
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {"ep2": 300000}

    def test_invalid_episode_entries_skipped(self):
        """Non-dict episode entries are skipped silently (defensive)."""
        ds = {
            "episodes": [
                "not a dict",
                {"episode_id": "ep1", "duration_minutes": 1.0},
                None,
                {"duration_minutes": 1.0},  # no episode_id
            ],
        }
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {"ep1": 60000}

    def test_empty_episodes_list_returns_empty_map(self):
        ds = {"episodes": []}
        with _patch_loader(ds):
            out = episode_duration_ms_map("x")
        assert out == {}

    def test_curated_5feeds_dev_v1_real_dataset(self):
        """End-to-end: the real prod-eval dataset has duration_minutes; the
        helper loads it without monkey-patching. Guards regression where the
        canonical dataset format changes shape.
        """
        out = episode_duration_ms_map("curated_5feeds_dev_v1")
        # 10 episodes in this dataset; all should have duration_minutes set.
        assert len(out) == 10
        for eid, ms in out.items():
            assert ms > 0, f"{eid} has non-positive duration_ms: {ms}"
