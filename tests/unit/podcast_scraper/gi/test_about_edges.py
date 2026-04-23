"""Unit tests for gi.about_edges semantic edge ranking (#664)."""

from __future__ import annotations

import numpy as np
import pytest

from podcast_scraper.gi.about_edges import (
    ABOUT_EDGE_DEFAULT_FLOOR,
    ABOUT_EDGE_DEFAULT_TOP_K,
    rank_about_edges,
)


class _StubEncoder:
    """Returns unit vectors by text lookup. Tests drive the cosine matrix
    by choosing the vectors, which makes every outcome deterministic."""

    def __init__(self, vectors_by_text):
        self._by_text = {k: np.asarray(v, dtype=float) for k, v in vectors_by_text.items()}

    def encode(self, texts, normalize_embeddings=True):
        arr = np.stack([self._by_text[t] for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


@pytest.mark.unit
class TestRankAboutEdges:
    def test_top_k_respected(self):
        """With K=2 and three topics, only the top two cosines survive."""
        enc = _StubEncoder(
            {
                "insight text": [1.0, 0.0, 0.0],
                # cosines relative to insight: 0.9, 0.6, 0.3
                "topic A": [0.9, np.sqrt(1 - 0.81), 0.0],
                "topic B": [0.6, np.sqrt(1 - 0.36), 0.0],
                "topic C": [0.3, np.sqrt(1 - 0.09), 0.0],
            }
        )
        result = rank_about_edges(
            ["insight text"],
            [("t:a", "topic A"), ("t:b", "topic B"), ("t:c", "topic C")],
            top_k=2,
            floor=0.0,
            encoder=enc,
        )
        assert len(result) == 1
        assert [tid for tid, _ in result[0]] == ["t:a", "t:b"]
        assert result[0][0][1] == pytest.approx(0.9, abs=1e-3)

    def test_floor_filters_below_threshold(self):
        """Topics with cosine below floor are dropped entirely, even if top_k allows."""
        enc = _StubEncoder(
            {
                "x": [1.0, 0.0, 0.0],
                # cosines: 0.9, 0.1 — second is below 0.25 floor.
                "a": [0.9, np.sqrt(1 - 0.81), 0.0],
                "b": [0.1, np.sqrt(1 - 0.01), 0.0],
            }
        )
        result = rank_about_edges(
            ["x"],
            [("t:a", "a"), ("t:b", "b")],
            top_k=2,
            floor=0.25,
            encoder=enc,
        )
        assert len(result[0]) == 1
        assert result[0][0][0] == "t:a"

    def test_insight_with_no_topics_above_floor_returns_empty(self):
        """Orphan insights (no topic above floor) get zero ABOUT edges."""
        enc = _StubEncoder(
            {
                "drift": [1.0, 0.0, 0.0],
                "unrelated": [0.05, np.sqrt(1 - 0.0025), 0.0],
            }
        )
        result = rank_about_edges(
            ["drift"],
            [("t:unrelated", "unrelated")],
            top_k=2,
            floor=0.25,
            encoder=enc,
        )
        assert result == [[]]

    def test_empty_inputs_short_circuit(self):
        """Empty insights or topics return empty rows without invoking encoder."""

        class _ExplodingEncoder:
            def encode(self, texts, normalize_embeddings=True):
                raise AssertionError("should not be called on empty input")

        enc = _ExplodingEncoder()
        assert rank_about_edges([], [("t:a", "a")], encoder=enc) == []
        assert rank_about_edges(["x"], [], encoder=enc) == [[]]

    def test_result_ordering_is_descending_by_cosine(self):
        """Returned tuples are always sorted by cosine descending."""
        enc = _StubEncoder(
            {
                "i": [1.0, 0.0, 0.0],
                "low": [0.4, np.sqrt(1 - 0.16), 0.0],
                "high": [0.8, np.sqrt(1 - 0.64), 0.0],
                "mid": [0.6, np.sqrt(1 - 0.36), 0.0],
            }
        )
        result = rank_about_edges(
            ["i"],
            [("t:low", "low"), ("t:high", "high"), ("t:mid", "mid")],
            top_k=3,
            floor=0.0,
            encoder=enc,
        )
        cosines = [c for _, c in result[0]]
        assert cosines == sorted(cosines, reverse=True)
        assert [tid for tid, _ in result[0]] == ["t:high", "t:mid", "t:low"]

    def test_multiple_insights_scored_independently(self):
        """Each insight gets its own top-K ranking against the same topics."""
        enc = _StubEncoder(
            {
                "i1": [1.0, 0.0, 0.0],
                "i2": [0.0, 1.0, 0.0],
                "oil": [0.9, 0.1, 0.0],
                "tech": [0.1, 0.9, 0.0],
            }
        )
        result = rank_about_edges(
            ["i1", "i2"],
            [("t:oil", "oil"), ("t:tech", "tech")],
            top_k=1,
            floor=0.0,
            encoder=enc,
        )
        assert result[0][0][0] == "t:oil"
        assert result[1][0][0] == "t:tech"

    def test_defaults_are_k2_floor_025(self):
        """Guard that the module-level defaults match the chosen sweep winner."""
        assert ABOUT_EDGE_DEFAULT_TOP_K == 2
        assert ABOUT_EDGE_DEFAULT_FLOOR == 0.25
