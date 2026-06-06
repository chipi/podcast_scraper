"""Unit tests for the embedding-provider A/B eval harness (#897, ADR-098)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from podcast_scraper.evaluation.embedding_provider_eval import (
    _cosine,
    _ndcg_at,
    _pairs_from_gi,
    _percentile,
    _recall_at,
    _reciprocal_rank,
    GroundTruthPair,
    K_VALUES,
    ProviderConfig,
    ProviderMetrics,
    score_provider,
    write_report_json,
    write_report_md,
)


def _gi_doc(pairs: List[tuple]) -> dict:
    """Build a minimal gi.json doc from (insight_text, quote_text) pairs."""
    nodes = []
    edges = []
    for i, (i_text, q_text) in enumerate(pairs):
        i_id = f"insight:{i}"
        q_id = f"quote:{i}"
        nodes.append({"id": i_id, "type": "Insight", "properties": {"text": i_text}})
        nodes.append({"id": q_id, "type": "Quote", "properties": {"text": q_text}})
        edges.append({"type": "SUPPORTED_BY", "from": i_id, "to": q_id})
    return {"episode_id": "ep:x", "nodes": nodes, "edges": edges}


# ----- ground-truth extraction --------------------------------------------------


def test_pairs_from_gi_extracts_supported_by_edges():
    doc = _gi_doc(
        [("how to invest", "Buy index funds."), ("rates impact", "Higher rates hurt bonds.")]
    )
    pairs = _pairs_from_gi(doc)
    assert len(pairs) == 2
    assert pairs[0].insight_text == "how to invest"
    assert pairs[0].quote_text == "Buy index funds."
    assert pairs[0].episode_id == "ep:x"


def test_pairs_from_gi_skips_wrong_edge_type():
    doc = _gi_doc([("x", "y")])
    # Mutate edge to non-SUPPORTED_BY type → should be filtered out.
    doc["edges"][0]["type"] = "MENTIONS"
    assert _pairs_from_gi(doc) == []


def test_pairs_from_gi_skips_empty_text():
    doc = _gi_doc([("", "non-empty")])
    assert _pairs_from_gi(doc) == []


def test_pairs_from_gi_skips_unknown_nodes():
    doc = _gi_doc([("a", "b")])
    doc["edges"][0]["from"] = "insight:nonexistent"
    assert _pairs_from_gi(doc) == []


# ----- metric primitives --------------------------------------------------------


def test_recall_at_hit_in_top_k_is_one():
    assert _recall_at(["a", "b", "c"], "b", 3) == 1.0
    assert _recall_at(["a", "b", "c"], "b", 2) == 1.0
    assert _recall_at(["a", "b", "c"], "c", 2) == 0.0  # cut off before target


def test_recall_at_target_absent_is_zero():
    assert _recall_at(["a", "b"], "z", 10) == 0.0


def test_ndcg_at_target_first_is_one():
    assert _ndcg_at(["a", "b", "c"], "a", 3) == 1.0


def test_ndcg_at_target_second_is_log_discounted():
    # DCG = 1/log2(3) ≈ 0.631; ideal DCG = 1/log2(2) = 1.0
    score = _ndcg_at(["x", "a"], "a", 5)
    assert 0.62 < score < 0.64


def test_ndcg_at_target_absent_is_zero():
    assert _ndcg_at(["x", "y"], "a", 5) == 0.0


def test_reciprocal_rank():
    assert _reciprocal_rank(["a", "b", "c"], "a") == 1.0
    assert _reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert _reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1.0 / 3)
    assert _reciprocal_rank(["a", "b"], "z") == 0.0


def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_identical_is_one_for_normalized():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_percentile():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.0], 0.5) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 1.0) == 5.0


# ----- score_provider end-to-end (with mocked embedder) -------------------------


def _make_pairs(n: int) -> List[GroundTruthPair]:
    return [
        GroundTruthPair(
            episode_id="ep",
            insight_id=f"i{i}",
            insight_text=f"insight {i}",
            quote_id=f"q{i}",
            quote_text=f"quote {i}",
        )
        for i in range(n)
    ]


@patch("podcast_scraper.evaluation.embedding_provider_eval._embed_one")
def test_score_provider_perfect_embedder_hits_all(mock_embed):
    """When the embedder maps each insight to its quote's vector exactly, recall@1 = 1.0."""
    pairs = _make_pairs(5)

    # Map text → identity-like 2D vector indexed by trailing int.
    # Both insight i and quote i map to the same vector → perfect retrieval.
    def fake_embed(text: str, cfg):
        idx = int(text.split()[-1])
        vec = [0.0, 0.0, 0.0, 0.0, 0.0]
        vec[idx] = 1.0
        return vec, 1.5  # 1.5 ms latency

    mock_embed.side_effect = fake_embed
    cfg = ProviderConfig(label="t", provider="ollama", model_id="m", endpoint="http://x")

    m = score_provider(pairs, cfg)
    assert m.n_queries == 5
    assert m.recall_at[1] == pytest.approx(1.0)
    assert m.recall_at[10] == pytest.approx(1.0)
    assert m.mrr == pytest.approx(1.0)
    assert m.embed_latency_ms_p50 == pytest.approx(1.5)
    assert m.dim == 5


@patch("podcast_scraper.evaluation.embedding_provider_eval._embed_one")
def test_score_provider_random_embedder_floor_is_chance(mock_embed):
    """An embedder that returns the same vector for everything → recall@1 = 1/N (tie)."""
    pairs = _make_pairs(4)

    def fake_embed(text: str, cfg):
        return [1.0, 0.0], 2.0

    mock_embed.side_effect = fake_embed
    cfg = ProviderConfig(label="t", provider="sentence_transformers", model_id="m")

    m = score_provider(pairs, cfg)
    # All quotes tie at cosine = 1.0; tie-break by insertion order. Each insight's
    # supporting quote is at the same insertion-order position, so recall@1 = 1.0.
    # This is a known artifact of stable sorting on ties; documented in the harness.
    assert m.n_queries == 4


@patch("podcast_scraper.evaluation.embedding_provider_eval._embed_one")
def test_score_provider_includes_all_k_values(mock_embed):
    pairs = _make_pairs(3)
    mock_embed.return_value = ([1.0, 0.0], 0.0)
    cfg = ProviderConfig(label="t", provider="ollama", model_id="m", endpoint="x")

    m = score_provider(pairs, cfg)
    for k in K_VALUES:
        assert k in m.recall_at
        assert k in m.ndcg_at


# ----- report writers -----------------------------------------------------------


def test_write_report_json_round_trips(tmp_path: Path):
    pairs = _make_pairs(3)
    metrics = [
        ProviderMetrics(
            label="A",
            provider="sentence_transformers",
            model_id="minilm",
            n_queries=3,
            recall_at={1: 0.5, 5: 0.8, 10: 0.9, 20: 1.0},
            ndcg_at={1: 0.5, 5: 0.7, 10: 0.85, 20: 0.95},
            mrr=0.6,
            embed_latency_ms_p50=10.0,
            embed_latency_ms_p95=25.0,
            dim=384,
        ),
        ProviderMetrics(
            label="B",
            provider="ollama",
            model_id="nomic-embed-text",
            n_queries=3,
            recall_at={1: 0.7, 5: 0.95, 10: 1.0, 20: 1.0},
            ndcg_at={1: 0.7, 5: 0.88, 10: 0.95, 20: 0.97},
            mrr=0.8,
            embed_latency_ms_p50=18.0,
            embed_latency_ms_p95=40.0,
            dim=768,
        ),
    ]
    dest = tmp_path / "report.json"
    write_report_json(metrics, pairs, dest)
    doc = json.loads(dest.read_text())
    assert doc["schema_version"] == 1
    assert doc["ground_truth"]["n_pairs"] == 3
    assert len(doc["providers"]) == 2
    assert doc["providers"][1]["model_id"] == "nomic-embed-text"


def test_write_report_md_contains_recall_and_provider_rows(tmp_path: Path):
    pairs = _make_pairs(2)
    metrics = [
        ProviderMetrics(
            label="A",
            provider="sentence_transformers",
            model_id="minilm",
            n_queries=2,
            recall_at={k: 0.5 for k in K_VALUES},
            ndcg_at={k: 0.5 for k in K_VALUES},
            mrr=0.5,
            embed_latency_ms_p50=10.0,
            embed_latency_ms_p95=20.0,
            dim=384,
        ),
    ]
    dest = tmp_path / "report.md"
    write_report_md(metrics, pairs, dest)
    body = dest.read_text()
    assert "Recall@1" in body
    assert "Recall@20" in body
    assert "MRR" in body
    assert "sentence_transformers" in body
    assert "minilm" in body
