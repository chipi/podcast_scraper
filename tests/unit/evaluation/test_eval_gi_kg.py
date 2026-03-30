"""Unit tests for GIL/KG experiment helpers and scorers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from podcast_scraper.evaluation.eval_gi_kg_runtime import (
    runtime_config_for_grounded_insights_eval,
    runtime_config_for_knowledge_graph_eval,
)
from podcast_scraper.evaluation.experiment_config import (
    DataConfig,
    EvalStubBackendConfig,
    ExperimentConfig,
    load_experiment_config,
    OpenAIBackendConfig,
    PromptConfig,
)
from podcast_scraper.evaluation.gi_scorer import (
    compute_gil_prediction_stats,
    compute_gil_vs_reference_metrics,
)
from podcast_scraper.evaluation.kg_scorer import (
    compute_kg_prediction_stats,
    compute_kg_vs_reference_metrics,
)
from podcast_scraper.evaluation.scorer import score_run


def test_eval_stub_rejected_for_summarization_task() -> None:
    """eval_stub backend is only valid for grounded_insights / knowledge_graph."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            id="bad_stub_summarization",
            task="summarization",
            backend=EvalStubBackendConfig(),
            data=DataConfig(dataset_id="curated_5feeds_smoke_v1"),
        )


def test_grounded_insights_rejects_non_stub_backend() -> None:
    """GIL/KG eval tasks currently require eval_stub (see experiment_config validators)."""
    with pytest.raises(ValidationError):
        ExperimentConfig(
            id="bad_gi_openai",
            task="grounded_insights",
            backend=OpenAIBackendConfig(model="gpt-4o-mini"),
            prompts=PromptConfig(user="openai/summarization/long_v1"),
            data=DataConfig(dataset_id="curated_5feeds_smoke_v1"),
        )


def test_runtime_config_gi_has_gil_enabled() -> None:
    cfg = runtime_config_for_grounded_insights_eval({"gi_insight_source": "stub"})
    assert cfg.generate_gi is True
    assert cfg.generate_kg is False
    assert cfg.generate_summaries is False


def test_runtime_config_kg_has_kg_enabled() -> None:
    cfg = runtime_config_for_knowledge_graph_eval({"kg_extraction_source": "stub"})
    assert cfg.generate_kg is True
    assert cfg.generate_gi is False


def test_load_experiment_config_gil_stub_yaml() -> None:
    path = Path("data/eval/configs/gil_eval_stub_curated_5feeds_smoke_v1.yaml")
    if not path.exists():
        pytest.skip("eval config yaml not present")
    loaded = load_experiment_config(path)
    assert loaded.task == "grounded_insights"
    assert loaded.backend.type == "eval_stub"


def test_load_experiment_config_kg_stub_yaml() -> None:
    path = Path("data/eval/configs/kg_eval_stub_curated_5feeds_smoke_v1.yaml")
    if not path.exists():
        pytest.skip("eval config yaml not present")
    loaded = load_experiment_config(path)
    assert loaded.task == "knowledge_graph"
    assert loaded.backend.type == "eval_stub"


def test_compute_gil_prediction_stats() -> None:
    predictions = [
        {
            "episode_id": "e1",
            "output": {
                "gil": {
                    "nodes": [
                        {"type": "Insight", "id": "i1"},
                        {"type": "Quote", "id": "q1"},
                    ],
                    "edges": [{"from": "i1", "to": "q1", "type": "SUPPORTED_BY"}],
                }
            },
        }
    ]
    stats = compute_gil_prediction_stats(predictions)
    assert stats["episodes_with_gil"] == 1
    assert stats["avg_insight_nodes"] == 1.0
    assert stats["avg_quote_nodes"] == 1.0


def test_compute_kg_prediction_stats() -> None:
    predictions = [
        {
            "episode_id": "e1",
            "output": {"kg": {"nodes": [{"id": "n1"}], "edges": [{"from": "a", "to": "b"}]}},
        }
    ]
    stats = compute_kg_prediction_stats(predictions)
    assert stats["episodes_with_kg"] == 1
    assert stats["avg_nodes"] == 1.0
    assert stats["avg_edges"] == 1.0


def test_score_run_grounded_insights_task(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(
        json.dumps(
            {
                "episode_id": "p01_e01",
                "dataset_id": "curated_5feeds_smoke_v1",
                "output": {"gil": {"nodes": [], "edges": []}},
                "metadata": {"processing_time_seconds": 0.1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = score_run(
        pred_path,
        dataset_id="curated_5feeds_smoke_v1",
        run_id="test_run",
        task="grounded_insights",
    )
    assert metrics["task"] == "grounded_insights"
    assert metrics["schema"] == "metrics_gil_eval_run_v1"
    assert "gil" in metrics["intrinsic"]


def test_score_run_knowledge_graph_task(tmp_path: Path) -> None:
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(
        json.dumps(
            {
                "episode_id": "p01_e01",
                "dataset_id": "curated_5feeds_smoke_v1",
                "output": {"kg": {"nodes": [], "edges": []}},
                "metadata": {"processing_time_seconds": 0.1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = score_run(
        pred_path,
        dataset_id="curated_5feeds_smoke_v1",
        run_id="test_run",
        task="knowledge_graph",
    )
    assert metrics["task"] == "knowledge_graph"
    assert metrics["schema"] == "metrics_kg_eval_run_v1"
    assert "kg" in metrics["intrinsic"]


def test_compute_gil_vs_reference(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    gil_gold = {
        "nodes": [{"type": "Insight", "id": "x"}],
        "edges": [],
    }
    (gold_dir / "p01_e01.json").write_text(json.dumps(gil_gold), encoding="utf-8")
    predictions = [
        {
            "episode_id": "p01_e01",
            "output": {"gil": gil_gold},
        }
    ]
    out = compute_gil_vs_reference_metrics(
        predictions,
        "ref_v1",
        gold_dir,
        dataset_id="curated_5feeds_smoke_v1",
    )
    assert out["scored_episodes"]["scored"] == 1
    assert out["insight_quote_edge_count_exact_match_rate"] == 1.0


def test_compute_kg_vs_reference(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    kg_gold = {"nodes": [{"id": "ep"}], "edges": []}
    (gold_dir / "p01_e01.json").write_text(json.dumps(kg_gold), encoding="utf-8")
    predictions = [
        {
            "episode_id": "p01_e01",
            "output": {"kg": kg_gold},
        }
    ]
    out = compute_kg_vs_reference_metrics(
        predictions,
        "ref_v1",
        gold_dir,
        dataset_id="curated_5feeds_smoke_v1",
    )
    assert out["scored_episodes"]["scored"] == 1
    assert out["node_edge_count_exact_match_rate"] == 1.0
