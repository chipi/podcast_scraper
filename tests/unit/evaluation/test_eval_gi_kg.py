"""Unit tests for GIL/KG experiment helpers and scorers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from podcast_scraper.config import Config as RuntimeConfig
from podcast_scraper.evaluation.eval_gi_kg_runtime import (
    merge_eval_task_into_summarizer_config,
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


def test_grounded_insights_accepts_openai_backend() -> None:
    """GIL eval may use real summarization backends (regenerate summary, then GIL)."""
    cfg = ExperimentConfig(
        id="gi_openai_ok",
        task="grounded_insights",
        backend=OpenAIBackendConfig(model="gpt-4o-mini"),
        prompts=PromptConfig(user="openai/summarization/long_v1"),
        data=DataConfig(dataset_id="curated_5feeds_smoke_v1"),
    )
    assert cfg.backend.type == "openai"


def test_merge_eval_task_sets_gi_and_kg_flags() -> None:
    base = RuntimeConfig.model_validate(
        {
            "rss": "",
            "summary_provider": "transformers",
            "generate_summaries": True,
            "generate_metadata": True,
            "generate_gi": False,
            "generate_kg": False,
            "transcribe_missing": False,
        }
    )
    gi_cfg = merge_eval_task_into_summarizer_config(
        base, "grounded_insights", {"gi_require_grounding": False}
    )
    assert gi_cfg.generate_gi is True
    assert gi_cfg.generate_kg is False
    assert gi_cfg.gi_insight_source == "summary_bullets"
    assert gi_cfg.gi_require_grounding is False
    kg_cfg = merge_eval_task_into_summarizer_config(base, "knowledge_graph", None)
    assert kg_cfg.generate_kg is True
    assert kg_cfg.generate_gi is False
    assert kg_cfg.kg_extraction_source == "summary_bullets"


def test_merge_eval_task_unsupported_task_raises() -> None:
    base = RuntimeConfig.model_validate(
        {
            "rss": "",
            "summary_provider": "transformers",
            "generate_summaries": True,
            "generate_metadata": True,
            "generate_gi": False,
            "generate_kg": False,
            "transcribe_missing": False,
        }
    )
    with pytest.raises(ValueError, match="unsupported task"):
        merge_eval_task_into_summarizer_config(base, "summarization", None)


def test_merge_eval_task_coerces_invalid_gi_insight_source() -> None:
    base = RuntimeConfig.model_validate(
        {
            "rss": "",
            "summary_provider": "transformers",
            "generate_summaries": True,
            "generate_metadata": True,
            "generate_gi": False,
            "generate_kg": False,
            "transcribe_missing": False,
        }
    )
    gi_cfg = merge_eval_task_into_summarizer_config(
        base,
        "grounded_insights",
        {"gi_insight_source": "not_valid"},
    )
    assert gi_cfg.gi_insight_source == "summary_bullets"


def test_merge_eval_task_coerces_invalid_kg_extraction_source() -> None:
    base = RuntimeConfig.model_validate(
        {
            "rss": "",
            "summary_provider": "transformers",
            "generate_summaries": True,
            "generate_metadata": True,
            "generate_gi": False,
            "generate_kg": False,
            "transcribe_missing": False,
        }
    )
    kg_cfg = merge_eval_task_into_summarizer_config(
        base,
        "knowledge_graph",
        {"kg_extraction_source": "invalid"},
    )
    assert kg_cfg.kg_extraction_source == "summary_bullets"


def test_merge_eval_task_applies_numeric_params() -> None:
    base = RuntimeConfig.model_validate(
        {
            "rss": "",
            "summary_provider": "transformers",
            "generate_summaries": True,
            "generate_metadata": True,
            "generate_gi": False,
            "generate_kg": False,
            "transcribe_missing": False,
        }
    )
    gi_cfg = merge_eval_task_into_summarizer_config(
        base,
        "grounded_insights",
        {"gi_max_insights": 3},
    )
    assert gi_cfg.gi_max_insights == 3
    kg_cfg = merge_eval_task_into_summarizer_config(
        base,
        "knowledge_graph",
        {"kg_max_topics": 5, "kg_max_entities": 7},
    )
    assert kg_cfg.kg_max_topics == 5
    assert kg_cfg.kg_max_entities == 7


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


def test_score_run_grounded_insights_with_silver_reference_jsonl(tmp_path: Path) -> None:
    """score_run wires GI vs_reference to silver predictions.jsonl layout."""
    ref_dir = tmp_path / "silver_gil"
    ref_dir.mkdir()
    gil_ref = {
        "nodes": [{"type": "Insight", "id": "i1"}, {"type": "Quote", "id": "q1"}],
        "edges": [{"from": "i1", "to": "q1", "type": "SUPPORTED_BY"}],
    }
    line = json.dumps(
        {"episode_id": "p01_e01", "output": {"gil": gil_ref}},
    )
    (ref_dir / "predictions.jsonl").write_text(line + "\n", encoding="utf-8")
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text(line + "\n", encoding="utf-8")
    metrics = score_run(
        pred_path,
        dataset_id="curated_5feeds_smoke_v1",
        run_id="run_with_silver_gil",
        reference_paths={"silver_gil": ref_dir},
        task="grounded_insights",
    )
    vs = metrics.get("vs_reference") or {}
    assert "silver_gil" in vs
    blob = vs["silver_gil"]
    assert "error" not in blob
    assert blob.get("insight_quote_edge_count_exact_match_rate") == 1.0


def test_score_run_knowledge_graph_with_silver_reference_jsonl(tmp_path: Path) -> None:
    """score_run wires KG vs_reference to silver predictions.jsonl layout."""
    ref_dir = tmp_path / "silver_kg"
    ref_dir.mkdir()
    kg_ref = {
        "nodes": [{"id": "n1"}, {"id": "n2"}],
        "edges": [{"from": "n1", "to": "n2"}],
    }
    line = json.dumps({"episode_id": "p01_e01", "output": {"kg": kg_ref}})
    (ref_dir / "predictions.jsonl").write_text(line + "\n", encoding="utf-8")
    pred_path = tmp_path / "predictions_kg.jsonl"
    pred_path.write_text(line + "\n", encoding="utf-8")
    metrics = score_run(
        pred_path,
        dataset_id="curated_5feeds_smoke_v1",
        run_id="run_with_silver_kg",
        reference_paths={"silver_kg": ref_dir},
        task="knowledge_graph",
    )
    vs = metrics.get("vs_reference") or {}
    assert "silver_kg" in vs
    blob = vs["silver_kg"]
    assert "error" not in blob
    assert blob.get("node_edge_count_exact_match_rate") == 1.0


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


def test_compute_gil_vs_reference_silver_predictions_jsonl(tmp_path: Path) -> None:
    """Silver references use predictions.jsonl with output.gil per line."""
    ref_dir = tmp_path / "silver_run"
    ref_dir.mkdir()
    gil_ref = {
        "nodes": [{"type": "Insight", "id": "x"}],
        "edges": [],
    }
    line = json.dumps(
        {
            "episode_id": "p01_e01",
            "output": {"gil": gil_ref},
        }
    )
    (ref_dir / "predictions.jsonl").write_text(line + "\n", encoding="utf-8")
    predictions = [
        {
            "episode_id": "p01_e01",
            "output": {"gil": gil_ref},
        }
    ]
    out = compute_gil_vs_reference_metrics(
        predictions,
        "silver_smoke",
        ref_dir,
        dataset_id="curated_5feeds_smoke_v1",
    )
    assert out["scored_episodes"]["scored"] == 1
    assert out["insight_quote_edge_count_exact_match_rate"] == 1.0


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


def test_compute_kg_vs_reference_silver_predictions_jsonl(tmp_path: Path) -> None:
    ref_dir = tmp_path / "silver_kg"
    ref_dir.mkdir()
    kg_ref = {"nodes": [{"id": "ep"}], "edges": [{"from": "a", "to": "b"}]}
    (ref_dir / "predictions.jsonl").write_text(
        json.dumps({"episode_id": "e1", "output": {"kg": kg_ref}}) + "\n",
        encoding="utf-8",
    )
    predictions = [{"episode_id": "e1", "output": {"kg": kg_ref}}]
    out = compute_kg_vs_reference_metrics(
        predictions,
        "silver_kg",
        ref_dir,
        dataset_id="curated_5feeds_smoke_v1",
    )
    assert out["scored_episodes"]["scored"] == 1
    assert out["node_edge_count_exact_match_rate"] == 1.0


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
