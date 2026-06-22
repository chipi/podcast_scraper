"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §§3-5):
task_pipeline (postprocessor, kg_extraction_src, gi_insight_src,
gi_max_insights) + inference_args + inference_image now captured into
pipeline.stages.main. These are the knobs that materially affect output
(extraction mode, post-processing, vLLM serve flags) but used to be
invisible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.data.materialize_baseline import (  # noqa: E402
    generate_enhanced_fingerprint,
)

pytestmark = pytest.mark.unit


def _stub_provider() -> Any:
    p = MagicMock()
    p.__class__.__name__ = "OpenAISummarizationProvider"
    return p


def _experiment_cfg(
    task: str,
    params: dict[str, Any],
    *,
    postprocessor: str | None = None,
) -> Any:
    backend = SimpleNamespace(type="openai", model="autoresearch", base_url=None)
    prompts = SimpleNamespace(postprocessor=postprocessor) if postprocessor else None
    return SimpleNamespace(
        task=task,
        backend=backend,
        params=params,
        map_params=None,
        reduce_params=None,
        tokenize=None,
        chunking=None,
        transcript_cleaning_strategy=None,
        prompts=prompts,
        preprocessing_profile=None,
    )


def _common_mocks(mock_get_model_details: MagicMock, mock_get_provider_lib_info: MagicMock) -> None:
    mock_get_model_details.return_value = {
        "model_name": "autoresearch",
        "model_revision": None,
        "tokenizer_name": None,
        "tokenizer_revision": None,
        "framework": None,
        "endpoint": "chat.completions",
        "provider_type": "openai",
    }
    mock_get_provider_lib_info.return_value = {
        "provider_library": "openai",
        "provider_library_version": "2.15.0",
    }


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_task_pipeline_captures_postprocessor(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    _common_mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _experiment_cfg(
        "knowledge_graph",
        {"temperature": 0.7, "max_length": 4096},
        postprocessor="strip_r1_reasoning",
    )
    fp = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    tp = fp["pipeline"]["stages"]["main"]["task_pipeline"]
    assert tp["postprocessor"] == "strip_r1_reasoning"


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_task_pipeline_captures_extraction_src_and_max_insights(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    _common_mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _experiment_cfg(
        "grounded_insights",
        {
            "temperature": 0.0,
            "max_length": 800,
            "gi_insight_src": "provider",
            "gi_max_insights": 12,
        },
    )
    fp = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    tp = fp["pipeline"]["stages"]["main"]["task_pipeline"]
    assert tp["gi_insight_src"] == "provider"
    assert tp["gi_max_insights"] == 12


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_inference_args_and_image_from_env(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    """Operator runbook sets VLLM_INFERENCE_ARGS / VLLM_INFERENCE_IMAGE env
    vars; fingerprint captures them verbatim."""
    _common_mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _experiment_cfg("knowledge_graph", {"temperature": 0.0, "max_length": 800})
    args_payload = "[vllm serve google/gemma-4-26B-A4B-it --max-num-seqs 4 --enforce-eager]"
    img_payload = "nvcr.io/nvidia/vllm:26.05-py3"
    with patch.dict(
        os.environ,
        {"VLLM_INFERENCE_ARGS": args_payload, "VLLM_INFERENCE_IMAGE": img_payload},
    ):
        fp = generate_enhanced_fingerprint(
            baseline_id="a",
            dataset_id="d",
            experiment_config=cfg,
            provider=_stub_provider(),
            model_name="autoresearch",
            preprocessing_profile="cleaning_v4",
            git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
        )
    main_stage = fp["pipeline"]["stages"]["main"]
    assert main_stage["inference_args"] == args_payload
    assert main_stage["inference_image"] == img_payload


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_inference_args_default_none_when_env_absent(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    _common_mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _experiment_cfg("knowledge_graph", {"temperature": 0.0, "max_length": 800})
    # Clear the env vars in case they're set in CI
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("VLLM_INFERENCE_ARGS", None)
        os.environ.pop("VLLM_INFERENCE_IMAGE", None)
        fp = generate_enhanced_fingerprint(
            baseline_id="a",
            dataset_id="d",
            experiment_config=cfg,
            provider=_stub_provider(),
            model_name="autoresearch",
            preprocessing_profile="cleaning_v4",
            git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
        )
    main_stage = fp["pipeline"]["stages"]["main"]
    assert main_stage["inference_args"] is None
    assert main_stage["inference_image"] is None


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_task_pipeline_empty_when_no_relevant_fields(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    """No postprocessor, no extraction_src — task_pipeline is {} (still present)."""
    _common_mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _experiment_cfg("knowledge_graph", {"temperature": 0.0, "max_length": 800})
    fp = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    tp = fp["pipeline"]["stages"]["main"]["task_pipeline"]
    assert tp == {}
