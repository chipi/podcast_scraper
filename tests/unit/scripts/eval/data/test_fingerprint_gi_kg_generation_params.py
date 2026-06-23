"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §1):
GI/KG task fingerprints used to hardcode generation_params={}. Two materially
different GI runs (e.g. Magistral temp=0.7 max_length=4096 vs Qwen3.5 temp=0.0
max_length=800) produced identical-looking fingerprint blocks. Now they
mirror the (anthropic, mistral, grok, deepseek, gemini, ollama) summary
branch — the actual sampling lands in the fingerprint.
"""

from __future__ import annotations

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
    """Minimal provider stub for the fingerprint generator."""
    p = MagicMock()
    p.__class__.__name__ = "OpenAISummarizationProvider"
    return p


def _experiment_cfg(task: str, params: dict[str, Any]) -> Any:
    """Build a minimal experiment config matching the generator's read pattern."""
    backend = SimpleNamespace(type="openai", model="autoresearch", base_url=None)
    return SimpleNamespace(
        task=task,
        backend=backend,
        params=params,
        map_params=None,
        reduce_params=None,
        tokenize=None,
        chunking=None,
        transcript_cleaning_strategy=None,
        prompts=None,
        preprocessing_profile=None,
    )


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
def test_kg_task_fingerprint_captures_temperature_and_max_tokens(
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
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

    cfg = _experiment_cfg(
        "knowledge_graph",
        {"temperature": 0.7, "top_p": 0.95, "max_length": 4096, "seed": 42},
    )
    fp = generate_enhanced_fingerprint(
        baseline_id="autoresearch_prompt_vllm_magistral_small_2509_dev_knowledge_graph_v1",
        dataset_id="curated_5feeds_dev_v1",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "abc123", "branch": "feat/x", "is_dirty": False},
    )

    gen = fp["pipeline"]["stages"]["main"]["generation_params"]
    assert gen["temperature"] == 0.7
    assert gen["top_p"] == 0.95
    assert gen["max_tokens"] == 4096
    assert gen["seed"] == 42


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
def test_gi_task_fingerprint_captures_temperature_and_max_tokens(
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
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

    cfg = _experiment_cfg(
        "grounded_insights",
        {"temperature": 0.0, "max_length": 800},
    )
    fp = generate_enhanced_fingerprint(
        baseline_id="autoresearch_prompt_vllm_qwen3_5_35b_a3b_dev_grounded_insights_v1",
        dataset_id="curated_5feeds_dev_v1",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "abc123", "branch": "feat/x", "is_dirty": False},
    )

    gen = fp["pipeline"]["stages"]["main"]["generation_params"]
    assert gen["temperature"] == 0.0
    assert gen["max_tokens"] == 800


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
def test_kg_two_different_configs_produce_different_generation_params(
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    """The regression that motivated this fix: two materially different GI/KG
    configs USED to produce identical generation_params={}. They now differ."""
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

    cfg_magistral = _experiment_cfg(
        "knowledge_graph",
        {"temperature": 0.7, "top_p": 0.95, "max_length": 4096},
    )
    cfg_qwen = _experiment_cfg(
        "knowledge_graph",
        {"temperature": 0.0, "max_length": 800},
    )
    fp1 = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg_magistral,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    fp2 = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg_qwen,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )

    g1 = fp1["pipeline"]["stages"]["main"]["generation_params"]
    g2 = fp2["pipeline"]["stages"]["main"]["generation_params"]
    assert g1 != g2, "GI/KG fingerprints must differ when sampling differs"
    assert g1["temperature"] != g2["temperature"]
    assert g1["max_tokens"] != g2["max_tokens"]


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
def test_gi_kg_params_drops_none_for_hash_stability(
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    """Optional params (top_p, seed) absent in YAML must drop OUT of the
    fingerprint rather than appear as None — keeps the hash shape stable
    between configs that do and don't specify them."""
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
    cfg = _experiment_cfg(
        "knowledge_graph",
        {"temperature": 0.0, "max_length": 800},  # no top_p, no seed
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
    gen = fp["pipeline"]["stages"]["main"]["generation_params"]
    assert "temperature" in gen
    assert "max_tokens" in gen
    # None-valued optionals filtered out:
    assert "top_p" not in gen
    assert "seed" not in gen
    assert "min_tokens" not in gen
