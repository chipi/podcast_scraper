"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §7):
fingerprint_version: "2.0" + top-level fingerprint_hash that walks the FULL
fingerprint dict. v1.0 hash only covered model_name + a few fields, missing
generation_params, prompts, preprocessing, etc. — two runs differing only in
those previously-invisible fields had identical hashes despite producing
genuinely different outputs.
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
    p = MagicMock()
    p.__class__.__name__ = "OpenAISummarizationProvider"
    return p


def _cfg(task: str = "knowledge_graph", **params: Any) -> Any:
    backend = SimpleNamespace(type="openai", model="autoresearch", base_url=None, extra_body=None)
    return SimpleNamespace(
        task=task,
        backend=backend,
        params={"temperature": 0.0, "max_length": 800, **params},
        map_params=None,
        reduce_params=None,
        tokenize=None,
        chunking=None,
        transcript_cleaning_strategy=None,
        llm_pipeline_mode=None,
        prompts=None,
        preprocessing_profile=None,
    )


def _mocks(m_get_model_details, m_get_provider_lib_info) -> None:
    m_get_model_details.return_value = {
        "model_name": "autoresearch",
        "model_revision": None,
        "tokenizer_name": None,
        "tokenizer_revision": None,
        "framework": None,
        "endpoint": "chat.completions",
        "provider_type": "openai",
    }
    m_get_provider_lib_info.return_value = {
        "provider_library": "openai",
        "provider_library_version": "2.15.0",
    }


def _gen(cfg) -> dict:
    return generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_fingerprint_version_bumped_to_2_0(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp = _gen(_cfg())
    assert fp["fingerprint_version"] == "2.0"


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_fingerprint_hash_is_present_and_sha256(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp = _gen(_cfg())
    assert "fingerprint_hash" in fp
    assert isinstance(fp["fingerprint_hash"], str)
    assert len(fp["fingerprint_hash"]) == 64
    # sha256 hex
    int(fp["fingerprint_hash"], 16)


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_hash_differs_on_generation_params_change(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """The regression v2.0 prevents: GI/KG temp=0.7 vs temp=0.0 had identical
    v1.0 hashes (because v1.0 didn't walk generation_params). Now distinct."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp_cool = _gen(_cfg(temperature=0.0))
    fp_warm = _gen(_cfg(temperature=0.7))
    assert fp_cool["fingerprint_hash"] != fp_warm["fingerprint_hash"]


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_hash_differs_on_max_length_change(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """Magistral max_length 800 vs 4096 must produce distinct hashes."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp_800 = _gen(_cfg(max_length=800))
    fp_4096 = _gen(_cfg(max_length=4096))
    assert fp_800["fingerprint_hash"] != fp_4096["fingerprint_hash"]


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_hash_stable_for_identical_configs(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """Same inputs → same hash, every time (idempotent / reproducible)."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp1 = _gen(_cfg(temperature=0.0, max_length=800))
    fp2 = _gen(_cfg(temperature=0.0, max_length=800))
    assert fp1["fingerprint_hash"] == fp2["fingerprint_hash"]


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id")
def test_hash_differs_on_backing_model_id_change(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """The headline regression #2 closed: model_name='autoresearch' alias
    hides the backing model. Two runs at the same alias against different
    backing models now produce different hashes."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    cfg = _cfg()
    # Patch the cfg's base_url so backing_model_id capture path runs
    cfg.backend.base_url = "http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1"
    mock_probe.return_value = "Qwen/Qwen3.5-35B-A3B"
    fp_qwen = _gen(cfg)
    mock_probe.return_value = "mistralai/Magistral-Small-2509"
    fp_magistral = _gen(cfg)
    assert fp_qwen["fingerprint_hash"] != fp_magistral["fingerprint_hash"]
