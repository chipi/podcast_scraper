"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §2):
backing_model_id + base_url get captured into pipeline.stages.main.model so
the vLLM ``served-model-name: autoresearch`` alias doesn't hide which actual
HF model loaded. Probes GET /v1/models with a short timeout; failures fall
back to None.
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
    _probe_vllm_backing_model_id,
    generate_enhanced_fingerprint,
)

pytestmark = pytest.mark.unit


def _stub_provider() -> Any:
    p = MagicMock()
    p.__class__.__name__ = "OpenAISummarizationProvider"
    return p


def _experiment_cfg(base_url: str | None = None) -> Any:
    backend = SimpleNamespace(type="openai", model="autoresearch", base_url=base_url)
    return SimpleNamespace(
        task="knowledge_graph",
        backend=backend,
        params={"temperature": 0.0, "max_length": 800},
        map_params=None,
        reduce_params=None,
        tokenize=None,
        chunking=None,
        transcript_cleaning_strategy=None,
        prompts=None,
        preprocessing_profile=None,
    )


def test_probe_returns_none_for_empty_base_url() -> None:
    assert _probe_vllm_backing_model_id(None) is None
    assert _probe_vllm_backing_model_id("") is None


def test_probe_returns_none_on_network_error() -> None:
    """Server unreachable → None, fingerprint generation must not raise."""
    with patch("requests.get", side_effect=Exception("connection refused")):
        assert _probe_vllm_backing_model_id("http://nowhere:8003/v1") is None


def test_probe_returns_none_on_non_200_response() -> None:
    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.status_code = 503
    with patch("requests.get", return_value=mock_resp):
        assert _probe_vllm_backing_model_id("http://server:8003/v1") is None


def test_probe_extracts_root_field_from_vllm_response() -> None:
    """vLLM's GET /v1/models payload puts the HF id at data[0].root."""
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "autoresearch",
                "object": "model",
                "owned_by": "vllm",
                "root": "Qwen/Qwen3.5-35B-A3B",
                "parent": None,
            }
        ],
    }
    with patch("requests.get", return_value=mock_resp):
        assert _probe_vllm_backing_model_id("http://dgx:8003/v1") == "Qwen/Qwen3.5-35B-A3B"


def test_probe_handles_malformed_response() -> None:
    """Defensive against shape drift — anything unexpected → None."""
    cases: list[Any] = [
        {"object": "list"},  # no data
        {"data": []},  # empty data
        {"data": ["not_a_dict"]},  # wrong item type
        {"data": [{"id": "x"}]},  # missing root
        "not a dict",
    ]
    for payload in cases:
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = payload
        with patch("requests.get", return_value=mock_resp):
            assert _probe_vllm_backing_model_id("http://x:8003/v1") is None


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id")
def test_fingerprint_captures_backing_model_id_and_base_url(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    mock_probe.return_value = "mistralai/Magistral-Small-2509"
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
    cfg = _experiment_cfg(base_url="http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1")
    fp = generate_enhanced_fingerprint(
        baseline_id="autoresearch_prompt_vllm_magistral_small_2509_dev_knowledge_graph_v1",
        dataset_id="curated_5feeds_dev_v1",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="autoresearch",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    model_block = fp["pipeline"]["stages"]["main"]["model"]
    assert model_block["base_url"] == "http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1"
    assert model_block["backing_model_id"] == "mistralai/Magistral-Small-2509"


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id")
def test_fingerprint_handles_missing_base_url_gracefully(
    mock_probe: MagicMock,
    mock_get_model_details: MagicMock,
    mock_get_provider_lib_info: MagicMock,
) -> None:
    """When base_url is absent (OpenAI cloud, Gemini, etc), backing_model_id
    is None and base_url is None — fingerprint stays valid."""
    mock_probe.return_value = None
    mock_get_model_details.return_value = {
        "model_name": "gpt-4o-mini-2024-07-18",
        "model_revision": "2024-07-18",
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
    cfg = _experiment_cfg(base_url=None)
    fp = generate_enhanced_fingerprint(
        baseline_id="a",
        dataset_id="d",
        experiment_config=cfg,
        provider=_stub_provider(),
        model_name="gpt-4o-mini",
        preprocessing_profile="cleaning_v4",
        git_info={"commit_sha": "x", "branch": "y", "is_dirty": False},
    )
    model_block = fp["pipeline"]["stages"]["main"]["model"]
    assert model_block["base_url"] is None
    assert model_block["backing_model_id"] is None
