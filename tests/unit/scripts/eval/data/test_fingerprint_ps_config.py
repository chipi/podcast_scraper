"""RFC-097 fingerprint gap closure (FINGERPRINT_GAPS_ANALYSIS_2026-06-22.md §5):
podcast_scraper_config captures Config-side fields that materially affect
output. Operator case #3 (2026-06-22): "running different config for
podcast_scraper core configuration in test" → distinct fingerprints.
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


def _cfg(
    *,
    llm_pipeline_mode: str | None = None,
    transcript_cleaning_strategy: str | None = None,
    extra_body: dict | None = None,
) -> Any:
    backend = SimpleNamespace(
        type="openai", model="autoresearch", base_url=None, extra_body=extra_body
    )
    return SimpleNamespace(
        task="knowledge_graph",
        backend=backend,
        params={"temperature": 0.0, "max_length": 800},
        map_params=None,
        reduce_params=None,
        tokenize=None,
        chunking=None,
        transcript_cleaning_strategy=transcript_cleaning_strategy,
        llm_pipeline_mode=llm_pipeline_mode,
        prompts=None,
        preprocessing_profile=None,
    )


def _mocks(mock_get_model_details, mock_get_provider_lib_info) -> None:
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
def test_captures_llm_pipeline_mode(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp = _gen(_cfg(llm_pipeline_mode="staged"))
    psc = fp["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    assert psc["llm_pipeline_mode"] == "staged"


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_captures_transcript_cleaning_strategy(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp = _gen(_cfg(transcript_cleaning_strategy="cleaning_v4"))
    psc = fp["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    assert psc["transcript_cleaning_strategy"] == "cleaning_v4"


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_captures_openai_extra_body(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """Qwen-thinking models pass chat_template_kwargs via extra_body — that
    knob materially affects output (enable_thinking: false suppresses
    [THINK] block emission). Must land in the fingerprint."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    fp = _gen(_cfg(extra_body=extra))
    psc = fp["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    assert psc["openai_extra_body"] == extra


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_empty_when_no_relevant_fields(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp = _gen(_cfg())
    psc = fp["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    assert psc == {}


@patch("scripts.eval.data.materialize_baseline.get_provider_library_info")
@patch("scripts.eval.data.materialize_baseline.get_model_details")
@patch("scripts.eval.data.materialize_baseline._probe_vllm_backing_model_id", return_value=None)
def test_different_extra_body_produces_different_fingerprint(
    mock_probe, mock_get_model_details, mock_get_provider_lib_info
) -> None:
    """The regression this prevents: enable_thinking=true vs false on the same
    Qwen model produces materially different output. Different extra_body
    → distinct fingerprint values."""
    _mocks(mock_get_model_details, mock_get_provider_lib_info)
    fp_thinking = _gen(_cfg(extra_body={"chat_template_kwargs": {"enable_thinking": True}}))
    fp_no_thinking = _gen(_cfg(extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    psc1 = fp_thinking["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    psc2 = fp_no_thinking["pipeline"]["stages"]["main"]["podcast_scraper_config"]
    assert psc1 != psc2
