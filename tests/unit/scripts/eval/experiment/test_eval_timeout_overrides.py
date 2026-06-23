"""Eval-path fail-fast timeout overrides.

Production summarization_timeout/transcription_timeout default to 1200/1800s
to absorb provider 503 storms (see #697). The eval path (scripts/eval/experiment)
uses tighter defaults so client-side hangs surface in minutes rather than tens
of minutes — the Gemma GI 47-min Tailscale-drop hang on 2026-06-21 was the
burn that motivated this. Operator can override via env vars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Ensure scripts/eval/experiment is importable.
_REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.experiment.run_experiment import (  # noqa: E402
    _eval_podcast_scraper_config_overrides,
)

pytestmark = pytest.mark.unit


def _stub_experiment_cfg() -> Any:
    """Minimal stub that has the attributes _eval_podcast_scraper_config_overrides reads."""

    class _Stub:
        transcript_cleaning_strategy = None
        llm_pipeline_mode = None

    return _Stub()


def test_default_summarization_timeout_is_600_seconds() -> None:
    """Default (no env override) returns 600s for both timeouts."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("EVAL_OPENAI_TIMEOUT_SECONDS", None)
        os.environ.pop("EVAL_TRANSCRIPTION_TIMEOUT_SECONDS", None)
        out = _eval_podcast_scraper_config_overrides(_stub_experiment_cfg())
    assert out["summarization_timeout"] == 600
    assert out["transcription_timeout"] == 600


def test_env_var_override_summarization_timeout() -> None:
    with patch.dict(os.environ, {"EVAL_OPENAI_TIMEOUT_SECONDS": "180"}):
        out = _eval_podcast_scraper_config_overrides(_stub_experiment_cfg())
    assert out["summarization_timeout"] == 180


def test_env_var_override_transcription_timeout() -> None:
    with patch.dict(os.environ, {"EVAL_TRANSCRIPTION_TIMEOUT_SECONDS": "900"}):
        out = _eval_podcast_scraper_config_overrides(_stub_experiment_cfg())
    assert out["transcription_timeout"] == 900


def test_env_var_invalid_value_falls_back_to_default() -> None:
    """A non-integer env value must not break run_experiment; falls back to 600."""
    with patch.dict(os.environ, {"EVAL_OPENAI_TIMEOUT_SECONDS": "not-a-number"}):
        out = _eval_podcast_scraper_config_overrides(_stub_experiment_cfg())
    assert out["summarization_timeout"] == 600


def test_existing_overrides_still_threaded() -> None:
    """The new timeout fields don't displace the pre-existing override knobs."""

    class _Cfg:
        transcript_cleaning_strategy = "cleaning_v4"
        llm_pipeline_mode = "staged"

    # Cast through Any: the helper duck-types via getattr; we use a stub here.
    out = _eval_podcast_scraper_config_overrides(_Cfg())  # type: ignore[arg-type]
    assert out["transcript_cleaning_strategy"] == "cleaning_v4"
    assert out["llm_pipeline_mode"] == "staged"
    # Plus the new timeouts.
    assert "summarization_timeout" in out
    assert "transcription_timeout" in out
