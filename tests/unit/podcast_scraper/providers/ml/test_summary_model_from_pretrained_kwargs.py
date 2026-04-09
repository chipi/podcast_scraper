"""Exercise SummaryModel._load_model branches that unpack kwargs via cast(Any, ...) (mypy).

Installs a stub ``transformers`` module so ``@patch("transformers.X.from_pretrained")``
resolves without the real package. ``_detect_device`` and
``_load_model_move_to_device_and_pipeline`` are patched so CI (no ``torch``) never runs
auto-detect or pipeline creation; production
code avoids eager ``import torch`` on non-Pegasus paths after ``_load_model`` (see
``_load_model_pegasus_sanity_and_clear_config``).
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

if "transformers" not in sys.modules:
    _fake_tf = ModuleType("transformers")
    for _cls_name in (
        "AutoTokenizer",
        "AutoModelForSeq2SeqLM",
        "BartForConditionalGeneration",
        "LEDForConditionalGeneration",
        "Pipeline",
    ):
        setattr(_fake_tf, _cls_name, MagicMock())
    _fake_tf.utils = ModuleType("transformers.utils")  # type: ignore[attr-defined]
    _fake_tf.utils.logging = MagicMock()  # type: ignore[attr-defined]
    sys.modules["transformers"] = _fake_tf
    sys.modules["transformers.utils"] = _fake_tf.utils  # type: ignore[attr-defined]
    sys.modules["transformers.utils.logging"] = _fake_tf.utils.logging  # type: ignore[attr-defined]

from podcast_scraper.providers.ml.summarizer import SummaryModel  # noqa: E402


def _invoke_retry(fn, *_a, **_kw):
    return fn()


_DETECT = "podcast_scraper.providers.ml.summarizer.SummaryModel._detect_device"
_MOVE = (
    "podcast_scraper.providers.ml.summarizer.SummaryModel._load_model_move_to_device_and_pipeline"
)
_RETRY = "podcast_scraper.providers.ml.summarizer._load_with_retry_summarizer"


@pytest.mark.unit
@patch(_DETECT, return_value="cpu")
@patch(_MOVE)
@patch(_RETRY, side_effect=_invoke_retry)
@patch("transformers.BartForConditionalGeneration.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_bart_branch_unpacks_model_kwargs(
    mock_tok, mock_bart, mock_retry, mock_move_pipe, _mock_dev, tmp_path
) -> None:
    mock_tok.return_value = MagicMock()
    mock_bart.return_value = MagicMock()
    SummaryModel(
        "facebook/bart-large-cnn",
        device="cpu",
        cache_dir=str(tmp_path),
        revision=None,
    )
    assert mock_bart.called
    assert mock_tok.called
    assert mock_retry.called
    assert mock_move_pipe.called
    call_kw = mock_bart.call_args.kwargs
    assert call_kw.get("local_files_only") is True
    assert "cache_dir" in call_kw


@pytest.mark.unit
@patch(_DETECT, return_value="cpu")
@patch(_MOVE)
@patch(_RETRY, side_effect=_invoke_retry)
@patch("transformers.LEDForConditionalGeneration.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_led_branch_unpacks_led_model_kwargs(
    mock_tok, mock_led, mock_retry, mock_move_pipe, _mock_dev, tmp_path
) -> None:
    mock_tok.return_value = MagicMock()
    mock_led.return_value = MagicMock()
    SummaryModel(
        "allenai/led-base-16384",
        device="cpu",
        cache_dir=str(tmp_path),
        revision=None,
    )
    assert mock_led.called
    assert mock_tok.called
    assert mock_retry.called
    assert mock_move_pipe.called
    call_kw = mock_led.call_args.kwargs
    assert call_kw.get("use_safetensors") is False


@pytest.mark.unit
@patch(_DETECT, return_value="cpu")
@patch(_MOVE)
@patch(_RETRY, side_effect=_invoke_retry)
@patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_auto_branch_unpacks_model_kwargs(
    mock_tok, mock_auto, mock_retry, mock_move_pipe, _mock_dev, tmp_path
) -> None:
    mock_tok.return_value = MagicMock()
    mock_auto.return_value = MagicMock()
    SummaryModel(
        "google/long-t5-tglobal-base",
        device="cpu",
        cache_dir=str(tmp_path),
        revision=None,
    )
    assert mock_auto.called
    assert mock_tok.called
    assert mock_retry.called
    assert mock_move_pipe.called
    assert mock_auto.call_args.kwargs.get("local_files_only") is True
