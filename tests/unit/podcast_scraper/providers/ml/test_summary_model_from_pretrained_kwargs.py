"""Exercise SummaryModel._load_model branches that unpack kwargs via cast(Any, ...) (mypy)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.summarizer import SummaryModel


def _invoke_retry(fn, *_a, **_kw):
    return fn()


@pytest.mark.unit
@patch(
    "podcast_scraper.providers.ml.summarizer.SummaryModel._load_model_move_to_device_and_pipeline"
)
@patch(
    "podcast_scraper.providers.ml.summarizer._load_with_retry_summarizer",
    side_effect=_invoke_retry,
)
@patch("transformers.BartForConditionalGeneration.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_bart_branch_unpacks_model_kwargs(
    mock_tok, mock_bart, mock_retry, mock_move_pipe, tmp_path
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
@patch(
    "podcast_scraper.providers.ml.summarizer.SummaryModel._load_model_move_to_device_and_pipeline"
)
@patch(
    "podcast_scraper.providers.ml.summarizer._load_with_retry_summarizer",
    side_effect=_invoke_retry,
)
@patch("transformers.LEDForConditionalGeneration.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_led_branch_unpacks_led_model_kwargs(
    mock_tok, mock_led, mock_retry, mock_move_pipe, tmp_path
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
@patch(
    "podcast_scraper.providers.ml.summarizer.SummaryModel._load_model_move_to_device_and_pipeline"
)
@patch(
    "podcast_scraper.providers.ml.summarizer._load_with_retry_summarizer",
    side_effect=_invoke_retry,
)
@patch("transformers.AutoModelForSeq2SeqLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_summary_model_auto_branch_unpacks_model_kwargs(
    mock_tok, mock_auto, mock_retry, mock_move_pipe, tmp_path
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
