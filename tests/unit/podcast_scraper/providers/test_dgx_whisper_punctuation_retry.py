"""DGX Whisper: an unpunctuated transcript gets ONE prompted retry; nothing is ever lost (#2284).

Prod: 118 of 1,734 DGX Whisper transcripts had no punctuation and no capitals. Reproduced on the
shortest affected episode: the prod request gave the same 0-punctuation text twice; the same audio
with a punctuated ``prompt`` came back punctuated. These pin the retry strategy:

- a punctuated first result costs nothing extra (one call);
- an unpunctuated one gets exactly one more call, WITH the prompt (the identical request would
  return the identical text), and the better of the two is kept;
- a retry that errors, comes back unpunctuated, or echoes the prompt keeps the FIRST transcript,
  and never charges the circuit breaker -- the endpoint answered, the content is the problem;
- the English prompt is never sent for another language;
- the outcome is recorded on the result so the episode can be flagged.
"""

from __future__ import annotations

from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.tailnet_dgx import whisper_provider as wp
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)
from podcast_scraper.transcription.punctuation import PUNCTUATION_PROMPT

pytestmark = pytest.mark.unit

UNPUNCTUATED = "so the ports moved north over a decade and the trading families followed " * 30
PUNCTUATED = (
    "So the ports moved north over a decade. Why? The delta silted up, and trade moved. " * 25
)


@pytest.fixture(autouse=True)
def _isolate():
    wp._whisper_breaker.reset()
    with (
        patch(
            "podcast_scraper.providers.resilience.sockets._duration_via_ffprobe", return_value=None
        ),
        patch.object(wp, "check_faster_whisper_health", return_value=True),
        patch.object(wp.time, "sleep"),
    ):
        yield
    wp._whisper_breaker.reset()


def _provider(**cfg_kw: Any) -> TailnetDgxWhisperTranscriptionProvider:
    cfg = Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
            "openai_api_key": "sk-test",
            **cfg_kw,
        }
    )
    p = TailnetDgxWhisperTranscriptionProvider(cfg)
    p._initialized = True
    return p


class _Server:
    """Stands in for ``_transcribe_dgx``: answers each call from a script, recording prompts."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.prompts: List[Optional[str]] = []

    def __call__(self, audio_path, language, timeout_sec=None, model_override=None, prompt=None):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer, [{"start": 0.0, "end": 1.0, "text": answer[:20]}], 1.0


def _run(provider, server, tmp_path, language: Optional[str] = "en"):
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"\x00\x01")
    with patch.object(provider, "_transcribe_dgx", side_effect=server):
        result, _ = provider.transcribe_with_segments(str(audio), language)
    return result


def test_a_punctuated_transcript_costs_one_call(tmp_path) -> None:
    server = _Server(PUNCTUATED)
    result = _run(_provider(), server, tmp_path)
    assert server.prompts == [None]
    assert result["text"] == PUNCTUATED
    assert result["punctuation"]["unpunctuated"] is False
    assert result["punctuation"]["retried"] is False


def test_an_unpunctuated_transcript_is_retried_once_with_the_prompt(tmp_path) -> None:
    server = _Server(UNPUNCTUATED, PUNCTUATED)
    result = _run(_provider(), server, tmp_path)
    assert server.prompts == [None, PUNCTUATION_PROMPT]
    assert result["text"] == PUNCTUATED
    assert result["punctuation"]["retried"] is True
    assert result["punctuation"]["unpunctuated"] is False


@pytest.mark.parametrize(
    "second, rejected",
    [
        (UNPUNCTUATED, "still unpunctuated"),
        (PUNCTUATION_PROMPT + " " + PUNCTUATED, "echoed the prompt"),
    ],
)
def test_a_retry_that_does_not_help_keeps_the_first_transcript(tmp_path, second, rejected) -> None:
    server = _Server(UNPUNCTUATED, second)
    result = _run(_provider(), server, tmp_path)
    assert len(server.prompts) == 2, "one retry, never a loop"
    assert result["text"] == UNPUNCTUATED
    assert result["punctuation"]["unpunctuated"] is True
    assert result["punctuation"]["retry_rejected"] == rejected


def test_a_failed_retry_keeps_the_transcript_and_spares_the_breaker(tmp_path) -> None:
    provider = _provider()
    server = _Server(UNPUNCTUATED, RuntimeError("connection reset"))
    with patch.object(wp._whisper_breaker, "record_failure") as record_failure:
        result = _run(provider, server, tmp_path)
    assert result["text"] == UNPUNCTUATED
    assert result["punctuation"]["unpunctuated"] is True
    assert "connection reset" in result["punctuation"]["retry_error"]
    record_failure.assert_not_called()
    assert wp._whisper_breaker.allow()


def test_another_language_is_not_sent_the_english_prompt(tmp_path) -> None:
    server = _Server(UNPUNCTUATED)
    result = _run(_provider(), server, tmp_path, language="de")
    assert server.prompts == [None]
    assert result["punctuation"]["unpunctuated"] is True
    assert result["punctuation"]["skipped"] == "language=de"


def test_mode_off_detects_but_never_prompts(tmp_path) -> None:
    server = _Server(UNPUNCTUATED)
    result = _run(_provider(dgx_whisper_punctuation_prompt="off"), server, tmp_path)
    assert server.prompts == [None]
    assert result["punctuation"]["unpunctuated"] is True


def test_mode_always_prompts_the_first_request_and_does_not_repeat_it(tmp_path) -> None:
    server = _Server(UNPUNCTUATED)
    result = _run(_provider(dgx_whisper_punctuation_prompt="always"), server, tmp_path)
    assert server.prompts == [PUNCTUATION_PROMPT], "the same prompted request would not change"
    assert result["punctuation"]["prompted_first"] is True
    assert result["punctuation"]["unpunctuated"] is True


def test_the_reprocess_strategy_retries_too(tmp_path) -> None:
    provider = _provider(resilience_run_context="reprocess")
    assert provider._strategy is wp.FailureStrategy.HOLD
    server = _Server(UNPUNCTUATED, PUNCTUATED)
    result = _run(provider, server, tmp_path)
    assert server.prompts == [None, PUNCTUATION_PROMPT]
    assert result["text"] == PUNCTUATED


@patch("httpx.Client")
def test_the_prompt_is_sent_as_the_request_form_field(mock_client_cls: MagicMock, tmp_path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"abc")
    resp = MagicMock()
    resp.json.return_value = {"text": "Hello there. How are you?", "segments": []}
    client = MagicMock()
    client.__enter__.return_value = client
    client.post.return_value = resp
    mock_client_cls.return_value = client
    provider = _provider()

    provider._transcribe_dgx(str(audio), "en", prompt=PUNCTUATION_PROMPT)
    assert client.post.call_args.kwargs["data"]["prompt"] == PUNCTUATION_PROMPT

    provider._transcribe_dgx(str(audio), "en")
    assert "prompt" not in client.post.call_args.kwargs["data"], "prod request unchanged"
