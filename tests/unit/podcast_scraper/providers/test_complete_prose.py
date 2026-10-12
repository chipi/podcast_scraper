"""``complete_prose``: one prompt, plain text, costed under its own stage (RFC-127).

The board summary is the first LLM call outside the pipeline. Its cost must be counted as
``board_summary``, never as the pipeline stage the older ``complete_text`` reports
(``speaker_detection``), and an answer cut off at the token limit must never be returned.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from podcast_scraper import config
from podcast_scraper.providers.guardrails.exceptions import GuardrailViolation
from podcast_scraper.providers.vllm import VLLMProvider


def _response(text: str, finish: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def _provider(
    monkeypatch: pytest.MonkeyPatch, replies: list[Any]
) -> tuple[Any, list[dict], list[str]]:
    """The DGX profile's provider, with the chat call and the cost event recorded, not sent."""
    provider = VLLMProvider(config.Config(profile="prod_dgx_full"))
    provider._summarization_initialized = True
    calls: list[dict] = []
    stages: list[str] = []

    def chat_create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return replies.pop(0)

    monkeypatch.setattr(provider, "_chat_create", chat_create)
    monkeypatch.setattr(provider, "_emit_stage_cost", lambda **kw: stages.append(kw["stage"]))
    return provider, calls, stages


def test_returns_the_text_costed_under_its_own_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls, stages = _provider(monkeypatch, [_response("  A board about risk.  ")])
    out = provider.complete_prose("summarise", stage="board_summary", max_tokens=300)
    assert out == "A board about risk."
    assert stages == ["board_summary"]
    # Prose, not JSON: no response_format, and the token cap asked for.
    assert "response_format" not in calls[0]
    assert calls[0]["max_tokens"] == 300


def test_an_answer_cut_off_at_the_limit_is_never_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls, _stages = _provider(
        monkeypatch, [_response("A board about", "length"), _response("Still cut", "length")]
    )
    with pytest.raises(GuardrailViolation):
        provider.complete_prose("summarise", stage="board_summary")
    assert len(calls) == 2, "one re-roll, then it refuses"


def test_refuses_before_initialize(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _calls, _stages = _provider(monkeypatch, [])
    provider._summarization_initialized = False
    with pytest.raises(RuntimeError):
        provider.complete_prose("summarise", stage="board_summary")
