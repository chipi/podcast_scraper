"""Unit tests for the finale-tier judge clients (Sonnet 4.6 / Gemini 2.5 Pro / R1).

All transports are mocked — these tests verify:

- The judge composes the right API call (model id, temperature=0.0, message shape)
- Usage/cost is read from the mocked response
- Missing credentials raise ``JudgeUnavailableError`` (not bare ``RuntimeError``)
- R1 strips ``<think>`` reasoning tags before returning
- R1 reports zero marginal USD cost (it's local on DGX)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from podcast_scraper.evaluation.judges import (
    DeepSeekR1Judge,
    Gemini25ProJudge,
    JudgeUnavailableError,
    OpenAIChatJudge,
    Sonnet46Judge,
)
from podcast_scraper.evaluation.judges.deepseek_r1 import strip_reasoning_tags

# ---------------------------------------------------------------------------
# Sonnet 4.6


def _mock_anthropic_msg(text: str, *, input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """Build a fake ``anthropic.types.Message`` shape."""
    block = SimpleNamespace(text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")


def test_sonnet46_score_uses_temperature_zero_and_extracts_text() -> None:
    """Sonnet 4.6 judge calls Anthropic with the right shape and reads usage."""
    client = MagicMock()
    client.messages.create.return_value = _mock_anthropic_msg(
        '{"score": 4, "explanation": "ok"}',
        input_tokens=250,
        output_tokens=40,
    )

    judge = Sonnet46Judge(client=client, api_key="sk-test")
    result = judge.score("prompt text", max_tokens=512)

    # API call shape
    args = client.messages.create.call_args.kwargs
    assert args["model"] == "claude-sonnet-4-6"
    assert args["temperature"] == 0.0
    assert args["max_tokens"] == 512
    assert args["messages"] == [{"role": "user", "content": "prompt text"}]

    # Result
    assert result.text == '{"score": 4, "explanation": "ok"}'
    assert result.prompt_tokens == 250
    assert result.completion_tokens == 40
    # Cost = 250/1M * $3 + 40/1M * $15 ≈ $0.00135
    assert result.cost_usd == pytest.approx(
        250 / 1_000_000 * 3.0 + 40 / 1_000_000 * 15.0,
        rel=1e-6,
    )
    assert result.model == "claude-sonnet-4-6"


def test_sonnet46_missing_api_key_raises_judge_unavailable(monkeypatch) -> None:
    """No key + no injected client → JudgeUnavailableError (not bare RuntimeError)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    judge = Sonnet46Judge(api_key="")
    with pytest.raises(JudgeUnavailableError, match="ANTHROPIC_API_KEY"):
        judge.score("prompt")


def test_sonnet46_api_failure_wrapped_as_judge_unavailable() -> None:
    """A transport-level exception is wrapped so the finale runner can continue."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    judge = Sonnet46Judge(client=client, api_key="sk-test")
    with pytest.raises(JudgeUnavailableError, match="Anthropic API call failed"):
        judge.score("prompt")


# ---------------------------------------------------------------------------
# Gemini 2.5 Pro


def _mock_gemini_response(
    text: str, *, prompt_token_count: int, candidates_token_count: int
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_token_count=prompt_token_count,
        candidates_token_count=candidates_token_count,
    )
    return SimpleNamespace(text=text, usage_metadata=usage)


def test_gemini25pro_score_invokes_generate_content_with_temperature_zero() -> None:
    """Gemini 2.5 Pro judge sets temperature=0 + reads usage_metadata."""
    client = MagicMock()
    client.models.generate_content.return_value = _mock_gemini_response(
        '{"score": 5, "explanation": "great"}',
        prompt_token_count=300,
        candidates_token_count=50,
    )

    judge = Gemini25ProJudge(client=client, api_key="g-test")
    result = judge.score("prompt", max_tokens=256)

    args = client.models.generate_content.call_args.kwargs
    assert args["model"] == "gemini-2.5-pro"
    assert args["contents"] == "prompt"
    assert args["config"]["temperature"] == 0.0
    assert args["config"]["max_output_tokens"] == 256

    assert result.text == '{"score": 5, "explanation": "great"}'
    assert result.prompt_tokens == 300
    assert result.completion_tokens == 50
    # Cost = 300/1M * $1.25 + 50/1M * $10
    assert result.cost_usd == pytest.approx(
        300 / 1_000_000 * 1.25 + 50 / 1_000_000 * 10.0,
        rel=1e-6,
    )


def test_gemini25pro_missing_api_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    judge = Gemini25ProJudge(api_key="")
    with pytest.raises(JudgeUnavailableError, match="GEMINI_API_KEY"):
        judge.score("prompt")


# ---------------------------------------------------------------------------
# DeepSeek-R1 (DGX Ollama)


def _mock_openai_chat_response(
    text: str, *, prompt_tokens: int, completion_tokens: int
) -> SimpleNamespace:
    msg = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_strip_reasoning_tags_drops_think_blocks() -> None:
    """``<think>`` blocks are removed but the final JSON is kept verbatim."""
    raw = (
        "<think>I need to consider all four dimensions carefully.</think>\n"
        '{"score": 3, "explanation": "ok"}'
    )
    assert strip_reasoning_tags(raw) == '{"score": 3, "explanation": "ok"}'


def test_strip_reasoning_tags_handles_multiple_blocks_and_no_think() -> None:
    """Multiple ``<think>`` blocks and a plain reply with none."""
    assert strip_reasoning_tags("<think>a</think>X<think>b</think>Y") == "XY"
    assert strip_reasoning_tags("plain") == "plain"


def test_deepseek_r1_score_strips_think_and_reports_zero_cost() -> None:
    """R1 judge strips reasoning tags + reports $0 marginal (local on DGX)."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_chat_response(
        '<think>reasoning here</think>\n{"score": 4, "explanation": "good"}',
        prompt_tokens=1200,
        completion_tokens=300,
    )

    judge = DeepSeekR1Judge(client=client, api_base="http://fake:11434/v1")
    result = judge.score("prompt", max_tokens=2048)

    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "deepseek-r1:32b"
    assert args["temperature"] == 0.0
    assert args["max_tokens"] == 2048
    assert args["messages"] == [{"role": "user", "content": "prompt"}]

    assert result.text == '{"score": 4, "explanation": "good"}'
    assert result.prompt_tokens == 1200
    assert result.completion_tokens == 300
    assert result.cost_usd == 0.0  # local, $0 marginal
    assert result.model == "deepseek-r1:32b"


def test_deepseek_r1_resolves_dgx_tailnet_fqdn(monkeypatch) -> None:
    """Without ``OLLAMA_API_BASE``, the judge composes from ``DGX_TAILNET_FQDN``."""
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    monkeypatch.setenv("DGX_TAILNET_FQDN", "dgx.example.ts.net")
    judge = DeepSeekR1Judge()
    assert judge.api_base == "http://dgx.example.ts.net:11434/v1"


def test_deepseek_r1_api_failure_wrapped() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("connection refused")
    judge = DeepSeekR1Judge(client=client, api_base="http://fake:11434/v1")
    with pytest.raises(JudgeUnavailableError, match="Ollama API call failed"):
        judge.score("prompt")


# ---------------------------------------------------------------------------
# OpenAI GPT-5.4 chat-completions judge


def test_openai_chat_score_uses_max_completion_tokens_and_temperature_zero() -> None:
    """GPT-5.4 judge sends ``max_completion_tokens`` (not ``max_tokens``) + temp 0."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_openai_chat_response(
        '{"score": 5, "explanation": "great"}',
        prompt_tokens=400,
        completion_tokens=60,
    )

    judge = OpenAIChatJudge(client=client, api_key="sk-test")
    result = judge.score("prompt text", max_tokens=512)

    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "gpt-5.4"
    assert args["temperature"] == 0.0
    # GPT-5.x rejects ``max_tokens`` — judge must send ``max_completion_tokens``.
    assert args["max_completion_tokens"] == 512
    assert "max_tokens" not in args
    assert args["messages"] == [{"role": "user", "content": "prompt text"}]

    assert result.text == '{"score": 5, "explanation": "great"}'
    assert result.prompt_tokens == 400
    assert result.completion_tokens == 60
    # Cost = 400/1M * $3 + 60/1M * $15
    assert result.cost_usd == pytest.approx(
        400 / 1_000_000 * 3.0 + 60 / 1_000_000 * 15.0,
        rel=1e-6,
    )
    assert result.model == "gpt-5.4"


def test_openai_chat_missing_api_key_raises(monkeypatch) -> None:
    """No autoresearch-namespaced key → JudgeUnavailableError (plain key never consulted)."""
    monkeypatch.delenv("AUTORESEARCH_JUDGE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", raising=False)
    judge = OpenAIChatJudge(api_key="")
    with pytest.raises(JudgeUnavailableError, match="AUTORESEARCH_JUDGE_OPENAI_API_KEY"):
        judge.score("prompt")


def test_openai_chat_api_failure_wrapped_as_judge_unavailable() -> None:
    """Transport-level exception is wrapped so the finale runner can continue."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("rate limited")
    judge = OpenAIChatJudge(client=client, api_key="sk-test")
    with pytest.raises(JudgeUnavailableError, match="OpenAI API call failed"):
        judge.score("prompt")
