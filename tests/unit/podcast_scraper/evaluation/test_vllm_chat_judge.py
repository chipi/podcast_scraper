"""Unit tests for the vLLM chat judge transport.

Mirrors the existing :mod:`ollama_chat` contract: scalar float in [0, 10],
parsed from the model's reply. Adds vLLM-specific coverage for the
``enable_thinking`` chat_template_kwarg, env-var base URL precedence, and
the :class:`JudgeUnavailableError` wrapping on transient failures.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from podcast_scraper.evaluation.judges.base import JudgeUnavailableError
from podcast_scraper.evaluation.judges.vllm_chat import _resolve_vllm_base, VllmChatJudge

pytestmark = pytest.mark.unit


def _ok_response(text: str) -> MagicMock:
    """Build a stub httpx Response carrying an OpenAI-compat chat reply."""
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


def _stub_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.post.return_value = _ok_response(response_text)
    return client


def test_resolve_base_prefers_vllm_api_base_env(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_API_BASE", "http://explicit.example:9000/v1")
    monkeypatch.setenv("DGX_TAILNET_FQDN", "ignored.tailnet.ts.net")
    assert _resolve_vllm_base() == "http://explicit.example:9000/v1"


def test_resolve_base_falls_back_to_dgx_tailnet_on_port_8003(monkeypatch) -> None:
    """No VLLM_API_BASE → ``http://<DGX_TAILNET_FQDN>:8003/v1``. Port 8003 is
    the autoresearch slot (NOT 11434 — that's Ollama, not vLLM)."""
    monkeypatch.delenv("VLLM_API_BASE", raising=False)
    monkeypatch.setenv("DGX_TAILNET_FQDN", "dgx-llm-1.tail6d0ed4.ts.net")
    assert _resolve_vllm_base() == "http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1"


def test_resolve_base_falls_back_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_API_BASE", raising=False)
    monkeypatch.delenv("DGX_TAILNET_FQDN", raising=False)
    assert _resolve_vllm_base() == "http://localhost:8003/v1"


def test_score_extracts_float_from_json_reply() -> None:
    client = _stub_client('{"score": 7.5}')
    judge = VllmChatJudge(
        model="Qwen/Qwen3.5-35B-A3B",
        api_base="http://stub:8003/v1",
        client=client,
    )
    assert judge.score(user_content="rubric") == pytest.approx(7.5)


def test_score_extracts_bare_float_when_no_json() -> None:
    """Mirror :func:`ollama_chat._parse_score`: ``"7"`` is valid score input."""
    client = _stub_client("7")
    judge = VllmChatJudge(model="any", api_base="http://stub:8003/v1", client=client)
    assert judge.score(user_content="rubric") == pytest.approx(7.0)


def test_score_clips_out_of_range_to_zero_to_ten() -> None:
    client = _stub_client('{"score": 42}')
    judge = VllmChatJudge(model="any", api_base="http://stub:8003/v1", client=client)
    assert judge.score(user_content="rubric") == pytest.approx(10.0)


def test_thinking_flag_injects_chat_template_kwargs() -> None:
    """When ``thinking=True``, the request body must carry
    ``chat_template_kwargs={"enable_thinking": True}`` so vLLM forwards it to
    the tokenizer's chat template (Qwen3 / Qwen3.5 reasoning toggle)."""
    client = _stub_client('{"score": 5}')
    judge = VllmChatJudge(
        model="Qwen/Qwen3.5-35B-A3B",
        api_base="http://stub:8003/v1",
        client=client,
        thinking=True,
    )
    judge.score(user_content="rubric")
    sent_body: dict[str, Any] = client.post.call_args.kwargs["json"]
    assert sent_body["chat_template_kwargs"] == {"enable_thinking": True}


def test_thinking_default_off_omits_chat_template_kwargs() -> None:
    client = _stub_client('{"score": 5}')
    judge = VllmChatJudge(model="any", api_base="http://stub:8003/v1", client=client)
    judge.score(user_content="rubric")
    sent_body: dict[str, Any] = client.post.call_args.kwargs["json"]
    assert "chat_template_kwargs" not in sent_body


def test_no_choices_raises_judge_unavailable() -> None:
    """vLLM occasionally returns an empty ``choices`` array under heavy
    contention. That must surface as :class:`JudgeUnavailableError`, not
    silently degrade to a default score."""
    client = MagicMock()
    client.post.return_value.json.return_value = {"choices": []}
    judge = VllmChatJudge(model="any", api_base="http://stub:8003/v1", client=client)
    with pytest.raises(JudgeUnavailableError, match="returned no choices"):
        judge.score(user_content="rubric")


def test_api_key_sets_bearer_auth_header() -> None:
    """DGX vLLM autoresearch serves under ``--api-key <token>``; requests
    without the header get 401. Env-var / constructor VLLM_API_KEY plumbs
    the token into an Authorization: Bearer header."""
    client = _stub_client('{"score": 5}')
    judge = VllmChatJudge(
        model="autoresearch",
        api_base="http://stub:8003/v1",
        api_key="buddy-is-the-king",
        client=client,
    )
    judge.score(user_content="rubric")
    sent_headers = client.post.call_args.kwargs.get("headers") or {}
    assert sent_headers.get("Authorization") == "Bearer buddy-is-the-king"


def test_no_api_key_omits_auth_header() -> None:
    """No key set → no Authorization header. Keeps the transport usable
    against a bare local no-auth vLLM (dev, tests) without a stub header."""
    client = _stub_client('{"score": 5}')
    judge = VllmChatJudge(
        model="any",
        api_base="http://stub:8003/v1",
        api_key="",
        client=client,
    )
    judge.score(user_content="rubric")
    sent_headers = client.post.call_args.kwargs.get("headers") or {}
    assert "Authorization" not in sent_headers
