"""LLM calls carry their run / episode / trace ids to the LiteLLM gateway (Langfuse join keys).

The gateway's Langfuse callback reads the request ``metadata``: ``session_id`` becomes the trace's
session, ``trace_metadata`` the trace's metadata. Without them a Langfuse generation could not be
joined to the run, the episode or the OTel trace (none carried any, 2026-10-14). Verified live
against the homelab gateway and the DGX vLLM: both accept the nested form; OpenAI's own
``metadata`` takes flat strings only, so the OpenAI provider never sends it.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider
from podcast_scraper.providers.vllm import VLLMProvider
from podcast_scraper.utils import correlation

pytestmark = pytest.mark.unit

_MODEL = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
_TRACE = "0123456789abcdef0123456789abcdef"


def _vllm() -> VLLMProvider:
    p = VLLMProvider(
        Config(
            rss="https://example.com/feed.xml",
            summary_provider="vllm",
            generate_summaries=True,
            vllm_api_base="http://homelab:4001/v1",
            vllm_summary_model=_MODEL,
        )
    )
    p.client = MagicMock()
    return p


@pytest.fixture
def ids(monkeypatch):
    monkeypatch.setattr(correlation, "get_run_id", lambda: "run-20261014T120000Z")
    monkeypatch.setattr(correlation, "get_episode_id", lambda: "ep-1")
    monkeypatch.setattr(correlation, "current_trace_id", lambda: _TRACE)


def _sent(p) -> Dict[str, Any]:
    _, kwargs = p.client.chat.completions.create.call_args
    return dict(kwargs)


def test_gateway_call_carries_session_and_trace_metadata(ids):
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[], max_tokens=5)
    md = _sent(p)["extra_body"]["metadata"]
    assert md["session_id"] == "run-20261014T120000Z"
    assert md["trace_metadata"] == {
        "run_id": "run-20261014T120000Z",
        "episode_id": "ep-1",
        "otel_trace_id": _TRACE,
    }


def test_existing_extra_body_and_metadata_are_kept(ids):
    p = _vllm()
    p._chat_create(
        model=_MODEL,
        messages=[],
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "metadata": {"tags": ["x"]},
        },
    )
    body = _sent(p)["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["metadata"]["tags"] == ["x"]
    assert body["metadata"]["session_id"] == "run-20261014T120000Z"


def test_no_ids_no_metadata(monkeypatch):
    monkeypatch.setattr(correlation, "get_run_id", lambda: None)
    monkeypatch.setattr(correlation, "get_episode_id", lambda: None)
    monkeypatch.setattr(correlation, "current_trace_id", lambda: "-")
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[])
    assert "extra_body" not in _sent(p)


def test_a_server_that_rejects_metadata_self_heals(ids):
    # An older vLLM may validate `metadata` as string-only: retry once without it, remember.
    p = _vllm()

    def _reject(**kw):
        if "metadata" in (kw.get("extra_body") or {}):
            raise RuntimeError(
                "400 Bad Request: body.metadata.trace_metadata: Input should be a valid string"
            )
        return MagicMock()

    p.client.chat.completions.create.side_effect = _reject
    p._chat_create(model=_MODEL, messages=[])
    assert p.client.chat.completions.create.call_count == 2
    p._chat_create(model=_MODEL, messages=[])
    assert p.client.chat.completions.create.call_count == 3  # no metadata, no retry


def test_openai_never_sends_join_metadata(ids):
    p = OpenAIProvider(
        Config(
            rss="https://example.com/feed.xml",
            summary_provider="openai",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="test-api-key-123",
        )
    )
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[])
    assert "metadata" not in (_sent(p).get("extra_body") or {})
