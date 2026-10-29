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
            vllm_api_base="http://gateway.test:4001/v1",
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


# --- the Langfuse shape (2026-10-15) --------------------------------------------------------
#
# With only ``session_id``, LiteLLM keys the Langfuse trace on the run: one run's 76 calls sat in
# ONE trace whose metadata named only the first call's episode, and every generation was called
# ``litellm:proj-podcast-prod``. LiteLLM (1.95.0 prod, 1.96.2 homelab; integrations/langfuse)
# pops ``trace_id`` / ``trace_name`` / ``generation_name`` out of the request metadata. Verified
# on the homelab gateway: the trace id becomes the OTel trace id, so a VictoriaTraces id opens
# the Langfuse trace, the session stays the run, and each generation is named after its stage.


def test_the_langfuse_trace_is_the_otel_trace(ids):
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[])
    md = _sent(p)["extra_body"]["metadata"]
    assert md["trace_id"] == _TRACE
    assert md["trace_name"] == "episode ep-1"
    assert md["session_id"] == "run-20261014T120000Z"


def test_without_an_otel_trace_no_trace_id_is_sent_and_the_run_trace_is_named(monkeypatch):
    # LiteLLM then keys the trace on the session (the run). Unnamed, that trace showed up in
    # prod as "litellm-acompletion" (2026-10-15: run-level speaker_detection before any
    # episode span), so it is named after the run.
    monkeypatch.setattr(correlation, "get_run_id", lambda: "run-1")
    monkeypatch.setattr(correlation, "get_episode_id", lambda: "ep-1")
    monkeypatch.setattr(correlation, "current_trace_id", lambda: "-")
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[])
    md = _sent(p)["extra_body"]["metadata"]
    assert "trace_id" not in md
    assert md["trace_name"] == "run run-1"
    assert md["session_id"] == "run-1"


def test_the_generation_is_named_after_its_stage_and_the_stage_is_not_sent_to_the_api(ids):
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[], _lf_stage="kg")
    sent = _sent(p)
    assert "_lf_stage" not in sent
    assert sent["extra_body"]["metadata"]["generation_name"] == "kg"


def test_a_stage_alone_still_names_the_generation(monkeypatch):
    monkeypatch.delenv("PODCAST_RELEASE", raising=False)
    monkeypatch.setattr(correlation, "get_run_id", lambda: None)
    monkeypatch.setattr(correlation, "get_episode_id", lambda: None)
    monkeypatch.setattr(correlation, "current_trace_id", lambda: "-")
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[], _lf_stage="storyline_summary")
    assert _sent(p)["extra_body"]["metadata"] == {"generation_name": "storyline_summary"}


def test_openai_drops_the_stage_too(ids):
    p = OpenAIProvider(
        Config(
            rss="https://example.com/feed.xml",
            summary_provider="openai",
            openai_summary_model="gpt-4o-mini",
            openai_api_key="test-api-key-123",
        )
    )
    p.client = MagicMock()
    p._chat_create(model="gpt-4o-mini", messages=[], _lf_stage="kg")
    sent = _sent(p)
    assert "_lf_stage" not in sent
    assert "metadata" not in (sent.get("extra_body") or {})


def test_every_chat_call_names_its_stage():
    # A call site added without a stage would land in Langfuse as an unnamed generation.
    import ast
    import inspect

    from podcast_scraper.providers.openai import openai_provider

    tree = ast.parse(inspect.getsource(openai_provider))
    missing = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "_chat_create"
        and not any(kw.arg == "_lf_stage" for kw in node.keywords)
        and not any(kw.arg is None for kw in node.keywords)  # the retry passes **kwargs on
    ]
    starred = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "_chat_create"
        and any(kw.arg is None for kw in node.keywords)
        and not any(kw.arg == "_lf_stage" for kw in node.keywords)
    ]
    retry_lines = [
        c.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_chat_create"
        for c in ast.walk(n)
        if isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "_chat_create"
    ]
    assert missing == []
    assert sorted(set(starred) - set(retry_lines)) == []


def test_the_langfuse_trace_carries_the_release(ids, monkeypatch):
    # 2026-10-15 prod: every Langfuse trace had release=NULL while GlitchTip events carried
    # sha-f35a446. LiteLLM (1.95.0 integrations/langfuse) turns any ``trace_<key>`` in the request
    # metadata into the trace's ``<key>``, so ``trace_release`` sets it per call (ADR-0010).
    monkeypatch.setenv("PODCAST_RELEASE", "sha-f35a446")
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[])
    assert _sent(p)["extra_body"]["metadata"]["trace_release"] == "sha-f35a446"


def test_no_release_no_trace_release(ids, monkeypatch):
    monkeypatch.delenv("PODCAST_RELEASE", raising=False)
    p = _vllm()
    p._chat_create(model=_MODEL, messages=[])
    assert "trace_release" not in _sent(p)["extra_body"]["metadata"]
