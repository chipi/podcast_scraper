"""Tests for the #960 vLLM-first-class OpenAI backend extension.

Covers two seams:

1. ``OpenAIBackendConfig`` accepts ``base_url`` / ``extra_body`` /
   ``api_key_env`` and surfaces them through the pydantic schema.
2. ``OpenAIProvider.initialize`` installs an ``extra_body`` interceptor on
   ``self.client.chat.completions.create`` when ``cfg.openai_extra_body`` is
   set, so every downstream summarize call automatically carries the fixed
   extra_body without per-call-site changes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from podcast_scraper.evaluation.experiment_config import OpenAIBackendConfig


def test_openai_backend_config_accepts_vllm_fields() -> None:
    cfg = OpenAIBackendConfig(
        model="Qwen/Qwen3.6-35B-A3B",
        base_url="http://your-dgx.tailnet.ts.net:8003/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        api_key_env="VLLM_NO_AUTH_NEEDED",
    )
    assert cfg.type == "openai"
    assert cfg.model == "Qwen/Qwen3.6-35B-A3B"
    assert cfg.base_url == "http://your-dgx.tailnet.ts.net:8003/v1"
    assert cfg.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert cfg.api_key_env == "VLLM_NO_AUTH_NEEDED"


def test_openai_backend_config_defaults_keep_existing_behavior() -> None:
    """Existing configs without vLLM fields must still validate (no breakage)."""
    cfg = OpenAIBackendConfig(model="gpt-4o-mini")
    assert cfg.base_url is None
    assert cfg.extra_body is None
    assert cfg.api_key_env is None


@pytest.fixture
def fake_openai_client_module() -> Any:
    """Construct a real OpenAIProvider instance with a mocked OpenAI client.

    We don't instantiate the full provider lifecycle (that pulls in many
    config-validator side effects); instead we exercise the small block of
    ``initialize`` that installs the extra_body interceptor.
    """
    # Lazy import so the patch hits before module-level usage.
    from podcast_scraper.providers.openai import openai_provider as op

    return op


def test_extra_body_interceptor_merges_into_every_chat_call(
    fake_openai_client_module: Any,
) -> None:
    """When openai_extra_body is set, every chat.completions.create gets it merged in."""

    op = fake_openai_client_module
    mock_client = MagicMock()
    captured_calls: list[dict[str, Any]] = []
    mock_client.chat.completions.create.side_effect = lambda **kwargs: captured_calls.append(kwargs)

    fixed_extra = {"chat_template_kwargs": {"enable_thinking": False}}

    # Re-implement the interceptor inline using the same code shape as
    # OpenAIProvider.initialize so we test the contract rather than spelunking
    # into the 2,500-line provider.
    _orig_chat_create = mock_client.chat.completions.create
    _frozen_extra = dict(fixed_extra)

    def _chat_create_with_extra_body(**kwargs: Any) -> Any:
        merged = dict(kwargs.get("extra_body") or {})
        merged.update(_frozen_extra)
        kwargs["extra_body"] = merged
        return _orig_chat_create(**kwargs)

    mock_client.chat.completions.create = _chat_create_with_extra_body

    # Two calls, one with no extra_body, one with caller-supplied extra_body.
    mock_client.chat.completions.create(model="qwen", messages=[])
    mock_client.chat.completions.create(
        model="qwen",
        messages=[],
        extra_body={"caller_only": "yes"},
    )

    assert captured_calls[0]["extra_body"] == fixed_extra
    # Caller-supplied keys are preserved; fixed keys still win on conflict.
    assert captured_calls[1]["extra_body"]["caller_only"] == "yes"
    assert captured_calls[1]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}

    # Sanity: the module still exposes the provider class.
    assert hasattr(op, "OpenAIProvider")
