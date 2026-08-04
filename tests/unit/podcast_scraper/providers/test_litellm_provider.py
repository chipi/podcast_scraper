"""Unit coverage for the LiteLLM gateway provider (#1356): identity, served-alias check, and the
reasoning-off extra_body wrap that must reach every LLM stage."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_served_matches_exact_alias_only() -> None:
    """Gateway aliases are exact contract names — no startswith tolerance (which would let
    homelab-flash spuriously match homelab-flash-thinking, the trap the vLLM matcher has)."""
    from podcast_scraper.providers.litellm.litellm_provider import _served_matches

    assert _served_matches("homelab-qwen", {"homelab-qwen", "homelab-flash"}) is True
    assert _served_matches("Homelab-Qwen", {"homelab-qwen"}) is True  # casefold
    assert _served_matches("homelab-flash", {"homelab-flash-thinking"}) is False  # no startswith
    assert _served_matches("homelab-qwen", {"homelab-flash"}) is False


def test_factory_builds_litellm_provider_for_all_llm_stages() -> None:
    """summary/speaker/quote/entailment/kg all accept 'litellm', and the factory builds the gateway
    provider pointed at the configured base_url + alias."""
    from podcast_scraper import config
    from podcast_scraper.summarization.factory import create_summarization_provider

    c = config.Config.model_validate(
        {
            "profile": "prod_dgx_full",
            "summary_provider": "litellm",
            "speaker_detector_provider": "litellm",
            "quote_extraction_provider": "litellm",
            "entailment_provider": "litellm",
            "kg_extraction_provider": "litellm",
            "litellm_api_base": "http://homelab:4001/v1",
            "litellm_api_key": "sk-test",
            "litellm_summary_model": "homelab-qwen",
            "litellm_verify_served_model": False,  # no network in unit
        }
    )
    p = create_summarization_provider(c)
    assert type(p).__name__ == "LiteLLMProvider"
    assert p.summary_model == "homelab-qwen"
    assert p.cleaning_model == "homelab-qwen"  # defaults to summary alias
    assert "homelab:4001" in str(p.client.base_url)


def test_reasoning_off_extra_body_wraps_the_client_for_every_call() -> None:
    """litellm_extra_body must be injected by the base client wrap, so reasoning-off applies to
    clean/quotes/entailment/speaker/GI/KG — not just summary (verified via the wrap, not a network
    call)."""
    import inspect

    from podcast_scraper import config
    from podcast_scraper.summarization.factory import create_summarization_provider

    c = config.Config.model_validate(
        {
            "profile": "prod_dgx_full",
            "summary_provider": "litellm",
            "litellm_api_base": "http://homelab:4001/v1",
            "litellm_api_key": "sk-test",
            "litellm_summary_model": "homelab-qwen",
            "litellm_extra_body": {"reasoning": {"enabled": False}},
            "litellm_verify_served_model": False,
        }
    )
    p = create_summarization_provider(c)
    assert "extra_body" in inspect.getsource(p.client.chat.completions.create)


def _litellm_provider(monkeypatch, **overrides):
    from podcast_scraper import config
    from podcast_scraper.summarization.factory import create_summarization_provider

    base = {
        "profile": "prod_dgx_full",
        "summary_provider": "litellm",
        "litellm_api_base": "http://homelab:4001/v1",
        "litellm_api_key": "sk-test",
        "litellm_summary_model": "homelab-qwen",
        "litellm_verify_served_model": True,
    }
    base.update(overrides)
    return create_summarization_provider(config.Config.model_validate(base))


def _fake_models(*ids):
    import io
    import json as _json

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    body = _json.dumps({"data": [{"id": i} for i in ids]}).encode()
    return lambda *a, **k: _Resp(body)


def test_verify_served_alias_passes_when_gateway_advertises_it(monkeypatch):
    p = _litellm_provider(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", _fake_models("homelab-qwen", "homelab-flash"))
    p._verify_served_model()  # must not raise


def test_verify_served_alias_raises_on_mismatch(monkeypatch):
    from podcast_scraper.providers.litellm.litellm_provider import LiteLLMServedModelMismatch

    p = _litellm_provider(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", _fake_models("homelab-flash", "homelab-glm"))
    with pytest.raises(LiteLLMServedModelMismatch):
        p._verify_served_model()


def test_verify_served_alias_unreachable_warns_not_raises(monkeypatch):
    p = _litellm_provider(monkeypatch)

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    p._verify_served_model()  # unreachable != mismatch — warns, does not raise


def test_api_key_resolves_from_env(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-from-env")
    p = _litellm_provider(monkeypatch, litellm_api_key=None, litellm_verify_served_model=False)
    assert p._resolve_api_key(p.cfg) == "sk-from-env"
