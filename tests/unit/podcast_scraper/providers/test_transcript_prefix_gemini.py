"""RFC-115 Phase E: Gemini explicit context caching (cachedContent).

Gemini's implicit cache does not fire for our pattern (probed 0%), so the transcript goes into an
explicit cache handle referenced across the episode's stages. It is STATEFUL and storage-billed, so
it is OFF by default (its own gemini_context_cache_enabled flag, not the global one). Live-verified:
summary creates one handle, GI reuses it at ~93% cache, cleanup deletes it.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from podcast_scraper import config as cfgmod
from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

pytestmark = pytest.mark.unit

TRANSCRIPT = ("HOST: welcome. GUEST: reliable durable software for small teams. " * 120).strip()


def _provider(*, enabled: bool) -> GeminiProvider:
    cfg = cfgmod.Config(
        rss="https://x/f.xml",
        summary_provider="gemini",
        gemini_api_key="test-gemini-key",
        gemini_summary_model="gemini-2.5-flash",
        generate_summaries=True,
        gemini_context_cache_enabled=enabled,
    )
    p = GeminiProvider(cfg)
    p._summarization_initialized = True
    return p


def _mock_client(cache_name: str = "cachedContents/abc") -> Mock:
    client = Mock()
    cache_obj = Mock()
    cache_obj.name = cache_name  # ``name`` is a reserved Mock kwarg; set it as an attribute
    client.caches.create.return_value = cache_obj
    resp = Mock()
    resp.text = "a summary"
    resp.usage_metadata = Mock(
        prompt_token_count=100, candidates_token_count=20, cached_content_token_count=90
    )
    resp.candidates = [Mock()]
    client.models.generate_content.return_value = resp
    return client


def _last_call(client: Mock) -> dict:
    return dict(client.models.generate_content.call_args.kwargs)


def test_gemini_cache_on_uses_cached_content_and_removes_transcript() -> None:
    p = _provider(enabled=True)
    p.client = _mock_client()
    p.summarize(text=TRANSCRIPT, episode_title="Ep")

    p.client.caches.create.assert_called_once()  # one handle created for the transcript
    kwargs = _last_call(p.client)
    config = kwargs["config"]  # a dict from _merge_generate_content_config
    assert config.get("cached_content") == "cachedContents/abc"
    assert "system_instruction" not in config  # instructions folded into contents, not system
    # the transcript is in the cache, NOT in the request contents
    assert TRANSCRIPT not in str(kwargs["contents"])


def test_gemini_cache_off_is_legacy() -> None:
    p = _provider(enabled=False)
    p.client = _mock_client()
    p.summarize(text=TRANSCRIPT, episode_title="Ep")

    p.client.caches.create.assert_not_called()
    config = _last_call(p.client)["config"]
    assert "cached_content" not in config
    assert config.get("system_instruction")  # legacy keeps the stage system prompt here
    assert TRANSCRIPT in str(_last_call(p.client)["contents"])


def test_gemini_cache_reused_across_stages() -> None:
    p = _provider(enabled=True)
    p.client = _mock_client()
    p.summarize(text=TRANSCRIPT, episode_title="Ep")
    p.generate_insights(text=TRANSCRIPT, episode_title="Ep", max_insights=3)
    # same transcript + model -> the handle is created once and reused
    p.client.caches.create.assert_called_once()


def test_gemini_short_transcript_skips_cache() -> None:
    p = _provider(enabled=True)
    p.client = _mock_client()
    p.summarize(text="too short to cache", episode_title="Ep")
    p.client.caches.create.assert_not_called()  # below the min cacheable size -> legacy


def test_gemini_cleanup_deletes_handles() -> None:
    p = _provider(enabled=True)
    p.client = _mock_client()
    p.summarize(text=TRANSCRIPT, episode_title="Ep")
    assert p._gemini_cache_handles  # a handle is live
    p.cleanup()
    p.client.caches.delete.assert_called_once()
    assert not p._gemini_cache_handles


def test_gemini_cached_tokens_surface_in_llm_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cachedContent saving must be observable: summarize forwards the response so the llm_cost
    event carries the normalised cached_content token count."""
    import podcast_scraper.workflow.cost_monitoring as cm
    from podcast_scraper.workflow.token_accounting import extract_token_usage

    captured: dict = {}
    real = cm.emit_llm_cost_event

    def _spy(*a, **k):  # noqa: ANN002, ANN003
        captured.update(k)
        return real(*a, **k)

    monkeypatch.setattr(cm, "emit_llm_cost_event", _spy)

    p = _provider(enabled=True)
    p.client = _mock_client()  # usage_metadata.cached_content_token_count = 90
    p.summarize(text=TRANSCRIPT, episode_title="Ep")

    assert captured.get("response") is not None
    assert extract_token_usage("gemini", captured["response"]).cached_input_tokens == 90
