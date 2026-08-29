"""Unit tests for the first-class vLLM provider (ADR-147).

Covers the sibling-of-openai contract: distinct identity/namespace, optional bearer, open-model
token/temperature heuristics, real-model-id wiring, and factory dispatch — without any network.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import yaml

from podcast_scraper.config import Config
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from podcast_scraper.providers.vllm import VLLMProvider

_MODEL = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"


def _vllm_cfg(**overrides: Any) -> Config:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="vllm",
        speaker_detector_provider="vllm",
        generate_summaries=True,
        generate_metadata=True,
        vllm_api_base="http://dgx:8003/v1",
        vllm_summary_model=_MODEL,
        vllm_speaker_model=_MODEL,
    )
    base.update(overrides)
    return Config(**base)


class TestIdentity:
    def test_is_sibling_not_openai_subclass(self):
        # Shares the transport base, but is NOT an OpenAIProvider (ADR-147).
        assert issubclass(VLLMProvider, OpenAICompatibleProvider)
        assert not issubclass(VLLMProvider, OpenAIProvider)
        assert not issubclass(OpenAIProvider, VLLMProvider)

    def test_namespace_and_telemetry(self):
        p = VLLMProvider(_vllm_cfg())
        assert p._CONFIG_NS == "vllm"
        assert p._TELEMETRY_PROVIDER == "vllm"
        assert p._PROVIDER_LABEL == "vLLM"
        assert p.get_capabilities().provider_name == "vllm"

    def test_names_real_model_id_on_the_wire(self):
        # No served-name alias: the wire model is the real HF id (reproducibility, ADR-143/144).
        p = VLLMProvider(_vllm_cfg())
        assert p.summary_model == _MODEL
        assert "autoresearch" != p.summary_model
        # The wire endpoint is config-driven (asserting the live client's base_url is fragile —
        # a sibling test module installs a global openai.OpenAI mock that leaks into the suite).
        assert p.cfg.vllm_api_base == "http://dgx:8003/v1"


class TestOptionalBearer:
    def test_dummy_when_unset(self, monkeypatch):
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        p = VLLMProvider(_vllm_cfg())
        assert p._resolve_api_key(p.cfg) == "EMPTY"

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "tok-from-env")
        p = VLLMProvider(_vllm_cfg())
        assert p._resolve_api_key(p.cfg) == "tok-from-env"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "tok-from-env")
        p = VLLMProvider(_vllm_cfg(vllm_api_key="explicit-tok"))
        assert p._resolve_api_key(p.cfg) == "explicit-tok"

    def test_authenticate_never_raises_without_key(self, monkeypatch):
        # Unlike OpenAIProvider, a missing/blank bearer must not fail construction.
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
        VLLMProvider(_vllm_cfg())  # must not raise


class TestOpenModelHeuristics:
    def test_token_kwarg_always_max_tokens(self):
        p = VLLMProvider(_vllm_cfg())
        # Even a model whose NAME looks like an OpenAI reasoning model must not get the rename.
        assert p._token_kwarg(512, model="gpt-5-lookalike") == {"max_tokens": 512}
        assert p._token_kwarg(256) == {"max_tokens": 256}

    def test_temperature_not_fixed(self):
        p = VLLMProvider(_vllm_cfg())
        assert p._temp_fixed_at_default == set()

    def test_cleaning_defaults_to_summary_model(self):
        p = VLLMProvider(_vllm_cfg())
        assert p.cleaning_model == _MODEL

    def test_cleaning_model_explicit_pin_respected(self):
        p = VLLMProvider(_vllm_cfg(vllm_cleaning_model="some/other-model"))
        assert p.cleaning_model == "some/other-model"


class TestOpenAIProviderUnaffected:
    def test_openai_still_requires_sk_key(self, monkeypatch):
        # The base auth is unchanged for the OpenAI-native sibling.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(
                Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
            )


class TestFactoryDispatch:
    def test_summarization_factory_returns_vllm(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        p = create_summarization_provider(_vllm_cfg())
        assert isinstance(p, VLLMProvider)

    def test_speaker_factory_returns_vllm(self):
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        p = create_speaker_detector(_vllm_cfg())
        assert isinstance(p, VLLMProvider)


class TestServedModelVerification:
    """ADR-147 B3: fail-closed served-model check.

    A wrong model on the DGX slot stops the run; an unreachable endpoint only warns.
    """

    @staticmethod
    def _fake_urlopen(ids):
        payload = __import__("json").dumps({"data": [{"id": i} for i in ids]}).encode()

        class _Resp:
            def read(self_inner):
                return payload

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        def _open(req, timeout=10):
            return _Resp()

        return _open

    def test_match_passes(self, monkeypatch):
        import podcast_scraper.providers.vllm.vllm_provider as m

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen([_MODEL]))
        VLLMProvider(_vllm_cfg())._verify_served_model()  # must not raise

    def test_dated_suffix_tolerated(self, monkeypatch):
        import podcast_scraper.providers.vllm.vllm_provider as m

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen([_MODEL + "-20260101"]))
        VLLMProvider(_vllm_cfg())._verify_served_model()  # startswith tolerance

    def test_mismatch_raises(self, monkeypatch):
        import podcast_scraper.providers.vllm.vllm_provider as m
        from podcast_scraper.providers.vllm.vllm_provider import VLLMServedModelMismatch

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen(["someoneelse/other"]))
        with pytest.raises(VLLMServedModelMismatch):
            VLLMProvider(_vllm_cfg())._verify_served_model()

    def test_unreachable_warns_not_raises(self, monkeypatch):
        import podcast_scraper.providers.vllm.vllm_provider as m

        def _boom(req, timeout=10):
            raise OSError("connection refused")

        monkeypatch.setattr(m.urllib.request, "urlopen", _boom)
        VLLMProvider(_vllm_cfg())._verify_served_model()  # must NOT raise

    def test_initialize_gated_by_flag(self, monkeypatch):
        # Flag off -> initialize() must not run the served-model check; on -> it must.
        monkeypatch.setattr(OpenAICompatibleProvider, "initialize", lambda self: None)
        for flag, expected in ((False, 0), (True, 1)):
            called = {"n": 0}
            p = VLLMProvider(_vllm_cfg(vllm_verify_served_model=flag))
            monkeypatch.setattr(p, "_verify_served_model", lambda: called.__setitem__("n", 1))
            p.initialize()
            assert called["n"] == expected


def _dgx_profile_cfg(name: str) -> Config:
    with open(f"config/profiles/{name}.yaml") as f:
        data = {k: v for k, v in yaml.safe_load(f).items() if k != "profile"}
    data.setdefault("rss_url", "https://example.com/feed.xml")
    return Config(**data)


class TestGroundingStaysLocal:
    """ADR-147 B1: with summary=vllm, the GI grounding stages (quote/entailment) must ALSO be vllm.
    Left as 'openai', they build a separate cloud OpenAIProvider (api.openai.com + gpt-4o-mini) and
    the 'DGX-local' corpus gets its grounding from a cloud model — the corpus-corrupting bug."""

    @pytest.mark.parametrize("name", ["prod_dgx_full", "eval_default"])
    def test_dgx_profile_grounding_is_vllm_not_cloud(self, name: str):
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = _dgx_profile_cfg(name)
        # Every producing LLM stage is local — summary, naming, and grounding.
        assert cfg.summary_provider == "vllm"
        assert cfg.speaker_detector_provider == "vllm"
        assert cfg.quote_extraction_provider == "vllm"
        assert cfg.entailment_provider == "vllm"
        # Fully airgapped: no cloud provider in any fallback chain either.
        _CLOUD = {"openai", "gemini", "anthropic", "deepgram", "cohere", "grok", "mistral"}
        assert not (set(cfg.summary_fallback_providers or []) & _CLOUD)
        assert not (set(cfg.transcription_fallback_providers or []) & _CLOUD)
        assert not (set(cfg.diarization_fallback_providers or []) & _CLOUD)
        # The value gate stays ENABLED but fully local (airgapped): it rates with the same local
        # model as the extractor (ADR-147). No internet dependency.
        #
        # This asserted `is None` until 2026-08-29, using "nothing pinned" as a PROXY for "nothing
        # cloud" — which held only because the old resolver refused to name a rater for local
        # providers. Same-route resolution now records the local rater EXPLICITLY, which is what we
        # want (a self-grading run is visible instead of implied). So assert the property the test
        # is actually about: the rater is local, never cloud.
        assert cfg.gi_value_gate_enabled is True
        rater = getattr(cfg, "gi_value_gate_provider", None)
        assert rater in (
            None,
            "vllm",
            "ollama",
        ), f"{name}: airgapped profile would rate insights via {rater!r}, which is not local"
        assert rater not in _CLOUD
        # gi.deps routes the grounding stages by these provider keys; the 'vllm' key builds a
        # VLLMProvider (a DGX-local client), never the cloud OpenAIProvider — proven end-to-end in
        # TestFactoryDispatch (which builds it without needing a resolvable live endpoint).
        assert callable(create_summarization_provider)
