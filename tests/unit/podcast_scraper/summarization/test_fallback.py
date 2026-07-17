"""Unit tests for FallbackAwareSummarizationProvider (RFC-089 #5).

Validates the provider-swap fallback contract:
- Primary OK → no fallback, no metric, fallback is never instantiated.
- Primary raises → fallback called with same args, result returned, metric recorded once.
- Fallback also raises → primary exception bubbles (existing degradation path applies).
- Metric counter records once per run, not once per call.
- Pass-through of non-protocol attributes (cleaning_processor) goes to primary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.summarization.fallback import (
    FallbackAwareSummarizationProvider,
    wrap_with_fallback_if_configured,
)


class _FakeMetrics:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record_llm_summary_fallback_active(self, name: str) -> None:
        self.calls.append(name)


class _FakeProvider:
    def __init__(
        self,
        name: str,
        raises: Optional[Exception] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.raises = raises
        self.result = result or {"summary": f"from-{name}"}
        self.initialize_count = 0
        self.cleanup_count = 0
        self.warmup_count = 0
        self.summarize_calls: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []
        self.summarize_bundled_calls: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []
        self.extract_quotes_calls: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []
        self.score_entailment_calls: list[tuple[tuple[Any, ...], Dict[str, Any]]] = []
        self.cleaning_processor = f"cleaning-{name}"

    def initialize(self) -> None:
        self.initialize_count += 1

    def cleanup(self) -> None:
        self.cleanup_count += 1

    def warmup(self, timeout_s: int = 600) -> None:
        self.warmup_count += 1

    def summarize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.summarize_calls.append((args, kwargs))
        if self.raises:
            raise self.raises
        return self.result

    def summarize_bundled(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.summarize_bundled_calls.append((args, kwargs))
        if self.raises:
            raise self.raises
        return self.result

    def extract_quotes(self, *args: Any, **kwargs: Any) -> list[Dict[str, Any]]:
        self.extract_quotes_calls.append((args, kwargs))
        if self.raises:
            raise self.raises
        return [{"text": f"quote-from-{self.name}", "start": 0, "end": 10}]

    def score_entailment(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.score_entailment_calls.append((args, kwargs))
        if self.raises:
            raise self.raises
        return {"score": 0.95, "provider": self.name}


@pytest.fixture
def fake_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.summary_provider = "ollama"
    cfg.degradation_policy = {"fallback_provider_on_failure": "gemini"}
    return cfg


@pytest.fixture
def primary_ok() -> _FakeProvider:
    return _FakeProvider("primary")


@pytest.fixture
def primary_broken() -> _FakeProvider:
    return _FakeProvider("primary", raises=RuntimeError("DGX unreachable"))


@pytest.fixture
def fallback_ok() -> _FakeProvider:
    return _FakeProvider("fallback")


@pytest.fixture
def fallback_broken() -> _FakeProvider:
    return _FakeProvider("fallback", raises=ValueError("gemini quota"))


def _patch_factory(monkeypatch: pytest.MonkeyPatch, fallback: _FakeProvider) -> None:
    """Stub the lazy factory call so no real provider gets built."""

    def fake_create(cfg: Any, provider_type_override: Optional[str] = None) -> Any:
        return fallback

    monkeypatch.setattr(
        "podcast_scraper.summarization.factory.create_summarization_provider",
        fake_create,
    )


class TestPrimaryOK:
    """Happy path: primary works, fallback never gets touched."""

    def test_summarize_returns_primary_result(
        self,
        fake_cfg: MagicMock,
        primary_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch factory in case anything accidentally touches it.
        _patch_factory(monkeypatch, _FakeProvider("never-called"))
        wrapped = FallbackAwareSummarizationProvider(primary_ok, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        result = wrapped.summarize("text here", pipeline_metrics=metrics)
        assert result == {"summary": "from-primary"}
        assert primary_ok.summarize_calls
        assert metrics.calls == []  # no fallback metric

    def test_fallback_never_built_on_happy_path(
        self,
        fake_cfg: MagicMock,
        primary_ok: _FakeProvider,
    ) -> None:
        wrapped = FallbackAwareSummarizationProvider(primary_ok, "gemini", fake_cfg)
        with patch(
            "podcast_scraper.summarization.factory.create_summarization_provider"
        ) as factory:
            wrapped.summarize("text")
            factory.assert_not_called()


class TestPrimaryRaisesFallbackSucceeds:
    """Primary throws → fallback called with same args → metric recorded once."""

    def test_summarize_falls_back(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        result = wrapped.summarize("text", episode_title="t", pipeline_metrics=metrics)
        assert result == {"summary": "from-fallback"}
        assert primary_broken.summarize_calls  # tried
        assert fallback_ok.summarize_calls  # then fell back
        assert fallback_ok.summarize_calls[0][0] == ("text",)
        assert metrics.calls == ["gemini"]

    def test_metric_records_only_once_across_multiple_calls(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        for _ in range(3):
            wrapped.summarize("text", pipeline_metrics=metrics)
        assert len(fallback_ok.summarize_calls) == 3  # each call did fall back
        assert metrics.calls == ["gemini"]  # but only first one recorded

    def test_fallback_built_lazily_then_reused(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        call_count = {"n": 0}

        def fake_create(cfg: Any, provider_type_override: Optional[str] = None) -> Any:
            call_count["n"] += 1
            return fallback_ok

        monkeypatch.setattr(
            "podcast_scraper.summarization.factory.create_summarization_provider",
            fake_create,
        )
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        wrapped.summarize("a")
        wrapped.summarize("b")
        wrapped.summarize("c")
        assert call_count["n"] == 1  # built once, reused twice
        assert fallback_ok.initialize_count == 1

    def test_summarize_bundled_also_falls_back(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        result = wrapped.summarize_bundled("text", pipeline_metrics=metrics)
        assert result == {"summary": "from-fallback"}
        assert fallback_ok.summarize_bundled_calls
        assert metrics.calls == ["gemini"]


class TestGiEvidenceMethodsFallBack:
    """RFC-089 #5 extension: extract_quotes / score_entailment must fall back too.

    Without this coverage, GI evidence would silently lose its grounding
    when DGX is down — the same wrapped instance serves the summary / quote /
    entailment roles, and a bare ``__getattr__`` would forward these methods
    to the broken primary.
    """

    def test_extract_quotes_falls_back(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        result = wrapped.extract_quotes("transcript", "insight", pipeline_metrics=metrics)
        assert result == [{"text": "quote-from-fallback", "start": 0, "end": 10}]
        assert primary_broken.extract_quotes_calls  # tried
        assert fallback_ok.extract_quotes_calls  # fell back
        assert metrics.calls == ["gemini"]

    def test_score_entailment_falls_back(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        result = wrapped.score_entailment("premise", "hypothesis", pipeline_metrics=metrics)
        assert result == {"score": 0.95, "provider": "fallback"}
        assert primary_broken.score_entailment_calls
        assert fallback_ok.score_entailment_calls
        assert metrics.calls == ["gemini"]

    def test_mixed_methods_share_one_metric_record(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """summarize + extract_quotes + score_entailment in one run still
        records the fallback metric only once."""
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        wrapped.summarize("text", pipeline_metrics=metrics)
        wrapped.extract_quotes("t", "i", pipeline_metrics=metrics)
        wrapped.score_entailment("p", "h", pipeline_metrics=metrics)
        assert metrics.calls == ["gemini"]  # one record across three methods
        # But each method did actually run through the fallback:
        assert fallback_ok.summarize_calls
        assert fallback_ok.extract_quotes_calls
        assert fallback_ok.score_entailment_calls


class TestBothFail:
    """When both providers fail, primary's exception bubbles (existing degradation runs)."""

    def test_primary_exception_re_raised(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_broken: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_broken)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        metrics = _FakeMetrics()
        with pytest.raises(RuntimeError, match="DGX unreachable"):
            wrapped.summarize("text", pipeline_metrics=metrics)
        assert primary_broken.summarize_calls
        assert fallback_broken.summarize_calls  # tried fallback
        assert metrics.calls == []  # but didn't record success


class TestPassthrough:
    """Non-protocol attributes forward to primary; lifecycle hooks chain correctly."""

    def test_getattr_passthrough(self, fake_cfg: MagicMock, primary_ok: _FakeProvider) -> None:
        wrapped = FallbackAwareSummarizationProvider(primary_ok, "gemini", fake_cfg)
        assert wrapped.cleaning_processor == "cleaning-primary"

    def test_initialize_only_invokes_primary(
        self, fake_cfg: MagicMock, primary_ok: _FakeProvider, fallback_ok: _FakeProvider
    ) -> None:
        wrapped = FallbackAwareSummarizationProvider(primary_ok, "gemini", fake_cfg)
        wrapped.initialize()
        assert primary_ok.initialize_count == 1
        assert fallback_ok.initialize_count == 0  # never touched

    def test_warmup_only_invokes_primary(
        self, fake_cfg: MagicMock, primary_ok: _FakeProvider
    ) -> None:
        wrapped = FallbackAwareSummarizationProvider(primary_ok, "gemini", fake_cfg)
        wrapped.warmup(timeout_s=30)
        assert primary_ok.warmup_count == 1

    def test_cleanup_runs_primary_then_fallback_if_built(
        self,
        fake_cfg: MagicMock,
        primary_broken: _FakeProvider,
        fallback_ok: _FakeProvider,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_factory(monkeypatch, fallback_ok)
        wrapped = FallbackAwareSummarizationProvider(primary_broken, "gemini", fake_cfg)
        # Trigger fallback build
        wrapped.summarize("x", pipeline_metrics=_FakeMetrics())
        wrapped.cleanup()
        assert primary_broken.cleanup_count == 1
        assert fallback_ok.cleanup_count == 1


class TestWrapWithFallbackIfConfigured:
    """The factory helper returns the primary unchanged when no fallback is configured."""

    def test_no_policy_returns_primary_unchanged(self, primary_ok: _FakeProvider) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "ollama"
        cfg.degradation_policy = None
        assert wrap_with_fallback_if_configured(primary_ok, cfg) is primary_ok

    def test_policy_without_fallback_field_returns_primary(self, primary_ok: _FakeProvider) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "ollama"
        cfg.degradation_policy = {"continue_on_stage_failure": True}
        assert wrap_with_fallback_if_configured(primary_ok, cfg) is primary_ok

    def test_policy_with_fallback_returns_wrapped(self, primary_ok: _FakeProvider) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "ollama"
        cfg.degradation_policy = {"fallback_provider_on_failure": "gemini"}
        wrapped = wrap_with_fallback_if_configured(primary_ok, cfg)
        assert isinstance(wrapped, FallbackAwareSummarizationProvider)

    def test_fallback_matches_primary_is_no_op(self, primary_ok: _FakeProvider) -> None:
        # No point falling back to the same provider.
        cfg = MagicMock()
        cfg.summary_provider = "ollama"
        cfg.summary_fallback_providers = []
        cfg.degradation_policy = {"fallback_provider_on_failure": "ollama"}
        assert wrap_with_fallback_if_configured(primary_ok, cfg) is primary_ok


class TestRegistryChainSourcing:
    """RFC-105 (#1198): the registry-emitted ``summary_fallback_providers`` is the source of truth
    for the LLM failover ladder; the legacy ``degradation_policy`` is back-compat only."""

    def test_registry_chain_wraps(self, primary_ok: _FakeProvider) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "openai"  # vLLM served over the openai protocol
        cfg.summary_fallback_providers = ["gemini"]
        cfg.degradation_policy = None
        wrapped = wrap_with_fallback_if_configured(primary_ok, cfg)
        assert isinstance(wrapped, FallbackAwareSummarizationProvider)
        assert wrapped._fallback_names == ["gemini"]

    def test_registry_chain_takes_precedence_over_legacy_policy(
        self, primary_ok: _FakeProvider
    ) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "ollama"
        cfg.summary_fallback_providers = ["gemini"]
        cfg.degradation_policy = {"fallback_provider_on_failure": "openai"}  # ignored
        wrapped = wrap_with_fallback_if_configured(primary_ok, cfg)
        assert isinstance(wrapped, FallbackAwareSummarizationProvider)
        assert wrapped._fallback_names == ["gemini"]

    def test_chain_drops_the_tier_equal_to_primary(self, primary_ok: _FakeProvider) -> None:
        cfg = MagicMock()
        cfg.summary_provider = "gemini"
        cfg.summary_fallback_providers = ["gemini"]  # same as primary -> dropped -> no wrap
        cfg.degradation_policy = None
        assert wrap_with_fallback_if_configured(primary_ok, cfg) is primary_ok

    def test_ordered_chain_tries_second_tier_when_first_fails(
        self, primary_broken: _FakeProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tier1 = _FakeProvider("gemini", raises=ValueError("gemini quota"))
        tier2 = _FakeProvider("openai")
        built: dict[str, _FakeProvider] = {"gemini": tier1, "openai": tier2}

        def fake_create(cfg: Any, provider_type_override: Optional[str] = None) -> Any:
            return built[str(provider_type_override)]

        monkeypatch.setattr(
            "podcast_scraper.summarization.factory.create_summarization_provider", fake_create
        )
        wrapped = FallbackAwareSummarizationProvider(
            primary_broken, ["gemini", "openai"], MagicMock()
        )
        metrics = _FakeMetrics()
        result = wrapped.summarize("text", pipeline_metrics=metrics)
        assert result == {"summary": "from-openai"}  # first tier failed, second served
        assert metrics.calls == ["openai"]  # attribution names the tier that actually won
