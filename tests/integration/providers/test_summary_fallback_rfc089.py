"""Integration test for RFC-089 #5 summarization provider-swap fallback.

Verifies the full path: profile → orchestration._create_summarization_provider →
FallbackAwareSummarizationProvider wrapper → fallback provider invoked when
primary fails at the summarize() boundary.

This is distinct from the older TestSummarizationProviderFallback in
test_fallback_behavior.py — that one is about graceful failure at provider
CREATION time. This one is about runtime swap when the primary's HTTP call
to e.g. DGX Ollama fails.

Local-only (no network); the cloud fallback provider is stubbed at the lazy
factory boundary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.summarization.fallback import (
    FallbackAwareSummarizationProvider,
)
from podcast_scraper.workflow import orchestration
from podcast_scraper.workflow.metrics import Metrics


class _PrimaryThatFails:
    """Stand-in for an Ollama provider whose HTTP call to DGX times out."""

    def __init__(self) -> None:
        self.initialize = Mock()
        self.cleanup = Mock()
        self.warmup = Mock()
        self.summarize_calls = 0

    def summarize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.summarize_calls += 1
        raise ConnectionError("DGX Ollama unreachable on the tailnet")


class _FallbackOK:
    """Stand-in for a cloud provider that successfully serves the request."""

    def __init__(self) -> None:
        self.initialize = Mock()
        self.cleanup = Mock()
        self.summarize_calls = 0

    def summarize(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.summarize_calls += 1
        return {"summary": "served by fallback", "summary_short": "fallback"}


@pytest.mark.integration
class TestRfc089SummaryFallbackWiring:
    """Validates the orchestration → wrapper → metric wiring end-to-end."""

    def _make_cfg(self, fallback_provider: Optional[str] = "gemini") -> config.Config:
        policy: Optional[Dict[str, Any]] = (
            {"fallback_provider_on_failure": fallback_provider} if fallback_provider else None
        )
        return config.Config(
            rss="https://example.com/feed.xml",
            summary_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
            dry_run=False,
            degradation_policy=policy,
        )

    def test_orchestration_returns_wrapper_when_policy_set(self) -> None:
        cfg = self._make_cfg(fallback_provider="gemini")
        primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            result = orchestration._create_summarization_provider(cfg)
        assert isinstance(result, FallbackAwareSummarizationProvider)

    def test_orchestration_returns_primary_when_policy_absent(self) -> None:
        cfg = self._make_cfg(fallback_provider=None)
        primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            result = orchestration._create_summarization_provider(cfg)
        assert result is primary

    def test_dgx_unreachable_falls_back_and_records_metric(self) -> None:
        cfg = self._make_cfg(fallback_provider="gemini")
        primary = _PrimaryThatFails()
        fallback = _FallbackOK()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            wrapper = orchestration._create_summarization_provider(cfg)

        assert wrapper is not None

        # Lazy factory inside the wrapper should produce our fake fallback.
        with patch(
            "podcast_scraper.summarization.factory.create_summarization_provider",
            return_value=fallback,
        ):
            metrics = Metrics()
            result = wrapper.summarize(
                text="t" * 100,
                episode_title="title",
                pipeline_metrics=metrics,
            )

        assert result["summary"] == "served by fallback"
        assert primary.summarize_calls == 1
        assert fallback.summarize_calls == 1
        assert metrics.llm_summary_fallback_active_count == 1
        assert metrics.llm_summary_fallback_provider == "gemini"

    def test_no_fallback_when_policy_absent_primary_exception_bubbles(self) -> None:
        cfg = self._make_cfg(fallback_provider=None)
        primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            wrapper = orchestration._create_summarization_provider(cfg)

        assert wrapper is not None

        with pytest.raises(ConnectionError, match="DGX Ollama unreachable"):
            wrapper.summarize(text="t" * 100, pipeline_metrics=Metrics())

    def test_local_dgx_balanced_profile_emits_primary_no_wrap(self) -> None:
        """local_dgx_balanced is a laptop-driven dev profile — no cloud fallback.

        Operator decision (2026-06-07): for laptop-to-DGX runs, a DGX outage
        should be visible (summary missing) rather than silently routing to a
        paid cloud provider. Cloud fallback is reserved for the prod profile
        (cloud_with_dgx_whisper_primary), where it's enabled at the Whisper
        layer (transcription_fallback_provider) — the LLM is already cloud
        Gemini there.

        Regression guard: if someone re-adds
        degradation_policy.fallback_provider_on_failure to local_dgx_balanced,
        this fails loudly so the operator gets pinged before quietly routing
        DGX outages to cloud.
        """
        import os

        os.environ.setdefault("GEMINI_API_KEY", "x")
        os.environ.setdefault("OPENAI_API_KEY", "x")
        from podcast_scraper.cli import _build_config, parse_args

        args = parse_args(
            [
                "--profile",
                "local_dgx_balanced",
                "https://example.com/feed.xml",
                "--output-dir",
                "/tmp/_t",
            ]
        )
        cfg = _build_config(args)
        primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            result = orchestration._create_summarization_provider(cfg)
        assert result is primary
        assert not isinstance(result, FallbackAwareSummarizationProvider)

    def test_local_dgx_full_profile_emits_primary_no_wrap(self) -> None:
        """local_dgx_full is the 'pure measurement' profile — no cloud fallback.

        Regression guard: if someone accidentally adds a fallback line to
        local_dgx_full, this fails. That profile must abort cleanly on DGX
        failure, not silently route to cloud.
        """
        import os

        os.environ.setdefault("GEMINI_API_KEY", "x")
        os.environ.setdefault("OPENAI_API_KEY", "x")
        from podcast_scraper.cli import _build_config, parse_args

        args = parse_args(
            [
                "--profile",
                "local_dgx_full",
                "https://example.com/feed.xml",
                "--output-dir",
                "/tmp/_t",
            ]
        )
        cfg = _build_config(args)
        primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.workflow.orchestration._get_factory_function",
            return_value=Mock(return_value=primary),
        ):
            result = orchestration._create_summarization_provider(cfg)
        assert result is primary
        assert not isinstance(result, FallbackAwareSummarizationProvider)


@pytest.mark.integration
class TestRfc089GiEvidenceFallbackWiring:
    """RFC-089 #5 also applies to GI evidence quote / entailment providers.

    Three cases:

    1. Same backend as summary → shared wrapped instance (the common
       local_dgx_balanced setup). Verified by checking gi.deps returns the
       wrapped object that orchestration already produced.
    2. Different backend for quote / entailment → freshly built by gi.deps;
       must get wrapped there too.
    3. local_dgx_full → no fallback configured; providers must NOT be wrapped.
    """

    def _make_cfg_balanced(self) -> config.Config:
        return config.Config(
            rss="https://example.com/feed.xml",
            summary_provider="ollama",
            quote_extraction_provider="ollama",
            entailment_provider="ollama",
            generate_summaries=True,
            generate_metadata=True,
            generate_gi=True,
            gi_insight_source="provider",
            degradation_policy={"fallback_provider_on_failure": "gemini"},
        )

    def _make_cfg_mixed_backend(self) -> config.Config:
        """summary on ollama, quote on deepseek, entailment on gemini —
        forces the separate-build branches in create_gil_evidence_providers."""
        import os

        os.environ.setdefault("GEMINI_API_KEY", "x")
        os.environ.setdefault("DEEPSEEK_API_KEY", "x")
        return config.Config(
            rss="https://example.com/feed.xml",
            summary_provider="ollama",
            quote_extraction_provider="deepseek",
            entailment_provider="gemini",
            generate_summaries=True,
            generate_metadata=True,
            generate_gi=True,
            gi_insight_source="provider",
            degradation_policy={"fallback_provider_on_failure": "gemini"},
        )

    def test_same_backend_reuses_wrapped_summary_provider(self) -> None:
        from podcast_scraper.gi.deps import create_gil_evidence_providers

        cfg = self._make_cfg_balanced()
        # Build a wrapped summary instance the way orchestration would.
        wrapped_summary = FallbackAwareSummarizationProvider(_PrimaryThatFails(), "gemini", cfg)
        quote, entail = create_gil_evidence_providers(cfg, summary_provider=wrapped_summary)
        # Both should be the same wrapped object.
        assert quote is wrapped_summary
        assert entail is wrapped_summary
        assert isinstance(quote, FallbackAwareSummarizationProvider)

    def test_separate_backend_quote_provider_is_wrapped(self) -> None:
        from podcast_scraper.gi.deps import create_gil_evidence_providers

        cfg = self._make_cfg_mixed_backend()
        # Stub the underlying factory so we don't build a real cloud provider.
        # Each fresh-built quote/entail provider should come out wrapped.
        stub_primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.summarization.factory.create_summarization_provider",
            return_value=stub_primary,
        ):
            quote, entail = create_gil_evidence_providers(cfg, summary_provider=Mock())
        assert isinstance(quote, FallbackAwareSummarizationProvider)
        assert isinstance(entail, FallbackAwareSummarizationProvider)

    def test_local_dgx_full_does_not_wrap_gi_providers(self) -> None:
        """local_dgx_full has no degradation_policy.fallback_provider_on_failure
        → no wrapping anywhere, including GI evidence factory."""
        import os

        os.environ.setdefault("GEMINI_API_KEY", "x")
        os.environ.setdefault("DEEPSEEK_API_KEY", "x")
        from podcast_scraper.gi.deps import create_gil_evidence_providers

        cfg = config.Config(
            rss="https://example.com/feed.xml",
            summary_provider="ollama",
            quote_extraction_provider="deepseek",
            entailment_provider="gemini",
            generate_summaries=True,
            generate_metadata=True,
            generate_gi=True,
            gi_insight_source="provider",
            degradation_policy=None,
        )
        stub_primary = _PrimaryThatFails()
        with patch(
            "podcast_scraper.summarization.factory.create_summarization_provider",
            return_value=stub_primary,
        ):
            quote, entail = create_gil_evidence_providers(cfg, summary_provider=Mock())
        assert not isinstance(quote, FallbackAwareSummarizationProvider)
        assert not isinstance(entail, FallbackAwareSummarizationProvider)
        assert quote is stub_primary
        assert entail is stub_primary
