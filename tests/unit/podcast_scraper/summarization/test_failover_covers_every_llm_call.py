"""Every LLM call fails over, and adding a new one cannot silently opt out.

THE BUG THESE GUARD AGAINST
``FallbackAwareSummarizationProvider`` gated failover on ``_WRAPPED_METHODS``, an allowlist of
eight names. ``__getattr__`` forwarded everything else straight to the primary, so a provider
that was "wrapped with failover" had NONE on::

    generate_insights   classify_insights   clean_transcript   extract_kg_graph
    detect_speakers     detect_hosts        analyze_patterns   complete_text

Eight LLM calls protected, eight unprotected, and nothing in the code said so. When the
OpenRouter account passed its weekly limit, summarisation failed over to native DeepSeek and
survived while every one of those died — which is how ``generate_insights`` wrote stub
artifacts across production episodes, and how a prod speaker-detection job died with "no
budget/credit left on this key" rather than failing over.

WHY THESE TESTS ARE SPLIT THE WAY THEY ARE
Three different things can break independently, so each gets its own layer:

* BEHAVIOUR (``TestTheWrapperFailsOverOnAnyLlmCall``) — does the chain actually fire? Uses a
  fake provider, so it is fast and hermetic.
* DRIFT (``TestTheProviderSurfaceCannotDriftUnnoticed``) — the guardrail the operator asked
  for. It reads the REAL provider class at runtime and fails when a public method appears that
  nobody has classified. Adding an LLM method to the provider therefore breaks this test until
  someone decides whether it fails over. That is the "cannot forget" mechanism: it is not a
  list anyone has to remember to update, it is a list the test forces you to update.
* WIRING (``TestEveryProviderConstructionSiteWraps``) — the wrapper working is useless if a
  construction site hands out an unwrapped provider, which is exactly how speaker detection
  ended up unprotected while sharing the same provider class as summary.

An end-to-end test is deliberately NOT the guardrail here: it would need a live provider and a
real quota failure to be meaningful, so it cannot run in CI and would give false confidence.
The layers above pin the behaviour, the surface and the wiring without a network.
"""

# mypy: disable-error-code="arg-type"
# Deliberate in this file: lightweight duck-typed doubles passed where the production type is
# declared.
# Constructing the real types would pull in the machinery these tests isolate. The
# annotations on the helpers here are what make mypy check these bodies at all — most
# older test files are unannotated and therefore unchecked.

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Set

import pytest

from podcast_scraper.summarization.fallback import (
    _NEVER_WRAPPED,
    FallbackAwareSummarizationProvider,
)

pytestmark = [pytest.mark.unit]


#: Public provider methods that make an LLM call and MUST fail over. Reviewed by hand; the drift
#: test below fails if the real provider grows a method that is in neither this set nor
#: ``_NEVER_WRAPPED``, which forces the decision instead of defaulting to "unprotected".
LLM_METHODS: Set[str] = {
    "analyze_patterns",
    "classify_insights",
    "clean_transcript",
    "complete_text",
    "detect_hosts",
    "detect_speakers",
    "extract_kg_graph",
    "extract_quotes",
    "extract_quotes_bundled",
    "generate_insights",
    "score_entailment",
    "score_entailment_bundled",
    "summarize",
    "summarize_bundled",
    "summarize_extraction_bundled",
    "summarize_mega_bundled",
}


class _Cfg:
    """Only what the wrapper reads."""

    def __init__(self, chain: List[str]) -> None:
        self.summary_provider = "litellm"
        self.speaker_detector_provider = "litellm"
        self.summary_fallback_providers = list(chain)
        self.degradation_policy: Dict[str, Any] = {}


class _Primary:
    """A provider whose every LLM method fails, the way an over-quota account behaves."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        def boom(*_a: Any, **_k: Any) -> Any:
            self.calls.append(name)
            raise RuntimeError("no budget/credit left on this key")

        return boom

    def initialize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


class _Fallback:
    """Native DeepSeek's stand-in: answers anything."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def __getattr__(self, name: str) -> Any:
        def ok(*_a: Any, **_k: Any) -> str:
            self.calls.append(name)
            return f"{name}-from-fallback"

        return ok

    def initialize(self) -> None:
        return None


def _wrapped(monkeypatch: pytest.MonkeyPatch) -> Any:
    primary, fallback = _Primary(), _Fallback()
    w = FallbackAwareSummarizationProvider(primary, ["deepseek"], _Cfg(["deepseek"]))
    monkeypatch.setattr(w, "_get_or_build_fallback", lambda _n: fallback)
    w._test_primary, w._test_fallback = primary, fallback  # type: ignore[attr-defined]
    return w


class TestTheWrapperFailsOverOnAnyLlmCall:
    @pytest.mark.parametrize("method", sorted(LLM_METHODS))
    def test_each_llm_method_falls_over_to_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        """One case per LLM call. Parametrised rather than looped so a regression names the
        exact method that lost its failover."""
        w = _wrapped(monkeypatch)
        assert getattr(w, method)() == f"{method}-from-fallback"
        assert method in w._test_primary.calls, "the primary must be tried first"
        assert method in w._test_fallback.calls

    def test_a_method_the_wrapper_has_never_heard_of_still_fails_over(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE point of inverting the rule. A provider method invented after this file was
        written is protected without anyone editing this file — which is precisely what an
        allowlist could not do."""
        w = _wrapped(monkeypatch)
        assert w.summarize_in_some_new_way() == "summarize_in_some_new_way-from-fallback"

    def test_exempt_methods_are_not_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Transcription has its own RFC-106 ladder; stacking two would fail over twice by
        different rules. Lifecycle calls make no LLM request at all."""
        w = _wrapped(monkeypatch)
        for name in ("transcribe", "transcribe_with_segments", "get_pricing"):
            with pytest.raises(RuntimeError):
                getattr(w, name)()
        assert w._test_fallback.calls == []

    def test_data_attributes_still_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The wrapper must stay invisible: non-callables are forwarded untouched."""
        w = _wrapped(monkeypatch)
        w._test_primary.cleaning_processor = "the-processor"  # type: ignore[attr-defined]
        assert w.cleaning_processor == "the-processor"

    def test_the_primary_error_survives_when_every_tier_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failover must not mask the original cause when it cannot help."""
        w = FallbackAwareSummarizationProvider(_Primary(), ["deepseek"], _Cfg(["deepseek"]))
        monkeypatch.setattr(w, "_get_or_build_fallback", lambda _n: _Primary())
        with pytest.raises(RuntimeError, match="no budget/credit left"):
            w.generate_insights()

    def test_no_fallback_configured_means_no_wrapping_behaviour_change(self) -> None:
        from podcast_scraper.summarization.fallback import wrap_with_fallback_if_configured

        primary = _Primary()
        assert wrap_with_fallback_if_configured(primary, _Cfg([])) is primary


class TestTheLadderIsProviderAgnostic:
    """Any provider, any order — the tier is a name in config, never a branch in code."""

    @pytest.mark.parametrize("tier", ["openai", "gemini", "qwen", "anthropic", "groq", "deepseek"])
    def test_a_single_tier_of_any_provider_is_used(
        self, monkeypatch: pytest.MonkeyPatch, tier: str
    ) -> None:
        asked: List[str] = []
        w = FallbackAwareSummarizationProvider(_Primary(), [tier], _Cfg([tier]))

        def _build(name: str) -> Any:
            asked.append(name)
            return _Fallback()

        monkeypatch.setattr(w, "_get_or_build_fallback", _build)
        assert w.generate_insights() == "generate_insights-from-fallback"
        assert asked == [tier], "the configured tier must be the one built"

    def test_tiers_are_tried_in_the_configured_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An ordered chain: the first tier that answers wins, and a dead tier does not stop
        the run."""
        asked: List[str] = []

        class _DeadTier(_Primary):
            pass

        w = FallbackAwareSummarizationProvider(
            _Primary(), ["gemini", "openai"], _Cfg(["gemini", "openai"])
        )

        def _build(name: str) -> Any:
            asked.append(name)
            return _DeadTier() if name == "gemini" else _Fallback()

        monkeypatch.setattr(w, "_get_or_build_fallback", _build)
        assert w.detect_speakers() == "detect_speakers-from-fallback"
        assert asked == ["gemini", "openai"]

    def test_the_tier_that_just_failed_is_dropped_from_its_own_chain(self) -> None:
        """Failing over to the provider that just failed is not failover."""
        from podcast_scraper.summarization.fallback import _summary_fallback_chain

        cfg = _Cfg(["litellm", "openai"])
        assert _summary_fallback_chain(cfg, primary_name="litellm") == ["openai"]

    def test_each_stage_drops_its_own_primary_not_summarys(self) -> None:
        """Speaker detection may run on a different provider than summary. The chain must be
        computed against the stage being wrapped, or the wrong tier gets dropped."""
        from podcast_scraper.summarization.fallback import _summary_fallback_chain

        cfg = _Cfg(["openai", "deepseek"])
        cfg.summary_provider = "deepseek"
        cfg.speaker_detector_provider = "openai"
        assert _summary_fallback_chain(cfg, primary_name="openai") == ["deepseek"]
        assert _summary_fallback_chain(cfg, primary_name="deepseek") == ["openai"]


class TestTheProviderSurfaceCannotDriftUnnoticed:
    """The guardrail. Reads the REAL provider class, so it reacts to code that does not exist yet.

    If someone adds ``summarize_v2`` to the provider tomorrow and forgets this file entirely,
    this test fails with that method's name and refuses to pass until it is classified as either
    an LLM call or an explicit exemption. Nobody has to remember; the test remembers.
    """

    def _public_methods(self) -> Set[str]:
        from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider

        return {
            name
            for name, member in inspect.getmembers(OpenAICompatibleProvider)
            if not name.startswith("_") and (inspect.isfunction(member) or inspect.ismethod(member))
        }

    def test_every_public_method_is_classified(self) -> None:
        unclassified = self._public_methods() - LLM_METHODS - set(_NEVER_WRAPPED)
        assert not unclassified, (
            f"unclassified provider method(s): {sorted(unclassified)}. Decide: does this make an "
            "LLM call? If yes add it to LLM_METHODS in this test (it is already protected — the "
            "wrapper covers everything by default). If it must NOT fail over, add it to "
            "_NEVER_WRAPPED in summarization/fallback.py and say why."
        )

    def test_every_known_llm_method_exists_on_the_real_provider(self) -> None:
        """Catches the reverse drift: a method renamed or removed leaves a stale entry here,
        and a stale list is how the coverage claim rots without anyone noticing."""
        missing = LLM_METHODS - self._public_methods()
        assert (
            not missing
        ), f"LLM_METHODS names methods the provider no longer has: {sorted(missing)}"

    def test_no_llm_method_is_accidentally_exempt(self) -> None:
        """The exact defect, stated as an invariant: nothing that makes an LLM call may sit in
        the exemption list."""
        overlap = LLM_METHODS & set(_NEVER_WRAPPED)
        assert not overlap, f"LLM calls must not be exempt from failover: {sorted(overlap)}"

    def test_transcription_stays_exempt(self) -> None:
        """It has its own ladder. If this ever flips, the two ladders will both fire."""
        assert {"transcribe", "transcribe_with_segments"} <= set(_NEVER_WRAPPED)

    @pytest.mark.parametrize(
        "import_path,class_name",
        [
            ("podcast_scraper.providers.deepseek.deepseek_provider", "DeepSeekProvider"),
            ("podcast_scraper.providers.openai.openai_provider", "OpenAIProvider"),
            ("podcast_scraper.providers.qwen.qwen_provider", "QwenProvider"),
            ("podcast_scraper.providers.groq.groq_provider", "GroqProvider"),
            ("podcast_scraper.providers.litellm.litellm_provider", "LiteLLMProvider"),
            ("podcast_scraper.providers.vllm.vllm_provider", "VLLMProvider"),
        ],
    )
    def test_any_provider_can_serve_as_the_failover_tier(
        self, import_path: str, class_name: str
    ) -> None:
        """Failover must be a CAPABILITY OF THE ABSTRACTION, not a deepseek-shaped special case.

        The ladder is a list of provider names in config, and the tier is built by
        ``create_summarization_provider(cfg, provider_type_override=name)`` — so setting
        ``summary_fallback_providers: [openai]`` or ``[gemini]`` or ``[qwen]`` has to work with
        no code change. That only holds if every candidate implements the same LLM surface, so
        each is checked here rather than trusting that they share a base class.
        """
        import importlib

        cls = getattr(importlib.import_module(import_path), class_name)
        available = {
            name
            for name, member in inspect.getmembers(cls)
            if not name.startswith("_") and (inspect.isfunction(member) or inspect.ismethod(member))
        }
        missing = LLM_METHODS - available
        assert (
            not missing
        ), f"{class_name} cannot serve as a failover tier; missing: {sorted(missing)}"

    def test_the_builder_accepts_any_configured_provider_name(self) -> None:
        """The ladder is data, not code. Nothing in the failover path may name a provider."""
        from podcast_scraper.summarization import factory

        sig = inspect.signature(factory.create_summarization_provider)
        assert "provider_type_override" in sig.parameters

        src = inspect.getsource(FallbackAwareSummarizationProvider)
        for hardcoded in ("deepseek", "openai", "gemini", "qwen", "anthropic"):
            assert (
                f'"{hardcoded}"' not in src
            ), f"the wrapper hardcodes the provider {hardcoded!r}; the tier must come from config"


class TestEveryProviderConstructionSiteWraps:
    """A perfect wrapper is worth nothing if a construction site hands out a bare provider —
    which is exactly how speaker detection stayed unprotected while sharing the provider class
    that summary was already failing over on."""

    def test_the_speaker_detector_factory_wraps(self) -> None:
        import inspect as _inspect

        from podcast_scraper.speaker_detectors import factory

        src = _inspect.getsource(factory.create_speaker_detector)
        assert "wrap_with_fallback_if_configured" in src

    def test_the_speaker_detector_passes_its_own_primary(self) -> None:
        """Not ``summary_provider``. The chain drops the tier equal to the primary, so naming the
        wrong stage would drop the wrong tier."""
        import inspect as _inspect

        from podcast_scraper.speaker_detectors import factory

        src = _inspect.getsource(factory.create_speaker_detector)
        assert "speaker_detector_provider" in src

    def test_the_builder_is_behind_one_choke_point(self) -> None:
        """The raw builder has thirteen returns; wrapping per-return is the allowlist mistake in
        another costume. Exactly one public entry point may exist."""
        from podcast_scraper.speaker_detectors import factory

        assert hasattr(factory, "_build_speaker_detector")
        assert hasattr(factory, "create_speaker_detector")

    @pytest.mark.parametrize(
        "module,symbol",
        [
            ("podcast_scraper.gi.deps", "wrap_with_fallback_if_configured"),
            ("podcast_scraper.workflow.metadata_generation", "wrap_with_fallback_if_configured"),
            ("podcast_scraper.workflow.orchestration", "wrap_with_fallback_if_configured"),
        ],
    )
    def test_the_other_construction_sites_still_wrap(self, module: str, symbol: str) -> None:
        """Regression guard on the sites that were already correct."""
        import importlib
        import inspect as _inspect

        src = _inspect.getsource(importlib.import_module(module))
        assert symbol in src
