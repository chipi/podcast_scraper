"""GemmaTranslateProvider: the prompt it sends, the model it refuses, per-unit failure, telemetry.

NOTHING HERE TOUCHES THE NETWORK — the OpenAI SDK client on the provider is replaced with a
fake. A unit test that could reach `dgx-llm-1:8005` would be the #1527 bug class: a suite that
silently depends on a GPU box being up, and that translates real text on someone's laptop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper import config
from podcast_scraper.providers.vllm.translate_provider import (
    GemmaTranslateProvider,
    MODEL_INPUT_TOKEN_LIMIT,
    PROMPT_NAME,
    TranslateServedModelMismatch,
    TranslationUnavailable,
)
from podcast_scraper.translation import (
    create_translation_provider,
    is_translation_configured,
    TranslationProvider,
    TranslationProviderUnavailable,
)

pytestmark = pytest.mark.unit

MODEL = "google/translategemma-12b-it"


def _cfg(
    *,
    translate_api_base: str = "http://translator.invalid:8005/v1",
    translate_model: str = MODEL,
    translate_verify_served_model: bool = False,
) -> config.Config:
    """Explicit keywords rather than `**dict`, so the typed Config signature is actually checked."""
    return config.Config(
        rss="https://example.com/feed.xml",
        translate_api_base=translate_api_base,
        translate_model=translate_model,
        translate_verify_served_model=translate_verify_served_model,
    )


class _Choice:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, p: int = 50, c: int = 20) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _Resp:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.choices = [_Choice(text, finish_reason)] if text is not None else []
        self.usage = _Usage()


class _FakeCompletions:
    def __init__(self, resp: Any = None, raises: Optional[Exception] = None) -> None:
        self._resp = resp
        self._raises = raises
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._resp


class _FakeModels:
    def __init__(self, ids: List[str]) -> None:
        self._ids = ids

    def list(self) -> Any:
        class _D:
            def __init__(self, ids: List[str]) -> None:
                self.data = [type("M", (), {"id": i})() for i in ids]

        return _D(self._ids)


class _FakeClient:
    """Stands in for the whole OpenAI client.

    Replacing the client wholesale rather than assigning to `client.completions`: those are
    read-only properties on the real SDK object, so poking them type-checks only under an
    ignore and would hide a genuine signature change behind it.
    """

    def __init__(self, completions: Any, models: Any, base_url: str) -> None:
        self.completions = completions
        self.models = models
        self.base_url = base_url


def _provider(
    text: str = "Hello world.",
    finish_reason: str = "stop",
    raises: Optional[Exception] = None,
    served: Optional[List[str]] = None,
    verify: bool = False,
    models: Any = None,
) -> GemmaTranslateProvider:
    cfg = _cfg(translate_verify_served_model=verify)
    p = GemmaTranslateProvider(cfg)
    fake = _FakeCompletions(_Resp(text, finish_reason), raises)
    p.client = _FakeClient(  # type: ignore[assignment]
        fake,
        models if models is not None else _FakeModels(served if served is not None else [MODEL]),
        str(cfg.translate_api_base),
    )
    return p


class TestItIsAProperProvider:
    def test_it_satisfies_the_translation_protocol(self) -> None:
        assert isinstance(_provider(), TranslationProvider)

    def test_it_inherits_the_shared_openai_compatible_transport(self) -> None:
        """Not a bespoke HTTP client. Retries, the temperature/context self-healing and the
        per-request timeout bound (#1852/#1894) come from the base rather than being
        re-implemented — which is the whole reason this is a provider and not a script."""
        from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider

        assert issubclass(GemmaTranslateProvider, OpenAICompatibleProvider)
        assert _provider()._chat_request_timeout is not None

    def test_it_has_its_own_config_namespace_and_telemetry_identity(self) -> None:
        """Distinct identity so cost is attributed to the translator, not to `vllm` — the two are
        different models on different ports doing different operations."""
        assert GemmaTranslateProvider._CONFIG_NS == "translate"
        assert GemmaTranslateProvider._TELEMETRY_PROVIDER == "gemma_translate"

    def test_the_client_points_at_the_translate_endpoint_not_the_summary_one(self) -> None:
        p = _provider()
        assert "8005" in str(p.client.base_url)


class TestThePromptComesFromTheStore:
    def test_the_rendered_prompt_is_byte_identical_to_the_models_template(self) -> None:
        """This string IS the interface: the chat route cannot deliver TranslateGemma's
        structured content (vLLM strips the custom keys and returns 400), so the prompt is
        rendered here. A stray space is a silent quality change with no other symptom."""
        got = _provider().build_prompt("Hola mundo.", source_language="es")
        assert got == (
            "<start_of_turn>user\n"
            "You are a professional Spanish (es) to English (en) translator. Your goal is to "
            "accurately convey the meaning and nuances of the original Spanish text while "
            "adhering to English grammar, vocabulary, and cultural sensitivities.\n"
            "Produce only the English translation, without any additional explanations or "
            "commentary. Please translate the following Spanish text into English:\n\n\n"
            "Hola mundo.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

    def test_the_anti_commentary_instruction_is_present(self) -> None:
        """The one clause whose absence has no other symptom, asserted by name.

        Measured on the real service with it MISSING: a 22-word unit returned 197 tokens
        beginning "Here's a translation that aims for accuracy and nuance:", and "Mmm." returned
        "Okay, I understand. I will do my best to provide accurate and nuanced English
        translations of any Spanish text you provide." With it restored: 19 tokens of clean
        translation, and "Hmm.". The byte-identity test above would also fail on a harmless
        reflow; this one fails only on the thing that changes behaviour.
        """
        prompt = _provider().build_prompt("Hola.", source_language="es")
        assert "without any additional explanations or commentary" in prompt

    def test_it_ends_with_the_generation_cue(self) -> None:
        """`render_prompt` strips trailing whitespace — right for its other callers, whose prompt
        is a chat message. For a RAW completions prompt the newline after `<start_of_turn>model`
        is the cue the model was trained to continue from, so the provider re-adds it."""
        assert (
            _provider()
            .build_prompt("Hola.", source_language="es")
            .endswith("<start_of_turn>model\n")
        )

    def test_the_prompts_name_and_sha256_ride_along_in_the_metadata(self) -> None:
        """What ties an English artifact to the prompt that produced it (S2.11). A prompt living
        inline in a client could not be hashed, which is why it lives in the store."""
        meta = _provider().translate("Hola.", source_language="es")["metadata"]
        assert meta["prompt"]["name"] == PROMPT_NAME
        assert len(meta["prompt"]["sha256"]) == 64
        assert meta["model"] == MODEL
        assert meta["provider"] == "gemma_translate"

    def test_an_undeclared_language_refuses_rather_than_guessing_a_name(self) -> None:
        """A guessed language name is followed confidently and produces fluent wrong output that
        no artifact reveals. So a language absent from config/languages.yaml cannot be sent."""
        with pytest.raises(ValueError, match="not declared in config/languages.yaml"):
            _provider().build_prompt("Kaixo.", source_language="eu")

    def test_translating_a_language_into_itself_is_refused(self) -> None:
        with pytest.raises(ValueError, match="into itself"):
            _provider().build_prompt("Hello.", source_language="en", target_language="en")


class TestTheServedModelCheck:
    def test_a_different_model_on_the_slot_raises(self) -> None:
        """ADR-143/144: a corpus attributed to the wrong translation model cannot be told from a
        correct one afterwards, so this fails the run rather than producing one."""
        p = _provider(served=["some/other-model"], verify=True)
        with pytest.raises(TranslateServedModelMismatch, match="not 'google/translategemma"):
            p.translate("Hola.", source_language="es")

    def test_an_unreachable_endpoint_only_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Unreachable is not a mismatch. The real call surfaces connectivity anyway, and hard
        failing here would make importing the module offline impossible."""

        class _Boom:
            def list(self) -> Any:
                raise OSError("no route to host")

        # Built with a failing models endpoint from the start, rather than assigning onto the
        # client afterwards — `models` is a read-only property on the real SDK object.
        p = _provider(verify=True, models=_Boom())
        got = p.translate("Hola.", source_language="es")
        assert got["text"] == "Hello world."
        assert any("could not verify the served model" in r.message for r in caplog.records)


class TestTheOperation:
    def test_a_translation_comes_back_stripped_with_usage(self) -> None:
        got = _provider(text="  Hello world.  ").translate("Hola mundo.", source_language="es")
        assert got["text"] == "Hello world."
        assert got["metadata"]["prompt_tokens"] == 50
        assert got["metadata"]["completion_tokens"] == 20
        assert got["metadata"]["finish_reason"] == "stop"

    def test_the_request_is_deterministic_in_intent_and_stops_at_the_turn_end(self) -> None:
        p = _provider()
        p.translate("Hola.", source_language="es")
        sent = p.client.completions.calls[0]  # type: ignore[attr-defined]
        assert sent["temperature"] == 0.0
        assert sent["stop"] == ["<end_of_turn>"]
        assert sent["model"] == MODEL
        assert sent["timeout"] is not None, "the base's per-request bound must be applied"

    def test_a_transport_failure_is_a_RESULT_not_an_exception(self) -> None:
        """The caller counts failures across an episode and applies RFC-124 §5.3's completeness
        gate. A provider that raised would be deciding the episode's fate from inside one call,
        unable to see the other 229."""
        got = _provider(raises=OSError("connection reset")).translate("Hola.", source_language="es")
        assert got["text"] is None
        assert "OSError" in got["metadata"]["error"]

    def test_an_empty_completion_is_a_failure_not_a_translation(self) -> None:
        got = _provider(text="   ").translate("Hola.", source_language="es")
        assert got["text"] is None
        assert "no completion text" in got["metadata"]["error"]

    def test_a_truncated_completion_is_a_failure_not_a_short_translation(self) -> None:
        """Non-empty, so every check built on "did we get text back" passes it. A silently short
        translation is worse than a missing one: the completeness gate never fires."""
        got = _provider(text="Hello wor", finish_reason="length").translate(
            "Hola mundo largo.", source_language="es"
        )
        assert got["text"] is None
        assert "truncated" in got["metadata"]["error"]

    def test_empty_input_needs_no_call_at_all(self) -> None:
        p = _provider()
        got = p.translate("   ", source_language="es")
        assert got["text"] == ""
        assert p.client.completions.calls == []  # type: ignore[attr-defined]

    def test_an_unconfigured_provider_raises_rather_than_passing_text_through(self) -> None:
        """Returning the source unchanged would put Spanish into `.en.txt`, which every later
        stage then reads as English."""
        p = GemmaTranslateProvider(config.Config(rss="https://e.com/f.xml"))
        with pytest.raises(TranslationUnavailable):
            p.translate("Hola.", source_language="es")

    def test_an_oversized_unit_is_REFUSED_rather_than_mistranslated(self) -> None:
        """The worst failure shape there is, caught before the request.

        The served container's window is larger than the model's documented 2K input context, so
        the server ACCEPTS an oversized unit and the model answers with a translation of its
        first sentence and `finish_reason: stop`. Measured on the real service: 4,800 prompt
        tokens in, 32 out — non-empty, "successful", 99% of the content gone. Nothing downstream
        could tell. So the provider refuses it here, where it becomes an ordinary failed unit.
        """
        p = _provider()
        huge = "Hola mundo. " * 3000
        got = p.translate(huge, source_language="es")
        assert got["text"] is None
        assert "input context" in got["metadata"]["error"]
        assert p.client.completions.calls == [], "nothing should have been sent"

    def test_the_fallback_estimate_is_pessimistic(self) -> None:
        """With no tokenizer reachable, the estimate must OVER-count so the guard errs toward
        refusing. Every context-overflow bug in this repo came from an optimistic constant."""
        p = _provider()
        text = "a" * 2200
        tokens, how = p.estimate_prompt_tokens(text)
        assert how == "estimate", "the fake client has no /tokenize"
        # 2.2 chars/token is the FEWEST measured over 141 real Spanish units, so the estimate is
        # at least as large as the real count.
        assert tokens >= len(text) / 4.06, "must not use the median ratio"
        assert tokens == int(len(text) / 2.2) + 1

    def test_a_normal_unit_passes_the_budget_check(self) -> None:
        """The guard must not refuse ordinary work — the real transcript's longest turn was 41
        words."""
        p = _provider()
        got = p.translate("Hola mundo, esto es una frase de longitud normal.", source_language="es")
        assert got["text"] == "Hello world."
        assert got["metadata"]["prompt_tokens_precheck_source"] == "estimate"

    def test_the_documented_2k_input_limit_is_exposed_for_unit_packing(self) -> None:
        """Measured: a 4,800-prompt-token unit returned a translation of its FIRST SENTENCE with
        `finish_reason: stop` — silent 99% content loss. The model card documents 2K, and the
        served container's larger window is not the same as the model supporting it."""
        assert MODEL_INPUT_TOKEN_LIMIT == 2048


class TestObservability:
    def test_every_unit_is_recorded_on_the_run_metrics(self) -> None:
        """Same shape as every other LLM operation: the provider funnels through a recorder on
        pipeline_metrics, so run totals and per-episode isolation both work."""
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        p = _provider()
        p.pipeline_metrics = m  # type: ignore[attr-defined]
        for _ in range(3):
            p.translate("Hola.", source_language="es")
        assert m.llm_translation_calls == 3
        assert m.llm_translation_input_tokens == 150
        assert m.llm_translation_output_tokens == 60
        # Local GPU: a measured zero, not an unmeasured None.
        assert m.llm_translation_cost_usd == 0.0

    def test_the_per_episode_probe_isolates_units_and_tokens(self) -> None:
        """Run-level accumulators are shared across parallel episodes, so a before/after delta
        on them is racy — the probe is what makes a per-episode number trustworthy."""
        from podcast_scraper.workflow.metrics import Metrics
        from podcast_scraper.workflow.processing_manifest import EpisodeCostProbe

        inner = Metrics()
        probe = EpisodeCostProbe(inner)
        p = _provider()
        p.pipeline_metrics = probe  # type: ignore[attr-defined]
        for _ in range(2):
            p.translate("Hola.", source_language="es")
        assert probe.translation_units == 2
        assert probe.translation_input_tokens == 100
        assert inner.llm_translation_calls == 2, "and it still forwards to the run totals"

    def test_a_missing_recorder_does_not_break_translation(self) -> None:
        """Telemetry never breaks the operation."""
        p = _provider()
        p.pipeline_metrics = object()  # type: ignore[attr-defined]
        assert p.translate("Hola.", source_language="es")["text"] == "Hello world."


class TestTheFactory:
    def test_it_builds_the_provider_from_config(self) -> None:
        p = create_translation_provider(_cfg())
        assert isinstance(p, GemmaTranslateProvider)

    def test_an_unknown_provider_id_is_refused(self) -> None:
        with pytest.raises(TranslationProviderUnavailable, match="unknown translation provider"):
            create_translation_provider(_cfg(), provider_type="nope")

    def test_no_endpoint_configured_is_refused_with_a_reason(self) -> None:
        with pytest.raises(TranslationProviderUnavailable, match="registry-governed"):
            create_translation_provider(config.Config(rss="https://e.com/f.xml"))

    def test_configured_is_separate_from_wanted(self) -> None:
        """`multilingual_ingest` says whether we WANT to translate; this says whether we COULD.
        Keeping them apart is what lets the stage record `flag_off` distinctly from a broken
        endpoint — one is a decision, the other a defect."""
        assert is_translation_configured(_cfg()) is True
        assert is_translation_configured(config.Config(rss="https://e.com/f.xml")) is False


class TestSentenceAlignment:
    """RFC-124 §5.1: the unit is the CONTEXT, the sentence is the alignment atom.

    The request sends numbered sentences and requires numbered output of the same length. That
    the model actually does this was VERIFIED against the live service (2026-09-30: 2-, 3- and
    5-sentence units all returned matching numbered output), not assumed from the RFC — it would
    otherwise have been the second untested assumption of the day.
    """

    @staticmethod
    def _unit(texts: List[str], *, oversized: bool = False) -> Any:
        from podcast_scraper.translation.units import TranslationUnit, UnitSentence

        cursor = 0
        sents = []
        for i, txt in enumerate(texts, start=1):
            sents.append(UnitSentence(f"t0000.s{i:02d}", txt, cursor, cursor + len(txt)))
            cursor += len(txt) + 1
        return TranslationUnit(
            unit_id="t0000.u01",
            turn_id="t0000",
            speaker_label="Maya",
            sentences=sents,
            oversized=oversized,
            content_key="deadbeef",
        )

    def test_numbered_output_aligns_back_to_sentence_ids(self) -> None:
        p = _provider(text="1. Welcome back.\n2. Today we talk trails.")
        got = p.translate_unit(self._unit(["Bienvenidos.", "Hoy hablamos."]), source_language="es")
        assert got["alignment"] == "sentence"
        assert got["sentences"] == [
            {"sent_id": "t0000.s01", "en_text": "Welcome back."},
            {"sent_id": "t0000.s02", "en_text": "Today we talk trails."},
        ]
        assert got["metadata"]["unit_id"] == "t0000.u01"
        assert got["metadata"]["content_key"] == "deadbeef"

    def test_a_single_sentence_unit_needs_no_numbering(self) -> None:
        """There is nothing to align, so the plain payload goes — and no numbering can confuse
        the model into emitting a list for one sentence."""
        p = _provider(text="Welcome back.")
        got = p.translate_unit(self._unit(["Bienvenidos."]), source_language="es")
        assert got["alignment"] == "sentence"
        assert got["sentences"] == [{"sent_id": "t0000.s01", "en_text": "Welcome back."}]
        sent_prompt = p.client.completions.calls[0]["prompt"]  # type: ignore[attr-defined]
        assert "1. " not in sent_prompt

    def test_a_length_mismatch_falls_back_to_the_whole_unit_and_SAYS_SO(self) -> None:
        """Coarser alignment, not wrong text — and recorded, so a corpus can be asked how often
        it happened rather than it being invisible."""
        p = _provider(text="Only one line came back.")
        got = p.translate_unit(
            self._unit(["Bienvenidos.", "Hoy hablamos.", "Tercera frase."]), source_language="es"
        )
        assert got["alignment"] == "unit"
        assert len(got["sentences"]) == 1
        assert got["sentences"][0]["sent_id"] == "t0000.s01"
        assert got["metadata"]["attempts"] == 3, "two numbered tries, then the block fallback"
        assert "alignment_mismatch" in got["metadata"]

    def test_an_oversized_unit_is_refused_without_a_request(self) -> None:
        p = _provider()
        got = p.translate_unit(self._unit(["x" * 50], oversized=True), source_language="es")
        assert got["alignment"] == "failed"
        assert p.client.completions.calls == []  # type: ignore[attr-defined]

    def test_a_dead_translator_reports_failed_not_a_silent_block(self) -> None:
        p = _provider(raises=OSError("connection reset"))
        got = p.translate_unit(self._unit(["Uno.", "Dos."]), source_language="es")
        assert got["alignment"] == "failed"
        assert got["sentences"] == []


class TestNumberedParsing:
    def test_a_wrapped_continuation_line_is_appended_not_dropped(self) -> None:
        """A translation that wraps is still one sentence. Dropping the tail would silently
        shorten it — the exact failure shape this arc keeps meeting."""
        from podcast_scraper.providers.vllm.translate_provider import _parse_numbered

        assert _parse_numbered("1. First part\n   and its continuation\n2. Second") == [
            "First part and its continuation",
            "Second",
        ]

    def test_both_dot_and_paren_numbering_parse(self) -> None:
        from podcast_scraper.providers.vllm.translate_provider import _parse_numbered

        assert _parse_numbered("1) One\n2) Two") == ["One", "Two"]

    def test_unnumbered_output_yields_nothing_rather_than_a_wrong_alignment(self) -> None:
        """Better to trip the mismatch path than to guess which sentence each line was."""
        from podcast_scraper.providers.vllm.translate_provider import _parse_numbered

        assert _parse_numbered("Just a paragraph with no numbers.") == []
