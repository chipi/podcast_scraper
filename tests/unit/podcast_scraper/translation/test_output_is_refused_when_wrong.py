"""The translator's dangerous failures REPORT SUCCESS, so the output has to be validated.

WHAT THIS GUARDS, and every string below is one the model actually produced — recorded in
MULTILINGUAL_ARC §9 from 174 real requests, not invented:

- **Commentary instead of a translation.** `Here are a few options for translating the Spanish
  text, depending on the specific context and desired emphasis: Option 1...` — 366 tokens, HTTP
  200, `finish_reason: stop`, identical across three passes, on a REAL 22-word unit. **1 of 47
  real units, 2.1%.** That text would land in `.en.txt` as though somebody had spoken it, and
  every stage downstream treats `.en.txt` as speech: the summariser, GI's claims, KG's entities,
  and the subtitle cues a listener reads.
- **The model answering the instruction.** Content-free units returned *"Okay, I understand. I'm
  ready to translate Spanish text into English. Please provide the text"*.
- **Silent truncation.** A 4,060-word unit (4,800 prompt tokens) returned **32 completion
  tokens** — its first sentence, 99.3% of the content gone, `stop`, non-empty. The
  `finish_reason == "length"` check cannot catch it because the model said `stop`.

WHY A GUARD AND NOT JUST THE PROMPT. The prompt already says "Produce only the English
translation, without any additional explanations or commentary", and adding that line is what
stopped most of it — the 2.1% was measured AFTER. And the numbered-alignment contract catches
commentary only when the line count mismatches, at which point the whole-unit fallback accepted
whatever came back, unvalidated. That fallback is where it landed.

WHY REFUSING IS THE RIGHT ANSWER. A refused unit is an ordinary failed unit, so §5.3's
completeness gate withholds the episode's whole English set. An episode absent from the English
surfaces is recoverable by a re-run; a corpus with fabricated speech in it is not. That is #876's
asymmetry — a wrong answer costs more than a missing one — applied to text instead of names.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.providers.vllm.translate_provider import reject_translation_output

pytestmark = pytest.mark.unit

_SPANISH = (
    "Sinceramente, el drenaje es la decisión de diseño con más impacto en cualquier sendero. "
    "Lo hemos visto en equipos que usan la Alianza Cascadia."
)
_GOOD = (
    "Honestly, drainage is the highest-leverage design choice on any trail. "
    "We have seen it on teams using the Cascadia Alliance."
)


class TestWhatIsAccepted:
    def test_a_real_translation_passes(self) -> None:
        assert reject_translation_output(_SPANISH, _GOOD) is None

    def test_a_TERSE_translation_passes(self) -> None:
        """The floor must not punish concision, because a false refusal costs the whole episode
        its English set (§5.3 withholds the set when any unit fails).

        Measured over the 103 sentences of the first real run: min 0.72, p05 0.81, p50 1.00,
        max 1.55. Zero would have been refused, and the tightest sits 2.9x above the floor.
        """
        assert (
            reject_translation_output(
                "¿Cómo encaja RockShox en este panorama?", "Where does RockShox fit?"
            )
            is None
        )

    def test_the_floor_has_MARGIN_over_the_measured_minimum(self) -> None:
        """Pinned as a number, because the floor is only defensible while it sits below what
        real translations produce. 0.72 was the tightest of 103 real sentences."""
        from podcast_scraper.providers.vllm.translate_provider import _MIN_OUTPUT_RATIO

        assert _MIN_OUTPUT_RATIO < 0.72 / 2, (
            f"the floor {_MIN_OUTPUT_RATIO} is within 2x of the tightest real translation "
            "(0.72) — tighten it only with a new measurement"
        )

    def test_a_None_is_left_to_the_caller(self) -> None:
        """`None` already means "the request failed"; the guard must not relabel it."""
        assert reject_translation_output(_SPANISH, None) is None

    def test_text_that_merely_MENTIONS_translation_passes(self) -> None:
        """Someone can talk about translation on a podcast. Only the model's own narration of
        its work is refused, which is why the markers are phrases and not the word."""
        assert (
            reject_translation_output(
                "Hablamos de la traducción del libro.", "We talked about the book's translation."
            )
            is None
        )


class TestTheMeasuredCommentaryIsRefused:
    def test_the_exact_string_the_model_returned(self) -> None:
        """MULTILINGUAL_ARC §9, unit u0014, 3/3 passes."""
        got = reject_translation_output(
            _SPANISH,
            "Here are a few options for translating the Spanish text, depending on the "
            "specific context and desired emphasis: Option 1: Honestly, drainage is...",
        )
        assert got is not None and "commentary" in got

    def test_the_chat_reply_to_a_content_free_unit(self) -> None:
        got = reject_translation_output(
            "Mmm.",
            "Okay, I understand. I'm ready to translate Spanish text into English. "
            "Please provide the text",
        )
        assert got is not None and "commentary" in got

    @pytest.mark.parametrize(
        "text",
        [
            "Here's a translation that aims for accuracy and nuance: Honestly, drainage...",
            "Here is a translation of the text: Honestly, drainage is the choice.",
            "As an AI, I cannot translate that.",
            "Note that this translation preserves the original emphasis.",
        ],
    )
    def test_other_recorded_shapes(self, text: str) -> None:
        assert reject_translation_output(_SPANISH, text) is not None

    def test_an_option_LIST_is_refused_even_without_a_marker(self) -> None:
        """The enumeration is the tell: each option can read like a plausible translation, so
        the individual lines would pass every other check."""
        got = reject_translation_output(
            _SPANISH,
            "Option 1: Honestly, drainage is the highest-leverage choice on any trail.\n"
            "Option 2: Frankly, drainage is the most impactful design decision.",
        )
        assert got is not None and "options" in got


class TestSilentTruncationIsRefused:
    def test_the_measured_shape(self) -> None:
        """4,060 words in, its first sentence out. `finish_reason: stop`, so nothing else
        catches it."""
        source = "Sinceramente, el drenaje es la decisión de diseño. " * 80
        got = reject_translation_output(source, "Honestly, drainage is the design choice.")
        assert got is not None and "truncated" in got

    def test_the_reason_carries_the_NUMBERS(self) -> None:
        """An operator reading the ledger has to be able to see how bad it was."""
        source = "x" * 1000
        got = reject_translation_output(source, "y" * 100) or ""
        assert "1000" in got and "100" in got and "10%" in got

    def test_an_empty_output_is_refused(self) -> None:
        assert reject_translation_output(_SPANISH, "   ") == "empty output"

    def test_no_source_means_no_ratio_check(self) -> None:
        """Nothing to compare against — refusing on a ratio we cannot compute would be a guess."""
        assert reject_translation_output("", "Honestly, drainage.") is None


class TestEveryAcceptancePathIsGuarded:
    """The guard is only as good as the paths that call it, and the whole-unit fallback — the
    one the measured commentary actually landed on — was previously unvalidated."""

    @staticmethod
    def _unit(texts: List[str]) -> Any:
        from podcast_scraper.translation.units import TranslationUnit, UnitSentence

        return TranslationUnit(
            unit_id="u0001",
            turn_id="t0001",
            speaker_label="SPEAKER_00",
            sentences=[
                UnitSentence(sent_id=f"t0001.s{i:02d}", text=x, char_start=0, char_end=len(x))
                for i, x in enumerate(texts)
            ],
        )

    @staticmethod
    def _provider(responses: List[Optional[str]]) -> Any:
        from podcast_scraper import config
        from podcast_scraper.providers.vllm.translate_provider import GemmaTranslateProvider

        cfg = config.Config(
            rss="https://e.com/f.xml",
            translate_api_base="http://translator.invalid:8005/v1",
            translate_model="google/translategemma-12b-it",
            translate_verify_served_model=False,
        )
        p = GemmaTranslateProvider(cfg)
        queue = list(responses)

        def fake_translate(text: str, **_kw: Any) -> Dict[str, Any]:
            out = queue.pop(0) if queue else None
            return {"text": out, "metadata": {"model": "stub"}}

        p.translate = fake_translate  # type: ignore[method-assign]
        return p

    COMMENTARY = (
        "Here are a few options for translating the Spanish text, depending on the specific "
        "context: Option 1: Honestly, drainage."
    )

    def test_the_SINGLE_sentence_path_refuses(self) -> None:
        p = self._provider([self.COMMENTARY])
        got = p.translate_unit(self._unit([_SPANISH]), source_language="es")
        assert got["alignment"] == "failed"
        assert "refused" in (got["metadata"].get("error") or "")

    def test_the_WHOLE_UNIT_FALLBACK_refuses(self) -> None:
        """Two numbered attempts mismatch — which is exactly when the model is ignoring the
        instruction — and then the fallback used to accept anything."""
        p = self._provider(["1. only one line", "1. only one line", self.COMMENTARY])
        got = p.translate_unit(self._unit([_SPANISH, "Otra frase aquí."]), source_language="es")
        assert got["alignment"] == "failed", got
        assert "refused" in (got["metadata"].get("error") or "")

    def test_the_NUMBERED_path_refuses_one_bad_line_among_good_ones(self) -> None:
        """A commentary response can come back with the right line count, and then the
        alignment contract passes it."""
        p = self._provider([f"1. {_GOOD}\n2. {self.COMMENTARY}"])
        got = p.translate_unit(self._unit([_SPANISH, "Otra frase aquí."]), source_language="es")
        assert got["alignment"] == "failed", got

    def test_a_GOOD_response_still_passes_every_path(self) -> None:
        """The control. Without it, a guard that refused everything would look like a pass."""
        p = self._provider([f"1. {_GOOD}\n2. Another sentence here."])
        got = p.translate_unit(self._unit([_SPANISH, "Otra frase aquí."]), source_language="es")
        assert got["alignment"] == "sentence", got
        assert len(got["sentences"]) == 2


class TestSpeechThatSoundsLikeCommentaryIsAccepted:
    """Found on El Hilo (es, 2026-10-09): a correct 11-sentence unit was refused on both passes
    because its first sentence — "Los resultados de las pruebas PISA pueden ser leídos de formas
    bien distintas según el contexto" — translates to "...depending on the context". One refused
    unit withholds the episode's whole English set (§5.3), so the episode lost summary, GI and KG
    over a faithful translation. Phrases that are ordinary speech are commentary only when the
    output also narrates translating."""

    @pytest.mark.parametrize(
        "source,english",
        [
            (
                "Los resultados pueden ser leídos de formas bien distintas según el contexto.",
                "The results can be interpreted in very different ways depending on the context.",
            ),
            ("Vale, entiendo. Sigamos.", "Okay, I understand. Let's go on."),
            (
                "Como investigadora de IA, lo veo a diario.",
                "As an AI researcher, I see it every day.",
            ),
        ],
    )
    def test_a_faithful_translation_is_not_refused(self, source: str, english: str) -> None:
        assert reject_translation_output(source, english) is None

    def test_the_same_phrase_inside_translator_narration_is_still_refused(self) -> None:
        got = reject_translation_output(
            _SPANISH, "Depending on the context, this could be translated as: Honestly, drainage."
        )
        assert got is not None and "commentary" in got


class TestARefusedUnitIsRetried:
    """The two other El Hilo refusals were one-off: re-sent, both units came back clean on two
    passes. A refusal was final on the first attempt, and one refusal costs the episode its
    English set."""

    _unit = staticmethod(TestEveryAcceptancePathIsGuarded._unit)
    _provider = staticmethod(TestEveryAcceptancePathIsGuarded._provider)
    COMMENTARY = TestEveryAcceptancePathIsGuarded.COMMENTARY

    def test_a_single_sentence_is_retried_after_a_refusal(self) -> None:
        p = self._provider([self.COMMENTARY, _GOOD])
        got = p.translate_unit(self._unit([_SPANISH]), source_language="es")
        assert got["alignment"] == "sentence" and got["metadata"]["attempts"] == 2

    def test_a_numbered_unit_is_retried_after_a_refusal(self) -> None:
        p = self._provider(
            [f"1. {_GOOD}\n2. {self.COMMENTARY}", f"1. {_GOOD}\n2. Another sentence here."]
        )
        got = p.translate_unit(self._unit([_SPANISH, "Otra frase aquí."]), source_language="es")
        assert got["alignment"] == "sentence" and got["metadata"]["attempts"] == 2

    def test_a_refusal_on_every_attempt_still_fails_and_says_why(self) -> None:
        p = self._provider([self.COMMENTARY, self.COMMENTARY])
        got = p.translate_unit(self._unit([_SPANISH]), source_language="es")
        assert got["alignment"] == "failed"
        assert "refused" in (got["metadata"].get("error") or "")


class TestARefusedSentenceIsRetriedInTheNumberedForm:
    """The retest (2026-10-09) showed the plain retry repeats the refusal: at temperature 0 the
    same request gets the same reply. Re-sent to the live translator, all three still-failing El
    Hilo / Radio Ambulante units — one garbled line of a reggaeton promo — came back as "Here's the
    translation:" (twice with an invented story after it) when sent plain, and as a clean single
    numbered line when sent as "1. <sentence>". The retry has to be a DIFFERENT request."""

    _unit = staticmethod(TestEveryAcceptancePathIsGuarded._unit)

    def test_the_second_attempt_sends_the_numbered_form_and_reads_one_line(self) -> None:
        sent: List[str] = []
        p = TestEveryAcceptancePathIsGuarded._provider([])

        def fake_translate(text: str, **_kw: Any) -> Dict[str, Any]:
            sent.append(text)
            out = (
                "Here's the translation: Honestly, drainage."
                if not text.startswith("1.")
                else f"1. {_GOOD}"
            )
            return {"text": out, "metadata": {"model": "stub"}}

        p.translate = fake_translate  # type: ignore[method-assign]
        got = p.translate_unit(self._unit([_SPANISH]), source_language="es")
        assert sent[0] == _SPANISH and sent[1].startswith("1. ")
        assert got["alignment"] == "sentence"
        assert got["sentences"][0]["en_text"] == _GOOD
