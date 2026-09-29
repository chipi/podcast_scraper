"""S2.14: English-only NLP declines non-English text rather than returning wrong results.

WHY A GUARD AND NOT A COPING STAGE. §5.2 measured what English models do to Spanish prose, and
in none of the three cases did they fail:

- ad excision found 0 patterns where the English render found 6
- the sniff gate OVER-counted, 98 against the English control's 65
- English NER kept recall at 2/2 while precision fell from 67% to 18%

A missing result is visible. A wrong one is not. That asymmetry is the argument.
"""

from __future__ import annotations

import pytest

from podcast_scraper.languages_guard import (
    is_english_text_language,
    REASON_INPUT_NOT_ENGLISH,
    refuse_non_english,
)

pytestmark = pytest.mark.unit


class TestWhatPasses:
    @pytest.mark.parametrize("language", ["en", "EN", "en-US", "en-gb", " en "])
    def test_english_and_its_regional_subtags_pass(self, language: str) -> None:
        assert is_english_text_language(language) is True
        assert refuse_non_english("naming", language) is None

    @pytest.mark.parametrize("language", [None, "", "   "])
    def test_an_unknown_language_PASSES(self, language: object) -> None:
        """Most of the corpus predates language resolution. Refusing those would stop the
        English pipeline that works today in order to protect a Spanish one that does not exist
        yet — the same reasoning the transcription guard records."""
        assert is_english_text_language(language) is True  # type: ignore[arg-type]
        assert refuse_non_english("naming", language) is None  # type: ignore[arg-type]


class TestWhatIsRefused:
    @pytest.mark.parametrize("language", ["es", "es-ES", "de", "sr", "pt-BR"])
    def test_every_non_english_tag_is_refused(self, language: str) -> None:
        assert is_english_text_language(language) is False
        assert refuse_non_english("naming", language) is not None

    def test_the_reason_names_the_stage_and_the_language(self) -> None:
        """The log line, the manifest entry and the metric carry the same sentence, so an
        operator reading any one of them learns which model was pointed at which language."""
        reason = refuse_non_english("transcript-intro host detection", "es")
        assert reason is not None
        assert "transcript-intro host detection" in reason
        assert "'es'" in reason

    def test_the_reason_cites_the_measurement_rather_than_asserting(self) -> None:
        """A guard whose justification is "it seemed wrong" gets removed by the next person who
        finds it inconvenient. The numbers are in the sentence."""
        reason = refuse_non_english("naming", "es") or ""
        assert "0 of 6" in reason
        assert "98 vs 65" in reason
        assert "67% to 18%" in reason

    def test_it_declines_rather_than_raising(self) -> None:
        """A guard that raised would turn a stage-ordering mistake into a lost episode. The
        episode is recoverable; a corpus of confidently wrong claims is not."""
        assert isinstance(refuse_non_english("naming", "es"), str)

    def test_the_vocabulary_is_closed(self) -> None:
        assert REASON_INPUT_NOT_ENGLISH == "input_not_english"


class TestTheNerEntryPointHonoursIt:
    def test_spanish_prose_yields_no_candidates(self) -> None:
        """The measured hazard: English NER over Spanish does not find nothing, it finds wrong
        things that survive `_looks_like_person` and become people."""
        from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_transcript_intro

        class _FakeNlp:
            def __call__(self, _text: str) -> object:
                raise AssertionError("the NER model must not be invoked on non-English text")

        got = detect_hosts_from_transcript_intro(
            "Bienvenidos de nuevo. Soy Maya Koster y hoy hablamos de senderos.",
            nlp=_FakeNlp(),
            text_language="es",
        )
        assert got == set()

    def test_english_prose_is_unaffected(self) -> None:
        """The path 678 episodes take. The guard must be inert for them."""
        from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_transcript_intro

        called: list[str] = []

        class _Nlp:
            def __call__(self, text: str) -> object:
                called.append(text)

                class _Doc:
                    ents: list = []

                return _Doc()

        detect_hosts_from_transcript_intro(
            "Welcome back. I'm Maya Koster and today we talk trails.",
            nlp=_Nlp(),
            text_language="en",
        )
        assert called, "English text must still reach the model"

    def test_no_language_offered_keeps_the_old_behaviour(self) -> None:
        """Every existing caller passes no language, and none of them may change behaviour."""
        from podcast_scraper.speaker_detectors.hosts import detect_hosts_from_transcript_intro

        called: list[str] = []

        class _Nlp:
            def __call__(self, text: str) -> object:
                called.append(text)

                class _Doc:
                    ents: list = []

                return _Doc()

        detect_hosts_from_transcript_intro("Welcome back. I'm Maya.", nlp=_Nlp())
        assert called


class TestTheSniffGateHonoursIt:
    def test_a_non_english_episode_goes_deep_only_without_a_sniff_pass(self) -> None:
        """Cheaper AND correct: the gate's entity count uses an English NER model, so on
        Spanish it routes on a number that means nothing — measured over-counting 98 vs 65. It
        skips the sniff transcription entirely rather than paying for one to feed a broken
        count."""
        from podcast_scraper import config
        from podcast_scraper.workflow import sniff_gate

        calls: list[dict] = []

        class _Provider:
            def transcribe_with_segments(self, path: str, **kw: object) -> tuple:
                calls.append(dict(kw))
                return {"text": "hola", "segments": []}, 1.0

        cfg = config.Config(
            rss="https://e.com/f.xml",
            language="es",
            dgx_whisper_sniff_model="tiny",
            dgx_whisper_model="large-v3",
        )
        result, _elapsed = sniff_gate.transcribe_with_sniff_gate(
            media_path="/tmp/a.mp3", cfg=cfg, provider=_Provider()
        )
        assert result["sniff_gate"]["decision"] == sniff_gate.GATE_DECISION_LANGUAGE_NOT_EN
        assert result["sniff_gate"]["language"] == "es"
        assert len(calls) == 1, "exactly one transcription — the deep one, no sniff pass"
        assert "model_override" not in calls[0]
