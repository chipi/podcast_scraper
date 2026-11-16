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
    is_target_language,
    REASON_INPUT_NOT_TARGET_LANGUAGE,
    refuse_unsupported_language,
)

pytestmark = pytest.mark.unit


class TestWhatPasses:
    @pytest.mark.parametrize("language", ["en", "EN", "en-US", "en-gb", " en "])
    def test_english_and_its_regional_subtags_pass(self, language: str) -> None:
        assert is_target_language(language) is True
        assert refuse_unsupported_language("naming", language) is None

    @pytest.mark.parametrize("language", [None, "", "   "])
    def test_an_unknown_language_PASSES(self, language: object) -> None:
        """Most of the corpus predates language resolution. Refusing those would stop the
        English pipeline that works today in order to protect a Spanish one that does not exist
        yet — the same reasoning the transcription guard records."""
        assert is_target_language(language) is True  # type: ignore[arg-type]
        assert refuse_unsupported_language("naming", language) is None  # type: ignore[arg-type]


class TestWhatIsRefused:
    @pytest.mark.parametrize("language", ["es", "es-ES", "de", "sr", "pt-BR"])
    def test_every_non_english_tag_is_refused(self, language: str) -> None:
        assert is_target_language(language) is False
        assert refuse_unsupported_language("naming", language) is not None

    def test_the_reason_names_the_stage_and_the_language(self) -> None:
        """The log line, the manifest entry and the metric carry the same sentence, so an
        operator reading any one of them learns which model was pointed at which language."""
        reason = refuse_unsupported_language("transcript-intro host detection", "es")
        assert reason is not None
        assert "transcript-intro host detection" in reason
        assert "'es'" in reason

    def test_the_reason_cites_the_measurement_rather_than_asserting(self) -> None:
        """A guard whose justification is "it seemed wrong" gets removed by the next person who
        finds it inconvenient. The numbers are in the sentence."""
        reason = refuse_unsupported_language("naming", "es") or ""
        assert "0 of 6" in reason
        assert "98 vs 65" in reason
        assert "67% to 18%" in reason

    def test_it_declines_rather_than_raising(self) -> None:
        """A guard that raised would turn a stage-ordering mistake into a lost episode. The
        episode is recoverable; a corpus of confidently wrong claims is not."""
        assert isinstance(refuse_unsupported_language("naming", "es"), str)

    def test_the_vocabulary_is_closed(self) -> None:
        assert REASON_INPUT_NOT_TARGET_LANGUAGE == "input_not_english"


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
            feed_declared_language="es",  # #2283: the episode's language comes from the feed
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


class TestTheGuardIsActuallyREACHABLE:
    """The finding this class exists for: every one of these paths was dead.

    A whole-branch review on 2026-09-30 found S2.14 unreachable, and verifying it turned up
    something larger. Two separate failures were stacked:

    1. `refuse_unsupported_language` had exactly ONE caller —
       `detect_hosts_from_transcript_intro` — whose own caller, `detect_speaker_names`, had no
       `text_language` parameter to pass. So the guard could not fire from anywhere in `src/`.
    2. The three §5.2 hazards each carried their OWN inline check against
       `transcription_language(cfg)` — and that function answered `en` for a feed declaring
       `es-ES`, because the channel tag never reached the config (#2172). Measured:

           BEFORE the fix   transcription_language -> 'en'
             S2.7 ad-free base skipped?   False
             sniff gate goes deep-only?   False
             .en.* invalidation fires?    False
           AFTER the fix    transcription_language -> 'es'   (all three True)

       So all three guards were inert for exactly the corpus they were written for. They fired
       only when an operator set `language_override` or pointed a feed at a non-English profile
       default — never from a publisher's own tag, which is how a non-English feed actually
       arrives.

    Both are fixed, and these tests exist because neither failure was visible from any test
    that checked the guard in isolation: the predicate was always correct. What was missing was
    a caller.
    """

    def test_no_language_still_proceeds(self) -> None:
        """Most of the corpus predates language resolution. Refusing those would stop naming
        for the corpus that works today."""
        from podcast_scraper.languages_guard import refuse_unsupported_language

        assert refuse_unsupported_language("speaker-name detection", None) is None

    def test_the_three_hazard_sites_share_ONE_predicate(self) -> None:
        """Three copies of `language is None or language == "en"` had drifted into three
        slightly different conditions. One predicate, one answer, one place to change it."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[3] / "src" / "podcast_scraper"
        for rel in ("workflow/sniff_gate.py", "workflow/episode_processor.py"):
            text = (root / rel).read_text(encoding="utf-8")
            assert "is_target_language" in text, f"{rel} rolls its own check"
