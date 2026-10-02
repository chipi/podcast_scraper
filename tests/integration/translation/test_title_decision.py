"""S2.4's title decision: the EPISODE title is translated, the SHOW name is not.

WHY IT HAD TO BE DECIDED RATHER THAN DRIFTED INTO. ADR-157's evidence includes the deployed model
turning `Sesiones de Sendero` into `Trail Sessions`. A show's NAME is its identity — renaming it
would change what the feed IS on every surface, in search, and in every listener's saved library.
An EPISODE title describes that episode's content, which is what translation is for.

AND THE EPISODE TITLE HAS TO BE IN ENGLISH, per §5.4 C-6. The roster reads the title for
host/guest context and for NER candidate discovery, so passing a Spanish title to an English NER
points the §5.2 hazard — recall held at 2/2 while precision fell 67% to 18% — at the one input
naming trusts most. It is the same silent mismatch `naming_text` exists to prevent one level
down: both are strings, so nothing errors, the roster just resolves fewer voices.

BEST EFFORT, NEVER FATAL. A title is one short unit and is not part of the §5.3 completeness
gate. A failed title costs the roster some context and naming falls back to the source title; it
must never cost the episode its translation.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.translation.artifacts import TranslationDocument
from podcast_scraper.workflow.translation_stage import _translate_title

pytestmark = pytest.mark.integration


class _Provider:
    """Returns a fixed English string for whatever unit it is given."""

    def __init__(self, out: str = "Trail Sessions, episode four") -> None:
        self.out = out
        self.seen: List[str] = []

    def translate_unit(self, unit: Any, **_kw: Any) -> Dict[str, Any]:
        self.seen.append(unit.source_text)
        return {
            "alignment": "sentence",
            "sentences": [{"sent_id": unit.sentences[0].sent_id, "en_text": self.out}],
            "metadata": {},
        }


class _Failing(_Provider):
    def translate_unit(self, unit: Any, **_kw: Any) -> Dict[str, Any]:
        raise RuntimeError("the translator is down")


class _Empty(_Provider):
    def translate_unit(self, unit: Any, **_kw: Any) -> Dict[str, Any]:
        return {"alignment": "failed", "sentences": [], "metadata": {"error": "no output"}}


def _cfg() -> config.Config:
    return config.Config(rss="https://e.com/f.xml", language="es")


class TestTheEpisodeTitleIsTranslated:
    def test_it_returns_the_english_title(self) -> None:
        provider = _Provider()
        got = _translate_title(_cfg(), provider, "Sesiones de Sendero, episodio cuatro", "es")
        assert got == "Trail Sessions, episode four"

    def test_the_title_is_what_was_SENT(self) -> None:
        """And it is sent with NO speaker label, like every other unit (D-24)."""
        provider = _Provider()
        _translate_title(_cfg(), provider, "Sesiones de Sendero, episodio cuatro", "es")
        assert provider.seen == ["Sesiones de Sendero, episodio cuatro"]

    def test_an_empty_title_is_not_sent_at_all(self) -> None:
        provider = _Provider()
        for title in ("", "   ", None):
            assert _translate_title(_cfg(), provider, title, "es") is None
        assert provider.seen == []


class TestItIsNeverFatal:
    def test_a_raising_provider_returns_None(self) -> None:
        """A failed title must not cost the episode its translation."""
        assert _translate_title(_cfg(), _Failing(), "Sesiones de Sendero", "es") is None

    def test_an_empty_translation_returns_None(self) -> None:
        assert _translate_title(_cfg(), _Empty(), "Sesiones de Sendero", "es") is None

    def test_a_whitespace_only_translation_returns_None(self) -> None:
        """Otherwise the roster gets a blank title, which is worse than the source one — it
        cannot tell "not translated" from "the episode has no title"."""
        assert _translate_title(_cfg(), _Provider(out="   "), "Sesiones", "es") is None


class TestTheTitleIsRecordedInTheLedger:
    def test_the_document_carries_it(self) -> None:
        doc = TranslationDocument(source_language="es", title_en="Trail Sessions")
        assert doc.to_dict()["title_en"] == "Trail Sessions"

    def test_it_survives_a_round_trip(self) -> None:
        raw = TranslationDocument(source_language="es", title_en="Trail Sessions").to_dict()
        assert TranslationDocument.from_dict(raw).title_en == "Trail Sessions"

    def test_an_absent_title_reads_as_None(self) -> None:
        """A ledger written before this field existed, and an English episode."""
        assert TranslationDocument.from_dict({"version": "1"}).title_en is None

    def test_the_title_unit_is_NOT_in_the_units_list(self) -> None:
        """It must not enter `doc.units`: the completeness gate counts units, so a failed title
        would make an otherwise-complete episode read as `failed` and withhold its English set.
        And `resolve_units_for_span` would be able to see a unit with char offsets 0-0, so a
        claim could be provenanced to the title.
        """
        from pathlib import Path as _P

        src = (
            _P(__file__).resolve().parents[3] / "src/podcast_scraper/workflow/translation_stage.py"
        ).read_text(encoding="utf-8")
        # The title result is assigned to `doc.title_en`, never appended to `doc.units`.
        assert "doc.title_en = _translate_title(" in src
        assert (
            'doc.units.append(\n                UnitRecord(\n                    unit_id="title"'
            not in src
        )


# `TestTheNamingStageReadsIt` lived here and is gone (2026-10-02). It asserted that the naming
# STAGE reads `title_en` from the ledger — and that stage was a reordering of the generic pipeline
# made for this feature, reverted on the operator's instruction. D-42 itself is untouched: the
# EPISODE title is still translated and still recorded as `title_en`, which the classes above and
# below this comment are what prove. Whether any consumer reads it is now an open question, not an
# assertion.


class TestTheShowNameIsNotTranslated:
    def test_nothing_passes_the_FEED_title_to_the_translator(self) -> None:
        """The decision, asserted where it could regress. `run_translation_stage` takes an
        `episode_title` and no feed/show title, so there is no channel for the show name to
        reach the translator by."""
        import inspect

        from podcast_scraper.workflow.translation_stage import run_translation_stage

        params = set(inspect.signature(run_translation_stage).parameters)
        assert "episode_title" in params
        assert not {"feed_title", "show_title", "show_name"} & params

    def test_the_helper_is_documented_as_episode_only(self) -> None:
        from podcast_scraper.workflow.translation_stage import _translate_title

        doc = _translate_title.__doc__ or ""
        assert "SHOW NAME IS NOT TRANSLATED" in doc


class TestTheTitleTravelsWithContext:
    """A one-sentence unit has no context, and a title is the shortest string in the episode.

    MEASURED, not theorised. Sent alone, the German fixture's `Wege Bauen, Die Bleiben` came back
    from the live model as "Building bridges, creating connections that last" — both nouns
    invented — while the same conversation in es/it/fr/pt produced the correct "Building Trails
    That Last." With the episode's opening sentences in the same unit it became "Building Paths
    That Last.", and all four others were unchanged.

    This module's own contract is "the unit is the translation CONTEXT; the sentence is the
    alignment atom" — the title simply was not using the mechanism that already existed.
    """

    def test_context_sentences_ride_in_the_same_unit(self) -> None:
        provider = _Provider()
        _translate_title(
            _cfg(),
            provider,
            "Wege Bauen, Die Bleiben",
            "de",
            [
                "Heute sprechen wir über Wegebau und Entwässerung auf steilen Hängen.",
                "Bankette am Hang sind nicht optional.",
            ],
        )
        # The provider records source_text; the context must be IN the text it was given.
        sent = provider.seen[-1]
        assert "Wege Bauen" in sent, sent
        assert "Entwässerung" in sent, "the context sentence never reached the model"

    def test_the_title_is_the_FIRST_sentence_so_the_answer_is_unambiguous(self) -> None:
        """Only `sentences[0]` is read back, so the title must be sent first."""
        provider = _Provider()
        _translate_title(
            _cfg(), provider, "Wege Bauen", "de", ["Ein Kontextsatz der lang genug ist."]
        )
        unit_text = provider.seen[-1]
        assert unit_text.index("Wege Bauen") < unit_text.index("Kontextsatz")

    def test_no_context_still_works(self) -> None:
        """Context is additive — an episode with none must behave exactly as before."""
        provider = _Provider()
        assert _translate_title(_cfg(), provider, "Sesiones de Sendero", "es", None) == provider.out
        assert _translate_title(_cfg(), provider, "Sesiones de Sendero", "es", []) == provider.out

    def test_blank_context_sentences_are_dropped(self) -> None:
        """A whitespace-only sentence would add an alignment slot carrying nothing."""
        provider = _Provider()
        _translate_title(_cfg(), provider, "Wege Bauen", "de", ["", "   ", "Echter Kontext hier."])
        assert provider.seen[-1].count("\n") <= 2, provider.seen[-1]
