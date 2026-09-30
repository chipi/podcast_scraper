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

import json
from pathlib import Path
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


class TestTheNamingStageReadsIt:
    def test_it_prefers_the_translated_title(self, tmp_path: Path) -> None:
        """The whole reason the field exists. Asserted by behaviour: the naming stage is given a
        Spanish title and a ledger holding the English one, and the ENGLISH one must reach the
        roster."""
        from podcast_scraper.workflow import naming_stage

        d = tmp_path / "transcripts"
        d.mkdir(parents=True)
        rel = "transcripts/01 - ep.txt"
        src_rows = [
            {
                "id": 0,
                "start": 0.0,
                "end": 30.0,
                "speaker": "SPEAKER_00",
                "speaker_label": "SPEAKER_00",
                "text": "Soy Dana Reyes.",
            }
        ]
        en_rows = [
            {
                "id": 0,
                "start": 0.0,
                "end": 30.0,
                "speaker_label": "SPEAKER_00",
                "text": "I'm Dana Reyes.",
            }
        ]
        (tmp_path / rel).write_text("SPEAKER_00: Soy Dana Reyes.\n", encoding="utf-8")
        (tmp_path / "transcripts/01 - ep.en.txt").write_text(
            "SPEAKER_00: I'm Dana Reyes.\n", encoding="utf-8"
        )
        (d / "01 - ep.segments.json").write_text(json.dumps(src_rows), encoding="utf-8")
        (d / "01 - ep.en.segments.json").write_text(json.dumps(en_rows), encoding="utf-8")
        (d / "01 - ep.translation.json").write_text(
            json.dumps({"version": "1", "title_en": "Trail Sessions with Dana Reyes"}),
            encoding="utf-8",
        )

        seen: Dict[str, Any] = {}
        real = naming_stage.__dict__["run_naming_stage"]

        import podcast_scraper.providers.ml.diarization.pipeline as dp

        original = dp.resolve_names_on_result

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen["episode_title"] = kwargs.get("episode_title")
            return original(*args, **kwargs)

        dp.resolve_names_on_result = spy  # type: ignore[assignment]
        try:
            real(
                config.Config(
                    rss="https://e.com/f.xml",
                    output_dir=str(tmp_path),
                    language="es",
                    speaker_resolution_llm=False,
                    translate_api_base="http://translator.invalid:8005/v1",
                    translate_model="google/translategemma-12b-it",
                ),
                transcript_relpath=rel,
                effective_output_dir=str(tmp_path),
                episode_title="Sesiones de Sendero con Dana Reyes",
            )
        finally:
            dp.resolve_names_on_result = original  # type: ignore[assignment]

        assert seen["episode_title"] == "Trail Sessions with Dana Reyes", (
            "the roster was given the SOURCE title while every per-voice sample was English — "
            "the same silent mismatch `naming_text` exists to prevent"
        )

    def test_it_falls_back_to_the_source_title(self, tmp_path: Path) -> None:
        """No ledger, or no `title_en`. A missing title costs the roster its role context
        entirely, which is worse than a title in the wrong language."""
        from podcast_scraper.workflow import naming_stage

        d = tmp_path / "transcripts"
        d.mkdir(parents=True)
        rel = "transcripts/01 - ep.txt"
        rows = [
            {
                "id": 0,
                "start": 0.0,
                "end": 30.0,
                "speaker": "SPEAKER_00",
                "speaker_label": "SPEAKER_00",
                "text": "Soy Dana Reyes.",
            }
        ]
        en = [
            {
                "id": 0,
                "start": 0.0,
                "end": 30.0,
                "speaker_label": "SPEAKER_00",
                "text": "I'm Dana Reyes.",
            }
        ]
        (tmp_path / rel).write_text("x\n", encoding="utf-8")
        (tmp_path / "transcripts/01 - ep.en.txt").write_text("y\n", encoding="utf-8")
        (d / "01 - ep.segments.json").write_text(json.dumps(rows), encoding="utf-8")
        (d / "01 - ep.en.segments.json").write_text(json.dumps(en), encoding="utf-8")

        import podcast_scraper.providers.ml.diarization.pipeline as dp

        seen: Dict[str, Any] = {}
        original = dp.resolve_names_on_result

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen["episode_title"] = kwargs.get("episode_title")
            return original(*args, **kwargs)

        dp.resolve_names_on_result = spy  # type: ignore[assignment]
        try:
            naming_stage.run_naming_stage(
                config.Config(
                    rss="https://e.com/f.xml",
                    output_dir=str(tmp_path),
                    language="es",
                    speaker_resolution_llm=False,
                    translate_api_base="http://translator.invalid:8005/v1",
                    translate_model="google/translategemma-12b-it",
                ),
                transcript_relpath=rel,
                effective_output_dir=str(tmp_path),
                episode_title="Sesiones de Sendero",
            )
        finally:
            dp.resolve_names_on_result = original  # type: ignore[assignment]

        assert seen["episode_title"] == "Sesiones de Sendero"


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
