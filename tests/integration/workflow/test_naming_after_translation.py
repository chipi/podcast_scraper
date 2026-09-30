"""D-34: naming runs AFTER translation, reading the ENGLISH render.

WHY THE ORDER MATTERS, and it is not "slightly better names". Every cue the roster reads is
English — the self-intro patterns (`I'm X`), the interview cues `corroborate_guests` requires,
and `en_core_web_sm` behind the NER. §5.2 measured what they do to Spanish prose: recall held at
2/2 while precision fell 67% to 18%, so they invent people rather than finding none.

And `corroborate_guests` needs an English interview cue in the title or description, so on a
Spanish feed every guest is REJECTED and only bare metadata names survive. Guests carry the
positions, so a translated episode yields no position-bearing insights at all. That is D-34's own
stated reasoning, and it is what this stage exists to fix.

THE HINGE is D-24: the English render carries each source turn's `speaker_label` verbatim, and at
translation time that label is still the anonymous `SPEAKER_NN`. Without it there would be no way
back from an English sentence to the voice that spoke it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper import config
from podcast_scraper.workflow.naming_stage import (
    align_english_to_voices,
    diarization_from_segments,
    naming_is_deferred,
    relabel_segments,
    run_naming_stage,
    STATUS_NAMED,
    STATUS_NO_ENGLISH,
    STATUS_NO_VOICES,
    STATUS_NOT_DEFERRED,
    STATUS_UNRESOLVED,
)

pytestmark = pytest.mark.integration

REL = "transcripts/01 - ep.txt"
EN_REL = "transcripts/01 - ep.en.txt"

# Spanish source: the self-intro is there, but in Spanish, where the English cue cannot see it.
_ES = [
    (
        0.0,
        30.0,
        "SPEAKER_00",
        "Bienvenidos de nuevo. Soy Dana Reyes, y hoy me acompaña Marcus Webb.",
    ),
    (30.0, 60.0, "SPEAKER_01", "Gracias por invitarme, Dana."),
    (60.0, 90.0, "SPEAKER_00", "Cuéntame qué construyes."),
]
# The English render of the same turns, labels carried verbatim (D-24).
_EN = [
    (0.0, 30.0, "SPEAKER_00", "Welcome back. I'm Dana Reyes, and today I'm joined by Marcus Webb."),
    (30.0, 60.0, "SPEAKER_01", "Thanks for having me, Dana."),
    (60.0, 90.0, "SPEAKER_00", "Tell me what you build."),
]


def _rows(spec: List[Any], *, with_speaker: bool) -> List[Dict[str, Any]]:
    out = []
    for i, (start, end, voice, text) in enumerate(spec):
        row: Dict[str, Any] = {
            "id": i,
            "start": start,
            "end": end,
            "speaker_label": voice,
            "text": text,
        }
        if with_speaker:
            row["speaker"] = voice
        out.append(row)
    return out


def _lay_down(tmp_path: Path) -> None:
    """A translated Spanish episode, as the pipeline leaves it just before naming."""
    d = tmp_path / "transcripts"
    d.mkdir(parents=True, exist_ok=True)
    src = _rows(_ES, with_speaker=True)
    en = _rows(_EN, with_speaker=False)
    (tmp_path / REL).write_text(
        "".join(f"{r['speaker_label']}: {r['text']}\n\n" for r in src), encoding="utf-8"
    )
    (tmp_path / EN_REL).write_text(
        "".join(f"{r['speaker_label']}: {r['text']}\n\n" for r in en), encoding="utf-8"
    )
    (d / "01 - ep.segments.json").write_text(json.dumps(src, indent=2), encoding="utf-8")
    (d / "01 - ep.en.segments.json").write_text(json.dumps(en, indent=2), encoding="utf-8")


def _cfg(tmp_path: Path, **kw: Any) -> config.Config:
    base: Dict[str, Any] = {
        "rss": "https://example.com/f.xml",
        "output_dir": str(tmp_path),
        "language": "es",
        "speaker_resolution_llm": False,
        "translate_api_base": "http://translator.invalid:8005/v1",
        "translate_model": "google/translategemma-12b-it",
    }
    base.update(kw)
    return config.Config(**base)  # type: ignore[arg-type]


class TestWhenNamingIsDeferred:
    def test_a_non_english_episode_with_a_translator_defers(self, tmp_path: Path) -> None:
        assert naming_is_deferred(_cfg(tmp_path)) is True

    def test_an_english_episode_does_NOT_defer(self, tmp_path: Path) -> None:
        """Nothing to wait for, and deferring would move the 678-episode corpus onto a new path
        to buy nothing."""
        assert naming_is_deferred(_cfg(tmp_path, language="en")) is False

    def test_no_translator_means_no_deferral(self, tmp_path: Path) -> None:
        """The English render will never arrive, so deferring would leave the episode
        permanently anonymous. Naming in place is worse (§5.2) but recoverable; never naming at
        all is not — and the §5.3 gate already stops the analysis stages trusting that
        transcript."""
        cfg = config.Config(
            rss="https://example.com/f.xml", output_dir=str(tmp_path), language="es"
        )
        assert naming_is_deferred(cfg) is False

    def test_an_episode_with_no_language_does_not_defer(self, tmp_path: Path) -> None:
        cfg = _cfg(tmp_path, language="en")
        assert naming_is_deferred(cfg) is False


class TestItNamesFromTheEnglishText:
    def test_the_voices_are_named(self, tmp_path: Path) -> None:
        """The headline. The self-intro exists only in the ENGLISH body — the Spanish says "Soy
        Dana Reyes", which the English cue regex cannot match — so a name here proves the stage
        read the translation."""
        _lay_down(tmp_path)
        got = run_naming_stage(
            _cfg(tmp_path),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
            episode_title="Trail Sessions",
        )
        assert got.status == STATUS_NAMED, got.reason
        assert got.renamed, "no voice was named"
        assert "Dana Reyes" in set(got.renamed.values())

    def test_the_SOURCE_transcript_is_re_rendered_with_the_names(self, tmp_path: Path) -> None:
        """Both bodies get the names. The source keeps its Spanish text."""
        _lay_down(tmp_path)
        run_naming_stage(_cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path))
        body = (tmp_path / REL).read_text(encoding="utf-8")
        assert "Dana Reyes:" in body
        assert "Bienvenidos de nuevo" in body, "the SOURCE text must survive verbatim"
        assert "SPEAKER_00:" not in body

    def test_the_ENGLISH_render_is_re_rendered_with_the_names(self, tmp_path: Path) -> None:
        _lay_down(tmp_path)
        run_naming_stage(_cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path))
        body = (tmp_path / EN_REL).read_text(encoding="utf-8")
        assert "Dana Reyes:" in body
        assert "Welcome back." in body, "the English text must survive verbatim"
        assert "SPEAKER_00:" not in body

    def test_the_frozen_voice_id_is_NEVER_rewritten(self, tmp_path: Path) -> None:
        """`speaker` is what `.anon.txt`, the roster and every later stage key on. Only
        `speaker_label` moves."""
        _lay_down(tmp_path)
        run_naming_stage(_cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path))
        rows = json.loads(
            (tmp_path / "transcripts" / "01 - ep.segments.json").read_text(encoding="utf-8")
        )
        # The re-render goes through the offset formatter, which emits `speaker_label`; the
        # point here is that no row claims a NAME as its voice id.
        for row in rows:
            assert not str(row.get("speaker", "")).startswith("Dana")

    def test_both_sidecars_are_rewritten_with_char_offsets(self, tmp_path: Path) -> None:
        """The bodies and their sidecars must describe the same text, or every GI quote in the
        episode points at the wrong characters."""
        _lay_down(tmp_path)
        run_naming_stage(_cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path))
        for body_rel, seg_rel in (
            (REL, "transcripts/01 - ep.segments.json"),
            (EN_REL, "transcripts/01 - ep.en.segments.json"),
        ):
            text = (tmp_path / body_rel).read_text(encoding="utf-8")
            rows = json.loads((tmp_path / seg_rel).read_text(encoding="utf-8"))
            assert rows, seg_rel
            for row in rows:
                cs, ce = int(row["char_start"]), int(row["char_end"])
                assert text[cs:ce] == row["text"], f"{seg_rel}: offsets do not describe the body"


class TestItDoesNothingWhenItShouldNot:
    def test_an_english_episode_is_a_no_op(self, tmp_path: Path) -> None:
        _lay_down(tmp_path)
        before = (tmp_path / REL).read_text(encoding="utf-8")
        got = run_naming_stage(
            _cfg(tmp_path, language="en"),
            transcript_relpath=REL,
            effective_output_dir=str(tmp_path),
        )
        assert got.status == STATUS_NOT_DEFERRED
        assert got.ran is False
        assert (tmp_path / REL).read_text(encoding="utf-8") == before

    def test_a_missing_english_render_leaves_the_labels_alone(self, tmp_path: Path) -> None:
        """The §5.3 gate already stops the analysis stages here. Naming simply has nothing to
        read, and the episode keeps anonymous labels rather than getting Spanish-derived ones."""
        d = tmp_path / "transcripts"
        d.mkdir(parents=True)
        (tmp_path / REL).write_text("SPEAKER_00: Hola.\n", encoding="utf-8")
        (d / "01 - ep.segments.json").write_text(
            json.dumps(_rows(_ES, with_speaker=True), indent=2), encoding="utf-8"
        )
        got = run_naming_stage(
            _cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert got.status == STATUS_NO_ENGLISH
        assert (tmp_path / REL).read_text(encoding="utf-8") == "SPEAKER_00: Hola.\n"

    def test_undiarized_source_segments_are_refused(self, tmp_path: Path) -> None:
        _lay_down(tmp_path)
        (tmp_path / "transcripts" / "01 - ep.segments.json").write_text(
            json.dumps(_rows(_ES, with_speaker=False), indent=2), encoding="utf-8"
        )
        got = run_naming_stage(
            _cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert got.status == STATUS_NO_VOICES

    def test_running_TWICE_does_not_re_attribute(self, tmp_path: Path) -> None:
        """The second pass sees NAMES in the English label position, not voice ids. Aligning
        those would attribute an already-named voice to a voice called "Dana Reyes"."""
        _lay_down(tmp_path)
        first = run_naming_stage(
            _cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert first.status == STATUS_NAMED
        after_first = (tmp_path / REL).read_text(encoding="utf-8")

        second = run_naming_stage(
            _cfg(tmp_path), transcript_relpath=REL, effective_output_dir=str(tmp_path)
        )
        assert second.status in (STATUS_NO_VOICES, STATUS_UNRESOLVED), second.reason
        assert (tmp_path / REL).read_text(encoding="utf-8") == after_first


class TestThePureHelpers:
    def test_the_diarization_comes_from_the_SOURCE_times(self) -> None:
        """The roster weighs talk time, and the English cues' durations are the source
        sentences' only because the renderer copies them. Deriving timing from the English side
        would make "who talked longest" a fact about the translation."""
        dz = diarization_from_segments(_rows(_ES, with_speaker=True))
        assert dz is not None
        assert dz.num_speakers == 2
        assert [s.speaker for s in dz.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        assert dz.segments[0].end == 30.0

    def test_no_speaker_field_means_no_diarization(self) -> None:
        assert diarization_from_segments(_rows(_ES, with_speaker=False)) is None

    def test_the_alignment_only_trusts_anonymous_labels(self) -> None:
        rows = _rows(_EN, with_speaker=False)
        rows[0]["speaker_label"] = "Dana Reyes"
        aligned = align_english_to_voices(rows)
        assert [v for _s, v in aligned] == ["SPEAKER_01", "SPEAKER_00"]

    def test_relabel_keys_on_the_frozen_voice_id_when_present(self) -> None:
        rows = _rows(_ES, with_speaker=True)
        out, changed = relabel_segments(rows, {"SPEAKER_00": "Dana Reyes"})
        assert changed == 2
        assert [r["speaker_label"] for r in out] == ["Dana Reyes", "SPEAKER_01", "Dana Reyes"]
        assert [r["speaker"] for r in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

    def test_relabel_falls_back_to_the_label_for_the_english_cues(self) -> None:
        """The English cues carry only `speaker_label`, so that is the key there."""
        rows = _rows(_EN, with_speaker=False)
        out, changed = relabel_segments(rows, {"SPEAKER_01": "Marcus Webb"})
        assert changed == 1
        assert out[1]["speaker_label"] == "Marcus Webb"


class TestBothStagesResolveTheLanguageTheSameWay:
    """Found by the FIRST REAL translation run (2026-09-30), not by any test.

    `run_translation_stage` accepted a `feed_language` argument and `run_naming_stage` did not,
    so the only way naming could see the feed's tag was `cfg.feed_declared_language`. Production
    sets that field in `run_pipeline`, so the two agreed there — but a caller that passed the tag
    to translation without it being on the config got:

        TRANSLATION: translated  source_language=es (rss)  units=47 failed=0  english_ready=True
        NAMING:      not_deferred  voices=0  renamed={}

    A complete English set, and an episode left with anonymous labels. No guest, no SPOKEN_BY
    edge, no position-bearing insights — the exact failure D-34 exists to prevent, reached by a
    different route. And SILENT, because `not_deferred` is a legitimate state meaning "naming
    already ran inside diarization".

    This is the second time in this arc that one question answered through two channels has bitten
    (the first: `transcription_language` vs the metadata writer, #2172). Both stages resolve
    through `resolve_config_language` from the same inputs now.
    """

    def test_both_stages_accept_the_feeds_language(self) -> None:
        import inspect

        from podcast_scraper.workflow.naming_stage import run_naming_stage
        from podcast_scraper.workflow.translation_stage import run_translation_stage

        for fn in (run_translation_stage, run_naming_stage):
            assert "feed_language" in inspect.signature(fn).parameters, fn.__name__

    def test_the_tag_alone_is_enough_to_defer(self, tmp_path: Path) -> None:
        """The config carries the profile default `en` and nothing else — exactly the shape the
        real run had. The tag must be sufficient."""
        from podcast_scraper.workflow.naming_stage import naming_is_deferred

        cfg = config.Config(
            rss="https://e.com/f.xml",
            language="en",
            translate_api_base="http://translator.invalid:8005/v1",
            translate_model="google/translategemma-12b-it",
        )
        assert naming_is_deferred(cfg) is False, "control: nothing says this is Spanish"
        assert naming_is_deferred(cfg, feed_language="es-ES") is True

    def test_the_two_stages_agree_on_every_shape(self, tmp_path: Path) -> None:
        """The property that matters, asserted directly: for the same inputs, "translation is
        owed" and "naming is deferred" must never disagree."""
        from podcast_scraper.workflow.naming_stage import naming_is_deferred
        from podcast_scraper.workflow.translation_stage import decide_translation

        for language, feed, translator in (
            ("en", None, True),
            ("en", "es-ES", True),
            ("es", None, True),
            ("en", "es-ES", False),
            ("en", "en-US", True),
        ):
            kw: dict[str, Any] = {"rss": "https://e.com/f.xml", "language": language}
            if translator:
                kw["translate_api_base"] = "http://translator.invalid:8005/v1"
                kw["translate_model"] = "google/translategemma-12b-it"
            cfg = config.Config(**kw)  # type: ignore[arg-type]
            owed = (
                decide_translation(cfg, feed_language=feed, transcript_relpath=REL).status
                == "pending"
            )
            deferred = naming_is_deferred(cfg, feed_language=feed)
            assert owed == deferred, (
                f"language={language} feed={feed} translator={translator}: translation "
                f"owed={owed} but naming deferred={deferred} — one of them will run against "
                "the wrong assumption"
            )

    def test_the_seam_passes_it_to_BOTH(self) -> None:
        """Asserted on the source, because the seam is inside a 200-line function that also
        writes metadata. It is the line whose absence produced the real-run failure."""
        src = (
            Path(__file__).resolve().parents[3]
            / "src/podcast_scraper/workflow/metadata_generation.py"
        ).read_text(encoding="utf-8")
        assert (
            src.count('feed_language=getattr(feed, "language", None),') >= 2
        ), "translation and naming must both receive the feed's tag at the seam"
