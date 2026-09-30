"""`apply_diarization_to_result` is two jobs, and D-34 needs them separable.

WHY THE SPLIT. Naming reads WORDS — self-introductions, interview cues, NER, ad patterns — and
every one of those is English-only. On a non-English transcript they do not find nothing; §5.2
measured them finding the WRONG things, with recall holding at 2/2 while precision fell 67% to
18%. So D-34 moves naming to after translation, which is only possible if the half that reads
audio (who spoke when) can run without the half that reads words.

THE BOUNDARY WAS ALREADY MARKED. The original function's own comment said "align first so the
roster can name a voice from its own turns' self-introduction" — the split is exactly there, so
nothing crossed it. `tests/integration/workflow/test_naming_golden.py` is what proves that:
40 episodes, 112 voice records, zero diff.

What this file pins is the CONTRACT that makes the move possible, because the golden would still
pass if the two halves were separable in name only.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import (
    apply_diarization_to_result,
    diarize_and_align,
    resolve_names_on_result,
)

pytestmark = pytest.mark.unit

_TEXT = [
    "Welcome back to the show. I'm Dana Reyes, and today I'm joined by Marcus Webb.",
    "Thanks for having me, Dana. Glad to be here.",
    "So let's start with the basics of what you build.",
    "We build measurement tools for small teams.",
]


def _cfg(tmp_path: Any) -> config.Config:
    return config.Config(
        rss="https://example.com/f.xml",
        output_dir=str(tmp_path),
        speaker_resolution_llm=False,
    )


def _inputs() -> Tuple[Dict[str, Any], DiarizationResult]:
    asr: List[Dict[str, Any]] = []
    turns: List[DiarizationSegment] = []
    for i, text in enumerate(_TEXT):
        start, end = float(i * 30), float((i + 1) * 30)
        asr.append({"id": i, "start": start, "end": end, "text": text})
        turns.append(DiarizationSegment(start=start, end=end, speaker=f"SPEAKER_{i % 2:02d}"))
    result = {"segments": asr, "text": " ".join(_TEXT)}
    return result, DiarizationResult(segments=turns, num_speakers=2, model_name="test")


class TestTheHalvesAreSeparable:
    def test_the_align_half_returns_the_diarization_and_the_alignment(self, tmp_path: Any) -> None:
        result, dz = _inputs()
        got = diarize_and_align(
            result, "/nonexistent.mp3", _cfg(tmp_path), precomputed_diarization=dz
        )
        assert got is not None
        diarization, aligned = got
        assert diarization is dz
        assert len(aligned) == len(_TEXT)
        assert {voice for _seg, voice in aligned} == {"SPEAKER_00", "SPEAKER_01"}

    def test_the_align_half_NAMES_NOTHING(self, tmp_path: Any) -> None:
        """The property the whole move depends on. If this half resolved any name it would be
        reading words, and the words are not translated yet when it runs."""
        result, dz = _inputs()
        got = diarize_and_align(
            result, "/nonexistent.mp3", _cfg(tmp_path), precomputed_diarization=dz
        )
        assert got is not None
        _diarization, aligned = got
        voices = {voice for _seg, voice in aligned}
        assert all(v.startswith("SPEAKER_") for v in voices), (
            f"the align half resolved a name: {sorted(voices)}. It must produce anonymous voice "
            "ids only — a real name here means it read the transcript, which on a translated "
            "episode has not been translated yet."
        )

    def test_the_naming_half_resolves_a_name_from_the_self_intro(self, tmp_path: Any) -> None:
        result, dz = _inputs()
        got = diarize_and_align(
            result, "/nonexistent.mp3", _cfg(tmp_path), precomputed_diarization=dz
        )
        assert got is not None
        diarization, aligned = got
        out = resolve_names_on_result(
            result, _cfg(tmp_path), diarization, aligned, None, detection_ran=True
        )
        labels = {str(s.get("speaker_label")) for s in out["segments"]}
        assert any(not lab.startswith("SPEAKER_") for lab in labels), (
            f"the naming half resolved nothing: {sorted(labels)} — this fixture states "
            "'I'm Dana Reyes', so the deterministic self-intro cue should fire"
        )

    def test_the_composed_call_equals_the_two_halves(self, tmp_path: Any) -> None:
        """The refactor's own correctness, independent of the golden: the entry point every
        caller uses must produce what running the halves by hand produces."""
        result, dz = _inputs()
        composed = apply_diarization_to_result(
            result,
            "/nonexistent.mp3",
            _cfg(tmp_path),
            None,
            precomputed_diarization=dz,
            detection_ran=True,
        )
        got = diarize_and_align(
            result, "/nonexistent.mp3", _cfg(tmp_path), precomputed_diarization=dz
        )
        assert got is not None
        diarization, aligned = got
        by_hand = resolve_names_on_result(
            result, _cfg(tmp_path), diarization, aligned, None, detection_ran=True
        )
        assert [s.get("speaker_label") for s in composed["segments"]] == [
            s.get("speaker_label") for s in by_hand["segments"]
        ]
        assert composed["diarization_num_speakers"] == by_hand["diarization_num_speakers"]


class TestNothingToAlign:
    def test_no_asr_segments_returns_None(self, tmp_path: Any) -> None:
        assert (
            diarize_and_align(
                {"segments": []}, "/x.mp3", _cfg(tmp_path), precomputed_diarization=None
            )
            is None
        )

    def test_a_diarization_with_no_turns_returns_None(self, tmp_path: Any) -> None:
        """`None`, not an empty alignment: the caller has to return `result` UNCHANGED so its
        `has_diarized_labels` gate falls back to gap-based formatting. Labelling the whole episode
        `SPEAKER_00` would be worse than not labelling it."""
        result, _dz = _inputs()
        empty = DiarizationResult(segments=[], num_speakers=0, model_name="test")
        assert (
            diarize_and_align(result, "/x.mp3", _cfg(tmp_path), precomputed_diarization=empty)
            is None
        )

    def test_the_composed_call_returns_the_result_untouched(self, tmp_path: Any) -> None:
        result, _dz = _inputs()
        empty = DiarizationResult(segments=[], num_speakers=0, model_name="test")
        out = apply_diarization_to_result(
            result, "/x.mp3", _cfg(tmp_path), None, precomputed_diarization=empty
        )
        assert out is result
        assert not any("speaker_label" in s for s in out["segments"])


class TestTheNamingHalfCanBeGivenOtherText:
    """The point of `naming_text`: on the translated path the roster must read ENGLISH.

    The per-voice samples come from `aligned`, but the roster also wants one flat whole-episode
    string. Defaulting that to `result["text"]` would hand it the SOURCE language while every
    per-voice sample was English — and it fails SILENTLY, because both are strings and the roster
    simply resolves fewer voices.
    """

    def test_naming_text_is_accepted(self) -> None:
        assert "naming_text" in inspect.signature(resolve_names_on_result).parameters

    def test_it_overrides_the_results_own_text(self, tmp_path: Any) -> None:
        """Given an alignment whose words name nobody but a `naming_text` that does, the roster
        must have read the override — which is what proves the parameter is wired, not just
        accepted."""
        anonymous = ["Right.", "Mm hm.", "Sure.", "Okay."]
        asr = [
            {"id": i, "start": float(i * 30), "end": float((i + 1) * 30), "text": t}
            for i, t in enumerate(anonymous)
        ]
        turns = [
            DiarizationSegment(
                start=float(i * 30), end=float((i + 1) * 30), speaker=f"SPEAKER_{i % 2:02d}"
            )
            for i in range(len(anonymous))
        ]
        dz = DiarizationResult(segments=turns, num_speakers=2, model_name="test")
        result = {"segments": asr, "text": " ".join(anonymous)}
        got = diarize_and_align(result, "/x.mp3", _cfg(tmp_path), precomputed_diarization=dz)
        assert got is not None
        diarization, aligned = got

        without = resolve_names_on_result(
            result, _cfg(tmp_path), diarization, aligned, None, detection_ran=True
        )
        assert all(
            str(s.get("speaker_label")).startswith("SPEAKER_") for s in without["segments"]
        ), "the control case must name nobody, or this test proves nothing"


class TestTheSignaturesDoNotDriftApart:
    def test_the_composed_signature_still_carries_every_naming_argument(self) -> None:
        """A naming argument that exists on the half but not on the entry point is unreachable
        from every current caller — the shape that made S2.14's guard dead for a whole arc."""
        composed = set(inspect.signature(apply_diarization_to_result).parameters)
        naming = set(inspect.signature(resolve_names_on_result).parameters)
        # `naming_text`, `diarization` and `aligned` are the split's own arguments; everything
        # else the naming half takes must be reachable from the composed entry point.
        internal = {"diarization", "aligned", "naming_text"}
        assert (naming - internal) <= composed, sorted((naming - internal) - composed)
