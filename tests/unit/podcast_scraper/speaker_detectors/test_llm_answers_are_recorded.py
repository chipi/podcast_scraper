"""Both LLM answers are on record in full, refused parts included (#2276).

The sidecar used to keep only the names that SURVIVED: the resolver's verdicts were reduced to the
accepted ones, and detection's raw answer, its filter drops and corroboration's refusals were log
lines. A refused proposal is exactly what an audit of the model needs, so each is now recorded with
what became of it. Recording is pure observation: the answers returned are unchanged.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.speaker_detectors.corroboration import corroborate_guests
from podcast_scraper.speaker_detectors.resolution import resolve_voices_and_roles
from podcast_scraper.workflow.stages import processing

pytestmark = pytest.mark.unit

HOST, GUEST = "Tobias Wren", "Maria Lindqvist"
VOICES = {
    "SPEAKER_00": f"Hello and welcome, I'm {HOST}. Today I talk with {GUEST} about ports.",
    "SPEAKER_01": "Thanks for having me. The merchant ledgers run for three centuries.",
}


def _answer(voices: Dict[str, Any]) -> str:
    return "<think>reasoning about who is who</think>" + json.dumps({"voices": voices})


def _resolve(raw: str, voices: Optional[Dict[str, str]] = None, **kw: Any) -> Tuple[Any, Dict]:
    report: Dict[str, Any] = {}
    out = resolve_voices_and_roles(
        [HOST, GUEST], voices or VOICES, lambda _p: raw, known_hosts=[HOST], report=report, **kw
    )
    plain = resolve_voices_and_roles(
        [HOST, GUEST], voices or VOICES, lambda _p: raw, known_hosts=[HOST], **kw
    )
    assert out == plain, "recording the answer changed the answer"
    return out, report


class TestTheResolversFullAnswerIsRecorded:
    def test_the_raw_text_and_every_verdict_are_kept(self) -> None:
        raw = _answer(
            {
                "SPEAKER_00": {"name": HOST, "role": "host"},
                "SPEAKER_01": {"name": GUEST, "role": "guest"},
            }
        )
        out, rep = _resolve(raw)
        assert out["SPEAKER_01"].name == GUEST
        assert rep["raw"] == raw and rep["raw_chars"] == len(raw)
        assert rep["stated_names"] == [HOST, GUEST] and rep["prompt_chars"] > 0
        outcomes = {v["said_voice"]: v["outcome"] for v in rep["verdicts"]}
        assert outcomes == {"SPEAKER_00": "accepted", "SPEAKER_01": "accepted"}

    def test_an_invented_name_is_on_record_though_it_names_nobody(self) -> None:
        out, rep = _resolve(_answer({"SPEAKER_01": {"name": "Elon Musk", "role": "guest"}}))
        assert out["SPEAKER_01"].name is None
        [v] = rep["verdicts"]
        assert v == {
            "said_voice": "SPEAKER_01",
            "voice": "SPEAKER_01",
            "name": "Elon Musk",
            "role": "guest",
            "matched": None,
            "outcome": "invented",
        }

    def test_a_third_person_name_and_the_complement_it_triggers_are_recorded(self) -> None:
        # The model puts the guest's name on the host, who only talks ABOUT her.
        out, rep = _resolve(_answer({"SPEAKER_00": {"name": GUEST, "role": "host"}}))
        [v] = rep["verdicts"]
        assert v["outcome"] == "third_person" and v["matched"] == GUEST
        assert rep["complement"] == [{"kind": "two_voice", "name": GUEST, "voice": "SPEAKER_01"}]
        assert out["SPEAKER_01"].name == GUEST

    def test_a_duplicate_an_unmapped_voice_and_a_role_only_answer(self) -> None:
        voices = {**VOICES, "SPEAKER_02": "And I am also here, briefly."}
        _, rep = _resolve(
            _answer(
                {
                    "SPEAKER_01": {"name": GUEST, "role": "guest"},
                    "SPEAKER_02": {"name": GUEST, "role": None},
                    "SPEAKER_09": {"name": HOST, "role": "host"},
                    "SPEAKER_00": {"name": None, "role": "host"},
                }
            ),
            voices,
        )
        outcomes = {v["said_voice"]: v["outcome"] for v in rep["verdicts"]}
        assert outcomes == {
            "SPEAKER_01": "accepted",
            "SPEAKER_02": "duplicate",
            "SPEAKER_09": "unmapped_voice",
            "SPEAKER_00": "role_only",
        }

    def test_a_failed_call_is_recorded_as_the_error(self) -> None:
        def boom(_p: str) -> str:
            raise TimeoutError("vLLM did not answer")

        report: Dict[str, Any] = {}
        assert resolve_voices_and_roles([HOST], VOICES, boom, report=report) == {}
        assert report["error"] == "TimeoutError: vLLM did not answer"

    def test_a_call_that_never_happens_says_why(self) -> None:
        report: Dict[str, Any] = {}
        resolve_voices_and_roles([], VOICES, lambda _p: "{}", report=report)
        assert report == {"skipped": "no_candidates_and_no_role_context"}

    def test_a_long_reasoning_preamble_is_capped_but_its_length_kept(self) -> None:
        raw = "<think>" + "x" * 40_000 + "</think>" + json.dumps({"voices": {}})
        _, rep = _resolve(raw)
        assert len(rep["raw"]) == 16_000 and rep["raw_chars"] == len(raw)


class TestCorroborationSaysWhyItRefused:
    def test_each_dropped_name_carries_its_reason_and_the_result_is_unchanged(self) -> None:
        refused: List[Dict[str, str]] = []
        kw: Dict[str, Any] = {
            "episode_title": f"The ports of the Baltic, with {GUEST}",
            "episode_description": f"{GUEST} joins us. We also discuss Ada Quill's old book.",
            "known_hosts": {HOST},
        }
        kept = corroborate_guests([GUEST, "Ada Quill", HOST], rejected_out=refused, **kw)
        assert kept == corroborate_guests([GUEST, "Ada Quill", HOST], **kw)
        assert kept == [GUEST]
        assert refused == [
            {"name": "Ada Quill", "reason": "no_interview_cue"},
            {"name": HOST, "reason": "is_a_host"},
        ]


class _Detector:
    def __init__(self, guests: List[str], hosts: Set[str], raw: Optional[str]) -> None:
        self._guests, self._hosts, self._raw = guests, hosts, raw

    def detect_speakers(
        self, *, episode_title: str, episode_description: str, known_hosts: Set[str]
    ) -> Tuple[List[str], Set[str], bool, bool]:
        if self._raw is not None:
            self.last_speaker_detection_raw = self._raw  # what a provider's parser stores
        return list(self._guests), set(self._hosts), bool(self._guests or self._hosts), False


class _Cfg:
    auto_speakers = True
    known_hosts: List[str] = []
    cache_detected_hosts = False
    screenplay_speaker_names: List[str] = []
    speaker_detector_provider = "openai"
    dry_run = False


class _Episode:
    idx = 1
    title = f"The ports of the Baltic, with {GUEST}"
    item = object()
    speaker_detection_report: Optional[Dict[str, Any]] = None


def _detect(detector: _Detector, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    monkeypatch.setattr(
        processing,
        "extract_episode_description",
        lambda _item: f"{GUEST} joins {HOST}. We also discuss Ada Quill's old book.",
    )
    ep = _Episode()
    hd = processing.HostDetectionResult(set(), {}, detector)
    processing._detect_speakers_for_episode(ep, _Cfg(), hd, None)  # type: ignore[arg-type]
    assert ep.speaker_detection_report is not None
    return ep.speaker_detection_report


class TestDetectionIsRecordedOnTheEpisode:
    def test_the_raw_answer_the_drops_and_the_refusals(self, monkeypatch: pytest.MonkeyPatch):
        raw = json.dumps({"hosts": [HOST], "guests": [GUEST, "Ada Quill", "Host"]})
        rep = _detect(_Detector([GUEST, "Ada Quill", "Host"], {HOST}, raw), monkeypatch)
        assert rep["detector"] == "_Detector"
        assert rep["raw"] == raw and rep["raw_chars"] == len(raw)
        assert rep["returned"] == {
            "speakers": [GUEST, "Ada Quill", "Host"],
            "hosts": [HOST],
            "succeeded": True,
        }
        assert rep["dropped_placeholders"] == ["Host"]
        assert rep["proposed_guests"] == [GUEST, "Ada Quill"]
        assert rep["corroborated_guests"] == [GUEST]
        assert rep["corroboration_rejected"] == [
            {"name": "Ada Quill", "reason": "no_interview_cue"}
        ]
        assert rep["outcome"] == "ran"

    def test_a_previous_episodes_raw_answer_never_stands_in(self, monkeypatch: pytest.MonkeyPatch):
        det = _Detector([GUEST], set(), raw=None)
        stale = "the previous episode's answer"
        det.last_speaker_detection_raw = stale  # type: ignore[attr-defined]
        rep = _detect(det, monkeypatch)
        assert "raw" not in rep

    def test_a_skipped_detection_is_recorded_as_skipped(self, monkeypatch: pytest.MonkeyPatch):
        cfg = _Cfg()
        cfg.auto_speakers = False  # type: ignore[misc]
        ep = _Episode()
        hd = processing.HostDetectionResult(set(), {}, _Detector([], set(), None))
        processing._detect_speakers_for_episode(ep, cfg, hd, None)  # type: ignore[arg-type]
        assert ep.speaker_detection_report == {
            "outcome": "skipped",
            "reason": "auto_speakers_disabled",
        }


def test_both_records_reach_the_decision_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The pipeline puts the resolver's report and the episode's detection record in the trace."""
    from podcast_scraper.config import Config

    def fake_llm(*_a: Any, report: Optional[Dict[str, Any]] = None, **_k: Any) -> Any:
        assert report is not None, "the pipeline must hand the resolver a report to fill"
        report.update({"raw": "the model said", "verdicts": [{"outcome": "invented"}]})
        return {}, {}

    monkeypatch.setattr(P, "_resolve_voices_via_llm", fake_llm)
    P._copresence_cache.clear()
    detection = {"outcome": "ran", "raw": '{"guests": []}'}
    out = P.apply_diarization_to_result(
        {"text": "Hi.", "segments": [{"start": 0, "end": 30, "text": f"Hi, I'm {HOST}."}]},
        "",
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False),
        [],
        precomputed_diarization=DiarizationResult(
            segments=[DiarizationSegment(0, 30, "SPEAKER_00")], num_speakers=1
        ),
        detection_report=detection,
    )
    inputs = out["speaker_diagnostics"]["decision_trace"]["inputs"]
    assert inputs["llm_resolution"] == {
        "raw": "the model said",
        "verdicts": [{"outcome": "invented"}],
    }
    assert inputs["detection"] == detection


def test_the_pipeline_says_why_the_resolver_was_not_asked(tmp_path: Path) -> None:
    report: Dict[str, Any] = {}
    from podcast_scraper.config import Config

    P._resolve_voices_via_llm(
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False),
        stated_names=[HOST],
        voice_texts=VOICES,
        known_hosts=[HOST],
        ordered_turns=[],
        report=report,
    )
    assert report == {"skipped": "speaker_resolution_llm_off"}


def test_a_person_the_detector_names_only_as_a_speaker_is_stated_never_a_guest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2276 problem 14: stated (a resolver candidate, counted when unplaced), never corroborated —
    so no count-based placement can paint them on a voice."""
    det = _Detector([GUEST], set(), raw="{}")
    det.last_speaker_detection_stated_only = ["Cleo Marsh"]  # type: ignore[attr-defined]

    def detect(**kw: Any) -> Tuple[List[str], Set[str], bool, bool]:
        det.last_speaker_detection_stated_only = ["Cleo Marsh"]  # type: ignore[attr-defined]
        return [GUEST], set(), True, False

    det.detect_speakers = detect  # type: ignore[method-assign]
    monkeypatch.setattr(
        processing, "extract_episode_description", lambda _item: f"{GUEST} joins {HOST}."
    )
    ep = _Episode()
    hd = processing.HostDetectionResult(set(), {}, det)
    out = processing._detect_speakers_for_episode(ep, _Cfg(), hd, None)  # type: ignore[arg-type]
    assert out is not None
    assert "Cleo Marsh" in out.stated and "Cleo Marsh" not in out.guests
    assert ep.speaker_detection_report is not None
    assert ep.speaker_detection_report["stated_only"] == ["Cleo Marsh"]
