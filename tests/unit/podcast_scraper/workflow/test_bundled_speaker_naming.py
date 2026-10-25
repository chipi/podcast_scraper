"""Transcript naming inside the bundled calls: the field, its parse, its mapping and its sidecar.

Off (or no speaker labels) must leave every bundled prompt byte-identical; on, the model judges
each transcript label, the answer maps back to voices through the speaker diagnostics, and it is
recorded beside the ladder without changing what is published. Synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from podcast_scraper.prompting.megabundle import (
    build_extraction_bundle_prompt,
    build_megabundle_prompt,
)
from podcast_scraper.providers.common.megabundle_parser import (
    parse_extraction_bundle_response,
    parse_megabundle_response,
)
from podcast_scraper.providers.ml.diarization import transcript_naming as tn
from podcast_scraper.workflow import bundled_speaker_naming as bsn

pytestmark = pytest.mark.unit

TRANSCRIPT = (
    "Tobias Wren: Welcome to the show.\nSPEAKER_01: Thanks for having me, I'm Maria Lindqvist."
)
LABELS = ["SPEAKER_01", "Tobias Wren"]


def _diag() -> Dict[str, Any]:
    return {
        "voices": [
            {
                "voice": "SPEAKER_00",
                "resolved_name": "Tobias Wren",
                "named": True,
                "role": "host",
                "source": "self_intro",
                "talk_time_s": 600,
            },
            {
                "voice": "SPEAKER_01",
                "resolved_name": "SPEAKER_01",
                "named": False,
                "role": "guest",
                "source": "raw",
                "talk_time_s": 500,
            },
            {
                "voice": "SPEAKER_02",
                "resolved_name": "SPEAKER_02",
                "named": False,
                "role": "unknown",
                "source": "raw",
                "talk_time_s": 5,
            },
        ]
    }


SPEAKERS = {
    "Tobias Wren": {"name": "Tobias Wren", "role": "host", "confidence": "high", "evidence": ""},
    "SPEAKER_01": {
        "name": "Maria Lindqvist",
        "role": "guest",
        "confidence": "high",
        "evidence": "I'm Maria Lindqvist",
    },
    "SPEAKER_02": {"name": "Ad Person", "role": "guest", "confidence": "low", "evidence": ""},
}


# --- prompts -------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [build_megabundle_prompt, build_extraction_bundle_prompt])
def test_no_labels_leaves_the_bundle_prompt_byte_identical(build: Any) -> None:
    assert build(TRANSCRIPT) == build(TRANSCRIPT, speaker_labels=None)
    assert build(TRANSCRIPT) == build(TRANSCRIPT, speaker_labels=[])


@pytest.mark.parametrize("build", [build_megabundle_prompt, build_extraction_bundle_prompt])
def test_labels_add_the_speakers_field_before_the_transcript(build: Any) -> None:
    _, user = build(TRANSCRIPT, speaker_labels=LABELS)
    assert "SPEAKER LABELS TO JUDGE: SPEAKER_01, Tobias Wren" in user
    assert user.index('Also return "speakers"') < user.index("Transcript:\n---")
    assert "A wrong name is worse than a missing one" in user


def test_the_bundled_summary_prompt_gets_the_field_only_when_asked() -> None:
    assert tn.with_bundled_field("USER", None) == "USER"
    assert tn.with_bundled_field("USER", {"other": 1}) == "USER"
    out = tn.with_bundled_field("USER", {tn.PARAMS_LABELS: LABELS})
    assert out.startswith('USER\n\nAlso return "speakers"')
    assert out.endswith("SPEAKER LABELS TO JUDGE: SPEAKER_01, Tobias Wren")


def test_the_providers_read_the_labels_under_the_shared_key() -> None:
    # Providers read params["speaker_naming_labels"] without importing transcript_naming.
    assert tn.PARAMS_LABELS == "speaker_naming_labels"


# --- parse ---------------------------------------------------------------------------------


def _mega(extra: Dict[str, Any]) -> str:
    body = {
        "title": "T",
        "summary": "S",
        "bullets": ["b"],
        "insights": [{"text": "x", "insight_type": "claim"}],
        "topics": ["soil"],
        "entities": [],
    }
    body.update(extra)
    return json.dumps(body)


def test_the_bundle_parsers_keep_the_speakers_field() -> None:
    assert parse_megabundle_response(_mega({"speakers": SPEAKERS})).speakers == SPEAKERS
    assert parse_extraction_bundle_response(_mega({"speakers": SPEAKERS})).speakers == SPEAKERS
    assert parse_megabundle_response(_mega({})).speakers is None
    assert parse_megabundle_response(_mega({"speakers": "no"})).speakers is None


# --- mapping and the sidecar ---------------------------------------------------------------


def test_labels_map_back_to_voices_and_a_shared_label_is_left_out() -> None:
    diag = _diag()
    diag["voices"].append(dict(diag["voices"][0], voice="SPEAKER_03"))  # Tobias Wren twice
    label_map = tn.labels_from_diagnostics(diag)
    assert label_map["Tobias Wren"] == ["SPEAKER_00", "SPEAKER_03"]
    answer = tn.answer_by_voice(SPEAKERS, label_map)
    assert answer is not None and "SPEAKER_00" not in answer and "SPEAKER_03" not in answer
    assert answer["SPEAKER_01"]["name"] == "Maria Lindqvist"
    assert tn.answer_by_voice("not an object", label_map) is None


def test_the_bundled_sidecar_compares_with_the_floor_and_the_publish_gate() -> None:
    side = tn.bundled_sidecar(
        "shadow", "mega_bundled", SPEAKERS, _diag(), floor=0.05, renames=None, language="en"
    )
    assert side["call"] == "mega_bundled" and side["mode"] == "shadow" and side["ok"]
    v = side["voices"]
    assert v["SPEAKER_00"]["outcome"] == "agree"
    assert v["SPEAKER_01"]["model_result"]["name"] == "Maria Lindqvist"
    assert v["SPEAKER_01"]["outcome"] == "adds"
    # 5 s of 1,105: under the 5% floor, so the model's name does not count.
    assert v["SPEAKER_02"]["model_result"]["name"] is None
    assert side["outcomes"] == {"agree": 2, "adds": 1, "removes": 0, "changes": 0}
    bad = tn.bundled_sidecar(
        "shadow", "bundled", None, _diag(), floor=0, renames=None, language="en"
    )
    assert bad["ok"] is False and bad["report"]["error"] == "no readable speakers field"


# --- the metadata stage's side ---------------------------------------------------------------


def _episode(tmp_path: Path) -> str:
    rel = "transcripts/0001 - ep.txt"
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "0001 - ep.speakers.diagnostics.json").write_text(
        json.dumps(_diag()), encoding="utf-8"
    )
    return rel


@pytest.mark.parametrize(
    "naming, mode, asked",
    [
        ("off", "mega_bundled", False),
        ("shadow", "staged", False),
        ("shadow", "bundled", True),
        ("shadow", "mega_bundled", True),
        ("publish", "extraction_bundled", True),
    ],
)
def test_labels_are_asked_only_in_a_bundled_mode_with_naming_on(
    tmp_path: Path, naming: str, mode: str, asked: bool
) -> None:
    rel = _episode(tmp_path)
    got = bsn.labels_for_call(SimpleNamespace(transcript_naming=naming), mode, str(tmp_path), rel)
    assert (got == ["SPEAKER_01", "SPEAKER_02", "Tobias Wren"]) is asked
    if not asked:
        assert got is None


def test_an_episode_without_a_sidecar_asks_nothing(tmp_path: Path) -> None:
    cfg = SimpleNamespace(transcript_naming="shadow")
    assert bsn.labels_for_call(cfg, "bundled", str(tmp_path), "transcripts/none.txt") is None


def test_the_speakers_field_is_taken_out_of_the_summary_json() -> None:
    result: Dict[str, Any] = {
        "summary": json.dumps(
            {"title": "T", "summary": "S", "bullets": ["b"], "speakers": SPEAKERS}
        ),
        "metadata": {},
    }
    assert bsn.take_speakers(result) == SPEAKERS
    assert "speakers" not in json.loads(result["summary"])
    assert bsn.take_speakers(result) is None
    assert bsn.take_speakers({"summary": "plain prose"}) is None


def test_record_writes_the_key_beside_the_ladder_and_marks_it_not_applied(tmp_path: Path) -> None:
    rel = _episode(tmp_path)
    cfg = SimpleNamespace(transcript_naming="publish", speaker_name_min_talk_share=0.05)
    bsn.record(cfg, str(tmp_path), rel, "extraction_bundled", SPEAKERS)
    diag = json.loads(
        (tmp_path / "transcripts" / "0001 - ep.speakers.diagnostics.json").read_text()
    )
    assert diag["voices"] == _diag()["voices"]
    side = diag[tn.SIDECAR_KEY]
    assert side["call"] == "extraction_bundled" and side["mode"] == "publish"
    assert side["applied"] is False
