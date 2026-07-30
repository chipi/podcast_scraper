"""#876: content.speakers + diarization_num_speakers derived from diarized segments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.workflow.metadata_generation import (
    _build_speakers_from_diarized_segments,
)

pytestmark = pytest.mark.unit


def _write_segments(root: Path, rel_txt: str, segments, suffix=".segments.json") -> None:
    base = root / rel_txt
    base.parent.mkdir(parents=True, exist_ok=True)
    (root / (rel_txt[: -len(".txt")] + suffix)).write_text(json.dumps(segments), encoding="utf-8")


def _seg(label, raw, text="x"):
    return {"start": 0.0, "end": 1.0, "text": text, "speaker_label": label, "speaker": raw}


def test_derives_host_and_guest_with_num_speakers(tmp_path: Path) -> None:
    rel = "transcripts/ep.txt"
    _write_segments(
        tmp_path,
        rel,
        [
            _seg("Patrick O'Shaughnessy", "SPEAKER_00"),
            _seg("Brian Chesky", "SPEAKER_01"),
            _seg("Patrick O'Shaughnessy", "SPEAKER_00"),
        ],
    )
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, ["Brian Chesky"])
    assert num == 2
    assert speakers is not None
    by_role = {(s.role, s.name) for s in speakers}
    # Host named from the segments even though it's NOT in detected_hosts (network feed).
    assert ("host", "Patrick O'Shaughnessy") in by_role
    assert ("guest", "Brian Chesky") in by_role


def test_roster_role_on_segments_wins_over_empty_detected_guests(tmp_path: Path) -> None:
    """The guest-as-host bug (The Journal / Karpathy / ~47% of the corpus).

    The roster labeled the guest correctly, but the reader defaulted every named voice to host
    because the pre-diarization ``detected_guests`` was empty. Now each segment carries the roster's
    authoritative ``speaker_role`` and it wins over the empty hint.
    """
    rel = "transcripts/ep.txt"
    segs = [
        {
            "start": 0,
            "end": 1,
            "text": "a",
            "speaker": "SPEAKER_05",
            "speaker_label": "Jessica Mendoza",
            "speaker_role": "host",
        },
        {
            "start": 1,
            "end": 2,
            "text": "b",
            "speaker": "SPEAKER_02",
            "speaker_label": "Ryan Knutson",
            "speaker_role": "host",
        },
        {
            "start": 2,
            "end": 3,
            "text": "c",
            "speaker": "SPEAKER_04",
            "speaker_label": "Benjamin Brundage",
            "speaker_role": "guest",
        },
    ]
    _write_segments(tmp_path, rel, segs)
    # detected_guests=[] is the exact observed input that used to make everyone a host.
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, [])
    assert num == 3
    by_role = {(s.role, s.name) for s in (speakers or [])}
    assert ("host", "Jessica Mendoza") in by_role
    assert ("host", "Ryan Knutson") in by_role
    assert ("guest", "Benjamin Brundage") in by_role  # the fix: guest stays guest
    assert ("host", "Benjamin Brundage") not in by_role


def test_roster_role_survives_adfree_build_end_to_end(tmp_path: Path) -> None:
    """F1 (advisor): the reader PREFERS the ad-free sidecar, so the role must survive the ad-free
    build — otherwise the guest-as-host bug resurfaces on real runs. End-to-end: build the ad-free
    segments via build_adfree_artifacts (as the pipeline does), write them, then read.
    """
    from podcast_scraper.providers.ml.diarization.formatting import (
        format_diarized_screenplay_with_offsets,
    )
    from podcast_scraper.workflow.adfree_transcript import build_adfree_artifacts

    raw = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "Welcome to the show.",
            "speaker": "SPEAKER_05",
            "speaker_label": "Jessica Mendoza",
            "speaker_role": "host",
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "Thanks for having me.",
            "speaker": "SPEAKER_04",
            "speaker_label": "Benjamin Brundage",
            "speaker_role": "guest",
        },
    ]
    text, _ = format_diarized_screenplay_with_offsets(raw)
    art = build_adfree_artifacts(text, raw)
    assert art is not None
    role_on_adfree = {s["speaker_label"]: s.get("speaker_role") for s in art.segments}
    assert role_on_adfree == {
        "Jessica Mendoza": "host",
        "Benjamin Brundage": "guest",
    }, "speaker_role must survive the ad-free build (the reader's preferred sidecar)"

    # The reader opens the ad-free sidecar preferentially — write ONLY that one, no roles elsewhere.
    rel = "transcripts/ep.txt"
    _write_segments(tmp_path, rel, art.segments, suffix=".adfree.segments.json")
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, [])
    by_role = {(s.role, s.name) for s in (speakers or [])}
    assert ("guest", "Benjamin Brundage") in by_role
    assert ("host", "Benjamin Brundage") not in by_role


def test_num_speakers_counts_int_zero_native_speaker(tmp_path: Path) -> None:
    """A native diarizer's speaker id can be int 0 — the old ``or`` chain dropped it, undercounting
    num_speakers. Segments with no names still return the correct voice count."""
    rel = "transcripts/ep.txt"
    _write_segments(
        tmp_path,
        rel,
        [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": 0},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": 1},
        ],
    )
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, [])
    assert num == 2  # not 1 — speaker 0 must be counted
    assert speakers is None  # unnamed segments -> no named roster (caller falls back)


def test_prefers_adfree_segments(tmp_path: Path) -> None:
    rel = "transcripts/ep.txt"
    # raw segments say one thing, ad-free says another → ad-free wins (#974 base)
    _write_segments(tmp_path, rel, [_seg("Wrong", "SPEAKER_00")], suffix=".segments.json")
    _write_segments(
        tmp_path,
        rel,
        [_seg("Maya", "SPEAKER_00"), _seg("Liam", "SPEAKER_01")],
        suffix=".adfree.segments.json",
    )
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, ["Liam"])
    assert num == 2
    names = {s.name for s in (speakers or [])}
    assert names == {"Maya", "Liam"}
    assert "Wrong" not in names


def test_unnamed_segments_keep_count_no_roster(tmp_path: Path) -> None:
    rel = "transcripts/ep.txt"
    _write_segments(
        tmp_path, rel, [_seg("SPEAKER_00", "SPEAKER_00"), _seg("SPEAKER_01", "SPEAKER_01")]
    )
    speakers, num = _build_speakers_from_diarized_segments(str(tmp_path), rel, [])
    assert num == 2
    assert speakers is None  # diarized but unnamed → caller falls back to detected names


def test_no_segments_returns_none(tmp_path: Path) -> None:
    assert _build_speakers_from_diarized_segments(str(tmp_path), "transcripts/x.txt", []) == (
        None,
        None,
    )
    assert _build_speakers_from_diarized_segments(str(tmp_path), None, []) == (None, None)


def test_panel_multiple_guests(tmp_path: Path) -> None:
    rel = "transcripts/ep.txt"
    _write_segments(
        tmp_path,
        rel,
        [
            _seg("Host Hank", "SPEAKER_00"),
            _seg("Guest A", "SPEAKER_01"),
            _seg("Guest B", "SPEAKER_02"),
        ],
    )
    speakers, num = _build_speakers_from_diarized_segments(
        str(tmp_path), rel, ["Guest A", "Guest B"]
    )
    assert num == 3
    roles = sorted((s.role, s.name) for s in (speakers or []))
    assert roles == [("guest", "Guest A"), ("guest", "Guest B"), ("host", "Host Hank")]
