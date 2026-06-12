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
