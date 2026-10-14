"""Unit tests for diarization alignment."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.alignment import (
    align_segments_to_speakers,
    carry_speaker_identity,
)
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = pytest.mark.unit


def test_align_segments_assigns_majority_overlap_speaker() -> None:
    whisper_segments = [
        {"start": 0.0, "end": 5.0, "text": "hello"},
        {"start": 5.0, "end": 10.0, "text": "world"},
    ]
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=6.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=6.0, end=10.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_00"
    assert aligned[1][1] == "SPEAKER_01"


def test_align_segments_carries_forward_when_no_overlap() -> None:
    whisper_segments = [{"start": 12.0, "end": 14.0, "text": "after gap"}]
    diarization = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=5.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_00"


def test_align_ignores_sub_epsilon_boundary_overlap() -> None:
    """A sub-microsecond boundary touch must not flip the speaker (A2 epsilon)."""
    whisper_segments = [{"start": 5.0, "end": 10.0, "text": "second"}]
    diarization = DiarizationResult(
        segments=[
            # SPEAKER_00 touches the segment start by 1e-7s — noise, not real speech.
            DiarizationSegment(start=0.0, end=5.0000001, speaker="SPEAKER_00"),
            DiarizationSegment(start=5.0, end=10.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    assert aligned[0][1] == "SPEAKER_01"


def test_align_tie_break_is_deterministic_and_prefers_continuity() -> None:
    """Equal overlaps resolve deterministically: continuity first, else lexicographic (A2)."""
    # First segment establishes SPEAKER_01 as the running speaker.
    whisper_segments = [
        {"start": 0.0, "end": 4.0, "text": "intro"},
        {"start": 4.0, "end": 8.0, "text": "tie"},
    ]
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=4.0, speaker="SPEAKER_01"),
            # Second whisper segment overlaps SPEAKER_00 and SPEAKER_01 equally (2.0s each).
            DiarizationSegment(start=4.0, end=6.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=6.0, end=8.0, speaker="SPEAKER_00"),
        ],
        num_speakers=2,
        model_name="test",
    )

    aligned = align_segments_to_speakers(whisper_segments, diarization)

    # Tie resolved to the previous speaker (continuity), not dict-insertion order.
    assert aligned[1][1] == "SPEAKER_01"
    # Determinism: same input, same output across repeated calls.
    assert align_segments_to_speakers(whisper_segments, diarization)[1][1] == "SPEAKER_01"


# --- carry_speaker_identity: re-transcribed audio keeps the voices it already had -----------------


def _ref(start, end, label, *, speaker=None, role=None, stated=None, voice_type=None):
    return {
        "start": start,
        "end": end,
        "text": "old",
        "speaker": speaker,
        "speaker_label": label,
        "speaker_role": role,
        "voice_type": voice_type,
        "stated_speaker": stated,
    }


def test_carry_identity_copies_every_identity_field_from_the_overlapped_reference() -> None:
    reference = [
        _ref(0, 10, "Alice Host", speaker="SPEAKER_00", role="host", voice_type="person"),
        _ref(10, 20, "Bob Guest", speaker="SPEAKER_01", role="guest", stated="Robert Guest"),
    ]
    new = [{"start": 1.0, "end": 9.0, "text": "a"}, {"start": 11.0, "end": 19.0, "text": "b"}]

    out = carry_speaker_identity(new, reference)

    assert out[0]["speaker_label"] == "Alice Host"
    assert (out[0]["speaker"], out[0]["speaker_role"]) == ("SPEAKER_00", "host")
    assert out[0]["voice_type"] == "person"
    assert (out[1]["speaker_label"], out[1]["stated_speaker"]) == ("Bob Guest", "Robert Guest")
    assert out[0]["text"] == "a" and out[1]["start"] == 11.0, "the new text and times are kept"


def test_carry_identity_works_when_the_corpus_only_kept_speaker_label() -> None:
    """A finished corpus stores ``speaker`` as None and the identity in ``speaker_label``; keying
    on ``speaker`` alone would have found no voice at all and labelled nothing."""
    reference = [_ref(0, 5, "Alice Host"), _ref(5, 10, "Bob Guest")]

    out = carry_speaker_identity([{"start": 6.0, "end": 9.0, "text": "x"}], reference)

    assert out[0]["speaker_label"] == "Bob Guest"
    assert "speaker" not in out[0], "an absent field stays absent rather than becoming None"


def test_carry_identity_gives_a_straddling_segment_the_majority_overlap() -> None:
    reference = [_ref(0, 10, "Alice Host"), _ref(10, 20, "Bob Guest")]

    out = carry_speaker_identity(
        [
            {"start": 6.0, "end": 12.0, "text": "mostly alice"},
            {"start": 8.0, "end": 16.0, "text": "mostly bob"},
        ],
        reference,
    )

    assert [o["speaker_label"] for o in out] == ["Alice Host", "Bob Guest"]


def test_carry_identity_ties_go_to_the_previous_segments_voice_then_the_earliest_voice() -> None:
    reference = [_ref(0, 10, "Bob Guest"), _ref(10, 20, "Carol Guest"), _ref(20, 30, "Bob Guest")]
    straddle = {"start": 15.0, "end": 25.0, "text": "five seconds of each"}

    alone = carry_speaker_identity([straddle], reference)
    after_carol = carry_speaker_identity(
        [{"start": 11.0, "end": 12.0, "text": "carol"}, straddle], reference
    )

    # With nobody before it the earliest reference voice wins (Bob); after Carol it stays Carol.
    assert alone[0]["speaker_label"] == "Bob Guest"
    assert after_carol[1]["speaker_label"] == "Carol Guest"
    assert carry_speaker_identity([straddle], reference) == alone, "deterministic"


def test_carry_identity_a_hole_takes_the_previous_new_segments_voice() -> None:
    """Speech the old decode dropped overlaps no reference segment: the speaker before takes it."""
    reference = [_ref(0, 5, "Alice Host"), _ref(40, 50, "Bob Guest")]

    out = carry_speaker_identity(
        [
            {"start": 1.0, "end": 4.0, "text": "inside alice"},
            {"start": 10.0, "end": 20.0, "text": "in the hole"},
            {"start": 41.0, "end": 49.0, "text": "inside bob"},
            {"start": 60.0, "end": 70.0, "text": "after the last reference"},
        ],
        reference,
    )

    assert [o["speaker_label"] for o in out] == [
        "Alice Host",
        "Alice Host",
        "Bob Guest",
        "Bob Guest",
    ]


def test_carry_identity_keeps_two_voices_apart_that_share_a_label_but_not_a_role() -> None:
    reference = [_ref(0, 10, "Sam", role="host"), _ref(10, 20, "Sam", role="guest")]

    out = carry_speaker_identity(
        [{"start": 2.0, "end": 4.0, "text": "a"}, {"start": 12.0, "end": 14.0, "text": "b"}],
        reference,
    )

    assert [o["speaker_role"] for o in out] == ["host", "guest"]


def test_carry_identity_returns_copies_in_the_input_order_and_does_not_mutate() -> None:
    reference = [_ref(0, 10, "Alice Host"), _ref(10, 20, "Bob Guest")]
    late = {"start": 15.0, "end": 18.0, "text": "late"}
    early = {"start": 1.0, "end": 3.0, "text": "early"}

    out = carry_speaker_identity([late, early], reference)

    assert [o["text"] for o in out] == [
        "late",
        "early",
    ], "alignment sorts internally, the caller's order stays"
    assert [o["speaker_label"] for o in out] == ["Bob Guest", "Alice Host"]
    assert "speaker_label" not in late and "speaker_label" not in early
    assert out[0] is not late


@pytest.mark.parametrize(
    "reference",
    [
        [],
        [{"start": 0, "end": 5, "text": "no identity at all"}],
        [{"speaker_label": "Alice Host", "text": "no times"}],
    ],
    ids=["empty", "no-identity", "no-times"],
)
def test_carry_identity_with_no_usable_reference_labels_nothing(reference) -> None:
    new = [{"start": 1.0, "end": 2.0, "text": "x"}]

    assert carry_speaker_identity(new, reference) == new
