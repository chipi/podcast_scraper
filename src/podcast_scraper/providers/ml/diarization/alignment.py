"""Align Whisper transcription segments to diarization speaker IDs."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .base import DiarizationResult, DiarizationSegment

# Overlaps below this (seconds) are float-boundary noise, not real shared speech.
# Without it, a segment ending at 5.0000001 vs a turn starting at 5.0 yields a
# spurious ~1e-7s overlap that can flip the assigned speaker.
_OVERLAP_EPS = 1e-6


def _overlap_duration(seg_start: float, seg_end: float, dia_start: float, dia_end: float) -> float:
    return max(0.0, min(seg_end, dia_end) - max(seg_start, dia_start))


def align_segments_to_speakers(
    whisper_segments: List[dict],
    diarization: DiarizationResult,
) -> List[Tuple[dict, str]]:
    """Assign a diarization speaker ID to each Whisper segment by overlap."""
    if not whisper_segments:
        return []

    diar_segments = sorted(diarization.segments, key=lambda seg: seg.start)
    default_speaker = diar_segments[0].speaker if diar_segments else "SPEAKER_00"
    last_speaker = default_speaker
    aligned: List[Tuple[dict, str]] = []

    for whisper_segment in sorted(whisper_segments, key=lambda seg: float(seg.get("start") or 0.0)):
        seg_start = float(whisper_segment.get("start") or 0.0)
        seg_end = float(whisper_segment.get("end") or seg_start)
        overlap_by_speaker: Dict[str, float] = {}
        for dia_segment in diar_segments:
            overlap = _overlap_duration(seg_start, seg_end, dia_segment.start, dia_segment.end)
            if overlap > _OVERLAP_EPS:
                overlap_by_speaker[dia_segment.speaker] = (
                    overlap_by_speaker.get(dia_segment.speaker, 0.0) + overlap
                )
        if overlap_by_speaker:
            # Deterministic tie-break: among speakers within _OVERLAP_EPS of the
            # max overlap (cross-talk / exact ties), prefer the previous segment's
            # speaker for continuity, else the lexicographically-smallest id. Avoids
            # the non-deterministic dict-insertion-order pick that bare max() gave.
            best = max(overlap_by_speaker.values())
            tied = sorted(
                spk for spk, ov in overlap_by_speaker.items() if best - ov <= _OVERLAP_EPS
            )
            speaker_id = last_speaker if last_speaker in tied else tied[0]
            last_speaker = speaker_id
        else:
            speaker_id = last_speaker
        aligned.append((whisper_segment, speaker_id))

    return aligned


#: What names a voice on a stored segment. ``speaker`` is the diarizer's voice id and is None on
#: corpora that only kept ``speaker_label``; the rest are what naming added on top of it.
IDENTITY_FIELDS = ("speaker", "speaker_label", "speaker_role", "voice_type", "stated_speaker")


def carry_speaker_identity(
    new_segments: Sequence[Mapping[str, Any]],
    reference_segments: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Copies of ``new_segments`` with the voice identity of the reference segments they overlap.

    For re-transcribed audio whose speakers are already known: no diarizer is run and nothing is
    renamed. Each distinct identity (all of ``IDENTITY_FIELDS`` together, so two voices that share a
    label but differ in role stay apart) becomes one voice in a :class:`DiarizationResult` built
    from the reference segments' own times, and ``align_segments_to_speakers`` assigns by overlap.
    A new segment that overlaps no reference segment (speech the old decode dropped) takes the
    identity of the previous new segment, as in any alignment; the first takes the earliest
    reference voice. Reference segments without times or without any identity field are ignored;
    with no usable reference the segments come back unlabelled.
    """
    out = [dict(seg) for seg in new_segments]
    voices: Dict[Tuple[Any, ...], str] = {}
    turns: List[DiarizationSegment] = []
    for ref in reference_segments:
        identity = tuple(ref.get(field) for field in IDENTITY_FIELDS)
        if all(value is None for value in identity):
            continue
        try:
            start, end = float(ref["start"]), float(ref["end"])
        except (KeyError, TypeError, ValueError):
            continue
        voice = voices.setdefault(identity, f"VOICE_{len(voices):03d}")
        turns.append(DiarizationSegment(start=start, end=end, speaker=voice))
    if not turns:
        return out
    by_voice = {voice: identity for identity, voice in voices.items()}
    diarization = DiarizationResult(segments=turns, num_speakers=len(voices))
    placed = {id(seg): voice for seg, voice in align_segments_to_speakers(out, diarization)}
    for seg in out:
        for field, value in zip(IDENTITY_FIELDS, by_voice[placed[id(seg)]]):
            if value is not None:
                seg[field] = value
    return out
