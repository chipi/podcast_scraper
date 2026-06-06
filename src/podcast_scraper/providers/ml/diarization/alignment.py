"""Align Whisper transcription segments to diarization speaker IDs."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import DiarizationResult

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
