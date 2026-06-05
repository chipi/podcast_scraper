"""Align Whisper transcription segments to diarization speaker IDs."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import DiarizationResult


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
            if overlap > 0:
                overlap_by_speaker[dia_segment.speaker] = (
                    overlap_by_speaker.get(dia_segment.speaker, 0.0) + overlap
                )
        if overlap_by_speaker:
            speaker_id = max(overlap_by_speaker, key=lambda key: overlap_by_speaker[key])
            last_speaker = speaker_id
        else:
            speaker_id = last_speaker
        aligned.append((whisper_segment, speaker_id))

    return aligned
