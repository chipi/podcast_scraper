"""Speaker diarization providers and helpers."""

from .alignment import align_segments_to_speakers
from .base import DiarizationResult, DiarizationSegment
from .factory import create_diarization_provider, resolve_hf_token
from .formatting import (
    format_diarized_screenplay_from_segments,
    format_diarized_screenplay_with_offsets,
)
from .pipeline import apply_diarization_to_result
from .pyannote_provider import PyAnnoteDiarizationProvider
from .roster import resolve_speaker_roster, SpeakerRole, SpeakerRoster

__all__ = [
    "DiarizationResult",
    "DiarizationSegment",
    "PyAnnoteDiarizationProvider",
    "SpeakerRole",
    "SpeakerRoster",
    "align_segments_to_speakers",
    "apply_diarization_to_result",
    "create_diarization_provider",
    "format_diarized_screenplay_from_segments",
    "format_diarized_screenplay_with_offsets",
    "resolve_hf_token",
    "resolve_speaker_roster",
]
