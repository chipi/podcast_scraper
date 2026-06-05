"""Diarization provider protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class DiarizationSegment:
    """One diarized speaker turn with time bounds."""

    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    """Complete diarization output for an audio file."""

    segments: List[DiarizationSegment] = field(default_factory=list)
    num_speakers: int = 0
    model_name: str = ""


class DiarizationProvider(Protocol):
    """Protocol for speaker diarization backends."""

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Run speaker diarization on an audio file."""
        ...
