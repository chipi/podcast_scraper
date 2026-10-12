"""Diarization provider protocol and result types."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol


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
    # Per-episode diarization cost in USD. None/0.0 for local diarizers (pyannote/DGX/MOSS); cloud
    # diarizers (Deepgram/Gemini) set the billed cost so the processing manifest can record it
    # (RFC-109 / ADR-132). Provider-agnostic: consumers read this field, not a provider specific.
    cost_usd: Optional[float] = None
    # One embedding per diarized voice, keyed by its label (cross-episode voice identity, #2304;
    # docs/wip/VOICE-IDENTITY-DESIGN-2026-10-12.md). Empty when the diarizer gives none (cloud,
    # MOSS, a relabel of an episode diarized before capture, an older DGX server).
    speaker_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    # The model that made ``speaker_embeddings`` when it is not ``model_name``: a relabel carries
    # stored vectors over without re-diarizing, and must not claim a diarization model it never ran.
    speaker_embeddings_model: str = ""


#: Decimals kept per embedding value: plenty for cosine distance, a third of the bytes.
EMBEDDING_DECIMALS = 5


def speaker_embeddings_from_output(
    output: Any, labels: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """``{label: embedding}`` from a pyannote 4.x ``DiarizeOutput``; ``{}`` for anything else.

    ``speaker_embeddings`` is a ``(num_speakers, dim)`` array in ``speaker_diarization.labels()``
    order. A row that is not finite (pyannote leaves NaN for a speaker it could not embed) is
    skipped rather than stored as noise.
    """
    matrix = getattr(output, "speaker_embeddings", None)
    annotation = getattr(output, "speaker_diarization", None)
    if matrix is None or annotation is None:
        return {}
    order = [str(x) for x in (labels if labels is not None else annotation.labels())]
    out: Dict[str, List[float]] = {}
    for label, row in zip(order, list(matrix)):
        values = [float(x) for x in row]
        if values and all(math.isfinite(x) for x in values):
            out[label] = [round(x, EMBEDDING_DECIMALS) for x in values]
    return out


def clean_embeddings(raw: Any) -> Dict[str, List[float]]:
    """``{label: [float, ...]}`` from untrusted JSON (a DGX response, a cache file, a sidecar)."""
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, List[float]] = {}
    for label, row in raw.items():
        if not isinstance(row, list) or not row:
            continue
        try:
            values = [float(x) for x in row]
        except (TypeError, ValueError):
            continue
        if all(math.isfinite(x) for x in values):
            out[str(label)] = values
    return out


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
