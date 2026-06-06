"""pyannote.audio diarization provider."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ....exceptions import ProviderDependencyError
from .base import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)


def _create_pyannote_pipeline(hf_token: str, model_name: str) -> Any:
    """Construct pyannote Pipeline (isolated for unit-test patching)."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise ProviderDependencyError(
            message="pyannote.audio is required for diarize=true",
            provider="PyAnnoteDiarizationProvider",
            dependency="pyannote.audio",
            suggestion="Install with: pip install -e '.[ml]'",
        ) from exc
    try:
        return Pipeline.from_pretrained(model_name, token=hf_token)
    except TypeError as exc:
        # Older huggingface_hub/pyannote used use_auth_token=; types only expose token=.
        # Only retry when the failure is actually about the token kwarg — otherwise a
        # TypeError raised inside model loading would be masked by a second attempt.
        if "token" not in str(exc):
            raise
        return Pipeline.from_pretrained(  # type: ignore[call-arg]
            model_name,
            use_auth_token=hf_token,
        )


def _resolve_device(device: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise ProviderDependencyError(
            message="torch is required for diarize=true",
            provider="PyAnnoteDiarizationProvider",
            dependency="torch",
            suggestion="Install with: pip install -e '.[ml]'",
        ) from exc
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _load_waveform(audio_path: str) -> tuple[Any, int]:
    try:
        import torchaudio
    except ImportError as exc:
        raise ProviderDependencyError(
            message="torchaudio is required for diarize=true",
            provider="PyAnnoteDiarizationProvider",
            dependency="torchaudio",
            suggestion="Install with: pip install -e '.[ml]'",
        ) from exc
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, int(sample_rate)


class PyAnnoteDiarizationProvider:
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        hf_token: str,
        *,
        device: str = "auto",
        model_name: str = "pyannote/speaker-diarization-3.1",
    ) -> None:
        self.model_name = model_name
        self._pipeline = _create_pyannote_pipeline(hf_token, model_name)
        resolved = _resolve_device(device)
        import torch

        self._pipeline.to(torch.device(resolved))
        logger.debug("Loaded pyannote diarization model %s on %s", model_name, resolved)

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
    ) -> DiarizationResult:
        """Run diarization on audio and return speaker-attributed segments."""
        waveform, sample_rate = _load_waveform(audio_path)
        params: dict[str, int] = {}
        if num_speakers is not None:
            if num_speakers < 1:
                raise ValueError(f"diarization_num_speakers must be >= 1, got {num_speakers}")
            params["num_speakers"] = num_speakers
        else:
            if min_speakers < 1 or min_speakers > max_speakers:
                raise ValueError(
                    "invalid diarization speaker bounds: " f"min={min_speakers}, max={max_speakers}"
                )
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate}, **params)
        segments: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(turn.start), end=float(turn.end), speaker=str(speaker)
                )
            )
        unique_speakers = {segment.speaker for segment in segments}
        return DiarizationResult(
            segments=segments,
            num_speakers=len(unique_speakers),
            model_name=self.model_name,
        )
