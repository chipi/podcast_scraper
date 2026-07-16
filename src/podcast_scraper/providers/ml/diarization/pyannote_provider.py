"""pyannote.audio diarization provider."""

from __future__ import annotations

import inspect
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


def _apply_clustering_overrides(
    pipeline: Any,
    *,
    threshold: Optional[float] = None,
    min_cluster_size: Optional[int] = None,
) -> None:
    """Override agglomerative-clustering hyperparameters on an instantiated pyannote pipeline.

    ``clustering.threshold`` (higher → merge more → fewer speakers) and
    ``clustering.min_cluster_size`` (higher → drop small over-segmentation fragments) are the two
    over-segmentation levers.
    ``pipeline.instantiate`` replaces ALL hyperparameters, so read the current instantiated params,
    deep-merge only the provided override(s), and re-instantiate. Best-effort: a pipeline that
    exposes no ``clustering`` block is left unchanged (logged), never broken — these are
    diarization-quality levers, not a correctness requirement.
    """
    if threshold is None and min_cluster_size is None:
        return
    try:
        current = dict(pipeline.parameters(instantiated=True))
    except Exception as exc:  # pragma: no cover - pyannote version / model dependent
        logger.warning("Cannot read pyannote params to set clustering overrides: %s", exc)
        return
    clustering = current.get("clustering")
    if not isinstance(clustering, dict):
        logger.warning(
            "pyannote pipeline exposes no 'clustering' block; clustering overrides "
            "(threshold=%s, min_cluster_size=%s) ignored.",
            threshold,
            min_cluster_size,
        )
        return
    merged = dict(clustering)
    if threshold is not None:
        merged["threshold"] = float(threshold)
    if min_cluster_size is not None:
        merged["min_cluster_size"] = int(min_cluster_size)
    current["clustering"] = merged
    pipeline.instantiate(current)


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

    # pyannote is NOT MPS-compatible: the pipeline requests float64, which Apple's Metal (MPS)
    # backend rejects ("Cannot convert a MPS Tensor to float64"). Coerce MPS -> CPU — the same
    # posture the QA evidence backend takes (resolve_evidence_device(mps_supported=False)) — so a
    # Mac dev box diarizes on CPU (slower) instead of crashing. CUDA / CPU are unaffected, so CI
    # (Linux/CUDA) and the DGX are unchanged.
    def _no_mps(dev: str) -> str:
        if dev == "mps":
            logger.warning(
                "pyannote does not support MPS (its pipeline uses float64, which Metal rejects); "
                "diarizing on CPU instead"
            )
            return "cpu"
        return dev

    if device != "auto":
        return _no_mps(device)
    if torch.cuda.is_available():
        return "cuda"
    # Deliberately skip MPS here (see _no_mps): pyannote is float64-incompatible with Metal.
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


def _apply_segment_squelch(
    segments: list[DiarizationSegment], min_segment_ms: Optional[int]
) -> list[DiarizationSegment]:
    """Drop speakers whose LONGEST segment is shorter than ``min_segment_ms`` (a squelch).

    pyannote over-segmentation spawns a phantom extra speaker out of a handful of sub-second
    snippets (audited: 0.3–0.9s fragments, <1% of audio). A real brief cameo, by contrast, is one
    contiguous multi-second turn. So the discriminator is the *longest* segment per speaker, not
    total talk-time (which overlaps: a 2.5s fragment cluster can out-total a <1.5s cameo). Speakers
    below the gate have all their segments removed; ``None``/0 disables the filter. See #1170.
    """
    if not min_segment_ms:
        return segments
    threshold_s = min_segment_ms / 1000.0
    longest: dict[str, float] = {}
    for seg in segments:
        dur = seg.end - seg.start
        if dur > longest.get(seg.speaker, 0.0):
            longest[seg.speaker] = dur
    kept = {speaker for speaker, dur in longest.items() if dur >= threshold_s}
    if len(kept) == len(longest):
        return segments
    dropped = sorted(set(longest) - kept)
    logger.debug(
        "diarization squelch (min_segment_ms=%s) dropped %d phantom speaker(s): %s",
        min_segment_ms,
        len(dropped),
        dropped,
    )
    return [seg for seg in segments if seg.speaker in kept]


class PyAnnoteDiarizationProvider:
    """Speaker diarization using pyannote.audio."""

    def __init__(
        self,
        hf_token: str,
        *,
        device: str = "auto",
        model_name: str = "pyannote/speaker-diarization-community-1",
        clustering_threshold: Optional[float] = None,
        min_cluster_size: Optional[int] = None,
        min_segment_ms: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self._min_segment_ms = min_segment_ms
        self._pipeline = _create_pyannote_pipeline(hf_token, model_name)
        _apply_clustering_overrides(
            self._pipeline,
            threshold=clustering_threshold,
            min_cluster_size=min_cluster_size,
        )
        resolved = _resolve_device(device)
        import torch

        self._pipeline.to(torch.device(resolved))
        logger.debug(
            "Loaded pyannote diarization model %s on %s "
            "(clustering_threshold=%s, min_cluster_size=%s)",
            model_name,
            resolved,
            clustering_threshold,
            min_cluster_size,
        )

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
        # pyannote 4.0.6 regression: ``Pipeline.__call__`` added a batch-input branch
        # with ``yield``, making the whole method a generator function — even for
        # single-file input, where it still does ``return prediction`` at the bottom.
        # In a generator that becomes ``StopIteration(prediction)`` on first next().
        # Unwrap so 4.0.4 / 4.0.5 (plain return) and 4.0.6+ (generator return) both work.
        if inspect.isgenerator(diarization):
            try:
                next(diarization)
            except StopIteration as stop:
                diarization = stop.value
            else:
                raise RuntimeError(
                    "pyannote Pipeline.__call__ yielded for single-file input; "
                    "expected a single prediction (DiarizeOutput or Annotation)."
                )
        # pyannote 4.x returns a DiarizeOutput wrapper; its ``speaker_diarization``
        # is the Annotation. pyannote 3.x returned the Annotation directly.
        annotation = getattr(diarization, "speaker_diarization", diarization)
        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(turn.start), end=float(turn.end), speaker=str(speaker)
                )
            )
        segments = _apply_segment_squelch(segments, self._min_segment_ms)
        unique_speakers = {segment.speaker for segment in segments}
        return DiarizationResult(
            segments=segments,
            num_speakers=len(unique_speakers),
            model_name=self.model_name,
        )
