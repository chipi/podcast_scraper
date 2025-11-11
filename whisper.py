"""Whisper integration helpers: model loading, transcription, screenplay formatting."""

from __future__ import annotations

import importlib
import logging
import sys
import time
import warnings
from types import ModuleType
from typing import Any, List, Optional, Tuple

from . import config, progress

logger = logging.getLogger(__name__)


def format_screenplay_from_segments(
    segments: List[dict],
    num_speakers: int,
    speaker_names: List[str],
    gap_s: float,
) -> str:
    if not segments:
        return ""

    segs = sorted(segments, key=lambda s: float(s.get("start") or 0.0))
    current_speaker_idx = 0
    lines: List[Tuple[int, str]] = []
    prev_end: Optional[float] = None

    for segment in segs:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if prev_end is not None and start - prev_end > gap_s:
            current_speaker_idx = (current_speaker_idx + 1) % max(config.MIN_NUM_SPEAKERS, num_speakers)
        prev_end = end
        if lines and lines[-1][0] == current_speaker_idx:
            lines[-1] = (lines[-1][0], lines[-1][1] + (" " if lines[-1][1] else "") + text)
        else:
            lines.append((current_speaker_idx, text))

    def speaker_label(idx: int) -> str:
        if 0 <= idx < len(speaker_names):
            return speaker_names[idx]
        return f"SPEAKER {idx + 1}"

    return "\n".join(f"{speaker_label(idx)}: {txt}" for idx, txt in lines) + "\n"


def _import_third_party_whisper() -> ModuleType:
    """Import the external whisper library without colliding with this module."""
    existing = sys.modules.get("whisper")
    # If another module already provides whisper, reuse it when it's not us.
    if existing is not None and existing is not sys.modules.get(__name__):
        return existing

    removed = None
    if existing is sys.modules.get(__name__):
        removed = sys.modules.pop("whisper")
    try:
        return importlib.import_module("whisper")
    finally:
        if removed is not None:
            sys.modules["whisper"] = removed


def load_whisper_model(cfg: config.Config) -> Optional[Any]:
    if not cfg.transcribe_missing:
        return None
    try:
        whisper_lib = _import_third_party_whisper()
        logger.info(f"Loading Whisper model ({cfg.whisper_model})...")
        model = whisper_lib.load_model(cfg.whisper_model)
        logger.info("Whisper model loaded successfully.")
        device = getattr(model, "device", None)
        device_type = getattr(device, "type", None)
        is_cpu_device = device_type == "cpu"
        setattr(model, "_is_cpu_device", is_cpu_device)
        if is_cpu_device:
            logger.info("Whisper is running on CPU; FP16 is unavailable so FP32 will be used.")
        return model
    except ImportError:
        logger.warning(
            "openai-whisper not installed. Install with: pip install openai-whisper && brew install ffmpeg"
        )
        return None
    except (RuntimeError, OSError) as exc:
        logger.warning(f"Failed to load Whisper model: {exc}")
        return None


def transcribe_with_whisper(whisper_model: Any, temp_media: str, cfg: config.Config) -> Tuple[dict, float]:
    logger.info(f"    transcribing with Whisper ({cfg.whisper_model})...")
    start = time.time()
    with progress.progress_context(None, "Transcribing") as reporter:
        suppress_fp16_warning = getattr(whisper_model, "_is_cpu_device", False)
        if suppress_fp16_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU",
                    category=UserWarning,
                )
                result = whisper_model.transcribe(temp_media, task="transcribe", language="en", verbose=False)
        else:
            result = whisper_model.transcribe(temp_media, task="transcribe", language="en", verbose=False)
        reporter.update(1)
    return result, time.time() - start


__all__ = [
    "format_screenplay_from_segments",
    "load_whisper_model",
    "transcribe_with_whisper",
]
