"""Whisper integration helpers: model loading, transcription, screenplay formatting."""

from __future__ import annotations

import importlib
import logging
import sys
import time
import warnings
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union, cast

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
            current_speaker_idx = (current_speaker_idx + 1) % max(
                config.MIN_NUM_SPEAKERS, num_speakers
            )
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
    """Import the external whisper library from openai-whisper package."""
    # Check if whisper is already imported
    existing = sys.modules.get("whisper")
    if existing is not None:
        # Verify it's the real library by checking for load_model function
        if hasattr(existing, "load_model"):
            return existing
        # If it's not the real library, remove it and try again
        sys.modules.pop("whisper", None)

    # Import the whisper library from openai-whisper package
    try:
        whisper_lib = importlib.import_module("whisper")
        # Verify it's the real library by checking for load_model function
        if not hasattr(whisper_lib, "load_model"):
            raise ImportError(
                "Imported 'whisper' module does not have 'load_model' function. "
                "Make sure 'openai-whisper' is installed: pip install openai-whisper"
            )
        return whisper_lib
    except ImportError as exc:
        # Re-raise with clearer error message
        raise ImportError(
            f"Failed to import openai-whisper library: {exc}. "
            "Make sure 'openai-whisper' is installed: pip install openai-whisper"
        ) from exc


def load_whisper_model(cfg: config.Config) -> Optional[Any]:
    """Load Whisper model with failover to smaller models if requested model fails.

    Args:
        cfg: Configuration object with whisper_model and language settings

    Returns:
        Loaded Whisper model or None if all attempts fail
    """
    if not cfg.transcribe_missing:
        return None

    try:
        whisper_lib = _import_third_party_whisper()
    except ImportError:
        logger.error(
            "openai-whisper library not installed. Cannot load Whisper models. "
            "Install with: pip install openai-whisper && brew install ffmpeg"
        )
        return None

    # Language-aware model selection: prefer .en variants for English
    requested_model = cfg.whisper_model
    model_name = requested_model
    if cfg.language in ("en", "english"):
        # If user specified a base model without .en, prefer the .en variant
        if model_name in ("tiny", "base", "small", "medium"):
            model_name = f"{model_name}.en"
            logger.debug("Language is English, preferring %s over %s", model_name, requested_model)
    else:
        # For non-English, ensure we use multilingual models (no .en suffix)
        if model_name.endswith(".en"):
            logger.debug(
                "Language is %s, using multilingual model instead of %s",
                cfg.language,
                model_name,
            )
            model_name = model_name[:-3]  # Remove .en suffix

    # Build fallback chain: try requested model, then progressively smaller models
    fallback_models = [model_name]
    if cfg.language in ("en", "english"):
        # For English, try .en variants in order: base.en, tiny.en
        if model_name not in ("tiny.en", "base.en"):
            if model_name.startswith("base"):
                fallback_models.append("tiny.en")
            elif model_name.startswith("small"):
                fallback_models.extend(["base.en", "tiny.en"])
            elif model_name.startswith("medium"):
                fallback_models.extend(["small.en", "base.en", "tiny.en"])
            elif model_name.startswith("large"):
                fallback_models.extend(["medium.en", "small.en", "base.en", "tiny.en"])
    else:
        # For multilingual, try: base, tiny
        if model_name not in ("tiny", "base"):
            if model_name.startswith("base"):
                fallback_models.append("tiny")
            elif model_name.startswith("small"):
                fallback_models.extend(["base", "tiny"])
            elif model_name.startswith("medium"):
                fallback_models.extend(["small", "base", "tiny"])
            elif model_name.startswith("large"):
                fallback_models.extend(["medium", "small", "base", "tiny"])

    logger.info("Loading Whisper model: %s", model_name)
    if len(fallback_models) > 1:
        logger.debug("Fallback chain: %s", fallback_models)

    last_error: Optional[Union[FileNotFoundError, RuntimeError, OSError]] = None
    for attempt_model in fallback_models:
        try:
            if attempt_model != model_name:
                logger.info(
                    "Trying fallback Whisper model: %s (requested: %s)", attempt_model, model_name
                )
            model = whisper_lib.load_model(attempt_model)
            if attempt_model != model_name:
                logger.warning(
                    "Loaded fallback Whisper model: %s (requested %s was not available)",
                    attempt_model,
                    model_name,
                )
            logger.debug("Whisper model loaded: %s", attempt_model)
            device = getattr(model, "device", None)
            device_type = getattr(device, "type", None)
            dtype = getattr(model, "dtype", None)
            logger.debug(
                "Whisper model details: device=%s dtype=%s num_params=%s",
                device,
                dtype,
                (
                    getattr(model, "num_parameters", lambda: "n/a")()
                    if callable(getattr(model, "num_parameters", None))
                    else getattr(model, "num_parameters", "n/a")
                ),
            )
            is_cpu_device = device_type == "cpu"
            setattr(model, "_is_cpu_device", is_cpu_device)
            if is_cpu_device:
                logger.info("Whisper is running on CPU; FP16 is unavailable so FP32 will be used.")
            else:
                logger.debug("Whisper model is using accelerator device type=%s", device_type)
            return model
        except FileNotFoundError as exc:
            last_error = exc
            logger.warning("Whisper model %s not found: %s", attempt_model, exc)
            continue
        except (RuntimeError, OSError) as exc:
            # RuntimeError/OSError could be network errors, disk space issues,
            # corrupted model files, or GPU/CUDA errors
            last_error = exc
            logger.warning("Failed to load Whisper model %s: %s", attempt_model, exc)
            continue

    # All models failed
    logger.error(
        "Failed to load any Whisper model. Tried: %s. Last error: %s",
        fallback_models,
        last_error,
    )
    return None


def transcribe_with_whisper(
    whisper_model: Any, temp_media: str, cfg: config.Config
) -> Tuple[dict, float]:
    logger.info("    transcribing with Whisper (%s)...", cfg.whisper_model)
    start = time.time()
    with progress.progress_context(None, "Transcribing") as reporter:
        suppress_fp16_warning = getattr(whisper_model, "_is_cpu_device", False)
        logger.debug(
            "Invoking Whisper transcription: media=%s suppress_fp16_warning=%s dtype=%s",
            temp_media,
            suppress_fp16_warning,
            getattr(whisper_model, "dtype", None),
        )
        # Use configured language, defaulting to "en" for backwards compatibility
        language = cfg.language if cfg.language else "en"
        logger.debug("Transcribing with language=%s", language)
        if suppress_fp16_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU",
                    category=UserWarning,
                )
                result = whisper_model.transcribe(
                    temp_media, task="transcribe", language=language, verbose=False
                )
        else:
            result = whisper_model.transcribe(
                temp_media, task="transcribe", language=language, verbose=False
            )
        cast(progress.ProgressReporter, reporter).update(1)
    elapsed = time.time() - start
    segments = result.get("segments")
    logger.debug(
        "Whisper transcription finished in %.2fs (segments=%s text_chars=%s)",
        elapsed,
        len(segments) if isinstance(segments, list) else "n/a",
        len(result.get("text") or ""),
    )
    return result, elapsed


__all__ = [
    "format_screenplay_from_segments",
    "load_whisper_model",
    "transcribe_with_whisper",
]
