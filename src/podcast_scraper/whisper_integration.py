"""Whisper integration helpers: model loading, transcription, screenplay formatting."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from contextlib import contextmanager
from types import ModuleType
from typing import Any, List, Optional, Tuple, Union

from . import config, progress
from .cache_utils import get_whisper_cache_dir

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

    logger.debug("Loading Whisper model: %s", model_name)
    if len(fallback_models) > 1:
        logger.debug("Fallback chain: %s", fallback_models)

    # Get whisper cache directory (prefers local cache if it exists)
    whisper_cache = get_whisper_cache_dir()
    whisper_cache_str = str(whisper_cache)
    logger.debug("Whisper cache directory: %s", whisper_cache_str)

    last_error: Optional[Union[FileNotFoundError, RuntimeError, OSError]] = None
    for attempt_model in fallback_models:
        try:
            if attempt_model != model_name:
                logger.debug(
                    "Trying fallback Whisper model: %s (requested: %s)", attempt_model, model_name
                )
            # Use download_root parameter to specify cache directory directly
            # This ensures we use pre-cached models and avoid network calls
            model = whisper_lib.load_model(attempt_model, download_root=whisper_cache_str)
            if attempt_model != model_name:
                logger.debug(
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
                logger.debug("Whisper is running on CPU; FP16 is unavailable so FP32 will be used.")
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


@contextmanager
def _intercept_whisper_progress(progress_reporter: progress.ProgressReporter):
    """Intercept Whisper's tqdm progress calls and forward to our progress reporter.

    Whisper uses tqdm internally for progress reporting. This context manager
    temporarily overrides tqdm to capture Whisper's progress updates and forward
    them to our own progress reporter, preventing multiple progress bar lines.

    Args:
        progress_reporter: Our progress reporter to forward updates to
    """
    try:
        import tqdm
    except ImportError:
        # tqdm not available, can't intercept
        yield
        return

    original_tqdm = tqdm.tqdm

    class InterceptedTqdm(original_tqdm):  # type: ignore[misc,valid-type]
        """Custom tqdm that suppresses output and forwards progress to our reporter."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Suppress tqdm's own display
            # Store file handle to ensure it's closed properly
            self._devnull_file = open(os.devnull, "w")
            kwargs["file"] = self._devnull_file
            kwargs["disable"] = True  # Disable tqdm's display completely
            # Store total for percentage calculation
            self._whisper_total: Optional[int] = kwargs.get("total")
            self._whisper_n = 0
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> Optional[bool]:
            """Update progress and forward to our reporter."""
            result: Optional[bool] = super().update(n)
            self._whisper_n = getattr(self, "n", self._whisper_n + n)

            # Forward update to our progress reporter
            if progress_reporter:
                # Forward the increment to our progress reporter
                progress_reporter.update(n)

            return result

        def close(self) -> None:
            """Clean up and ensure final update."""
            if progress_reporter and self._whisper_total:
                # Ensure we're at 100% if we have a total
                remaining = self._whisper_total - self._whisper_n
                if remaining > 0:
                    progress_reporter.update(remaining)
            super().close()
            # Explicitly close the devnull file handle to prevent descriptor leaks
            if hasattr(self, "_devnull_file") and self._devnull_file:
                try:
                    self._devnull_file.close()
                except Exception:  # nosec B110
                    # Ignore errors during cleanup
                    pass

        def __del__(self) -> None:
            """Safety net: ensure file handle is closed even if close() wasn't called."""
            if hasattr(self, "_devnull_file") and self._devnull_file:
                try:
                    self._devnull_file.close()
                except Exception:  # nosec B110
                    # Ignore errors during cleanup
                    pass

    # Temporarily replace tqdm.tqdm with our interceptor
    tqdm.tqdm = InterceptedTqdm

    try:
        yield
    finally:
        # Restore original tqdm
        tqdm.tqdm = original_tqdm


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

        # Intercept Whisper's tqdm progress calls and forward to our progress reporter
        # This prevents multiple progress bar lines while showing real progress
        with _intercept_whisper_progress(reporter):
            result = whisper_model.transcribe(
                temp_media, task="transcribe", language=language, verbose=False
            )
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
