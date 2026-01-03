"""Unified ML provider for transcription, speaker detection, and summarization.

This module provides a single MLProvider class that implements all three protocols:
- TranscriptionProvider (using Whisper)
- SpeakerDetector (using spaCy)
- SummarizationProvider (using Transformers)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared ML libraries.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import time
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .. import config, models, progress, speaker_detection, summarizer
from ..cache_utils import get_whisper_cache_dir

logger = logging.getLogger(__name__)

# Protocol types imported for type hints (used in docstrings and type annotations)
# from ..speaker_detectors.base import SpeakerDetector  # noqa: F401
# from ..summarization.base import SummarizationProvider  # noqa: F401
# from ..transcription.base import TranscriptionProvider  # noqa: F401


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


class MLProvider:
    """Unified ML provider implementing TranscriptionProvider, SpeakerDetector, and SummarizationProvider.

    This provider initializes and manages:
    - Whisper models for transcription
    - spaCy models for speaker detection
    - Transformers models for summarization

    All three ML libraries are initialized together, similar to how OpenAI providers
    share the same OpenAI client. Models are lazy-loaded based on configuration flags.
    """  # noqa: E501

    def __init__(self, cfg: config.Config):
        """Initialize unified ML provider.

        Args:
            cfg: Configuration object with settings for all three capabilities
        """
        self.cfg = cfg

        # Whisper transcription state
        self._whisper_model: Optional[Any] = None
        self._whisper_initialized = False

        # spaCy speaker detection state
        self._spacy_nlp: Optional[Any] = None
        self._spacy_heuristics: Optional[Dict[str, Any]] = None
        self._spacy_initialized = False

        # Transformers summarization state
        self._map_model: Optional[summarizer.SummaryModel] = None
        self._reduce_model: Optional[summarizer.SummaryModel] = None
        self._transformers_initialized = False

        # Mark provider as requiring separate instances for thread safety
        # HuggingFace models are not thread-safe and cannot be shared across threads
        self._requires_separate_instances = True

    def initialize(self) -> None:
        """Initialize all ML models (Whisper, spaCy, Transformers).

        Models are lazy-loaded based on configuration flags:
        - Whisper: if transcribe_missing is True
        - spaCy: if auto_speakers is True
        - Transformers: if generate_summaries is True

        This method is idempotent and can be called multiple times safely.

        Note: If one component fails to initialize (e.g., Whisper due to missing cache),
        other components will still be initialized. This allows summarization to work
        even if transcription is unavailable.
        """
        # Initialize Whisper if transcription enabled
        # If Whisper fails, log warning but continue with other components
        if self.cfg.transcribe_missing and not self._whisper_initialized:
            try:
                self._initialize_whisper()
            except Exception as e:
                logger.warning(
                    "Failed to initialize Whisper (transcription will be unavailable): %s", e
                )
                # Don't raise - allow other components to initialize

        # Initialize spaCy if speaker detection enabled
        if self.cfg.auto_speakers and not self._spacy_initialized:
            try:
                self._initialize_spacy()
            except Exception as e:
                logger.warning(
                    "Failed to initialize spaCy (speaker detection will be unavailable): %s", e
                )
                # Don't raise - allow other components to initialize

        # Initialize Transformers if summarization enabled
        if self.cfg.generate_summaries and not self._transformers_initialized:
            self._initialize_transformers()

    def _initialize_whisper(self) -> None:  # noqa: C901
        """Initialize Whisper model for transcription."""
        logger.debug("Initializing Whisper transcription (model: %s)", self.cfg.whisper_model)

        try:
            whisper_lib = _import_third_party_whisper()
        except ImportError:
            logger.error(
                "openai-whisper library not installed. Cannot load Whisper models. "
                "Install with: pip install openai-whisper && brew install ffmpeg"
            )
            raise RuntimeError(
                "Failed to load Whisper model. "
                "Make sure 'openai-whisper' is installed: pip install openai-whisper"
            )

        # Language-aware model selection: prefer .en variants for English
        requested_model = self.cfg.whisper_model
        model_name = requested_model
        if self.cfg.language in ("en", "english"):
            # If user specified a base model without .en, prefer the .en variant
            if model_name in ("tiny", "base", "small", "medium"):
                model_name = f"{model_name}.en"
                logger.debug(
                    "Language is English, preferring %s over %s", model_name, requested_model
                )
        else:
            # For non-English, ensure we use multilingual models (no .en suffix)
            if model_name.endswith(".en"):
                logger.debug(
                    "Language is %s, using multilingual model instead of %s",
                    self.cfg.language,
                    model_name,
                )
                model_name = model_name[:-3]  # Remove .en suffix

        # Build fallback chain: try requested model, then progressively smaller models
        fallback_models = [model_name]
        if self.cfg.language in ("en", "english"):
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

        # Check cache directory for pre-cached models
        # Prefer local cache in project root, fallback to ~/.cache/whisper/
        whisper_cache = get_whisper_cache_dir()
        logger.debug(
            "Whisper cache directory: %s (exists: %s)", whisper_cache, whisper_cache.exists()
        )

        last_error: Optional[Union[FileNotFoundError, RuntimeError, OSError]] = None
        for attempt_model in fallback_models:
            # Check if model is cached before attempting to load
            # This helps avoid network calls when models are pre-cached
            model_file = whisper_cache / f"{attempt_model}.pt"
            if not model_file.exists():
                logger.debug(
                    "Whisper model %s not found in cache: %s (will try download if network available)",  # noqa: E501
                    attempt_model,
                    model_file,
                )
                # Continue to next model in fallback chain
                # Don't set last_error here - we want to try loading anyway
                # in case the cache path is different or the model is in a different location
            else:
                logger.debug(
                    "Whisper model %s found in cache: %s (size: %s MB)",
                    attempt_model,
                    model_file,
                    model_file.stat().st_size / (1024 * 1024) if model_file.exists() else 0,
                )

            try:
                # Use download_root parameter to specify cache directory directly
                # This is more reliable than environment variable
                whisper_cache_str = str(whisper_cache)
                if attempt_model != model_name:
                    logger.debug(
                        "Trying fallback Whisper model: %s (requested: %s)",
                        attempt_model,
                        model_name,
                    )
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
                    logger.debug(
                        "Whisper is running on CPU; FP16 is unavailable so FP32 will be used."
                    )
                else:
                    logger.debug("Whisper model is using accelerator device type=%s", device_type)
                self._whisper_model = model
                self._whisper_initialized = True
                logger.debug("Whisper transcription initialized successfully")
                return
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
        raise RuntimeError(
            f"Failed to load any Whisper model. Tried: {fallback_models}. "
            f"Last error: {last_error}"
        )

    def _initialize_spacy(self) -> None:
        """Initialize spaCy NER model for speaker detection."""
        if not self.cfg.auto_speakers:
            raise RuntimeError(
                "Cannot initialize spaCy: auto_speakers is False. "
                "Set auto_speakers=True to enable speaker detection."
            )
        logger.debug("Initializing spaCy NER model (model: %s)", self.cfg.ner_model)
        self._spacy_nlp = speaker_detection.get_ner_model(self.cfg)
        if self._spacy_nlp is None:
            logger.warning(
                "Failed to load spaCy NER model. Speaker detection may be limited. Model: %s",
                self.cfg.ner_model or "default",
            )
        else:
            logger.debug("spaCy NER model initialized successfully")
        self._spacy_initialized = True

    def _initialize_transformers(self) -> None:
        """Initialize Transformers models for summarization."""
        logger.debug(
            "Initializing Transformers summarization (model: %s)",
            self.cfg.summary_model or "default",
        )

        try:
            # Load MAP model (for chunk summarization)
            model_name = summarizer.select_summary_model(self.cfg)
            self._map_model = summarizer.SummaryModel(
                model_name=model_name,
                device=self.cfg.summary_device,
                cache_dir=self.cfg.summary_cache_dir,
            )
            logger.debug("Loaded MAP summary model: %s", model_name)

            # Load REDUCE model if different from MAP model (for final combine)
            reduce_model_name = summarizer.select_reduce_model(self.cfg, model_name)
            if reduce_model_name != model_name:
                self._reduce_model = summarizer.SummaryModel(
                    model_name=reduce_model_name,
                    device=self.cfg.summary_device,
                    cache_dir=self.cfg.summary_cache_dir,
                )
                logger.debug("Loaded REDUCE summary model: %s", reduce_model_name)
            else:
                # Use MAP model for REDUCE phase if they're the same
                self._reduce_model = self._map_model
                logger.debug("Using MAP model for REDUCE phase (same model)")

            self._transformers_initialized = True
            logger.debug("Transformers summarization initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize summarization models: %s", e)
            raise

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using Whisper.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or defaults to "en"

        Returns:
            Transcribed text as string

        Raises:
            RuntimeError: If provider is not initialized
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails
        """
        if not self._whisper_initialized or self._whisper_model is None:
            raise RuntimeError("MLProvider Whisper not initialized. Call initialize() first.")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or "en")

        logger.debug("Transcribing audio file: %s (language: %s)", audio_path, effective_language)

        # Call internal transcription method
        result_dict, elapsed = self._transcribe_with_whisper(audio_path, effective_language)

        # Extract text from result
        text = result_dict.get("text", "").strip()
        if not text:
            raise ValueError("Transcription returned empty text")

        logger.debug(
            "Transcription completed in %.2fs (text length: %d chars)",
            elapsed,
            len(text),
        )

        return str(text)  # Ensure we return str, not Any

    def transcribe_with_segments(
        self, audio_path: str, language: str | None = None
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Returns the complete Whisper transcription result including segments
        and timestamps for screenplay formatting.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or defaults to "en"

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start, end, text
            - Other Whisper metadata
        """
        if not self._whisper_initialized or self._whisper_model is None:
            raise RuntimeError("MLProvider Whisper not initialized. Call initialize() first.")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or "en")

        logger.debug(
            "Transcribing audio file with segments: %s (language: %s)",
            audio_path,
            effective_language,
        )

        # Call internal transcription method
        result_dict, elapsed = self._transcribe_with_whisper(audio_path, effective_language)

        logger.debug(
            "Transcription with segments completed in %.2fs (%d segments)",
            elapsed,
            len(result_dict.get("segments", [])),
        )

        return result_dict, elapsed

    def _transcribe_with_whisper(self, audio_path: str, language: str) -> Tuple[dict, float]:
        """Internal method to transcribe audio with Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code for transcription

        Returns:
            Tuple of (result_dict, elapsed_time)
        """
        logger.info("    transcribing with Whisper (%s)...", self.cfg.whisper_model)
        start = time.time()
        with progress.progress_context(None, "Transcribing") as reporter:
            suppress_fp16_warning = getattr(self._whisper_model, "_is_cpu_device", False)
            logger.debug(
                "Invoking Whisper transcription: media=%s suppress_fp16_warning=%s dtype=%s",
                audio_path,
                suppress_fp16_warning,
                getattr(self._whisper_model, "dtype", None),
            )
            logger.debug("Transcribing with language=%s", language)

            # Intercept Whisper's tqdm progress calls and forward to our progress reporter
            # This prevents multiple progress bar lines while showing real progress
            if self._whisper_model is None:
                raise RuntimeError("Whisper model not initialized")
            with _intercept_whisper_progress(reporter):
                result = self._whisper_model.transcribe(
                    audio_path, task="transcribe", language=language, verbose=False
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

    def format_screenplay_from_segments(
        self,
        segments: List[dict],
        num_speakers: int,
        speaker_names: List[str],
        gap_s: float,
    ) -> str:
        """Format transcription segments as screenplay with speaker attribution.

        Args:
            segments: List of segment dictionaries with start, end, text
            num_speakers: Number of speakers to rotate through
            speaker_names: List of speaker names (optional, uses indices if not provided)
            gap_s: Gap in seconds between segments to trigger speaker rotation

        Returns:
            Formatted screenplay text with speaker labels
        """
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

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> Tuple[list[str], Set[str], bool]:
        """Detect speaker names from episode metadata using spaCy NER.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

        Returns:
            Tuple of:
            - List of detected speaker names
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
        """
        # Ensure model is loaded (only if auto_speakers is enabled)
        if self._spacy_nlp is None:
            if not self.cfg.auto_speakers:
                raise RuntimeError(
                    "Cannot detect speakers: auto_speakers is False. "
                    "Set auto_speakers=True to enable speaker detection."
                )
            self._initialize_spacy()

        # Use detect_speaker_names with adapted parameters
        # Pass self._spacy_nlp directly (required parameter)
        speaker_names, detected_hosts_set, detection_succeeded = (
            speaker_detection.detect_speaker_names(
                episode_title=episode_title,
                episode_description=episode_description,
                nlp=self._spacy_nlp,  # Required: pass pre-loaded model
                cfg=self.cfg,
                known_hosts=None,  # Use cached_hosts instead
                cached_hosts=known_hosts,  # Map known_hosts to cached_hosts
                heuristics=self._spacy_heuristics,
            )
        )

        return speaker_names, detected_hosts_set, detection_succeeded

    def detect_hosts(
        self,
        feed_title: Optional[str],
        feed_description: Optional[str],
        feed_authors: Optional[List[str]] = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using spaCy NER.

        Args:
            feed_title: Feed title
            feed_description: Feed description (optional)
            feed_authors: List of author names from RSS feed (optional, preferred source)

        Returns:
            Set of detected host names
        """
        # Ensure model is loaded (only if auto_speakers is enabled)
        if self._spacy_nlp is None:
            if not self.cfg.auto_speakers:
                raise RuntimeError(
                    "Cannot detect speakers: auto_speakers is False. "
                    "Set auto_speakers=True to enable speaker detection."
                )
            self._initialize_spacy()

        return speaker_detection.detect_hosts_from_feed(
            feed_title=feed_title,
            feed_description=feed_description,
            feed_authors=feed_authors,
            nlp=self._spacy_nlp,
        )

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes using spaCy NER.

        Args:
            episodes: List of episodes to analyze
            known_hosts: Set of known host names

        Returns:
            Dictionary with pattern analysis results, or None if analysis fails
        """
        # Ensure model is loaded (only if auto_speakers is enabled)
        if self._spacy_nlp is None:
            if not self.cfg.auto_speakers:
                raise RuntimeError(
                    "Cannot detect speakers: auto_speakers is False. "
                    "Set auto_speakers=True to enable speaker detection."
                )
            self._initialize_spacy()

        if not self._spacy_nlp:
            return None

        # Analyze patterns and cache heuristics for use in detect_speakers
        self._spacy_heuristics = speaker_detection.analyze_episode_patterns(
            episodes=episodes,
            nlp=self._spacy_nlp,
            cached_hosts=known_hosts,
            sample_size=speaker_detection.DEFAULT_SAMPLE_SIZE,
        )

        return self._spacy_heuristics

    def clear_cache(self) -> None:
        """Clear cache (no-op since cache was removed).

        This method is kept for backward compatibility but does nothing.
        Models are managed by providers and cleaned up via cleanup().
        """
        logger.debug("clear_cache() called (no-op, cache removed)")

    # ============================================================================
    # SummarizationProvider Protocol Implementation
    # ============================================================================

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using Transformers MAP/REDUCE pattern.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title (not used by transformers provider)
            episode_description: Optional episode description (not used by transformers provider)
            params: Optional parameters dict with:
                - max_length: Maximum summary length (default from config)
                - min_length: Minimum summary length (default from config)
                - chunk_size: Chunk size in tokens (default from config)
                - chunk_parallelism: Number of chunks to process in parallel
                  (CPU only, default from config)
                - use_word_chunking: Use word-based chunking (default: auto-detected)
                - word_chunk_size: Chunk size in words (default from config)
                - word_overlap: Overlap in words (default from config)
                - prompt: Optional instruction/prompt

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {
                    "model_used": str,
                    "reduce_model_used": Optional[str],
                    ...
                }
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider not initialized
        """
        if not self._transformers_initialized or not self._map_model:
            raise RuntimeError("MLProvider Transformers not initialized. Call initialize() first.")

        # Extract parameters with defaults from config
        max_length = (params.get("max_length") if params else None) or self.cfg.summary_max_length
        min_length = (params.get("min_length") if params else None) or self.cfg.summary_min_length
        chunk_size = (params.get("chunk_size") if params else None) or self.cfg.summary_chunk_size
        # Chunk-level parallelism: Use chunk_parallelism from params, fallback to config
        chunk_parallelism = (
            params.get("chunk_parallelism") if params else None
        ) or self.cfg.summary_chunk_parallelism
        # batch_size is deprecated - use chunk_parallelism instead
        # Keep for backward compatibility if explicitly provided
        batch_size = params.get("batch_size") if params else None
        if batch_size is None:
            batch_size = chunk_parallelism if self._map_model.device == "cpu" else None
        use_word_chunking = params.get("use_word_chunking") if params else None
        word_chunk_size = (
            params.get("word_chunk_size") if params else None
        ) or self.cfg.summary_word_chunk_size
        word_overlap = (
            params.get("word_overlap") if params else None
        ) or self.cfg.summary_word_overlap
        prompt = params.get("prompt") if params else None

        # Auto-detect word chunking if not specified
        if use_word_chunking is None:
            model_name = (
                self._map_model.model_name if hasattr(self._map_model, "model_name") else ""
            )
            use_word_chunking = any(
                model_keyword in model_name.lower()
                for model_keyword in ["bart", "pegasus", "distilbart"]
            )

        # Use summarize_long_text for MAP/REDUCE pattern
        try:
            summary_text = summarizer.summarize_long_text(
                model=self._map_model,
                text=text,
                chunk_size=chunk_size or summarizer.BART_MAX_POSITION_EMBEDDINGS,
                max_length=max_length,
                min_length=min_length,
                batch_size=batch_size if self._map_model.device == "cpu" else None,
                prompt=prompt,
                use_word_chunking=use_word_chunking,
                word_chunk_size=word_chunk_size or summarizer.DEFAULT_WORD_CHUNK_SIZE,
                word_overlap=word_overlap or summarizer.DEFAULT_WORD_OVERLAP,
                reduce_model=self._reduce_model,
            )

            # Build metadata
            metadata: Dict[str, Any] = {
                "model_used": self._map_model.model_name,
                "reduce_model_used": (
                    self._reduce_model.model_name
                    if self._reduce_model and self._reduce_model != self._map_model
                    else None
                ),
                "device": self._map_model.device,
            }

            return {
                "summary": summary_text,
                "summary_short": None,  # Transformers provider doesn't generate short summaries
                "metadata": metadata,
            }
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            raise ValueError(f"Summarization failed: {e}") from e

    # ============================================================================
    # Cleanup Methods
    # ============================================================================

    def cleanup(self) -> None:
        """Cleanup all provider resources (unload models, close connections, etc.).

        This method releases all resources held by the provider.
        It may be called multiple times safely (idempotent).
        """
        # Cleanup Whisper
        if self._whisper_initialized:
            logger.debug("Cleaning up Whisper model")
            self._whisper_model = None
            self._whisper_initialized = False

        # Cleanup spaCy
        if self._spacy_initialized:
            logger.debug("Cleaning up spaCy model")
            self._spacy_nlp = None
            self._spacy_heuristics = None
            self._spacy_initialized = False

        # Cleanup Transformers
        if self._transformers_initialized:
            logger.debug("Cleaning up Transformers models")
            # Properly unload models to free memory and clean up threads
            if self._reduce_model and self._reduce_model != self._map_model:
                summarizer.unload_model(self._reduce_model)
            if self._map_model:
                summarizer.unload_model(self._map_model)
            self._map_model = None
            self._reduce_model = None
            self._transformers_initialized = False

    # ============================================================================
    # Properties for Backward Compatibility
    # ============================================================================

    @property
    def model(self) -> Optional[Any]:
        """Get the loaded Whisper model (for backward compatibility).

        Returns:
            Loaded Whisper model or None if not initialized
        """
        return self._whisper_model

    @property
    def nlp(self) -> Optional[Any]:
        """Get the loaded spaCy NER model (for backward compatibility).

        Returns:
            Loaded spaCy nlp object or None if not initialized
        """
        return self._spacy_nlp

    @property
    def heuristics(self) -> Optional[Dict[str, Any]]:
        """Get cached heuristics from pattern analysis (for backward compatibility).

        Returns:
            Heuristics dictionary or None if not analyzed
        """
        return self._spacy_heuristics

    @property
    def map_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the MAP model instance (for backward compatibility).

        Returns:
            SummaryModel instance or None if not initialized
        """
        return self._map_model

    @property
    def reduce_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the REDUCE model instance (for backward compatibility).

        Returns:
            SummaryModel instance or None if not initialized
        """
        return self._reduce_model

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component).

        Returns:
            True if any component is initialized, False otherwise
        """
        return (
            self._whisper_initialized or self._spacy_initialized or self._transformers_initialized
        )
