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

from ... import config, models
from ...cache import get_whisper_cache_dir
from ...exceptions import (
    ProviderConfigError,
    ProviderDependencyError,
    ProviderNotInitializedError,
    ProviderRuntimeError,
)
from ...utils import progress
from . import speaker_detection, summarizer
from .whisper_utils import normalize_whisper_model_name

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

    def create_worker_instances(self, num_workers: int) -> List["MLProvider"]:
        """Create worker provider instances for parallel processing.

        ML models (Transformers, Whisper) are not thread-safe, so each worker
        thread needs its own provider instance with its own model instances.

        Args:
            num_workers: Number of worker instances to create

        Returns:
            List of MLProvider instances, one per worker

        Raises:
            ProviderNotInitializedError: If this provider is not initialized
            ProviderRuntimeError: If worker instance creation fails
        """
        if not self._transformers_initialized:
            raise ProviderNotInitializedError(
                provider="MLProvider",
                capability="summarization",
            )

        workers = []
        for i in range(num_workers):
            try:
                logger.debug(f"Creating worker provider instance {i+1}/{num_workers}")
                # Create new provider with same config
                worker = MLProvider(self.cfg)
                # Initialize it (this loads models)
                worker.initialize()
                workers.append(worker)
            except Exception as e:
                logger.error(f"Failed to create worker provider instance {i+1}/{num_workers}: {e}")
                # Cleanup any successfully created workers
                for worker in workers:
                    try:
                        worker.cleanup()
                    except Exception:
                        pass
                raise ProviderRuntimeError(
                    message=f"Failed to create worker provider instances: {e}",
                    provider="MLProvider",
                    suggestion="Check model cache and device availability",
                ) from e

        logger.debug(f"Successfully created {len(workers)} worker provider instances")
        return workers

    def preload(self) -> None:
        """Preload ML models at startup if configured to use them.

        This method is called early in the pipeline to preload models before processing
        begins. It respects preload_models and dry_run configuration flags.

        Models are preloaded based on configuration:
        - Whisper: if transcribe_missing=True and transcription_provider="whisper"
        - Transformers: if generate_summaries=True and summary_provider="transformers"
        - spaCy: if auto_speakers=True and speaker_detector_provider="spacy"

        This method adds comprehensive logging and error handling for preloading,
        then calls initialize() to actually load the models.

        Raises:
            RuntimeError: If required model cannot be loaded
            ImportError: If ML dependencies are not installed
        """
        # Skip preloading if disabled or in dry run mode
        if not self.cfg.preload_models:
            logger.debug("Skipping model preloading (preload_models=False)")
            return

        if self.cfg.dry_run:
            logger.debug("Skipping model preloading (dry_run=True)")
            return

        # Determine which models need to be preloaded
        needs_whisper = self.cfg.transcribe_missing and self.cfg.transcription_provider == "whisper"
        needs_transformers = (
            self.cfg.generate_summaries and self.cfg.summary_provider == "transformers"
        )
        needs_spacy = self.cfg.auto_speakers and self.cfg.speaker_detector_provider == "spacy"

        # If no models need preloading, return early
        if not (needs_whisper or needs_transformers or needs_spacy):
            logger.debug("No ML models need preloading based on configuration")
            return

        # Log preloading start
        logger.info("Preloading ML models based on configuration...")
        preload_start = time.time()

        # Preload each model type with logging
        if needs_whisper:
            logger.info("Preloading Whisper model: %s", self.cfg.whisper_model)
            try:
                if not self._whisper_initialized:
                    self._initialize_whisper()
                logger.info("✓ Whisper model preloaded successfully")
            except Exception as e:
                error_msg = f"Failed to preload Whisper model: {e}"
                logger.error(error_msg)
                raise ProviderDependencyError(
                    message=error_msg,
                    provider="MLProvider/Whisper",
                    dependency="openai-whisper",
                    suggestion="Install with: pip install openai-whisper",
                ) from e

        if needs_spacy:
            from ...config import DEFAULT_NER_MODEL

            model_name = self.cfg.ner_model or DEFAULT_NER_MODEL
            logger.info("Preloading spaCy model: %s", model_name)
            try:
                if not self._spacy_initialized:
                    self._initialize_spacy()
                logger.info("✓ spaCy model preloaded successfully")
            except Exception as e:
                error_msg = f"Failed to preload spaCy model: {e}"
                logger.error(error_msg)
                raise ProviderDependencyError(
                    message=error_msg,
                    provider="MLProvider/spaCy",
                    dependency=model_name,
                    suggestion=f"Install with: python -m spacy download {model_name}",
                ) from e

        if needs_transformers:
            logger.info("Preloading Transformers models for summarization")
            try:
                if not self._transformers_initialized:
                    self._initialize_transformers()
                logger.info("✓ Transformers models preloaded successfully")
            except Exception as e:
                error_msg = f"Failed to preload Transformers models: {e}"
                logger.error(error_msg)
                raise ProviderDependencyError(
                    message=error_msg,
                    provider="MLProvider/Transformers",
                    dependency="transformers",
                    suggestion="Check model cache and network connectivity for model downloads",
                ) from e

        # Log completion with timing
        preload_elapsed = time.time() - preload_start
        logger.info("Model preloading completed in %.1fs", preload_elapsed)

    def _try_copy_from_preloaded(self) -> bool:
        """Try to copy model references from preloaded MLProvider instance.

        This avoids re-initializing models that were already loaded during preload,
        saving time and memory. Whisper and spaCy models are thread-safe for read
        operations, so we can share the model instances. Transformers models are not
        thread-safe, so we still need separate instances.

        Returns:
            True if models were copied from preloaded instance, False otherwise
        """
        try:
            from ...workflow import _preloaded_ml_provider

            if _preloaded_ml_provider is None:
                return False

            # Check if we ARE the preloaded instance - if so, skip copying (already initialized)
            if _preloaded_ml_provider is self:
                logger.debug("This is the preloaded instance, skipping copy")
                return True

            # Only copy if configurations match (same models, same device, etc.)
            if _preloaded_ml_provider.cfg.whisper_model != self.cfg.whisper_model:
                return False
            if _preloaded_ml_provider.cfg.whisper_device != self.cfg.whisper_device:
                return False
            if _preloaded_ml_provider.cfg.summary_model != self.cfg.summary_model:
                return False
            if _preloaded_ml_provider.cfg.summary_device != self.cfg.summary_device:
                return False

            copied = False

            # Copy Whisper model if preloaded and we need it
            # Whisper models are thread-safe for read operations, so we can share the instance
            if (
                self.cfg.transcribe_missing
                and not self._whisper_initialized
                and _preloaded_ml_provider._whisper_initialized
                and _preloaded_ml_provider._whisper_model is not None
            ):
                logger.debug("Reusing preloaded Whisper model instance (thread-safe for read)")
                self._whisper_model = _preloaded_ml_provider._whisper_model
                self._whisper_initialized = True
                copied = True

            # Copy spaCy model if preloaded and we need it
            # spaCy models are thread-safe for read operations, so we can share the instance
            if (
                self.cfg.auto_speakers
                and not self._spacy_initialized
                and _preloaded_ml_provider._spacy_initialized
                and _preloaded_ml_provider._spacy_nlp is not None
            ):
                logger.debug("Reusing preloaded spaCy model instance (thread-safe for read)")
                self._spacy_nlp = _preloaded_ml_provider._spacy_nlp
                self._spacy_heuristics = _preloaded_ml_provider._spacy_heuristics
                self._spacy_initialized = True
                copied = True

            # For Transformers, we cannot share instances (not thread-safe)
            # But we can note that models are preloaded to avoid redundant logging
            if (
                self.cfg.generate_summaries
                and not self._transformers_initialized
                and _preloaded_ml_provider._transformers_initialized
            ):
                logger.debug(
                    "Transformers models already preloaded, creating new instances "
                    "for thread safety (models will load faster from cache)"
                )
                copied = True

            return copied
        except (ImportError, AttributeError):
            # Preloaded instance not available or not accessible
            return False

    def initialize(self) -> None:
        """Initialize ML models based on provider configuration.

        Models are lazy-loaded based on configuration flags AND provider types:
        - Whisper: if transcribe_missing=True and transcription_provider="whisper"
        - spaCy: if auto_speakers=True and speaker_detector_provider="spacy"
        - Transformers: if generate_summaries=True and summary_provider="transformers"

        This method first tries to reuse models from a preloaded instance to avoid
        redundant initialization. If no preloaded instance is available, it initializes
        models normally.

        This method is idempotent and can be called multiple times safely.

        Note: If one component fails to initialize (e.g., Whisper due to missing cache),
        other components will still be initialized. This allows summarization to work
        even if transcription is unavailable.

        Note: Models are only loaded if the corresponding provider is set to use ML.
        For example, if summary_provider="openai", Transformers models will not be loaded
        even if generate_summaries=True.
        """
        # Check if we're already fully initialized (e.g., we're the preloaded instance)
        # If so, skip initialization entirely
        needs_whisper = self.cfg.transcribe_missing and self.cfg.transcription_provider == "whisper"
        needs_spacy = self.cfg.auto_speakers and self.cfg.speaker_detector_provider == "spacy"
        needs_transformers = (
            self.cfg.generate_summaries and self.cfg.summary_provider == "transformers"
        )

        already_initialized = (
            (not needs_whisper or self._whisper_initialized)
            and (not needs_spacy or self._spacy_initialized)
            and (not needs_transformers or self._transformers_initialized)
        )

        if already_initialized:
            logger.debug("Models already initialized, skipping redundant initialization")
            return

        # Try to copy from preloaded instance first (avoids redundant initialization)
        # This will actually copy model instances for Whisper and spaCy (thread-safe)
        preloaded_available = self._try_copy_from_preloaded()

        # Initialize Whisper if transcription enabled AND provider is Whisper
        # If Whisper was copied from preloaded, _whisper_initialized will be True and we skip
        # If Whisper fails, log warning but continue with other components
        if needs_whisper and not self._whisper_initialized:
            try:
                # If preloaded was available but copy didn't happen, model file is cached
                if preloaded_available:
                    logger.debug(
                        "Loading Whisper model (preloaded instance available, "
                        "using cached model file)"
                    )
                self._initialize_whisper()
            except Exception as e:
                logger.warning(
                    "Failed to initialize Whisper (transcription will be unavailable): %s", e
                )
                # Don't raise - allow other components to initialize

        # Initialize spaCy if speaker detection enabled AND provider is spaCy
        # If spaCy was copied from preloaded, _spacy_initialized will be True and we skip
        if needs_spacy and not self._spacy_initialized:
            try:
                # If preloaded was available but copy didn't happen, model is cached
                if preloaded_available:
                    logger.debug(
                        "Loading spaCy model (preloaded instance available, using cached model)"
                    )
                self._initialize_spacy()
            except Exception as e:
                logger.warning(
                    "Failed to initialize spaCy (speaker detection will be unavailable): %s", e
                )
                # Don't raise - allow other components to initialize

        # Initialize Transformers if summarization enabled AND provider is Transformers
        # Note: Transformers models cannot be shared across threads (not thread-safe)
        # so we always create new instances, but if preloaded, the model files are cached
        if needs_transformers and not self._transformers_initialized:
            if preloaded_available:
                logger.debug(
                    "Creating new Transformers instances (models already cached from preload)"
                )
            self._initialize_transformers()

    def _detect_whisper_device(self) -> str:
        """Detect the best device for Whisper transcription (Issue #387).

        Updated to check stage-level device config (transcription_device) first,
        which overrides provider-specific device (whisper_device).

        Returns:
            Device string: 'mps', 'cuda', or 'cpu'
        """
        # Stage-level device config takes precedence (Issue #387)
        if self.cfg.transcription_device:
            logger.debug(
                "Using stage-level transcription_device: %s", self.cfg.transcription_device
            )
            return self.cfg.transcription_device.lower()
        # Fallback to provider-specific device config
        if self.cfg.whisper_device:
            logger.debug("Using configured whisper_device: %s", self.cfg.whisper_device)
            return self.cfg.whisper_device

        # Auto-detect: prefer MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.debug("Auto-detected MPS (Apple Silicon) for Whisper")
                return "mps"
            if torch.cuda.is_available():
                logger.debug("Auto-detected CUDA for Whisper")
                return "cuda"
        except ImportError:
            pass

        logger.debug("Using CPU for Whisper (no GPU detected)")
        return "cpu"

    def _initialize_whisper(self) -> None:  # noqa: C901
        """Initialize Whisper model for transcription."""
        import time

        init_start = time.time()
        logger.debug("Initializing Whisper transcription (model: %s)", self.cfg.whisper_model)

        step_start = time.time()
        try:
            whisper_lib = _import_third_party_whisper()
        except ImportError:
            logger.error(
                "openai-whisper library not installed. Cannot load Whisper models. "
                "Install with: pip install openai-whisper && brew install ffmpeg"
            )
            raise ProviderDependencyError(
                message="openai-whisper library not installed",
                provider="MLProvider/Whisper",
                dependency="openai-whisper",
                suggestion="Install with: pip install openai-whisper && brew install ffmpeg",
            )
        import_time = time.time() - step_start
        logger.debug("  [TIMING] Import whisper library: %.3fs", import_time)

        # Use centralized fallback logic (config-driven, no hardcoded values)
        step_start = time.time()
        model_name, fallback_models = normalize_whisper_model_name(
            self.cfg.whisper_model, self.cfg.language
        )
        normalize_time = time.time() - step_start
        logger.debug("  [TIMING] Normalize model name: %.3fs", normalize_time)
        logger.debug("Loading Whisper model: %s", model_name)

        # Check cache directory for pre-cached models
        # Prefer local cache in project root, fallback to ~/.cache/whisper/
        step_start = time.time()
        whisper_cache = get_whisper_cache_dir()
        cache_dir_time = time.time() - step_start
        logger.debug("  [TIMING] Get cache directory: %.3fs", cache_dir_time)
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
                step_start = time.time()
                whisper_cache_str = str(whisper_cache)
                if attempt_model != model_name:
                    logger.debug(
                        "Trying fallback Whisper model: %s (requested: %s)",
                        attempt_model,
                        model_name,
                    )
                # Detect and use optimal device (MPS/CUDA/CPU)
                device_to_use = self._detect_whisper_device()
                device_detect_time = time.time() - step_start
                logger.debug(
                    "  [TIMING] Detect device: %.3fs (device: %s)",
                    device_detect_time,
                    device_to_use,
                )

                step_start = time.time()
                logger.debug("  [TIMING] Starting whisper_lib.load_model()...")
                # Suppress PyTorch's "Device set to use mps" stdout message
                import contextlib
                import io

                with contextlib.redirect_stdout(io.StringIO()):
                    model = whisper_lib.load_model(
                        attempt_model, download_root=whisper_cache_str, device=device_to_use
                    )
                load_model_time = time.time() - step_start
                logger.debug(
                    "  [TIMING] whisper_lib.load_model() completed: %.3fs", load_model_time
                )
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
                step_start = time.time()
                self._whisper_model = model
                self._whisper_initialized = True
                assign_time = time.time() - step_start
                total_time = time.time() - init_start
                logger.debug("  [TIMING] Assign model to instance: %.3fs", assign_time)
                logger.info(
                    "[TIMING BREAKDOWN] Whisper initialization total: %.3fs "
                    "(import: %.3fs, normalize: %.3fs, cache_dir: %.3fs, "
                    "device_detect: %.3fs, load_model: %.3fs, assign: %.3fs)",
                    total_time,
                    import_time,
                    normalize_time,
                    cache_dir_time,
                    device_detect_time,
                    load_model_time,
                    assign_time,
                )
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
        raise ProviderDependencyError(
            message=(
                f"Failed to load any Whisper model. Tried: {fallback_models}. "
                f"Last error: {last_error}"
            ),
            provider="MLProvider/Whisper",
            dependency="whisper-model",
            suggestion="Ensure models are cached with 'make preload-ml-models'",
        )

    def _initialize_spacy(self) -> None:
        """Initialize spaCy NER model for speaker detection."""
        if not self.cfg.auto_speakers:
            raise ProviderConfigError(
                message="Cannot initialize spaCy: auto_speakers is False",
                provider="MLProvider/spaCy",
                config_key="auto_speakers",
                suggestion="Set auto_speakers=True to enable speaker detection",
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
        logger.info("Initializing Transformers summarization models...")

        try:
            # Load MAP model (for chunk summarization)
            model_name = summarizer.select_summary_model(self.cfg)
            logger.info(f"Loading MAP model: {model_name}")
            # Stage-level device config takes precedence (Issue #387)
            device_to_use = (
                self.cfg.summarization_device
                if self.cfg.summarization_device
                else self.cfg.summary_device
            )
            self._map_model = summarizer.SummaryModel(
                model_name=model_name,
                device=device_to_use,
                cache_dir=self.cfg.summary_cache_dir,
            )
            logger.info(f"✓ MAP model loaded: {model_name}")

            # Load REDUCE model if different from MAP model (for final combine)
            reduce_model_name = summarizer.select_reduce_model(self.cfg, model_name)
            if reduce_model_name != model_name:
                logger.info(f"Loading REDUCE model: {reduce_model_name}")
                # Stage-level device config takes precedence (Issue #387)
                device_to_use = (
                    self.cfg.summarization_device
                    if self.cfg.summarization_device
                    else self.cfg.summary_device
                )
                self._reduce_model = summarizer.SummaryModel(
                    model_name=reduce_model_name,
                    device=device_to_use,
                    cache_dir=self.cfg.summary_cache_dir,
                )
                logger.info(f"✓ REDUCE model loaded: {reduce_model_name}")
            else:
                # Use MAP model for REDUCE phase if they're the same
                self._reduce_model = self._map_model
                logger.info(f"✓ Using MAP model for REDUCE phase (same model: {model_name})")

            self._transformers_initialized = True
            logger.info("Transformers summarization initialized successfully")
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
            ProviderNotInitializedError: If provider is not initialized
            FileNotFoundError: If audio file doesn't exist
            ProviderRuntimeError: If transcription fails
        """
        if not self._whisper_initialized or self._whisper_model is None:
            raise ProviderNotInitializedError(
                provider="MLProvider/Whisper",
                capability="transcription",
            )

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or "en")

        logger.debug("Transcribing audio file: %s (language: %s)", audio_path, effective_language)

        # Call internal transcription method
        result_dict, elapsed = self._transcribe_with_whisper(audio_path, effective_language)

        # Extract text from result
        text = result_dict.get("text", "").strip()
        if not text:
            raise ProviderRuntimeError(
                message="Transcription returned empty text",
                provider="MLProvider/Whisper",
                suggestion="Check audio file content and format",
            )

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
            raise ProviderNotInitializedError(
                provider="MLProvider/Whisper",
                capability="transcription",
            )

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
        total_start = time.time()

        step_start = time.time()
        with progress.progress_context(None, "Transcribing") as reporter:
            progress_setup_time = time.time() - step_start
            logger.debug("  [TIMING] Progress context setup: %.3fs", progress_setup_time)

            step_start = time.time()
            suppress_fp16_warning = getattr(self._whisper_model, "_is_cpu_device", False)
            logger.debug(
                "Invoking Whisper transcription: media=%s suppress_fp16_warning=%s dtype=%s",
                audio_path,
                suppress_fp16_warning,
                getattr(self._whisper_model, "dtype", None),
            )
            logger.debug("Transcribing with language=%s", language)
            prep_time = time.time() - step_start
            logger.debug("  [TIMING] Preparation (device check, logging): %.3fs", prep_time)

            # Intercept Whisper's tqdm progress calls and forward to our progress reporter
            # This prevents multiple progress bar lines while showing real progress
            if self._whisper_model is None:
                raise ProviderNotInitializedError(
                    provider="MLProvider/Whisper",
                    capability="transcription",
                )

            step_start = time.time()
            with _intercept_whisper_progress(reporter):
                intercept_setup_time = time.time() - step_start
                logger.debug("  [TIMING] Progress interceptor setup: %.3fs", intercept_setup_time)

                step_start = time.time()
                logger.debug("  [TIMING] Starting Whisper model.transcribe() call...")
                result = self._whisper_model.transcribe(
                    audio_path, task="transcribe", language=language, verbose=False
                )
                whisper_transcribe_time = time.time() - step_start
                logger.debug(
                    "  [TIMING] Whisper model.transcribe() completed: %.3fs",
                    whisper_transcribe_time,
                )

        elapsed = time.time() - total_start
        logger.info(
            "[TIMING BREAKDOWN] Transcription total: %.3fs "
            "(progress_setup: %.3fs, prep: %.3fs, intercept_setup: %.3fs, "
            "whisper_transcribe: %.3fs)",
            elapsed,
            progress_setup_time,
            prep_time,
            intercept_setup_time,
            whisper_transcribe_time,
        )
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
                raise ProviderConfigError(
                    message="Cannot detect speakers: auto_speakers is False",
                    provider="MLProvider/spaCy",
                    config_key="auto_speakers",
                    suggestion="Set auto_speakers=True to enable speaker detection",
                )
            self._initialize_spacy()

        # Use detect_speaker_names with adapted parameters
        # Pass self._spacy_nlp directly (required parameter)
        speaker_names, detected_hosts_set, detection_succeeded = (
            speaker_detection.detect_speaker_names(  # type: ignore[misc]
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
                raise ProviderConfigError(
                    message="Cannot detect speakers: auto_speakers is False",
                    provider="MLProvider/spaCy",
                    config_key="auto_speakers",
                    suggestion="Set auto_speakers=True to enable speaker detection",
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
                raise ProviderConfigError(
                    message="Cannot detect speakers: auto_speakers is False",
                    provider="MLProvider/spaCy",
                    config_key="auto_speakers",
                    suggestion="Set auto_speakers=True to enable speaker detection",
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
            ProviderNotInitializedError: If provider not initialized
        """
        if not self._transformers_initialized or not self._map_model:
            raise ProviderNotInitializedError(
                provider="MLProvider/Transformers",
                capability="summarization",
            )

        # Extract parameters with defaults from config
        chunk_size = (params.get("chunk_size") if params else None) or self.cfg.summary_chunk_size
        # Chunk-level parallelism: Use chunk_parallelism from params, fallback to config
        chunk_parallelism = (
            params.get("chunk_parallelism") if params else None
        ) or self.cfg.summary_chunk_parallelism
        # Use chunk_parallelism for batch_size when device is CPU
        batch_size = chunk_parallelism if self._map_model.device == "cpu" else None
        use_word_chunking = params.get("use_word_chunking") if params else None
        word_chunk_size = (
            params.get("word_chunk_size") if params else None
        ) or self.cfg.summary_word_chunk_size
        word_overlap = (
            params.get("word_overlap") if params else None
        ) or self.cfg.summary_word_overlap
        prompt = params.get("prompt") if params else None
        # Extract max_length and min_length from params (for backward compatibility)
        # These are legacy params that map to map_max_new_tokens/min_new_tokens
        max_length = params.get("max_length") if params else None
        min_length = params.get("min_length") if params else None

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
        # Check if we should return intermediates (for debugging/metrics)
        return_intermediates = params.get("return_intermediates", False) if params else False

        # All defaults come from Config (no hardcoded values)
        # Params dict can override Config defaults (for experiments)
        logger.info("=== Using ML Parameters from Config ===")
        logger.info("Map stage:")
        logger.info(f"  max_new_tokens: {self.cfg.summary_map_params.get('max_new_tokens')}")
        logger.info(f"  min_new_tokens: {self.cfg.summary_map_params.get('min_new_tokens')}")
        logger.info(f"  num_beams: {self.cfg.summary_map_params.get('num_beams')}")
        logger.info(
            f"  no_repeat_ngram_size: {self.cfg.summary_map_params.get('no_repeat_ngram_size')}"
        )
        logger.info(f"  length_penalty: {self.cfg.summary_map_params.get('length_penalty')}")
        logger.info(f"  early_stopping: {self.cfg.summary_map_params.get('early_stopping')}")
        logger.info(
            f"  repetition_penalty: {self.cfg.summary_map_params.get('repetition_penalty')}"
        )
        logger.info("Reduce stage:")
        logger.info(f"  max_new_tokens: {self.cfg.summary_reduce_params.get('max_new_tokens')}")
        logger.info(f"  min_new_tokens: {self.cfg.summary_reduce_params.get('min_new_tokens')}")
        logger.info(f"  num_beams: {self.cfg.summary_reduce_params.get('num_beams')}")
        logger.info(
            f"  no_repeat_ngram_size: {self.cfg.summary_reduce_params.get('no_repeat_ngram_size')}"
        )
        logger.info(f"  length_penalty: {self.cfg.summary_reduce_params.get('length_penalty')}")
        logger.info(f"  early_stopping: {self.cfg.summary_reduce_params.get('early_stopping')}")
        logger.info(
            f"  repetition_penalty: {self.cfg.summary_reduce_params.get('repetition_penalty')}"
        )
        logger.info("Tokenization:")
        logger.info(
            f"  map_max_input_tokens: {self.cfg.summary_tokenize.get('map_max_input_tokens')}"
        )
        logger.info(
            f"  reduce_max_input_tokens: {self.cfg.summary_tokenize.get('reduce_max_input_tokens')}"
        )
        logger.info(f"  truncation: {self.cfg.summary_tokenize.get('truncation')}")
        # Log chunking params if available
        if self.cfg.summary_word_chunk_size or word_chunk_size:
            logger.info("Chunking:")
            effective_word_chunk_size = word_chunk_size or self.cfg.summary_word_chunk_size
            effective_word_overlap = word_overlap or self.cfg.summary_word_overlap
            effective_use_word_chunking = use_word_chunking
            if effective_use_word_chunking is None:
                # Auto-detected
                effective_use_word_chunking = any(
                    model_keyword
                    in (
                        self._map_model.model_name if hasattr(self._map_model, "model_name") else ""
                    ).lower()
                    for model_keyword in ["bart", "pegasus", "distilbart"]
                )
            strategy = "word_chunking" if effective_use_word_chunking else "token_chunking"
            logger.info(f"  strategy: {strategy}")
            if effective_use_word_chunking:
                logger.info(f"  word_chunk_size: {effective_word_chunk_size}")
                logger.info(f"  word_overlap: {effective_word_overlap}")

        try:
            # Extract generation params: params dict overrides Config defaults
            # Start with Config defaults, then override with params dict if provided
            map_max_new_tokens = (
                params.get("map_max_new_tokens") if params else None
            ) or self.cfg.summary_map_params.get("max_new_tokens")
            map_min_new_tokens = (
                params.get("map_min_new_tokens") if params else None
            ) or self.cfg.summary_map_params.get("min_new_tokens")
            map_num_beams = (
                params.get("map_num_beams") if params else None
            ) or self.cfg.summary_map_params.get("num_beams")
            map_no_repeat_ngram_size = (
                params.get("map_no_repeat_ngram_size") if params else None
            ) or self.cfg.summary_map_params.get("no_repeat_ngram_size")
            map_length_penalty = (
                params.get("map_length_penalty") if params else None
            ) or self.cfg.summary_map_params.get("length_penalty")
            map_early_stopping = params.get("map_early_stopping") if params else None
            if map_early_stopping is None:
                map_early_stopping = self.cfg.summary_map_params.get("early_stopping")
            map_repetition_penalty = (
                params.get("map_repetition_penalty") if params else None
            ) or self.cfg.summary_map_params.get("repetition_penalty")
            map_encoder_no_repeat_ngram_size = (
                params.get("map_encoder_no_repeat_ngram_size") if params else None
            ) or self.cfg.summary_map_params.get("encoder_no_repeat_ngram_size")
            map_max_input_tokens = (
                params.get("map_max_input_tokens") if params else None
            ) or self.cfg.summary_tokenize.get("map_max_input_tokens")

            reduce_max_new_tokens = (
                params.get("reduce_max_new_tokens") if params else None
            ) or self.cfg.summary_reduce_params.get("max_new_tokens")
            reduce_min_new_tokens = (
                params.get("reduce_min_new_tokens") if params else None
            ) or self.cfg.summary_reduce_params.get("min_new_tokens")
            reduce_num_beams = (
                params.get("reduce_num_beams") if params else None
            ) or self.cfg.summary_reduce_params.get("num_beams")
            reduce_no_repeat_ngram_size = (
                params.get("reduce_no_repeat_ngram_size") if params else None
            ) or self.cfg.summary_reduce_params.get("no_repeat_ngram_size")
            reduce_length_penalty = (
                params.get("reduce_length_penalty") if params else None
            ) or self.cfg.summary_reduce_params.get("length_penalty")
            reduce_early_stopping = params.get("reduce_early_stopping") if params else None
            if reduce_early_stopping is None:
                reduce_early_stopping = self.cfg.summary_reduce_params.get("early_stopping")
            reduce_repetition_penalty = (
                params.get("reduce_repetition_penalty") if params else None
            ) or self.cfg.summary_reduce_params.get("repetition_penalty")
            reduce_encoder_no_repeat_ngram_size = (
                params.get("reduce_encoder_no_repeat_ngram_size") if params else None
            ) or self.cfg.summary_reduce_params.get("encoder_no_repeat_ngram_size")
            reduce_max_input_tokens = (
                params.get("reduce_max_input_tokens") if params else None
            ) or self.cfg.summary_tokenize.get("reduce_max_input_tokens")
            truncation = params.get("truncation") if params else None
            if truncation is None:
                truncation = self.cfg.summary_tokenize.get("truncation")
            preprocessing_profile = (
                params.get("preprocessing_profile") if params else None
            ) or "cleaning_v4"  # Default to cleaning_v4 (matches production baseline)

            result = summarizer.summarize_long_text(
                model=self._map_model,
                text=text,
                chunk_size=chunk_size or summarizer.BART_MAX_POSITION_EMBEDDINGS,
                max_length=max_length if max_length is not None else 150,
                min_length=min_length if min_length is not None else 30,
                batch_size=batch_size if self._map_model.device == "cpu" else None,
                prompt=prompt,
                use_word_chunking=use_word_chunking,
                word_chunk_size=word_chunk_size or summarizer.DEFAULT_WORD_CHUNK_SIZE,
                word_overlap=word_overlap or summarizer.DEFAULT_WORD_OVERLAP,
                reduce_model=self._reduce_model,
                return_intermediates=return_intermediates,
                # Pass new explicit generation params
                map_max_new_tokens=map_max_new_tokens,
                map_min_new_tokens=map_min_new_tokens,
                map_num_beams=map_num_beams,
                map_no_repeat_ngram_size=map_no_repeat_ngram_size,
                map_length_penalty=map_length_penalty,
                map_early_stopping=map_early_stopping,
                map_repetition_penalty=map_repetition_penalty,
                map_encoder_no_repeat_ngram_size=map_encoder_no_repeat_ngram_size,
                map_max_input_tokens=map_max_input_tokens,
                reduce_max_new_tokens=reduce_max_new_tokens,
                reduce_min_new_tokens=reduce_min_new_tokens,
                reduce_num_beams=reduce_num_beams,
                reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
                reduce_length_penalty=reduce_length_penalty,
                reduce_early_stopping=reduce_early_stopping,
                reduce_repetition_penalty=reduce_repetition_penalty,
                reduce_encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
                reduce_max_input_tokens=reduce_max_input_tokens,
                truncation=truncation,
                preprocessing_profile=preprocessing_profile,
                # Optional 2nd-pass distill parameters (Issue #387)
                enable_2nd_pass_distill=getattr(self.cfg, "summary_2nd_pass_distill", False),
                transcript_text=text,  # Use original transcript text
                episode_description=episode_description,
            )

            # Handle both return types (str or tuple)
            if isinstance(result, tuple):
                summary_text, intermediates = result
            else:
                summary_text = result
                intermediates = None

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

            result_dict: Dict[str, Any] = {
                "summary": summary_text,
                "summary_short": None,  # Transformers provider doesn't generate short summaries
                "metadata": metadata,
            }

            # Include intermediates if available
            if intermediates:
                result_dict["intermediates"] = intermediates

            return result_dict
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Handle "Already borrowed" error from Rust tokenizer in parallel execution
            if "already borrowed" in error_msg:
                logger.error(
                    f"Tokenizer threading error during summarization: {e}. "
                    "This can occur in parallel execution with Rust-based tokenizers. "
                    "Consider reducing parallelism or using sequential processing."
                )
                raise ProviderRuntimeError(
                    message=f"Summarization failed: {e}",
                    provider="MLProvider/Transformers",
                    suggestion=(
                        "Tokenizer threading error - try sequential processing "
                        "or reduce parallelism"
                    ),
                ) from e
            raise
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            raise ProviderRuntimeError(
                message=f"Summarization failed: {e}",
                provider="MLProvider/Transformers",
                suggestion="Check model cache and input text format",
            ) from e

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

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component).

        Returns:
            True if any component is initialized, False otherwise
        """
        return (
            self._whisper_initialized or self._spacy_initialized or self._transformers_initialized
        )

    @property
    def map_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the map model used for summarization.

        Returns:
            SummaryModel instance if initialized, None otherwise
        """
        return self._map_model

    @property
    def reduce_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the reduce model used for summarization.

        Returns:
            SummaryModel instance if initialized, None otherwise
        """
        return self._reduce_model

    @property
    def model(self) -> Optional[Any]:
        """Get the Whisper model used for transcription.

        Returns:
            Whisper model instance if initialized, None otherwise
        """
        return self._whisper_model

    @property
    def heuristics(self) -> Optional[Dict[str, Any]]:
        """Get the speaker detection heuristics.

        Returns:
            Heuristics dictionary if available, None otherwise
        """
        return self._spacy_heuristics
