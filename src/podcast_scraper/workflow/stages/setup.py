"""Setup stage for pipeline initialization.

This module handles environment setup, ML model preloading, and output directory
configuration.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Optional, Tuple

from ... import config, filesystem

logger = logging.getLogger(__name__)

# Module-level registry for preloaded MLProvider instance
# This allows factories to reuse the same instance across capabilities
_preloaded_ml_provider: Optional[Any] = None


def initialize_ml_environment() -> None:
    """Initialize environment variables for ML libraries.

    This function sets default environment variables that improve the behavior of ML libraries
    (PyTorch, Transformers, Hugging Face Hub) in production. These defaults can be overridden
    by setting the environment variables in `.env` file or system environment.

    Environment Variables:
    - HF_HUB_DISABLE_PROGRESS_BARS: Disables progress bars to avoid misleading
      "Downloading" messages when loading models from cache. Default: "1" (disabled)
      Can be overridden in `.env` file or system environment.
    - OMP_NUM_THREADS, MKL_NUM_THREADS, TORCH_NUM_THREADS: Not set here by default
      (allows full CPU utilization). Users can set these in `.env` file or system
      environment to limit thread usage if needed.

    This is called automatically when the pipeline starts, but can also be called
    manually if needed.

    Note:
        - These settings are different from test environment settings, which limit
          threads to reduce resource usage. In production, we want full performance.
        - User-set values (from `.env` or system environment) take precedence over
          these defaults (respects `override=False` in dotenv loading).
        - See `examples/.env.example` for documentation of all environment variables.

    See Also:
        - `examples/.env.example`: Template with all environment variables
        - `docs/api/CONFIGURATION.md`: Complete environment variable documentation
    """
    # Disable Hugging Face Hub progress bars to avoid misleading "Downloading" messages
    # when loading models from cache. This is especially important when models are
    # already cached and no actual download is happening.
    # Only set if not already set by user (respects .env file or system environment)
    if "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        logger.debug("Set HF_HUB_DISABLE_PROGRESS_BARS=1 to suppress misleading progress bars")

    # Note: We don't set OMP_NUM_THREADS, MKL_NUM_THREADS, or TORCH_NUM_THREADS here
    # because in production we want to use all available CPU cores for best performance.
    # Users can set these environment variables in `.env` file or system environment
    # if they want to limit threads (e.g., in Docker containers with limited resources).
    # In tests, these are set to "1" to reduce resource usage (see tests/conftest.py).
    # See `examples/.env.example` for documentation.


def should_preload_ml_models(cfg: config.Config) -> bool:
    """Check if ML models should be preloaded based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        True if any ML models need to be preloaded, False otherwise
    """
    # Check if any ML models are needed
    needs_whisper = cfg.transcribe_missing and cfg.transcription_provider == "whisper"
    needs_transformers = cfg.generate_summaries and cfg.summary_provider == "transformers"
    needs_spacy = cfg.auto_speakers and cfg.speaker_detector_provider == "spacy"

    return needs_whisper or needs_transformers or needs_spacy


def ensure_ml_models_cached(cfg: config.Config) -> None:
    """Ensure required ML models are cached, downloading them if needed.

    This function checks if required models are cached, and if not, downloads them
    using the centralized preload script logic. This is the ONLY place where models
    can be downloaded - libraries are never allowed to download on their own.

    Args:
        cfg: Configuration object

    Note:
        This only downloads models if they're not already cached. It uses the same
        preload logic as the preload script, ensuring all downloads go through our
        centralized mechanism. Models are then loaded with local_files_only=True
        to prevent libraries from attempting their own downloads.
    """
    # Skip if preloading is disabled or in dry run
    if not cfg.preload_models or cfg.dry_run:
        return

    # Skip if no ML models are needed
    if not should_preload_ml_models(cfg):
        return

    # Skip in test environments (tests should use pre-cached models)
    if config._is_test_environment():
        return

    try:
        from ...cache_utils import (
            get_transformers_cache_dir,
            get_whisper_cache_dir,
        )

        models_to_download = []

        # Check Whisper models
        if cfg.transcribe_missing and cfg.transcription_provider == "whisper":
            whisper_cache = get_whisper_cache_dir()
            model_file = whisper_cache / f"{cfg.whisper_model}.pt"
            if not model_file.exists():
                models_to_download.append(("whisper", cfg.whisper_model))
                logger.info(f"Whisper model {cfg.whisper_model} not cached, will download")

        # Check Transformers models
        if cfg.generate_summaries and cfg.summary_provider == "transformers":
            from ... import summarizer

            transformers_cache = get_transformers_cache_dir()
            # Get MAP model
            map_model = summarizer.select_summary_model(cfg)
            # Resolve alias to actual model ID
            resolved_map = summarizer.DEFAULT_SUMMARY_MODELS.get(map_model, map_model)
            model_cache_name = resolved_map.replace("/", "--")
            model_cache_path = transformers_cache / f"models--{model_cache_name}"
            if not model_cache_path.exists():
                models_to_download.append(("transformers", map_model))
                logger.info(f"Transformers model {map_model} not cached, will download")

            # Get REDUCE model (might be different)
            reduce_model = summarizer.select_reduce_model(cfg, map_model)
            if reduce_model != map_model:
                resolved_reduce = summarizer.DEFAULT_SUMMARY_MODELS.get(reduce_model, reduce_model)
                reduce_cache_name = resolved_reduce.replace("/", "--")
                reduce_cache_path = transformers_cache / f"models--{reduce_cache_name}"
                if not reduce_cache_path.exists():
                    models_to_download.append(("transformers", reduce_model))
                    logger.info(f"Transformers model {reduce_model} not cached, will download")

        # Download missing models using centralized preload functions
        if models_to_download:
            logger.info("Downloading missing ML models (this may take a few minutes)...")
            try:
                # Import preload functions from internal package module
                # This is the ONLY place where models can be downloaded
                from ...model_loader import (
                    preload_transformers_models,
                    preload_whisper_models,
                )

                # Group models by type
                whisper_models = [m[1] for m in models_to_download if m[0] == "whisper"]
                transformers_models = [m[1] for m in models_to_download if m[0] == "transformers"]

                if whisper_models:
                    preload_whisper_models(whisper_models)
                if transformers_models:
                    preload_transformers_models(transformers_models)

                logger.info("Missing models downloaded and cached successfully")
            except ImportError:
                logger.warning(
                    "Model loader module not available. "
                    "You may need to run 'make preload-ml-models' manually."
                )
            except Exception as e:
                logger.warning(
                    f"Could not automatically download models: {e}. "
                    "You may need to run 'make preload-ml-models' manually."
                )
                # Don't fail - let the normal loading process handle the error

    except ImportError:
        # Preload script not available - that's okay, we'll try to load anyway
        pass
    except Exception as e:
        # Don't fail on cache check errors - let normal loading handle it
        logger.debug(f"Error checking model cache: {e}")


def preload_ml_models_if_needed(cfg: config.Config) -> None:
    """Preload ML models early in the pipeline if configured to use them.

    This function creates an MLProvider instance and calls preload() on it,
    storing the instance in a module-level registry for reuse by factories.

    Args:
        cfg: Configuration object

    Raises:
        RuntimeError: If required model cannot be loaded
        ImportError: If ML dependencies are not installed
    """
    global _preloaded_ml_provider

    # Skip if preloading is disabled or in dry run
    if not cfg.preload_models or cfg.dry_run:
        return

    # Skip if no ML models are needed
    if not should_preload_ml_models(cfg):
        return

    # Ensure models are cached (download if needed, but only in production)
    ensure_ml_models_cached(cfg)

    # Create MLProvider instance and preload models
    try:
        from ..ml.ml_provider import MLProvider

        _preloaded_ml_provider = MLProvider(cfg)
        _preloaded_ml_provider.preload()
        logger.debug("ML models preloaded successfully, instance stored for reuse")
    except ImportError as e:
        logger.warning("ML dependencies not available, skipping model preloading: %s", e)
        _preloaded_ml_provider = None
    except Exception as e:
        logger.error("Failed to preload ML models: %s", e)
        _preloaded_ml_provider = None
        # Re-raise to fail fast for required models
        raise


def setup_pipeline_environment(cfg: config.Config) -> Tuple[str, Optional[str]]:
    """Setup output directory and handle cleanup if needed.

    Creates the output directory structure based on configuration, optionally
    adding a run ID subdirectory. If clean_output is enabled, removes any
    existing output directory before creating it.

    Args:
        cfg: Configuration object with output_dir, run_id, clean_output,
            and dry_run settings

    Returns:
        Tuple[str, Optional[str]]: A tuple containing:
            - effective_output_dir (str): Full path to output directory
              (may include run_id subdirectory)
            - run_suffix (Optional[str]): Run ID suffix if run_id was provided,
              None otherwise

    Raises:
        RuntimeError: If output directory cleanup fails when clean_output=True
        OSError: If directory creation fails

    Example:
        >>> cfg = Config(rss_url="...", output_dir="./out", run_id="test_run")
        >>> output_dir, run_suffix = setup_pipeline_environment(cfg)
        >>> print(output_dir)  # "./out/test_run"
        >>> print(run_suffix)  # "test_run"
    """
    effective_output_dir, run_suffix = filesystem.setup_output_directory(cfg)
    logger.debug("Effective output dir=%s (run_suffix=%s)", effective_output_dir, run_suffix)

    if cfg.clean_output and cfg.dry_run:
        if os.path.exists(effective_output_dir):
            logger.info(
                "Dry-run: would remove existing output directory (--clean-output): %s",
                effective_output_dir,
            )
    elif cfg.clean_output:
        try:
            if os.path.exists(effective_output_dir):
                shutil.rmtree(effective_output_dir)
                logger.info(
                    "Removed existing output directory (--clean-output): %s",
                    effective_output_dir,
                )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to clean output directory {effective_output_dir}: {exc}"
            ) from exc

    if cfg.dry_run:
        logger.info(f"Dry-run: not creating output directory {effective_output_dir}")
    else:
        os.makedirs(effective_output_dir, exist_ok=True)
        # Recreate subdirectories after cleanup (if clean_output was used)
        # or ensure they exist (setup_output_directory creates them, but they may have been removed)
        transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
        metadata_dir = os.path.join(effective_output_dir, filesystem.METADATA_SUBDIR)
        os.makedirs(transcripts_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

    return effective_output_dir, run_suffix


def get_preloaded_ml_provider() -> Optional[Any]:
    """Get the preloaded ML provider instance.

    Returns:
        The preloaded MLProvider instance, or None if not preloaded
    """
    return _preloaded_ml_provider
