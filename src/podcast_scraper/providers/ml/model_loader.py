"""Centralized ML model loading and downloading.

This module provides the ONLY place where ML models can be downloaded.
All model downloads go through these functions - libraries are never allowed
to download on their own (always use local_files_only=True when loading).

Both the CLI workflow and the preload script use these internal functions.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ... import config
from ...cache import (
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)

logger = logging.getLogger(__name__)


def preload_whisper_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Whisper models (centralized download function).

    This is the ONLY place where Whisper models can be downloaded.
    All other code must use local_files_only=True when loading.

    Args:
        model_names: List of Whisper model names to preload.
                    If None, uses WHISPER_MODELS env var or defaults to test default.
    """
    try:
        import whisper
    except ImportError:
        logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
        raise

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("WHISPER_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: tiny.en for local dev/tests (smallest, fastest)
            model_names = [config.TEST_DEFAULT_WHISPER_MODEL]

    if not model_names:
        logger.debug("Skipping Whisper model preloading (no models specified)")
        return

    logger.info("Preloading Whisper models...")

    # Use get_whisper_cache_dir() which checks for local cache first,
    # then falls back to ~/.cache/whisper/ (which CI caches between jobs)
    whisper_cache = get_whisper_cache_dir()

    # Ensure cache directory exists
    whisper_cache.mkdir(parents=True, exist_ok=True)

    whisper_cache_str = str(whisper_cache)
    for model_name in model_names:
        logger.info(f"  - {model_name}...")
        try:
            model_file = whisper_cache / f"{model_name}.pt"

            # Use download_root parameter to cache to local directory
            # This is the ONLY place where whisper downloads models
            model = whisper.load_model(model_name, download_root=whisper_cache_str)
            assert model is not None, f"Model {model_name} loaded but is None"
            assert (
                hasattr(model, "dims") and model.dims is not None
            ), f"Model {model_name} missing dims attribute"

            if model_file.exists():
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ Whisper {model_name} cached ({file_size_mb:.1f} MB)")
            else:
                logger.info(f"  ✓ Whisper {model_name} cached")
        except Exception as e:
            logger.error(f"  ✗ Failed to preload Whisper model {model_name}: {e}")
            raise


def preload_transformers_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Transformers models (centralized download function).

    This is the ONLY place where Transformers models can be downloaded.
    All other code must use local_files_only=True when loading.

    Args:
        model_names: List of Transformers model names to preload.
                    If None, uses TRANSFORMERS_MODELS env var or defaults to common models.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers torch")
        raise

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("TRANSFORMERS_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: preload only test defaults (small, fast models for local dev/testing)
            model_names = [
                config.TEST_DEFAULT_SUMMARY_MODEL,
                config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
            ]

    if not model_names:
        logger.debug("Skipping Transformers model preloading (no models specified)")
        return

    logger.info("Preloading Transformers models...")

    # Use get_transformers_cache_dir() which respects HF_HUB_CACHE env var
    cache_dir = get_transformers_cache_dir()

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        logger.info(f"  - {model_name}...")

        # Check if already cached
        model_cache_name = model_name.replace("/", "--")
        model_cache_path = cache_dir / f"models--{model_cache_name}"
        if model_cache_path.exists():
            existing_size = sum(
                f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
            )
            existing_size_mb = existing_size / (1024 * 1024)
            logger.info(f"    Status: Already cached ({existing_size_mb:.1f} MB)")
        else:
            logger.info("    Status: Not cached, downloading...")

        try:
            # Apply retry policy for transient errors (Issue #379)
            from requests.exceptions import (
                ConnectionError,
                HTTPError,
                RequestException,
                Timeout,
            )

            from ...utils.retry import retry_with_exponential_backoff

            # Define retryable exceptions for model loading
            retryable_exceptions = (
                ConnectionError,
                HTTPError,
                Timeout,
                RequestException,
                OSError,  # Network/IO errors
            )

            # Determine if model supports safetensors
            # Known models without safetensors: Pegasus, LED (LED disabled for different reason)
            model_lower = model_name.lower()
            use_safetensors = "pegasus" not in model_lower

            # Retry wrapper for tokenizer loading
            def _load_tokenizer():
                return AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    local_files_only=False,  # nosec B615
                    use_safetensors=use_safetensors,
                )

            # Retry wrapper for model loading
            def _load_model():
                return AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=str(cache_dir),
                    local_files_only=False,  # nosec B615
                    use_safetensors=use_safetensors,
                )

            # This is the ONLY place where transformers downloads models
            # Use retry with exponential backoff for transient errors
            retry_with_exponential_backoff(
                _load_tokenizer,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=retryable_exceptions,
            )

            retry_with_exponential_backoff(
                _load_model,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=retryable_exceptions,
            )

            # Calculate final size after download
            if model_cache_path.exists():
                total_size = sum(
                    f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                logger.info(f"  ✓ Downloaded and cached: {size_mb:.1f} MB")
            else:
                logger.info(f"  ✓ Downloaded and cached: {model_name}")
        except Exception as e:
            logger.error(f"  ✗ Failed to preload Transformers model {model_name}: {e}")
            raise
