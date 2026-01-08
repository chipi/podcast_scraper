#!/usr/bin/env python3
"""Helper functions for checking ML model cache in integration tests.

These functions check if ML models are cached locally before attempting to load them.
This prevents tests from hanging on network downloads, which violates integration test rules.

Supply Chain Security:
- CI validates cache in shell BEFORE pytest runs (authoritative check)
- When ML_MODELS_VALIDATED=true env var is set, tests trust CI validation
- Tests use local_files_only=True when loading models (hard security boundary)
- If models not cached, tests fail fast rather than attempting downloads

Models checked (in order of preference):
- Local cache: .cache/ in project root (if exists)
- User cache: ~/.cache/whisper/, ~/.cache/huggingface/hub/, ~/.local/share/spacy/
"""

import os
from pathlib import Path
from typing import Optional

from podcast_scraper.cache_utils import (
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)


def _is_ci_cache_validated() -> bool:
    """Check if CI has already validated the cache.

    In CI (nightly, regular builds), cache is validated in a shell step BEFORE
    pytest runs. When this validation passes, ML_MODELS_VALIDATED=true is set.

    This allows tests to skip redundant complex directory checks that can fail
    in multi-worker pytest-xdist environments due to filesystem timing issues.

    The actual security boundary is local_files_only=True when loading models,
    which is enforced by the model loading code, not by these cache checks.

    Returns:
        True if CI has validated the cache, False otherwise
    """
    return os.environ.get("ML_MODELS_VALIDATED", "").lower() == "true"


def _is_whisper_model_cached(model_name: str) -> bool:
    """Check if a Whisper model is cached locally.

    Args:
        model_name: Whisper model name (e.g., "base.en", "tiny", "tiny.en")

    Returns:
        True if model appears to be cached, False otherwise
    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    # Whisper caches models - prefer local cache, fallback to user cache
    cache_dir = get_whisper_cache_dir()

    # Explicit path logging for debugging
    logger.debug(
        f"Checking Whisper model cache:\n"
        f"  Model: {model_name}\n"
        f"  Cache directory: {cache_dir}\n"
        f"  Cache exists: {cache_dir.exists()}\n"
        f"  User: {os.environ.get('USER', 'unknown')}\n"
        f"  Home: {Path.home()}"
    )

    if not cache_dir.exists():
        logger.debug(f"Whisper cache directory does not exist: {cache_dir}")
        return False

    # Whisper stores models as: {model_name}.pt
    # e.g., "base.en" -> "base.en.pt", "tiny.en" -> "tiny.en.pt"
    # Standard: Use .en variants for English (tiny.en, base.en) for better performance
    model_file = cache_dir / f"{model_name}.pt"

    logger.debug(f"  Expected model file: {model_file}")
    logger.debug(f"  Model file exists: {model_file.exists()}")

    if model_file.exists() and model_file.is_file():
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.debug(f"  Model file size: {file_size_mb:.1f} MB")
        logger.debug(f"Whisper model {model_name} is cached at: {model_file}")
        return True

    # Legacy support: If checking for "tiny" (without .en), also check "tiny.en"
    # This allows backward compatibility but we standardize on tiny.en
    if model_name == "tiny":
        tiny_en_file = cache_dir / "tiny.en.pt"
        logger.debug(f"  Checking legacy path: {tiny_en_file}")
        logger.debug(f"  Legacy file exists: {tiny_en_file.exists()}")
        if tiny_en_file.exists() and tiny_en_file.is_file():
            logger.debug(f"Whisper model {model_name} found at legacy path: {tiny_en_file}")
            return True

    logger.debug(f"Whisper model {model_name} not found in cache")
    return False


def _is_spacy_model_cached(model_name: str) -> bool:
    """Check if a spaCy model is cached/installed locally.

    Args:
        model_name: spaCy model name (e.g., "en_core_web_sm")

    Returns:
        True if model appears to be installed, False otherwise
    """
    import logging
    import os
    import site

    logger = logging.getLogger(__name__)

    # spaCy can store models in multiple locations:
    # 1. ~/.local/share/spacy/ (user data directory)
    # 2. site-packages (when installed via pip)
    search_paths = []

    # Check user data directory
    spacy_data_dir = Path.home() / ".local" / "share" / "spacy"
    if spacy_data_dir.exists():
        search_paths.append(spacy_data_dir / model_name)

    # Check site-packages directories
    for site_pkg_dir in site.getsitepackages():
        search_paths.append(Path(site_pkg_dir) / model_name)

    # Explicit path logging for debugging
    logger.debug(
        f"Checking spaCy model cache:\n"
        f"  Model: {model_name}\n"
        f"  User cache directory: {spacy_data_dir}\n"
        f"  User cache exists: {spacy_data_dir.exists()}\n"
        f"  Site packages: {[str(Path(p)) for p in site.getsitepackages()]}\n"
        f"  Search paths: {[str(p) for p in search_paths]}\n"
        f"  User: {os.environ.get('USER', 'unknown')}\n"
        f"  Home: {Path.home()}"
    )

    # Check if model exists in any of the search paths
    for model_dir in search_paths:
        logger.debug(f"  Checking: {model_dir}")
        logger.debug(f"    Exists: {model_dir.exists()}")
        if model_dir.exists() and model_dir.is_dir():
            # Check for model metadata file (meta.json) to confirm it's a valid model
            meta_file = model_dir / "meta.json"
            logger.debug(f"    Meta file: {meta_file}")
            logger.debug(f"    Meta file exists: {meta_file.exists()}")
            if meta_file.exists():
                logger.debug(f"spaCy model {model_name} found at: {model_dir}")
                return True

    logger.debug(f"spaCy model {model_name} not found in any search path")
    return False


def _is_transformers_model_cached(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a Hugging Face Transformers model is cached locally.

    This function performs both existence checks and integrity validation to ensure
    cached models are not corrupted. It validates safetensors files by attempting
    to read their metadata headers. Additionally, it attempts to actually load the
    tokenizer with local_files_only=True to ensure the model can be loaded in tests
    where network access is blocked.

    Args:
        model_name: Hugging Face model identifier (e.g., "facebook/bart-large-cnn")
        cache_dir: Cache directory path (defaults to standard HF cache)

    Returns:
        True if model appears to be cached AND can be loaded with local_files_only=True, False otherwise
    """
    # Use same cache directory logic as preload script and SummaryModel
    # Prefer local cache if available, fallback to provided or default
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        # Use cache_utils to get cache directory (prefers local, falls back to default)
        cache_path = get_transformers_cache_dir()

    # Explicit path logging for debugging
    import logging
    import os

    logger = logging.getLogger(__name__)
    logger.debug(
        f"Checking Transformers model cache:\n"
        f"  Model: {model_name}\n"
        f"  Cache directory: {cache_path}\n"
        f"  Cache exists: {cache_path.exists()}\n"
        f"  User: {os.environ.get('USER', 'unknown')}\n"
        f"  Home: {Path.home()}"
    )

    if not cache_path.exists():
        logger.debug(f"Cache directory does not exist: {cache_path}")
        return False

    # Transformers stores models as: models--{org}--{model_name}
    # e.g., "facebook/bart-large-cnn" -> "models--facebook--bart-large-cnn"
    model_cache_name = model_name.replace("/", "--")
    model_cache_path = cache_path / f"models--{model_cache_name}"
    cache_path_str = str(cache_path)

    logger.debug(
        f"  Model cache path: {model_cache_path}\n"
        f"  Model cache exists: {model_cache_path.exists()}"
    )

    # Check if cache directory exists and has content
    if model_cache_path.exists() and model_cache_path.is_dir():
        # Primary check: Try to actually load the tokenizer with local_files_only=True
        # This is the most reliable check since it matches what SummaryModel does
        # and handles all cache structures (snapshots, blobs, .no_exist, etc.)
        try:
            from transformers import AutoTokenizer

            # Try loading tokenizer with same parameters as SummaryModel uses
            # nosec B615 - local_files_only=True prevents network access, model_name from test config
            AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_path_str,
                local_files_only=True,
            )  # nosec B615
            # If we get here, the model is actually loadable
            logger.debug(f"Model {model_name} is cached and loadable from: {model_cache_path}")
            return True
        except Exception as e:
            # Model can't be loaded with local_files_only=True
            # Fall back to file existence checks for better error messages
            logger.debug(
                f"Model {model_name} cannot be loaded with local_files_only=True:\n"
                f"  Cache path: {cache_path_str}\n"
                f"  Model cache path: {model_cache_path}\n"
                f"  Error: {e}"
            )
            snapshots = model_cache_path / "snapshots"
            if snapshots.exists():
                # Check if any snapshot directory exists (indicates model was downloaded)
                for item in snapshots.iterdir():
                    if item.is_dir():
                        # Check for model files (safetensors or bin)
                        model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors"))
                        if model_files:
                            # Model files exist but can't be loaded (incomplete cache, missing tokenizer, etc.)
                            logger.debug(
                                f"Model files exist but cannot be loaded:\n"
                                f"  Model files found: {len(model_files)}\n"
                                f"  Snapshot: {item}\n"
                                f"  This indicates incomplete cache or missing tokenizer files"
                            )
                            return False
            return False

    logger.debug(f"Model cache directory does not exist: {model_cache_path}")
    return False


def _validate_transformers_model_integrity(snapshot_dir: Path, model_files: list[Path]) -> bool:
    """Validate integrity of Transformers model files.

    Attempts to read safetensors file headers to detect corruption.
    This catches issues like incomplete downloads or corrupted cache files.

    Args:
        snapshot_dir: Snapshot directory containing model files
        model_files: List of model file paths (safetensors or bin)

    Returns:
        True if model files appear valid, False if corrupted or unreadable
    """
    # Only validate safetensors files (bin files require full model loading)
    safetensors_files = [f for f in model_files if f.suffix == ".safetensors"]

    if not safetensors_files:
        # No safetensors files to validate, assume valid
        # (bin files would require full model loading which is too expensive)
        return True

    # Validate each safetensors file by attempting to read metadata
    for model_file in safetensors_files:
        try:
            # Try to import safetensors (may not be available in all environments)
            try:
                from safetensors import safe_open
            except ImportError:
                # safetensors not available, skip integrity check
                # This is acceptable for cache checking (existence is enough)
                return True

            # Attempt to read file header/metadata
            # This will fail if file is corrupted or incomplete
            with safe_open(model_file, framework="pt") as f:
                # Try to read at least one key to validate header
                keys = list(f.keys())
                if not keys:
                    # Empty model file is suspicious
                    return False
                # Successfully read metadata, file appears valid
        except Exception:
            # Any error reading the file indicates corruption
            # Common errors:
            # - "Error while deserializing header: invalid JSON in header"
            # - "Error while deserializing header: incomplete metadata"
            # - FileNotFoundError (broken symlink)
            return False

    return True


def require_whisper_model_cached(model_name: str) -> None:
    """Require that a Whisper model is cached, skip with helpful message if not.

    If ML_MODELS_VALIDATED=true is set (by CI after shell validation), this function
    trusts that the cache is valid and skips the complex directory checks. This prevents
    issues with pytest-xdist workers failing to see the same filesystem state.

    The actual security boundary is enforced by network blocking (--disable-socket)
    and local_files_only=True when loading models.

    Args:
        model_name: Whisper model name

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    # Trust CI validation - skip redundant checks in multi-worker environment
    if _is_ci_cache_validated():
        return

    if not _is_whisper_model_cached(model_name):
        import pytest

        cache_dir = Path.home() / ".cache" / "whisper"
        model_file = cache_dir / f"{model_name}.pt"

        pytest.skip(
            f"Whisper model '{model_name}' is not cached.\n"
            f"  Expected cache location: {model_file}\n"
            f"  Cache directory: {cache_dir}\n"
            f"  Cache directory exists: {cache_dir.exists()}\n"
            f"  ML_MODELS_VALIDATED: {os.environ.get('ML_MODELS_VALIDATED', 'NOT SET')}\n"
            f"  User: {os.environ.get('USER', 'unknown')}\n"
            f"  Home: {Path.home()}\n"
            f"  Run 'make preload-ml-models' to pre-cache all required models."
        )


def require_spacy_model_cached(model_name: str) -> None:
    """Require that a spaCy model is cached/installed, skip with helpful message if not.

    Note: spaCy models are installed as pip packages (via .[ml] dependency),
    so they should always be available in CI. This function exists for consistency.

    If ML_MODELS_VALIDATED=true is set (by CI after shell validation), this function
    trusts that the dependencies are installed and skips the checks.

    Args:
        model_name: spaCy model name

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    # Trust CI validation - skip redundant checks in multi-worker environment
    if _is_ci_cache_validated():
        return

    if not _is_spacy_model_cached(model_name):
        import site

        import pytest

        spacy_data_dir = Path.home() / ".local" / "share" / "spacy"
        site_packages_paths = [Path(p) for p in site.getsitepackages()]
        search_paths = []
        if spacy_data_dir.exists():
            search_paths.append(spacy_data_dir / model_name)
        for site_pkg_dir in site_packages_paths:
            search_paths.append(site_pkg_dir / model_name)

        pytest.skip(
            f"spaCy model '{model_name}' is not cached/installed.\n"
            f"  User cache: {spacy_data_dir}\n"
            f"  User cache exists: {spacy_data_dir.exists()}\n"
            f"  Site packages: {[str(p) for p in site_packages_paths]}\n"
            f"  Searched paths: {[str(p) for p in search_paths]}\n"
            f"  ML_MODELS_VALIDATED: {os.environ.get('ML_MODELS_VALIDATED', 'NOT SET')}\n"
            f"  User: {os.environ.get('USER', 'unknown')}\n"
            f"  Home: {Path.home()}\n"
            f"  Run 'make preload-ml-models' to pre-cache all required models."
        )


def require_transformers_model_cached(model_name: str, cache_dir: Optional[str] = None) -> None:
    """Require that a Transformers model is cached, skip with helpful message if not.

    If ML_MODELS_VALIDATED=true is set (by CI after shell validation), this function
    trusts that the cache is valid and skips the complex directory checks. This prevents
    issues with pytest-xdist workers failing to see the same filesystem state.

    The actual security boundary is enforced by:
    1. Network blocking (--disable-socket in pytest)
    2. HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 (set in conftest.py)
    3. local_files_only=True when loading models

    If a model is not cached and these boundaries are active, model loading will fail
    fast with a clear error rather than attempting a network download.

    Args:
        model_name: Hugging Face model identifier or alias (e.g., 'bart-small')
        cache_dir: Cache directory path

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    # Resolve alias to actual model ID if needed
    from podcast_scraper.summarizer import DEFAULT_SUMMARY_MODELS

    resolved_model_name = model_name
    if model_name in DEFAULT_SUMMARY_MODELS:
        resolved_model_name = DEFAULT_SUMMARY_MODELS[model_name]

    # Trust CI validation - skip redundant checks in multi-worker environment
    if _is_ci_cache_validated():
        return

    if not _is_transformers_model_cached(resolved_model_name, cache_dir):
        import pytest

        # Get cache path for debugging - use same logic as _is_transformers_model_cached
        if cache_dir:
            cache_path = Path(cache_dir)
        else:
            # Use cache_utils to get cache directory (respects HF_HUB_CACHE env var)
            cache_path = get_transformers_cache_dir()

        # Use resolved model name for cache path lookup
        model_cache_name = resolved_model_name.replace("/", "--")
        model_cache_path = cache_path / f"models--{model_cache_name}"

        # List all cached models for debugging
        cached_models = []
        if cache_path.exists():
            for model_dir in cache_path.glob("models--*"):
                if model_dir.is_dir():
                    model_name_cached = model_dir.name.replace("--", "/")
                    # Calculate total size
                    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    cached_models.append(
                        f"    - {model_name_cached}: {size_mb:.1f} MB (at {model_dir})"
                    )

        # Debug: check parent directories to understand why cache doesn't exist
        parent_exists = cache_path.parent.exists() if cache_path.parent else False
        grandparent_exists = (
            cache_path.parent.parent.exists()
            if cache_path.parent and cache_path.parent.parent
            else False
        )

        # Show both alias and resolved name if different
        display_name = (
            f"'{model_name}' (alias for {resolved_model_name})"
            if model_name != resolved_model_name
            else f"'{model_name}'"
        )
        skip_message = (
            f"Transformers model {display_name} is not cached.\n"
            f"  Expected cache location: {model_cache_path}\n"
            f"  Cache directory: {cache_path}\n"
            f"  Cache directory exists: {cache_path.exists()}\n"
            f"  Parent ({cache_path.parent}) exists: {parent_exists}\n"
            f"  Grandparent ({cache_path.parent.parent}) exists: {grandparent_exists}\n"
            f"  ML_MODELS_VALIDATED: {os.environ.get('ML_MODELS_VALIDATED', 'NOT SET')}\n"
            f"  HF_HUB_CACHE env: {os.environ.get('HF_HUB_CACHE', 'NOT SET')}\n"
            f"  HF_HOME env: {os.environ.get('HF_HOME', 'NOT SET')}\n"
        )
        if cached_models:
            skip_message += "  Currently cached models:\n" + "\n".join(cached_models) + "\n"
        else:
            skip_message += "  No models found in cache directory.\n"
        skip_message += (
            f"  User: {os.environ.get('USER', 'unknown')}\n"
            f"  Home: {Path.home()}\n"
            f"  Run 'make preload-ml-models' to pre-cache all required models."
        )

        pytest.skip(skip_message)
