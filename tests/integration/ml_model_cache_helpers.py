#!/usr/bin/env python3
"""Helper functions for checking ML model cache in integration tests.

These functions check if ML models are cached locally before attempting to load them.
This prevents tests from hanging on network downloads, which violates integration test rules.

Models checked:
- Whisper models: ~/.cache/whisper/
- spaCy models: ~/.local/share/spacy/
- Transformers models: ~/.cache/huggingface/hub/
"""

from pathlib import Path
from typing import Optional


def _is_whisper_model_cached(model_name: str) -> bool:
    """Check if a Whisper model is cached locally.

    Args:
        model_name: Whisper model name (e.g., "base.en", "tiny", "tiny.en")

    Returns:
        True if model appears to be cached, False otherwise
    """
    # Whisper caches models in ~/.cache/whisper/
    cache_dir = Path.home() / ".cache" / "whisper"

    if not cache_dir.exists():
        return False

    # Whisper stores models as: {model_name}.pt
    # e.g., "base.en" -> "base.en.pt", "tiny.en" -> "tiny.en.pt"
    # Standard: Use .en variants for English (tiny.en, base.en) for better performance
    model_file = cache_dir / f"{model_name}.pt"
    if model_file.exists() and model_file.is_file():
        return True

    # Legacy support: If checking for "tiny" (without .en), also check "tiny.en"
    # This allows backward compatibility but we standardize on tiny.en
    if model_name == "tiny":
        tiny_en_file = cache_dir / "tiny.en.pt"
        if tiny_en_file.exists() and tiny_en_file.is_file():
            return True

    return False


def _is_spacy_model_cached(model_name: str) -> bool:
    """Check if a spaCy model is cached/installed locally.

    Args:
        model_name: spaCy model name (e.g., "en_core_web_sm")

    Returns:
        True if model appears to be installed, False otherwise
    """
    import site

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

    # Check if model exists in any of the search paths
    for model_dir in search_paths:
        if model_dir.exists() and model_dir.is_dir():
            # Check for model metadata file (meta.json) to confirm it's a valid model
            meta_file = model_dir / "meta.json"
            if meta_file.exists():
                return True

    return False


def _is_transformers_model_cached(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """Check if a Hugging Face Transformers model is cached locally.

    This function performs both existence checks and integrity validation to ensure
    cached models are not corrupted. It validates safetensors files by attempting
    to read their metadata headers.

    Args:
        model_name: Hugging Face model identifier (e.g., "facebook/bart-large-cnn")
        cache_dir: Cache directory path (defaults to standard HF cache)

    Returns:
        True if model appears to be cached AND valid, False otherwise
    """
    # Use same cache directory logic as SummaryModel
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        # Use same logic as summarizer module
        hf_cache_base = Path.home() / ".cache" / "huggingface"
        hf_cache_dir = hf_cache_base / "hub"
        hf_cache_dir_legacy = hf_cache_base / "transformers"

        if hf_cache_dir.exists() or not hf_cache_dir_legacy.exists():
            cache_path = hf_cache_dir
        else:
            cache_path = hf_cache_dir_legacy

    if not cache_path.exists():
        return False

    # Transformers stores models as: models--{org}--{model_name}
    # e.g., "facebook/bart-large-cnn" -> "models--facebook--bart-large-cnn"
    model_cache_name = model_name.replace("/", "--")
    model_cache_path = cache_path / f"models--{model_cache_name}"

    # Check if cache directory exists and has content
    if model_cache_path.exists() and model_cache_path.is_dir():
        # Check for tokenizer and model files (basic check)
        # Transformers caches in snapshots/{revision}/ directories
        snapshots = model_cache_path / "snapshots"
        if snapshots.exists():
            # Check if any snapshot directory exists (indicates model was downloaded)
            for item in snapshots.iterdir():
                if item.is_dir():
                    # Check for model files (safetensors or bin)
                    model_files = list(item.glob("*.bin")) + list(item.glob("*.safetensors"))
                    # Check for tokenizer files
                    # Transformers REQUIRES tokenizer.json for actual loading
                    # tokenizer_config.json alone is NOT sufficient
                    # Tokenizer files may be in snapshot or in blobs (symlinked)
                    tokenizer_json = list(item.glob("tokenizer.json")) + list(
                        (model_cache_path / "blobs").glob("*tokenizer.json")
                    )
                    # Model is cached if we have model files AND tokenizer.json
                    # Config.json indicates the model structure is present but is not sufficient
                    if model_files and tokenizer_json:
                        # Additional integrity check: validate safetensors files
                        # This catches corrupted files that pass existence checks
                        if not _validate_transformers_model_integrity(item, model_files):
                            return False
                        return True

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

    Args:
        model_name: Whisper model name

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    if not _is_whisper_model_cached(model_name):
        import pytest

        pytest.skip(
            f"Whisper model '{model_name}' is not cached. "
            f"Run 'make preload-ml-models' to pre-cache all required models."
        )


def require_spacy_model_cached(model_name: str) -> None:
    """Require that a spaCy model is cached/installed, skip with helpful message if not.

    Args:
        model_name: spaCy model name

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    if not _is_spacy_model_cached(model_name):
        import pytest

        pytest.skip(
            f"spaCy model '{model_name}' is not cached. "
            f"Run 'make preload-ml-models' to pre-cache all required models."
        )


def require_transformers_model_cached(model_name: str, cache_dir: Optional[str] = None) -> None:
    """Require that a Transformers model is cached, skip with helpful message if not.

    Args:
        model_name: Hugging Face model identifier
        cache_dir: Cache directory path

    Raises:
        pytest.skip: If model is not cached (for better test reporting)
    """
    if not _is_transformers_model_cached(model_name, cache_dir):
        import pytest

        pytest.skip(
            f"Transformers model '{model_name}' is not cached. "
            f"Run 'make preload-ml-models' to pre-cache all required models."
        )
