#!/usr/bin/env python3
"""Preload all ML models for podcast scraper.

This script preloads all required ML models:
- Whisper models (transcription)
- spaCy models (speaker detection)
- Transformers models (summarization)

It can be used by:
- Makefile: `make preload-ml-models`
- Dockerfile: During Docker image build
- CI/CD: For model caching

Usage:
    # Preload all models (default)
    python scripts/preload_ml_models.py

    # Preload only specific Whisper models
    WHISPER_MODELS=base.en,tiny.en python scripts/preload_ml_models.py

    # Skip Whisper preloading
    SKIP_WHISPER=1 python scripts/preload_ml_models.py

    # Skip Transformers preloading (faster, for fast builds)
    SKIP_TRANSFORMERS=1 python scripts/preload_ml_models.py
"""

import gc
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def preload_whisper_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Whisper models.

    Args:
        model_names: List of Whisper model names to preload.
                    If None, uses WHISPER_MODELS env var or defaults to ['tiny.en'].
    """
    try:
        import whisper
    except ImportError:
        print("ERROR: openai-whisper not installed. Install with: pip install openai-whisper")
        sys.exit(1)

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("WHISPER_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: tiny.en for local dev/tests (smallest, fastest)
            # Matches config.TEST_DEFAULT_WHISPER_MODEL
            # Docker and production use base.en (better quality, matches app default)
            model_names = ["tiny.en"]

    if not model_names:
        print("Skipping Whisper model preloading (no models specified)")
        return

    print("Preloading Whisper models...")
    for model_name in model_names:
        print(f"  - {model_name}...")
        try:
            model = whisper.load_model(model_name)
            assert model is not None, f"Model {model_name} loaded but is None"
            assert (
                hasattr(model, "dims") and model.dims is not None
            ), f"Model {model_name} missing dims attribute"
            print(f"  ✓ Whisper {model_name} cached and verified")
        except Exception as e:
            print(f"ERROR: Failed to preload Whisper {model_name}: {e}")
            print("       Install with: pip install openai-whisper")
            sys.exit(1)


def preload_spacy_models(model_names: Optional[List[str]] = None) -> None:
    """Preload/verify spaCy models.

    Args:
        model_names: List of spaCy model names to preload.
                    If None, uses SPACY_MODELS env var or defaults to ['en_core_web_sm'].
    """
    try:
        import spacy
    except ImportError:
        print("ERROR: spacy not installed. Install with: pip install spacy")
        sys.exit(1)

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("SPACY_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: en_core_web_sm (installed as dependency)
            # Matches config.DEFAULT_NER_MODEL (same for tests and production)
            model_names = ["en_core_web_sm"]

    if not model_names:
        print("Skipping spaCy model preloading (no models specified)")
        return

    print("Verifying spaCy models...")
    print("  (Models are installed as dependencies, no download needed)")
    for model_name in model_names:
        print(f"  - {model_name}...")
        try:
            nlp = spacy.load(model_name)
            assert nlp is not None, f"Model {model_name} loaded but is None"
            # Test that model works
            doc = nlp("Test text")
            assert (
                doc is not None and len(doc) > 0
            ), f"Model {model_name} loaded but doesn't process text"
            print(f"  ✓ spaCy {model_name} verified (installed as dependency)")
        except Exception as e:
            print(f"ERROR: spaCy model {model_name} not available: {e}")
            print("       Install with: pip install -e .[ml]")
            sys.exit(1)


def preload_transformers_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Transformers models.

    Args:
        model_names: List of Transformers model names to preload.
                    If None, uses TRANSFORMERS_MODELS env var or defaults to common models.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Install with: pip install transformers torch")
        sys.exit(1)

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("TRANSFORMERS_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: preload both test and production defaults
            # These match the constants in src/podcast_scraper/config.py:
            # - TEST_DEFAULT_SUMMARY_MODEL = "facebook/bart-base" (small, fast, used in tests)
            # - Production default (MAP): "facebook/bart-large-cnn" (large, quality, used in app)
            # - TEST_DEFAULT_SUMMARY_REDUCE_MODEL = "allenai/led-base-16384"
            #   (long-context, used in both)
            model_names = [
                # Test default (small, ~500MB) - matches config.TEST_DEFAULT_SUMMARY_MODEL
                "facebook/bart-base",
                "facebook/bart-large-cnn",  # Production default (large, ~2GB) - app runtime default
                "sshleifer/distilbart-cnn-12-6",  # Fast option (optional)
                # REDUCE default (long-context, ~1GB)
                # matches config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
                "allenai/led-base-16384",
            ]

    if not model_names:
        print("Skipping Transformers model preloading (no models specified)")
        return

    print("Preloading Transformers models...")
    for model_name in model_names:
        print(f"  - {model_name}...")
        try:
            print("    Downloading...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Verify model loads
            assert (
                model is not None and tokenizer is not None
            ), f"Model {model_name} loaded but is None"

            # Verify tokenizer works
            tokens = tokenizer.encode("Test text", return_tensors="pt")
            assert tokens is not None, f"Tokenizer {model_name} doesn't encode text"

            # Verify model structure
            assert (
                hasattr(model, "config") and model.config is not None
            ), f"Model {model_name} missing config"

            # Clean up memory
            del model, tokenizer
            gc.collect()

            # Verify files are cached to disk
            cache_path = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / f"models--{model_name.replace('/', '--')}"
            )
            snapshots = cache_path / "snapshots"
            if snapshots.exists():
                # Check for model files
                model_files = []
                for item in snapshots.iterdir():
                    if (snapshots / item.name).is_dir():
                        model_files.extend(
                            list((snapshots / item.name).glob("*.safetensors"))
                            + list((snapshots / item.name).glob("*.bin"))
                        )
                assert len(model_files) > 0, f"Model {model_name} files not found in cache"
                print(f"  ✓ {model_name} cached and verified")
            else:
                msg = (
                    f"  ⚠ {model_name} loaded but cache path not found "
                    "(may be in different location)"
                )
                print(msg)

        except Exception as e:
            print(f"ERROR: Failed to preload {model_name}: {e}")
            print("       Install with: pip install transformers torch")
            sys.exit(1)


def main() -> None:
    """Main entry point for model preloading."""
    print("Preloading ML models...")
    print("This will download and cache models to avoid network calls during testing.")
    print("")

    # Check skip flags
    skip_whisper = os.environ.get("SKIP_WHISPER", "").strip().lower() in ("1", "true", "yes")
    skip_spacy = os.environ.get("SKIP_SPACY", "").strip().lower() in ("1", "true", "yes")
    skip_transformers = os.environ.get("SKIP_TRANSFORMERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    # Preload models (unless skipped)
    if not skip_whisper:
        preload_whisper_models()
        print("")

    if not skip_spacy:
        preload_spacy_models()
        print("")

    if not skip_transformers:
        preload_transformers_models()
        print("")

    print("All models preloaded and verified successfully!")
    print("Models are cached in:")
    print("  - Whisper: ~/.cache/whisper/")
    print("  - spaCy: Installed as dependency (en_core_web_sm in pyproject.toml)")
    print("  - Transformers: ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
