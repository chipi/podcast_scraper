#!/usr/bin/env python3
"""Preload all ML models for podcast scraper.

This script preloads all required ML models:
- Whisper models (transcription)
- spaCy models (speaker detection)
- Transformers models (summarization)

It uses the centralized model_loader module from the package, ensuring
all downloads go through the same internal code path.

It can be used by:
- Makefile: `make preload-ml-models`
- Dockerfile: During Docker image build
- CI/CD: For model caching

Usage:
    # Preload test models (default - small, fast models for local dev/testing)
    python scripts/preload_ml_models.py

    # Preload only specific Whisper models
    WHISPER_MODELS=base.en,tiny.en python scripts/preload_ml_models.py

    # Preload production Transformers models (large, quality models)
    # TRANSFORMERS_MODELS=facebook/bart-large-cnn,allenai/led-base-16384 \\
    #     python scripts/preload_ml_models.py

    # Skip Whisper preloading
    SKIP_WHISPER=1 python scripts/preload_ml_models.py

    # Skip Transformers preloading (faster, for fast builds)
    SKIP_TRANSFORMERS=1 python scripts/preload_ml_models.py
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import centralized model loader functions (internal package code)
from podcast_scraper import config
from podcast_scraper.cache_utils import (
    get_project_root,
    get_spacy_cache_dir,
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)
from podcast_scraper.model_loader import (
    preload_transformers_models,
    preload_whisper_models,
)

# preload_whisper_models and preload_transformers_models are now imported
# from podcast_scraper.model_loader (internal package code)
# This ensures all downloads go through the same centralized functions


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

    # Explicit path logging for debugging
    # spaCy models can be in multiple locations
    spacy_cache_user = Path.home() / ".local" / "share" / "spacy"
    import site

    site_packages_paths = [Path(p) for p in site.getsitepackages()]
    print(f"  User cache directory: {spacy_cache_user}")
    print(f"  User cache exists: {spacy_cache_user.exists()}")
    print(f"  Site packages: {site_packages_paths}")
    print(f"  User: {os.environ.get('USER', 'unknown')}")
    print(f"  Home: {Path.home()}")

    for model_name in model_names:
        print(f"  - {model_name}...")
        try:
            # Try to find where the model is installed
            nlp = spacy.load(model_name)
            assert nlp is not None, f"Model {model_name} loaded but is None"

            # Get model path from loaded model
            model_path = None
            if hasattr(nlp, "path") and nlp.path:
                model_path = Path(nlp.path)
            elif hasattr(nlp.meta, "path") and nlp.meta.get("path"):
                model_path = Path(nlp.meta["path"])

            if model_path:
                print(f"    Model location: {model_path}")
                print(f"    Model location exists: {model_path.exists()}")

            # Test that model works
            doc = nlp("Test text")
            assert (
                doc is not None and len(doc) > 0
            ), f"Model {model_name} loaded but doesn't process text"

            print(f"  ✓ spaCy {model_name} verified (installed as dependency)")
            if model_path:
                print(f"    Location: {model_path}")
        except Exception as e:
            print(f"ERROR: spaCy model {model_name} not available: {e}")
            print(f"       User cache: {spacy_cache_user}")
            print(f"       Site packages: {site_packages_paths}")
            print("       Install with: pip install -e .[ml]")
            sys.exit(1)


# preload_transformers_models is now imported from podcast_scraper.model_loader
# (internal package code) - see imports at top of file


def main() -> None:
    """Main entry point for model preloading."""
    import argparse

    parser = argparse.ArgumentParser(description="Preload ML models for podcast scraper")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Preload production models (Whisper base.en, BART-large-cnn, LED-large-16384)",
    )
    args = parser.parse_args()

    if args.production:
        print("Preloading ALL ML models (test + production) for nightly tests...")
        print("")
        print("Test models (for regular tests in nightly):")
        print("  - Whisper: tiny.en")
        print("  - Transformers: facebook/bart-base, allenai/led-base-16384")
        print("")
        print("Production models (for nightly-only tests):")
        print("  - Whisper: base.en")
        print("  - Transformers: facebook/bart-large-cnn, allenai/led-large-16384")
        print("")
        print("Common: en_core_web_sm (spaCy)")
        print("")

        # Include BOTH test AND production models
        # Nightly workflow runs both regular tests (need test models) and
        # nightly-only tests (need production models)
        whisper_models = [
            "tiny.en",  # Test model (for regular tests in nightly)
            "base.en",  # Production model (for nightly-only tests)
        ]
        transformers_models = [
            # Test models
            config.TEST_DEFAULT_SUMMARY_MODEL,  # facebook/bart-base
            config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # allenai/led-base-16384
            # Production models
            "facebook/bart-large-cnn",  # Production MAP model
            "allenai/led-large-16384",  # Production REDUCE model (from issue #175)
        ]
        spacy_models = ["en_core_web_sm"]  # Same for both

        preload_whisper_models(whisper_models)
        print("")

        preload_spacy_models(spacy_models)
        print("")

        preload_transformers_models(transformers_models)
        print("")
    else:
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
    print("")

    # Show actual paths for debugging
    whisper_cache = get_whisper_cache_dir()
    project_root = get_project_root()
    local_whisper_cache = project_root / ".cache" / "whisper"
    print("  Whisper Models:")
    print(f"    Cache directory: {whisper_cache}")
    print(f"    Using local cache: {whisper_cache == local_whisper_cache}")
    print(f"    Cache exists: {whisper_cache.exists()}")
    if whisper_cache.exists():
        model_files = list(whisper_cache.glob("*.pt"))
        if model_files:
            print(f"    Cached models: {[f.stem for f in model_files]}")
            for model_file in sorted(model_files):
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"      - {model_file.name}: {size_mb:.1f} MB")
    print("")

    print("  spaCy Models:")
    spacy_cache = get_spacy_cache_dir()
    project_root = get_project_root()
    local_spacy_cache = project_root / ".cache" / "spacy"
    if spacy_cache:
        print(f"    Cache directory: {spacy_cache}")
        print(f"    Using local cache: {spacy_cache == local_spacy_cache}")
        print(f"    Cache exists: {spacy_cache.exists()}")
    else:
        print("    Cache directory: Not available (models installed as packages)")
    import site

    site_packages_paths = [Path(p) for p in site.getsitepackages()]
    print(f"    Site packages: {site_packages_paths}")
    print("    (Models installed as Python package dependencies)")
    print("")

    transformers_cache = get_transformers_cache_dir()
    project_root = get_project_root()
    local_transformers_cache = project_root / ".cache" / "huggingface" / "hub"
    print("  Transformers Models:")
    print(f"    Cache directory: {transformers_cache}")
    print(f"    Using local cache: {transformers_cache == local_transformers_cache}")
    print(f"    Cache exists: {transformers_cache.exists()}")
    if transformers_cache.exists():
        model_dirs = [
            d for d in transformers_cache.iterdir() if d.is_dir() and d.name.startswith("models--")
        ]
        if model_dirs:
            model_names = [d.name.replace("models--", "").replace("--", "/") for d in model_dirs]
            print(f"    Cached models ({len(model_names)}): {model_names}")
            for model_dir in sorted(model_dirs):
                # Calculate total size
                total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                size_mb = total_size / (1024 * 1024)
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                file_count = sum(1 for f in model_dir.rglob("*") if f.is_file())
                print(f"      - {model_name}: {size_mb:.1f} MB ({file_count} files) at {model_dir}")
    print("")
    print("  Environment:")
    print(f"    User: {os.environ.get('USER', 'unknown')}")
    print(f"    Home: {Path.home()}")


if __name__ == "__main__":
    main()
