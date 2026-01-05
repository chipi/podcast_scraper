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

import gc
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import config to use test defaults
from podcast_scraper import config
from podcast_scraper.cache_utils import (
    get_project_root,
    get_spacy_cache_dir,
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)


def preload_whisper_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Whisper models.

    Args:
        model_names: List of Whisper model names to preload.
                    If None, uses WHISPER_MODELS env var or defaults to test default.
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
            model_names = [config.TEST_DEFAULT_WHISPER_MODEL]

    if not model_names:
        print("Skipping Whisper model preloading (no models specified)")
        return

    print("Preloading Whisper models...")

    # Use local cache if available, otherwise fall back to user cache
    whisper_cache = get_whisper_cache_dir()
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "whisper"

    # Explicit path logging for debugging
    print(f"  Cache directory: {whisper_cache}")
    print(f"  Using local cache: {whisper_cache == local_cache}")
    print(f"  Cache directory exists: {whisper_cache.exists()}")
    print(f"  User: {os.environ.get('USER', 'unknown')}")
    print(f"  Home: {Path.home()}")

    # Ensure cache directory exists
    whisper_cache.mkdir(parents=True, exist_ok=True)

    # Use download_root parameter to specify cache directory directly
    # This is more reliable than environment variable
    whisper_cache_str = str(whisper_cache)
    for model_name in model_names:
        print(f"  - {model_name}...")
        try:
            # Show expected model file path
            model_file = whisper_cache / f"{model_name}.pt"
            print(f"    Expected model file: {model_file}")
            print(f"    Model file exists before load: {model_file.exists()}")

            # Use download_root parameter to cache to local directory
            model = whisper.load_model(model_name, download_root=whisper_cache_str)
            assert model is not None, f"Model {model_name} loaded but is None"
            assert (
                hasattr(model, "dims") and model.dims is not None
            ), f"Model {model_name} missing dims attribute"

            # Verify model file exists after load
            print(f"    Model file exists after load: {model_file.exists()}")
            if model_file.exists():
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"    Model file size: {file_size_mb:.1f} MB")

            print(f"  ✓ Whisper {model_name} cached and verified")
            print(f"    Location: {model_file}")
        except Exception as e:
            print(f"  ✗ Failed to preload Whisper model {model_name}: {e}")
            print(f"       Cache directory: {whisper_cache}")
            print(f"       Expected model file: {whisper_cache / f'{model_name}.pt'}")
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
            # Default: preload only test defaults (small, fast models for local dev/testing)
            # These match the constants in src/podcast_scraper/config.py:
            # - TEST_DEFAULT_SUMMARY_MODEL = "facebook/bart-base"
            #   (small, ~500MB, fast, used in tests)
            # - TEST_DEFAULT_SUMMARY_REDUCE_MODEL = "allenai/led-base-16384"
            #   (long-context, ~1GB, used in both)
            #
            # For production models (e.g., "facebook/bart-large-cnn", ~2GB), set TRANSFORMERS_MODELS
            # environment variable explicitly or use in Docker builds with custom model list.
            model_names = [
                # Test default (small, ~500MB)
                # Matches config.TEST_DEFAULT_SUMMARY_MODEL
                config.TEST_DEFAULT_SUMMARY_MODEL,
                # REDUCE default (long-context, ~1GB)
                # Matches config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
                # Used in both tests and production for long-context summarization
                config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
            ]

    if not model_names:
        print("Skipping Transformers model preloading (no models specified)")
        return

    print("Preloading Transformers models...")

    # Use local cache if available, otherwise fall back to default
    cache_dir = get_transformers_cache_dir()
    project_root = get_project_root()
    local_cache = project_root / ".cache" / "huggingface" / "hub"

    # Ensure cache directory exists before loading models
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Explicit path logging for debugging
    print(f"  Cache directory: {cache_dir}")
    print(f"  Using local cache: {cache_dir == local_cache}")
    print(f"  Cache directory exists: {cache_dir.exists()}")
    print(f"  User: {os.environ.get('USER', 'unknown')}")
    print(f"  Home: {Path.home()}")

    for model_name in model_names:
        print(f"  - {model_name}...")
        print(f"    Source: Hugging Face (https://huggingface.co/{model_name})")
        print(f"    Cache location: {cache_dir}")

        # Check if already cached
        model_cache_name = model_name.replace("/", "--")
        model_cache_path = cache_dir / f"models--{model_cache_name}"
        if model_cache_path.exists():
            # Calculate existing size
            existing_size = sum(
                f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
            )
            existing_size_mb = existing_size / (1024 * 1024)
            print(f"    Status: Already cached ({existing_size_mb:.1f} MB)")
            print(f"    Cache path: {model_cache_path}")
        else:
            print("    Status: Not cached, downloading...")

        try:
            # nosec B615 - Model names pinned in config, preload script for dev/testing
            # Use cache_dir to ensure models are cached to local cache directory
            # Force download of all files by not using local_files_only
            # This ensures all required files (including optional ones) are cached
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=str(cache_dir), local_files_only=False  # nosec B615
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, cache_dir=str(cache_dir), local_files_only=False  # nosec B615
            )

            # Calculate final size after download
            if model_cache_path.exists():
                total_size = sum(
                    f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                print(f"    ✓ Downloaded and cached: {size_mb:.1f} MB")
                print(f"    ✓ Cache path: {model_cache_path}")

            # Try to download any optional files that might be needed
            # Some models have optional files like tokenizer_config.json that might
            # be requested at runtime. Download them now to ensure they're cached.
            try:
                from huggingface_hub import hf_hub_download

                optional_files = ["tokenizer_config.json", "special_tokens_map.json"]
                for filename in optional_files:
                    try:
                        hf_hub_download(  # nosec B615
                            repo_id=model_name,
                            filename=filename,
                            cache_dir=str(cache_dir),
                            local_files_only=False,
                        )
                    except Exception:
                        # File doesn't exist for this model - that's fine, it's optional
                        pass
            except ImportError:
                # huggingface_hub not available - that's fine, we'll rely on transformers
                pass

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

            # Verify model is cached to disk AND can be loaded with local_files_only=True
            # This is critical: we need to verify the model is actually on disk and loadable,
            # not just that it was loaded into memory. This matches what tests do and ensures
            # the model will work in network-isolated environments (Docker, CI, etc.)
            # Use the cache directory we already determined (local cache if available)
            transformers_cache_base = cache_dir
            cache_dir_str = str(transformers_cache_base)

            # Explicit path logging for debugging
            model_cache_name = model_name.replace("/", "--")
            model_cache_path = transformers_cache_base / f"models--{model_cache_name}"
            print(f"    Model cache path: {model_cache_path}")
            print(f"    Model cache exists: {model_cache_path.exists()}")

            # Primary verification: Try to actually load the model from disk
            # with local_files_only=True. This is the most reliable check since it
            # matches what SummaryModel does in tests and production.
            try:
                # Try loading tokenizer with local_files_only=True
                # This verifies the model is truly cached and loadable from disk
                # nosec B615 - local_files_only=True prevents network access, model_name from config
                verified_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir_str,
                    local_files_only=True,
                )  # nosec B615
                # Try loading model with local_files_only=True
                # nosec B615 - local_files_only=True prevents network access, model_name from config
                verified_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir_str,
                    local_files_only=True,
                )  # nosec B615
                # Clean up verification models
                del verified_model, verified_tokenizer
                gc.collect()
                print(f"  ✓ {model_name} cached and verified (loadable from disk)")
                print(f"    Location: {model_cache_path}")
            except Exception as e:
                # Model can't be loaded from disk - this is a problem
                # Check if files exist for better error message
                cache_path = transformers_cache_base / f"models--{model_name.replace('/', '--')}"
                snapshots = cache_path / "snapshots"
                if snapshots.exists():
                    # Files exist but can't be loaded - incomplete cache or corruption
                    raise RuntimeError(
                        f"Model {model_name} files exist in cache but cannot be loaded "
                        f"with local_files_only=True. Cache may be incomplete or corrupted.\n"
                        f"  Cache directory: {cache_dir_str}\n"
                        f"  Model cache path: {model_cache_path}\n"
                        f"  Snapshots path: {snapshots}\n"
                        f"  Error: {e}"
                    )
                else:
                    # Files don't exist - model wasn't cached to disk
                    raise RuntimeError(
                        f"Model {model_name} was loaded but not cached to disk.\n"
                        f"  Cache directory: {cache_dir_str}\n"
                        f"  Model cache path: {model_cache_path}\n"
                        f"  User: {os.environ.get('USER', 'unknown')}\n"
                        f"  Home: {Path.home()}\n"
                        f"  Error: {e}"
                    )

        except Exception as e:
            print(f"ERROR: Failed to preload {model_name}: {e}")
            print("       Install with: pip install transformers torch")
            sys.exit(1)


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
        print("Preloading PRODUCTION ML models...")
        print("Models: Whisper base.en, BART-large-cnn, LED-large-16384, en_core_web_sm")
        print("This will download and cache production models for nightly tests.")
        print("")

        # Production models
        # Note: Use "base.en" (not "base") because the code converts "base" to "base.en"
        # for English language, and Whisper caches models with the exact name used
        whisper_models = ["base.en"]  # Production Whisper model (English variant)
        transformers_models = [
            "facebook/bart-large-cnn",  # Production MAP model
            "allenai/led-large-16384",  # Production REDUCE model (from issue #175)
        ]
        spacy_models = ["en_core_web_sm"]  # Same for production

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
