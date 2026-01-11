#!/usr/bin/env python3
"""Check that unit tests can import modules without ML dependencies.

This script verifies that all modules imported by unit tests can be imported
without ML dependencies (spacy, torch, transformers) installed. This ensures
that unit tests can run in CI without heavy ML dependencies.

Usage:
    python scripts/tools/check_unit_test_imports.py

Exit codes:
    0: All imports succeed without ML dependencies
    1: One or more imports failed (ML dependencies required at import time)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Remove ML dependencies from sys.modules if they exist
# In CI, these won't be installed, so imports will fail if modules import them at top level
ML_MODULES = ["spacy", "torch", "transformers", "accelerate", "sentencepiece"]
for module_name in ML_MODULES:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Note: In CI, ML dependencies won't be installed, so if a module tries to import
# them at the top level, the import will fail and we'll catch it.
# Locally, if ML deps are installed, imports will succeed, but that's okay -
# the real enforcement happens in CI.

# Track which modules we need to test
# These are modules that unit tests import
MODULES_TO_TEST = [
    "podcast_scraper",
    "podcast_scraper.config",
    "podcast_scraper.metadata",
    "podcast_scraper.rss_parser",
    "podcast_scraper.speaker_detection",  # This one imports spacy at top level
    "podcast_scraper.summarizer",  # This one imports torch at top level
    "podcast_scraper.speaker_detectors.factory",
    "podcast_scraper.summarization.factory",
]

failed_imports = []
successful_imports = []
warnings = []

print("Checking if modules can be imported without ML dependencies...")
print("=" * 70)
print("Note: In CI, ML dependencies (spacy, torch, etc.) are NOT installed.")
print("      If a module imports them at the top level, this check will fail.")
print("      Locally, if ML deps are installed, imports may succeed.")
print("=" * 70)

for module_name in MODULES_TO_TEST:
    try:
        # Clear any cached imports of this module
        if module_name in sys.modules:
            del sys.modules[module_name]

        # Try to import
        __import__(module_name)
        successful_imports.append(module_name)
        print(f"✓ {module_name}")
    except ImportError as e:
        # Check if the error is about ML dependencies
        error_msg = str(e).lower()
        is_ml_error = any(ml_mod in error_msg for ml_mod in ML_MODULES)

        if is_ml_error:
            failed_imports.append((module_name, str(e)))
            print(f"✗ {module_name}")
            print(f"  ERROR: {e}")
            print("  → This module imports ML dependencies at import time!")
            print("  → Fix: Use lazy imports or mock ML dependencies in tests")
        else:
            # Non-ML import error - might be expected (e.g., missing optional deps)
            warnings.append((module_name, str(e)))
            print(f"? {module_name} (non-ML import error: {e})")
    except Exception as e:
        # Other errors (syntax, etc.) - these are real problems
        failed_imports.append((module_name, str(e)))
        print(f"✗ {module_name}")
        print(f"  ERROR: {e}")

print("=" * 70)
print(f"\nResults: {len(successful_imports)} successful, {len(failed_imports)} failed")
if warnings:
    print(f"Warnings: {len(warnings)} (non-ML import issues)")

if failed_imports:
    print("\n❌ FAILED: The following modules require ML dependencies at import time:")
    for module_name, error in failed_imports:
        print(f"  - {module_name}: {error}")
    print("\nFix options:")
    print("  1. Use lazy imports (import ML deps inside functions, not at module level)")
    print("  2. Mock ML dependencies in unit tests (already done in some tests)")
    print("  3. Move ML-dependent code to separate modules that aren't imported by unit tests")
    sys.exit(1)
else:
    print("\n✓ SUCCESS: All modules can be imported without ML dependencies")
    sys.exit(0)
