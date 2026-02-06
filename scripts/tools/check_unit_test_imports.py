#!/usr/bin/env python3
"""Check that unit tests can import modules without optional dependencies.

This script verifies that all modules imported by unit tests can be imported
without optional dependencies (ML: spacy, torch, transformers; LLM: openai, etc.) installed.
This ensures that unit tests can run in CI without heavy optional dependencies.

Usage:
    python scripts/tools/check_unit_test_imports.py

Exit codes:
    0: All imports succeed without optional dependencies
    1: One or more imports failed (optional dependencies required at import time)
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Remove optional dependencies from sys.modules if they exist
# In CI, these won't be installed, so imports will fail if modules import them at top level
# ML dependencies (ml extra)
ML_MODULES = ["spacy", "torch", "transformers", "accelerate", "sentencepiece"]
# LLM dependencies (llm extra) - Issue #405
LLM_MODULES = ["openai"]
OPTIONAL_MODULES = ML_MODULES + LLM_MODULES
for module_name in OPTIONAL_MODULES:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Note: In CI, ML dependencies won't be installed, so if a module tries to import
# them at the top level, the import will fail and we'll catch it.
# Locally, if ML deps are installed, imports will succeed, but that's okay -
# the real enforcement happens in CI.

# Track which modules we need to test
# These are modules that unit tests import
# Updated to reflect current module structure (Issue #403)
MODULES_TO_TEST = [
    "podcast_scraper",
    "podcast_scraper.config",
    "podcast_scraper.metadata",
    "podcast_scraper.rss_parser",
    "podcast_scraper.speaker_detectors.factory",
    "podcast_scraper.summarization.factory",
    "podcast_scraper.transcription.factory",
    "podcast_scraper.providers.ml.ml_provider",  # Should use lazy imports
    "podcast_scraper.providers.openai.openai_provider",  # Should use lazy imports (Issue #405)
]

failed_imports = []
successful_imports = []
warnings = []

print("Checking if modules can be imported without optional dependencies...")
print("=" * 70)
print(
    "Note: In CI, optional dependencies (ML: spacy, torch, etc.; "
    "LLM: openai, etc.) are NOT installed."
)
print("      If a module imports them at the top level, this check will fail.")
print("      Locally, if optional deps are installed, imports may succeed.")
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
        # Check if the error is about optional dependencies (ML or LLM)
        error_msg = str(e).lower()
        is_optional_error = any(opt_mod in error_msg for opt_mod in OPTIONAL_MODULES)

        if is_optional_error:
            failed_imports.append((module_name, str(e)))
            print(f"✗ {module_name}")
            print(f"  ERROR: {e}")
            print("  → This module imports optional dependencies at import time!")
            print("  → Fix: Use lazy imports or mock optional dependencies in tests")
        else:
            # Non-optional import error - might be expected (e.g., missing other deps)
            warnings.append((module_name, str(e)))
            print(f"? {module_name} (non-optional import error: {e})")
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
    print("\n❌ FAILED: The following modules require optional dependencies at import time:")
    for module_name, error in failed_imports:
        print(f"  - {module_name}: {error}")
    print("\nFix options:")
    print("  1. Use lazy imports (import optional deps inside functions, not at module level)")
    print("  2. Mock optional dependencies in unit tests (already done in some tests)")
    print(
        "  3. Move optional-dependent code to separate modules that aren't imported by unit tests"
    )
    sys.exit(1)
else:
    print("\n✓ SUCCESS: All modules can be imported without optional dependencies")
    sys.exit(0)
