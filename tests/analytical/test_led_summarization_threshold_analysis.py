"""LED Summarization Threshold Analysis Tool.

This is a diagnostic/analysis tool for investigating summarization threshold behavior
with LED models. It captures detailed metrics (token counts, compression ratios, warnings)
to help diagnose issues like inconsistent summarization quality at threshold boundaries.

**NOT a regular test** - This is an analytical/diagnostic tool, not part of the regular
test suite. Run it explicitly when investigating threshold-related issues:

    make test-analytical
    # or
    pytest tests/analytical/ -v -s

Originally created for Issue #283: Inconsistent Summarization Quality at 4k Token Threshold.
Tests use p07 and p08 feeds which have long-form episodes that trigger threshold behavior.

See: https://github.com/chipi/podcast_scraper/issues/283
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from io import StringIO
from pathlib import Path

import pytest

# Set HF_HUB_CACHE BEFORE any transformers imports
# This ensures transformers uses the local cache consistently
from podcast_scraper.cache import get_project_root

_project_root = get_project_root()
_local_cache = _project_root / ".cache" / "huggingface" / "hub"
if _local_cache.exists():
    os.environ["HF_HUB_CACHE"] = str(_local_cache)

from podcast_scraper import config, workflow
from tests.conftest import create_test_config
from tests.integration.ml_model_cache_helpers import (
    require_transformers_model_cached,
)


@pytest.mark.analytical  # Analytical/diagnostic tool, not a regular test
@pytest.mark.ml_models
@pytest.mark.slow
class TestLEDSummarizationThresholdAnalysis:
    """LED summarization threshold analysis and diagnostics.

    This class provides diagnostic tests for analyzing LED model behavior at
    summarization threshold boundaries. Use these tests to:
    - Capture baseline metrics (token counts, compression ratios, warnings)
    - Validate threshold fixes
    - Investigate threshold-related issues
    - Compare behavior across different threshold configurations
    """

    def test_p07_under_threshold_analysis(self, e2e_server):
        """Test p07 feed baseline behavior (Episode 1 scenario - under 4k threshold).

        This test reproduces the issue where episodes with 3-4k combined summary tokens
        get verbose summaries with warnings, despite being under the threshold.

        Expected current behavior (before fix):
        - Uses hierarchical reduce (under 4k threshold)
        - LED model preserves too much detail → poor compression (~65%)
        - Warning triggered (> 60% validation threshold)
        - Verbose, less useful summary

        After fix:
        - Should use LED-specific thresholds (6k ceiling, 75% validation)
        - Better compression or earlier extractive fallback
        - No false positive warnings
        """
        # Debug: Check cache directory resolution and set explicitly
        import os

        from podcast_scraper.cache import get_project_root, get_transformers_cache_dir

        cache_dir = get_transformers_cache_dir()
        project_root = get_project_root()

        # Explicitly set HF_HUB_CACHE to ensure transformers uses the local cache
        # This matches what CI does and ensures consistent cache resolution
        os.environ["HF_HUB_CACHE"] = str(cache_dir)

        print("\n=== DEBUG: Cache Directory Resolution ===")
        print(f"Project root: {project_root}")
        print(f"Cache dir: {cache_dir}")
        print(f"Cache exists: {cache_dir.exists()}")
        print(f"HF_HUB_CACHE env (set): {os.environ.get('HF_HUB_CACHE')}")
        print(f"Working dir: {os.getcwd()}")
        print()

        # Require models to be cached
        # Resolve model aliases to actual model IDs
        from podcast_scraper.providers.ml import summarizer

        map_model = config.TEST_DEFAULT_SUMMARY_MODEL
        resolved_map = summarizer.DEFAULT_SUMMARY_MODELS.get(map_model, map_model)
        print(f"Checking MAP model: {map_model}")
        print(f"  → Resolves to: {resolved_map}")
        require_transformers_model_cached(map_model, None)
        print("✅ MAP model cache check passed")
        print()

        reduce_model = config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL
        resolved_reduce = summarizer.DEFAULT_SUMMARY_MODELS.get(reduce_model, reduce_model)
        print(f"Checking REDUCE model: {reduce_model}")
        print(f"  → Resolves to: {resolved_reduce}")
        require_transformers_model_cached(reduce_model, None)
        print("✅ REDUCE model cache check passed")
        print()

        rss_url = e2e_server.urls.feed("podcast7_sustainability")

        # Capture log output to extract metrics
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        # Get the actual logger used in summarizer.py
        logger = logging.getLogger("podcast_scraper.providers.ml.summarizer")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        # Also set root logger to ensure messages propagate
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=1,
                    generate_metadata=True,
                    generate_summaries=True,
                    summary_provider="transformers",
                    summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # BART for MAP
                    summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # LED for REDUCE
                    transcribe_missing=False,  # p07 has transcript URL
                )

                # Run pipeline
                count, summary = workflow.run_pipeline(cfg)

                # Verify pipeline completed
                assert count > 0, "Should process at least one episode"

                # Extract metrics from logs
                log_output = log_capture.getvalue()

                # Debug: Print log output if approach not found
                if "approach=" not in log_output:
                    print(f"\n=== DEBUG: Log output (first 2000 chars) ===\n{log_output[:2000]}\n")

                # Extract combined_tokens from log
                combined_tokens_match = re.search(r"combined_tokens \((\d+)\)", log_output)
                combined_tokens = (
                    int(combined_tokens_match.group(1)) if combined_tokens_match else None
                )

                # Extract approach from log
                # Look for "approach=..." in the log output
                # Format: "approach=abstractive (single-pass)" or
                # "approach=hierarchical reduce" or "approach=extractive"
                # The log format is: "[MAP-REDUCE VALIDATION] Reduce phase decision: "
                # "... approach={approach}"
                approach_match = re.search(r"approach=([^,\n\]]+)", log_output)
                if not approach_match:
                    # Try alternative patterns
                    approach_match = re.search(r'approach="([^"]+)"', log_output)
                if not approach_match:
                    # Try without quotes
                    approach_match = re.search(r"approach=(\w+(?:\s+\w+)*)", log_output)
                approach = approach_match.group(1).strip() if approach_match else None

                # Verify approach is a valid strategy
                assert approach in [
                    "abstractive (single-pass)",
                    "hierarchical reduce",
                    "extractive",
                ], (
                    f"Invalid reduce strategy: {approach}. "
                    "Expected one of: abstractive (single-pass), "
                    "hierarchical reduce, extractive"
                )

                # Check for validation warning
                has_warning = (
                    "suspiciously close to" in log_output or "Model may have failed" in log_output
                )

                # Extract compression ratio from log
                compression_match = re.search(r"overall_compression=([\d.]+)x", log_output)
                compression_ratio = float(compression_match.group(1)) if compression_match else None

                # Read metadata to get summary
                metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
                assert len(metadata_files) > 0, "Should create at least one metadata file"

                if metadata_files:
                    with open(metadata_files[0], "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    summary_data = metadata.get("summary")
                    if summary_data and isinstance(summary_data, dict):
                        summary_text = summary_data.get("short_summary", "")
                    else:
                        summary_text = ""
                    summary_length = len(summary_text)
                else:
                    summary_length = 0

                # Document baseline metrics
                print("\n=== p07 Under-Threshold Analysis (LED Model) ===")
                print(f"Combined tokens: {combined_tokens}")
                print(f"Approach: {approach}")
                print(f"Compression ratio: {compression_ratio}x")
                print(f"Summary length: {summary_length} chars")
                print(f"Warning triggered: {has_warning}")

                # Baseline assertions (current behavior)
                # These will be updated after fix is implemented
                assert combined_tokens is not None, "Should have combined_tokens in logs"
                assert approach is not None, "Should have approach in logs"

        finally:
            logger.removeHandler(handler)

    def test_p08_over_threshold_analysis(self, e2e_server):
        """Test p08 feed baseline behavior (Episode 2 scenario - over 4k threshold).

        This test reproduces the issue where episodes with >4k combined summary tokens
        get cleaner extractive summaries, despite being longer.

        Expected current behavior (before fix):
        - Falls back to extractive approach (over 4k threshold)
        - Selected representative chunks → concise summary
        - No warning
        - Cleaner, more useful summary

        After fix:
        - Should use LED-specific thresholds (6k ceiling)
        - May still use extractive if over 6k, but with better decision logic
        - Behavior should remain optimal
        """
        # Require models to be cached
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
        require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL, None)

        rss_url = e2e_server.urls.feed("podcast8_solar")

        # Capture log output to extract metrics
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        # Get the actual logger used in summarizer.py
        logger = logging.getLogger("podcast_scraper.providers.ml.summarizer")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        # Also set root logger to ensure messages propagate
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = create_test_config(
                    rss_url=rss_url,
                    output_dir=tmpdir,
                    max_episodes=1,
                    generate_metadata=True,
                    generate_summaries=True,
                    summary_provider="transformers",
                    summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,  # BART for MAP
                    summary_reduce_model=config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,  # LED for REDUCE
                    transcribe_missing=False,  # p08 has transcript URL
                )

                # Run pipeline
                count, summary = workflow.run_pipeline(cfg)

                # Verify pipeline completed
                assert count > 0, "Should process at least one episode"

                # Extract metrics from logs
                log_output = log_capture.getvalue()

                # Debug: Print log output if approach not found
                if "approach=" not in log_output:
                    print(f"\n=== DEBUG: Log output (first 2000 chars) ===\n{log_output[:2000]}\n")

                # Extract combined_tokens from log
                combined_tokens_match = re.search(r"combined_tokens \((\d+)\)", log_output)
                combined_tokens = (
                    int(combined_tokens_match.group(1)) if combined_tokens_match else None
                )

                # Extract approach from log
                # Look for "approach=..." in the log output
                # Format: "approach=abstractive (single-pass)" or
                # "approach=hierarchical reduce" or "approach=extractive"
                # The log format is: "[MAP-REDUCE VALIDATION] Reduce phase decision: "
                # "... approach={approach}"
                approach_match = re.search(r"approach=([^,\n\]]+)", log_output)
                if not approach_match:
                    # Try alternative patterns
                    approach_match = re.search(r'approach="([^"]+)"', log_output)
                if not approach_match:
                    # Try without quotes
                    approach_match = re.search(r"approach=(\w+(?:\s+\w+)*)", log_output)
                approach = approach_match.group(1).strip() if approach_match else None

                # Verify approach is a valid strategy
                assert approach in [
                    "abstractive (single-pass)",
                    "hierarchical reduce",
                    "extractive",
                ], (
                    f"Invalid reduce strategy: {approach}. "
                    "Expected one of: abstractive (single-pass), "
                    "hierarchical reduce, extractive"
                )

                # Check for validation warning
                has_warning = (
                    "suspiciously close to" in log_output or "Model may have failed" in log_output
                )

                # Extract compression ratio from log
                compression_match = re.search(r"overall_compression=([\d.]+)x", log_output)
                compression_ratio = float(compression_match.group(1)) if compression_match else None

                # Read metadata to get summary
                metadata_files = list(Path(tmpdir).rglob("*.metadata.json"))
                assert len(metadata_files) > 0, "Should create at least one metadata file"

                if metadata_files:
                    with open(metadata_files[0], "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    summary_data = metadata.get("summary")
                    if summary_data and isinstance(summary_data, dict):
                        summary_text = summary_data.get("short_summary", "")
                    else:
                        summary_text = ""
                    summary_length = len(summary_text)
                else:
                    summary_length = 0

                # Document baseline metrics
                print("\n=== p08 Over-Threshold Analysis (LED Model) ===")
                print(f"Combined tokens: {combined_tokens}")
                print(f"Approach: {approach}")
                print(f"Compression ratio: {compression_ratio}x")
                print(f"Summary length: {summary_length} chars")
                print(f"Warning triggered: {has_warning}")

                # Baseline assertions (current behavior)
                # These will be updated after fix is implemented
                assert combined_tokens is not None, "Should have combined_tokens in logs"
                assert approach is not None, "Should have approach in logs"

        finally:
            logger.removeHandler(handler)
