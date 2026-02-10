#!/usr/bin/env python3
"""E2E acceptance test runner.

This script runs multiple config files sequentially, collects structured data
(logs, outputs, timing, exit codes, resource usage), and saves results for analysis.

Usage:
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --output-dir .test_outputs/acceptance \
        [--compare-baseline baseline_id] \
        [--save-as-baseline baseline_id]

Examples:
    # Run example configs
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "examples/config.example.yaml"

    # Run with baseline comparison
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --compare-baseline baseline_v1

    # Save current run as baseline
    python scripts/acceptance/run_acceptance_tests.py \
        --configs "config/examples/config.example.yaml" \
        --save-as-baseline baseline_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# Add parent directory to path to import podcast_scraper modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from podcast_scraper import config

# Import E2E server
try:
    from tests.e2e.fixtures.e2e_http_server import E2EHTTPServer, E2EServerURLs
except ImportError:
    E2EHTTPServer = None  # type: ignore
    E2EServerURLs = None  # type: ignore

logger = logging.getLogger(__name__)


def find_config_files(pattern: str) -> List[Path]:
    """Find config files matching the pattern.

    Args:
        pattern: Glob pattern (e.g. "config/examples/config.example.yaml" or
            "config/acceptance/*.yaml")

    Returns:
        List of matching config file paths
    """
    # Convert pattern to Path and find matches
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        base_dir = pattern_path.parent
        glob_pattern = pattern_path.name
    else:
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        base_dir = project_root / pattern_path.parent
        glob_pattern = pattern_path.name

    if not base_dir.exists():
        logger.error(f"Directory not found: {base_dir}")
        return []

    matches = list(base_dir.glob(glob_pattern))
    matches.sort()  # Sort for consistent ordering

    if not matches:
        logger.warning(f"No config files found matching pattern: {pattern}")
        return []

    logger.info(f"Found {len(matches)} config file(s) matching pattern: {pattern}")
    return matches


def modify_config_for_fixtures(
    config_path: Path,
    e2e_server: Optional[E2EHTTPServer],
    session_dir: Path,
    run_output_dir: Path,
    use_fixtures: bool = True,
) -> Path:
    """Modify config for running (with optional fixture support).

    Args:
        config_path: Original config file path
        e2e_server: E2E server instance (None if using real feeds/APIs)
        session_dir: Session directory (for storing modified configs if using fixtures)
        run_output_dir: Specific run output directory (already created)
        use_fixtures: If True, use E2E server fixtures; if False, use real RSS/APIs

    Returns:
        Path to modified config file
    """
    # Load original config
    config_dict = config.load_config_file(str(config_path))

    # Set output directory to the run-specific location
    config_dict["output_dir"] = str(run_output_dir)

    if use_fixtures and e2e_server:
        # Replace RSS URL with E2E server feed URL
        # Use podcast1_multi_episode as default (5 short episodes)
        original_rss = config_dict.get("rss", "")
        if original_rss and not original_rss.startswith("http://127.0.0.1"):
            # Replace with E2E server feed
            # Default to podcast1_multi_episode for bulk tests
            config_dict["rss"] = e2e_server.urls.feed("podcast1_multi_episode")
            logger.debug(f"Replaced RSS URL: {original_rss} -> {config_dict['rss']}")

        # Set API base URLs to E2E server for all providers
        # This ensures we use mock APIs, not real ones
        os.environ["OPENAI_API_BASE"] = e2e_server.urls.openai_api_base()
        os.environ["GEMINI_API_BASE"] = e2e_server.urls.gemini_api_base()
        os.environ["MISTRAL_API_BASE"] = e2e_server.urls.mistral_api_base()
        os.environ["GROK_API_BASE"] = e2e_server.urls.grok_api_base()
        os.environ["DEEPSEEK_API_BASE"] = e2e_server.urls.deepseek_api_base()
        os.environ["OLLAMA_API_BASE"] = e2e_server.urls.ollama_api_base()
        os.environ["ANTHROPIC_API_BASE"] = e2e_server.urls.anthropic_api_base()

        # Set dummy API keys (required for config validation, but won't be used with mocks)
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-bulk-tests"
        if "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "MISTRAL_API_KEY" not in os.environ:
            os.environ["MISTRAL_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "GROK_API_KEY" not in os.environ:
            os.environ["GROK_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "DEEPSEEK_API_KEY" not in os.environ:
            os.environ["DEEPSEEK_API_KEY"] = "test-dummy-key-for-bulk-tests"
        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ["ANTHROPIC_API_KEY"] = "test-dummy-key-for-bulk-tests"

    else:
        # Using real RSS feeds and real APIs
        # Keep original RSS URL from config
        # Don't set API base URLs (use real APIs from environment)
        # Don't set dummy API keys (use real keys from environment)
        logger.info("Using real RSS feeds and real API providers")
        logger.info(f"RSS URL: {config_dict.get('rss', 'not set')}")

    # Save modified config in the run directory
    # This is needed for the service to run and also serves as a record of what was used
    modified_config_path = run_output_dir / "config.yaml"

    import yaml

    with open(modified_config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return modified_config_path


def get_config_hash(config_path: Path) -> str:
    """Get hash of config file for tracking changes.

    Args:
        config_path: Path to config file

    Returns:
        SHA256 hash of config file
    """
    with open(config_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def monitor_process_resources(process: subprocess.Popen, interval: float = 0.5) -> Dict[str, Any]:
    """Monitor process resource usage continuously.

    Args:
        process: Process object to monitor
        interval: Sampling interval in seconds (default: 0.5)

    Returns:
        Dict with resource usage metrics
    """
    if psutil is None:
        return {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }

    peak_memory_mb = 0.0
    cpu_samples = []
    cpu_times_total = None

    try:
        psutil_process = psutil.Process(process.pid)

        # Monitor continuously while process is running
        while process.poll() is None:
            try:
                # Get process memory
                mem_info = psutil_process.memory_info()
                mem_mb = mem_info.rss / (1024**2)

                # Get children memory (may fail on macOS due to permissions)
                children_mem_mb = 0.0
                try:
                    for child in psutil_process.children(recursive=True):
                        try:
                            child_mem = child.memory_info()
                            children_mem_mb += child_mem.rss / (1024**2)
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            PermissionError,
                            OSError,
                        ):
                            pass
                except (psutil.AccessDenied, PermissionError, OSError):
                    # macOS may restrict access to child processes - just use main process memory
                    # This is expected on macOS and not a critical error
                    pass

                total_mem_mb = mem_mb + children_mem_mb
                peak_memory_mb = max(peak_memory_mb, total_mem_mb)

                # Sample CPU percent
                try:
                    cpu_pct = psutil_process.cpu_percent(interval=None)
                    if cpu_pct is not None:
                        cpu_samples.append(cpu_pct)
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    PermissionError,
                    OSError,
                ):
                    pass

                # Get CPU times
                try:
                    cpu_times = psutil_process.cpu_times()
                    cpu_times_total = cpu_times.user + cpu_times.system
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    PermissionError,
                    OSError,
                ):
                    pass

                time.sleep(interval)
            except psutil.NoSuchProcess:
                # Process finished
                break
            except (psutil.AccessDenied, PermissionError, OSError) as e:
                # Permission issues - log but continue
                logger.debug(f"Permission issue monitoring process: {e}")
                time.sleep(interval)

        # Get final CPU times if not already captured
        if cpu_times_total is None:
            try:
                cpu_times = psutil_process.cpu_times()
                cpu_times_total = cpu_times.user + cpu_times.system
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                PermissionError,
                OSError,
            ):
                cpu_times_total = None

        # Calculate average CPU percent
        avg_cpu_percent = statistics.mean(cpu_samples) if cpu_samples else None

        return {
            "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb > 0 else None,
            "cpu_time_seconds": round(cpu_times_total, 2) if cpu_times_total else None,
            "cpu_percent": round(avg_cpu_percent, 2) if avg_cpu_percent else None,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError, OSError) as e:
        logger.debug(f"Error monitoring process resources: {e}")
        return {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }


def _detect_log_level(line_lower: str, line_stripped: str) -> tuple[bool | None, bool | None]:
    """Detect if line contains error or warning log level indicators.

    Args:
        line_lower: Lowercase version of the line
        line_stripped: Stripped version of the line

    Returns:
        Tuple of (is_error, is_warning) where None means no explicit log level found
    """
    # Check for structured log formats first (most explicit)
    has_error_level = any(
        level in line_lower
        for level in [
            "level=error",
            "level=critical",
            "levelname=error",
            "levelname=critical",
        ]
    )
    has_warning_level = any(
        level in line_lower
        for level in [
            "level=warning",
            "level=warn",
            "levelname=warning",
            "levelname=warn",
        ]
    )
    has_info_level = any(level in line_lower for level in ["level=info", "levelname=info"])
    has_debug_level = any(level in line_lower for level in ["level=debug", "levelname=debug"])

    # Check for uppercase log level indicators (standard Python logging format)
    has_error_uppercase = (
        " ERROR " in line_stripped
        or line_stripped.startswith("ERROR ")
        or " CRITICAL " in line_stripped
        or line_stripped.startswith("CRITICAL ")
    )
    has_warning_uppercase = (
        " WARNING " in line_stripped
        or line_stripped.startswith("WARNING ")
        or "FutureWarning:" in line_stripped
    )
    has_info_uppercase = " INFO " in line_stripped or line_stripped.startswith("INFO ")
    has_debug_uppercase = " DEBUG " in line_stripped or line_stripped.startswith("DEBUG ")

    # Classify based on log level (priority order: ERROR > WARNING > INFO/DEBUG)
    if has_error_level or has_error_uppercase:
        return (True, False)
    elif has_warning_level or has_warning_uppercase:
        return (False, True)
    elif has_info_level or has_info_uppercase or has_debug_level or has_debug_uppercase:
        # INFO/DEBUG logs are never errors or warnings
        return (False, False)
    else:
        # No explicit log level found
        return (None, None)  # type: ignore[return-value]


def _classify_line_by_patterns(
    line_lower: str,
    error_patterns: list[str],
    error_exclusions: list[str],
    warning_patterns: list[str],
    warning_exclusions: list[str],
) -> tuple[bool, bool]:
    """Classify line as error or warning using pattern matching.

    Args:
        line_lower: Lowercase version of the line
        error_patterns: Patterns that indicate errors
        error_exclusions: Patterns that exclude false positives
        warning_patterns: Patterns that indicate warnings
        warning_exclusions: Patterns that exclude false positives

    Returns:
        Tuple of (is_error, is_warning)
    """
    is_error = False
    is_warning = False

    # Check for Python traceback patterns (always errors)
    if "traceback (most recent call last)" in line_lower or "traceback:" in line_lower:
        return (True, False)

    # Check for error patterns (but exclude false positives)
    if any(pattern in line_lower for pattern in error_patterns):
        if not any(exclusion in line_lower for exclusion in error_exclusions):
            # Also exclude success indicators
            if not any(
                success_indicator in line_lower
                for success_indicator in [
                    "failed=0",
                    "failed: 0",
                    "failed= 0",
                    "ok=",
                    "ok:",
                    "result: episodes=",
                ]
            ):
                # Exclude degradation policy warnings
                if "degradation" in line_lower and "policy" in line_lower:
                    is_warning = True  # Degradation messages are warnings
                else:
                    is_error = True

    # Check for warning patterns (but exclude false positives)
    elif any(pattern in line_lower for pattern in warning_patterns):
        if not any(exclusion in line_lower for exclusion in warning_exclusions):
            # Also exclude parameter names and other non-warning contexts
            if not any(
                non_warning in line_lower
                for non_warning in [
                    "suppress_fp16_warning",
                    "warning=",
                    "warning_count",
                    "warnings total",
                    "no warning",
                    "disable_warning",
                    "ignore_warning",
                ]
            ):
                is_warning = True

    return (is_error, is_warning)


def _normalize_log_line(line_stripped: str) -> str:
    """Normalize log line for deduplication.

    Args:
        line_stripped: Stripped log line

    Returns:
        Normalized line (timestamp, log level, and logger name removed)
    """
    normalized = line_stripped
    # Try to remove timestamp prefix (format: "YYYY-MM-DD HH:MM:SS,mmm")
    normalized = re.sub(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} ", "", normalized)
    # Remove log level prefix if present
    normalized = re.sub(
        r"^(ERROR|CRITICAL|WARNING|INFO|DEBUG)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    # Remove logger name prefix (format: "logger_name: ")
    normalized = re.sub(r"^[a-zA-Z0-9_.]+\s*:\s*", "", normalized)
    return normalized.strip()


def collect_logs_from_output(output_dir: Path) -> Dict[str, Any]:
    """Collect log information from output directory.

    Args:
        output_dir: Output directory for the run

    Returns:
        Dict with log analysis (errors, warnings, info count)
    """
    errors = []
    warnings = []
    info_count = 0

    # Track seen error/warning lines to avoid duplicates (same error in stdout and stderr)
    seen_errors = set()
    seen_warnings = set()

    # Patterns that indicate actual errors (not just mentions of the word "error")
    error_patterns = [
        "traceback",
        "exception:",
        "error:",
        "failed",
        "failure",
        "critical",
        "fatal",
    ]

    # Patterns that indicate the word "error" is used in a non-error context
    error_exclusions = [
        "error_type: none",
        "error_message: none",
        "errors total: 0",
        "no error",
        "error_count: 0",
        "error: none",
        "error: null",
        "'error_type': none",
        "'error_message': none",
        '"error_type": null',
        '"error_message": null',
        "failed=0",  # Processing job with no failures
        "failed: 0",  # Processing job with no failures
        "failed= 0",  # Processing job with no failures
        "ok=",  # Success indicators (e.g., "ok=3, failed=0")
        "ok:",
        "result: episodes=",  # Result summary lines
        "degradation policy",  # Degradation warnings
        # (e.g., "Saving transcript without summary (degradation policy: ...)")
        "degradation:",  # Degradation logger messages
    ]

    # Patterns that indicate actual warnings
    warning_patterns = [
        "warning:",
        "warn:",
        "deprecated",
        "deprecation",
    ]

    # Patterns that indicate the word "warning" is used in a non-warning context
    warning_exclusions = [
        "warning_count: 0",
        "warnings total: 0",
        "no warning",
        "suppress_fp16_warning",  # Parameter name
        "warning=",  # Parameter assignment
        "disable_warning",
        "ignore_warning",
    ]

    # Look for log files in output directory
    log_files = list(output_dir.glob("*.log")) + list(output_dir.rglob("*.log"))

    for log_file in log_files:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    line_lower = line_stripped.lower()

                    # PRIORITY 1: Check log level first (most reliable indicator)
                    # This must be done before pattern matching to avoid misclassification
                    level_is_error, level_is_warning = _detect_log_level(line_lower, line_stripped)

                    if level_is_error is not None:
                        # Explicit log level found
                        is_error = level_is_error
                        is_warning = level_is_warning
                    else:
                        # No explicit log level found - use pattern matching as fallback
                        is_error, is_warning = _classify_line_by_patterns(
                            line_lower,
                            error_patterns,
                            error_exclusions,
                            warning_patterns,
                            warning_exclusions,
                        )

                    # Add to appropriate list (errors take priority if somehow both are True)
                    # Deduplicate by normalizing the line (remove timestamps, etc.) to avoid
                    # counting the same error from both stdout.log and stderr.log
                    if is_error:
                        normalized = _normalize_log_line(line_stripped)
                        # Only add if we haven't seen this exact error before
                        if normalized and normalized not in seen_errors:
                            seen_errors.add(normalized)
                            errors.append(line_stripped[:200])  # Limit length
                        continue
                    elif is_warning:
                        normalized = _normalize_log_line(line_stripped)
                        # Only add if we haven't seen this exact warning before
                        if normalized and normalized not in seen_warnings:
                            seen_warnings.add(normalized)
                            warnings.append(line_stripped[:200])
                        continue

                    # Count info messages
                    if "info" in line_lower or any(
                        level in line_lower for level in ["level=info", "levelname=info"]
                    ):
                        info_count += 1
        except Exception as e:
            logger.warning(f"Failed to read log file {log_file}: {e}")

    # Extract key error messages for known error types (e.g., Ollama not running)
    # This helps identify the root cause more quickly
    key_error_messages = []
    for error in errors:
        error_lower = error.lower()
        # Check for common, actionable error messages
        if "ollama server is not running" in error_lower:
            # Extract the full helpful message
            if "ollama serve" in error_lower or "ollama.ai" in error_lower:
                key_error_messages.append(
                    "Ollama server is not running - start with 'ollama serve'"
                )
        elif "connection refused" in error_lower and "ollama" in error_lower:
            key_error_messages.append("Ollama connection refused - server may not be running")
        elif "model" in error_lower and "not available" in error_lower and "ollama" in error_lower:
            key_error_messages.append(
                "Ollama model not available - may need to run 'ollama pull <model>'"
            )
        elif "api key" in error_lower and ("missing" in error_lower or "invalid" in error_lower):
            key_error_messages.append("API key missing or invalid - check environment variables")
        elif "rate limit" in error_lower:
            key_error_messages.append("Rate limit exceeded - consider reducing request frequency")

    # Remove duplicates from key_error_messages
    key_error_messages = list(dict.fromkeys(key_error_messages))  # Preserves order

    return {
        "errors": errors[:50],  # Limit to 50 errors
        "warnings": warnings[:50],  # Limit to 50 warnings
        "info_count": info_count,
        "key_error_messages": key_error_messages[:10],  # Top 10 key error messages
    }


def copy_service_outputs(service_output_dir: Path, run_output_dir: Path) -> None:
    """Copy service output files (metadata, transcripts, summaries) to run directory.

    The service creates output in a structure like:
    - service_output_dir/run_<suffix>/transcripts/
    - service_output_dir/run_<suffix>/metadata/
    - service_output_dir/run_<suffix>/*.json (run.json, index.json, etc.)

    This function copies all these files directly to run_output_dir (flattened structure).
    Metadata and transcripts folders are kept, but run.json, index.json, etc. are placed flat.

    Args:
        service_output_dir: The output directory configured in the service
            (where service wrote files)
        run_output_dir: The run directory where we want to copy the files
    """
    if not service_output_dir.exists():
        logger.warning(f"Service output directory does not exist: {service_output_dir}")
        return

    # Find the actual run subdirectory (service creates run_<suffix> subdirs)
    run_subdirs = [
        d for d in service_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    ]

    if not run_subdirs:
        # Service might have written directly to service_output_dir
        source_dir = service_output_dir
    else:
        # Use the most recent run subdirectory (in case there are multiple)
        source_dir = max(run_subdirs, key=lambda p: p.stat().st_mtime)

    logger.debug(f"Copying service outputs from: {source_dir} to {run_output_dir}")

    # Copy metadata files directly to run_output_dir/metadata/ (keep folder structure)
    metadata_source = source_dir / "metadata"
    if metadata_source.exists():
        metadata_dest = run_output_dir / "metadata"
        if metadata_dest.exists():
            shutil.rmtree(metadata_dest)
        shutil.copytree(metadata_source, metadata_dest)
        logger.debug(f"Copied metadata files to: {metadata_dest}")

    # Copy transcript files directly to run_output_dir/transcripts/ (keep folder structure)
    transcripts_source = source_dir / "transcripts"
    if transcripts_source.exists():
        transcripts_dest = run_output_dir / "transcripts"
        if transcripts_dest.exists():
            shutil.rmtree(transcripts_dest)
        shutil.copytree(transcripts_source, transcripts_dest)
        logger.debug(f"Copied transcript files to: {transcripts_dest}")

    # Copy run tracking files (run.json, index.json, run_manifest.json, metrics.json)
    # directly to run_output_dir (flat)
    for tracking_file in ["run.json", "index.json", "run_manifest.json", "metrics.json"]:
        source_file = source_dir / tracking_file
        if source_file.exists():
            dest_file = run_output_dir / tracking_file
            shutil.copy2(source_file, dest_file)
            logger.debug(f"Copied {tracking_file} to: {dest_file}")

    # Copy any markdown summary files (if they exist) directly to run_output_dir (flat)
    for md_file in source_dir.rglob("*.md"):
        # Skip if it's in a subdirectory we already copied
        if "metadata" in md_file.parts or "transcripts" in md_file.parts:
            continue
        # Copy directly to run_output_dir (flat structure)
        dest_file = run_output_dir / md_file.name
        shutil.copy2(md_file, dest_file)
        logger.debug(f"Copied markdown file {md_file.name} to: {dest_file}")

    # Clean up the original nested service output directory after copying
    # This prevents confusion and saves disk space (files are now in flattened structure)
    # Only remove if source_dir is different from run_output_dir (i.e., it's a nested subdirectory)
    if source_dir != run_output_dir and source_dir.exists():
        try:
            # Check if source_dir is actually inside run_output_dir (safety check)
            if run_output_dir in source_dir.parents or source_dir.parent == run_output_dir:
                logger.debug(f"Removing original nested service output directory: {source_dir}")
                shutil.rmtree(source_dir)
                logger.debug(f"Cleaned up nested directory: {source_dir}")
        except Exception as exc:
            # Don't fail if cleanup fails - files are already copied
            logger.warning(f"Failed to clean up nested directory {source_dir}: {exc}")


def collect_outputs(output_dir: Path) -> Dict[str, Any]:
    """Collect output file information.

    Args:
        output_dir: Output directory for the run

    Returns:
        Dict with output counts (transcripts, metadata, summaries)
    """
    # Search directly in run_output_dir (flattened structure)
    # Metadata and transcripts are in subfolders, but run.json, etc. are flat
    search_dirs = [output_dir]

    transcripts = 0
    metadata = 0
    summaries = 0

    # Track unique transcript files by their normalized name to avoid double-counting
    # (deduplication by filename ensures we count each unique transcript once)
    unique_txt_files = set()  # normalized_name (filename only)
    unique_srt_files = set()  # normalized_name (filename only)

    # Track unique metadata files by their normalized name to avoid double-counting
    # (in case files exist in multiple locations)
    unique_metadata_files = {}  # normalized_name -> Path (we'll use the first one we find)

    # First pass: collect all unique transcript and metadata files across all search directories
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        # Collect transcript files, excluding .cleaned.txt variants
        # Note: .cleaned.txt files are for quality tooling later to measure the effect of cleaning.
        # They should NOT be part of any stats or counting - only the primary .txt files count.
        txt_files = [
            p
            for p in search_dir.rglob("*.txt")
            if not p.name.endswith(
                ".cleaned.txt"
            )  # Exclude cleaned variants (quality tooling only)
        ]
        srt_files = list(search_dir.rglob("*.srt"))

        # Track unique transcript files by filename (not path) to avoid double-counting
        # (deduplication ensures we count each unique transcript once)
        for txt_file in txt_files:
            unique_txt_files.add(txt_file.name)  # Use filename only for deduplication
        for srt_file in srt_files:
            unique_srt_files.add(srt_file.name)  # Use filename only for deduplication

        # Collect metadata files (not run_data.json, run.json, etc.)
        metadata_files = [
            p
            for p in search_dir.rglob("*.json")
            if "metadata" in p.name
            or (p.parent.name == "metadata" and p.suffix in [".json", ".yaml"])
        ] + [
            p
            for p in search_dir.rglob("*.yaml")
            if "metadata" in p.name
            or (p.parent.name == "metadata" and p.suffix in [".json", ".yaml"])
        ]

        # Track unique metadata files by normalized filename
        # Use normalized filename to identify duplicates across directories
        # (e.g., "0001 - Episode Title.metadata.json" is the same file regardless of path)
        for metadata_file in metadata_files:
            normalized_name = metadata_file.name
            if normalized_name not in unique_metadata_files:
                unique_metadata_files[normalized_name] = metadata_file

    # Count unique transcript files (deduplicated by filename)
    transcripts = len(unique_txt_files) + len(unique_srt_files)

    # Count unique metadata files
    metadata = len(unique_metadata_files)

    # Count summaries: check if unique metadata files contain summaries
    # Summaries are stored inside metadata JSON/YAML files, not as separate files
    # Only process each unique metadata file once to avoid double-counting summaries
    for normalized_name, metadata_file in unique_metadata_files.items():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                if metadata_file.suffix == ".yaml":
                    import yaml

                    content = yaml.safe_load(f)
                else:
                    content = json.load(f)
                # Check if this metadata file has a summary field
                if content and isinstance(content, dict):
                    # Summary can be at top level or nested in content/summary
                    if content.get("summary") or (
                        content.get("content")
                        and isinstance(content.get("content"), dict)
                        and content["content"].get("summary")
                    ):
                        summaries += 1
        except Exception:
            # If we can't read the file, skip it
            pass

    # Also count separate markdown summary files (if any exist)
    # Track these separately to avoid double-counting
    summary_md_files = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for md_file in search_dir.rglob("*.md"):
            if "summary" in md_file.name.lower() or "summary" in str(md_file):
                # Use normalized name to avoid double-counting
                summary_md_files.add(md_file.name)
    summaries += len(summary_md_files)

    return {
        "transcripts": transcripts,
        "metadata": metadata,
        "summaries": summaries,
    }


def _detect_dry_run_from_config(config_files: list[Path]) -> bool:
    """Detect if dry-run mode is enabled in any of the config files.

    Args:
        config_files: List of config file paths to check

    Returns:
        True if dry-run mode is detected, False otherwise
    """
    for config_file in config_files:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                import yaml

                config_content = yaml.safe_load(f)
                # Check for dry_run flag (can be True, "true", 1, etc.)
                dry_run_value = config_content.get("dry_run") if config_content else None
                if dry_run_value is True or (
                    isinstance(dry_run_value, str) and dry_run_value.lower() in ["true", "1", "yes"]
                ):
                    logger.debug(f"Detected dry-run mode from config: {config_file}")
                    return True
        except Exception as e:
            logger.debug(f"Failed to read config for dry-run detection: {config_file}: {e}")
    return False


def _detect_dry_run_from_stdout(stdout_path: Path) -> bool:
    """Detect if dry-run mode is indicated in stdout.log.

    Args:
        stdout_path: Path to stdout.log file

    Returns:
        True if dry-run mode is detected, False otherwise
    """
    if not stdout_path.exists():
        return False

    # Try reading multiple times with small delay to handle race conditions
    for attempt in range(3):
        try:
            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
                stdout_content = f.read()
                # Check for various dry-run indicators
                stdout_lower = stdout_content.lower()
                if any(
                    indicator in stdout_lower
                    for indicator in [
                        "dry run complete",
                        "dry-run complete",
                        "(dry-run)",
                        "(dry run)",
                        "dry run mode",
                        "dry-run mode",
                    ]
                ):
                    logger.debug(f"Detected dry-run mode from stdout.log (attempt {attempt + 1})")
                    return True
                # If we got here, file was readable but no dry-run indicator found
                return False
        except (IOError, OSError) as e:
            # File might still be open/writing, wait a bit and retry
            if attempt < 2:
                time.sleep(0.1)  # 100ms delay
                continue
            logger.debug(f"Failed to read stdout.log after {attempt + 1} attempts: {e}")
            return False
        except Exception as e:
            logger.debug(f"Unexpected error reading stdout.log: {e}")
            return False
    return False


def _extract_episodes_from_dry_run(stdout_content: str) -> int:
    """Extract planned episode count from dry-run output.

    Args:
        stdout_content: Content of stdout.log

    Returns:
        Number of planned episodes, or 0 if not found
    """
    # Try multiple patterns to be robust
    patterns = [
        r"transcripts_planned[=:]?\s*(\d+)",  # transcripts_planned=3 or transcripts_planned: 3
        r"transcripts.*planned[=:]?\s*(\d+)",  # transcripts planned=3 (with space)
        r"planned.*transcripts[=:]?\s*(\d+)",  # planned transcripts=3
    ]
    for pattern in patterns:
        match = re.search(pattern, stdout_content, re.IGNORECASE)
        if match:
            try:
                count = int(match.group(1))
                logger.debug(
                    f"Extracted planned transcripts from dry-run output: "
                    f"{count} (pattern: {pattern})"
                )
                return count
            except (ValueError, IndexError):
                continue
    logger.debug("Could not extract transcripts_planned from dry-run output, defaulting to 0")
    return 0


def _extract_episodes_from_run_json(run_json_path: Path) -> int:
    """Extract episode count from run.json.

    Args:
        run_json_path: Path to run.json file

    Returns:
        Number of episodes processed, or 0 if not found
    """
    if not run_json_path.exists():
        return 0

    try:
        with open(run_json_path, "r") as f:
            run_data = json.load(f)
            # Try different possible field names
            return (
                run_data.get("episodes_scraped_total")
                or run_data.get("episodes_processed")
                or run_data.get("episodes")
                or run_data.get("total_episodes")
                or 0
            )
    except Exception as e:
        logger.debug(f"Failed to read episode count from run.json: {e}")
        return 0


def _extract_provider_info(config_path: Path) -> Dict[str, Any]:
    """Extract provider and model information from config file.

    Args:
        config_path: Path to config file

    Returns:
        Dict with provider/model information
    """
    try:
        config_dict = config.load_config_file(str(config_path))
    except Exception as e:
        logger.debug(f"Failed to load config for provider extraction: {e}")
        return {}

    provider_info: Dict[str, Any] = {}

    # Transcription provider
    transcription_provider = config_dict.get("transcription_provider", "whisper")
    provider_info["transcription_provider"] = transcription_provider
    if transcription_provider == "whisper":
        provider_info["transcription_model"] = config_dict.get("whisper_model", "base")
    elif transcription_provider == "openai":
        provider_info["transcription_model"] = config_dict.get(
            "openai_transcription_model", "whisper-1"
        )
    elif transcription_provider == "gemini":
        provider_info["transcription_model"] = config_dict.get(
            "gemini_transcription_model", "gemini-1.5-pro"
        )
    elif transcription_provider == "mistral":
        provider_info["transcription_model"] = config_dict.get(
            "mistral_transcription_model", "mistral-large-latest"
        )

    # Speaker detection provider
    speaker_provider = config_dict.get("speaker_detector_provider", "spacy")
    provider_info["speaker_provider"] = speaker_provider
    if speaker_provider == "spacy":
        provider_info["speaker_model"] = config_dict.get("ner_model", "en_core_web_sm")
    elif speaker_provider == "openai":
        provider_info["speaker_model"] = config_dict.get("openai_speaker_model", "gpt-4o-mini")
    elif speaker_provider == "gemini":
        provider_info["speaker_model"] = config_dict.get("gemini_speaker_model", "gemini-1.5-pro")
    elif speaker_provider == "anthropic":
        provider_info["speaker_model"] = config_dict.get(
            "anthropic_speaker_model", "claude-3-5-haiku-latest"
        )
    elif speaker_provider == "mistral":
        provider_info["speaker_model"] = config_dict.get(
            "mistral_speaker_model", "mistral-large-latest"
        )
    elif speaker_provider == "grok":
        provider_info["speaker_model"] = config_dict.get("grok_speaker_model", "grok-beta")
    elif speaker_provider == "deepseek":
        provider_info["speaker_model"] = config_dict.get("deepseek_speaker_model", "deepseek-chat")
    elif speaker_provider == "ollama":
        provider_info["speaker_model"] = config_dict.get("ollama_speaker_model", "llama3.1:8b")

    # Summarization provider
    summary_provider = config_dict.get("summary_provider", "transformers")
    provider_info["summary_provider"] = summary_provider
    if summary_provider in ("transformers", "local"):
        provider_info["summary_map_model"] = config_dict.get(
            "summary_model", "facebook/bart-large-cnn"
        )
        provider_info["summary_reduce_model"] = config_dict.get("summary_reduce_model")
    elif summary_provider == "openai":
        provider_info["summary_model"] = config_dict.get("openai_summary_model", "gpt-4o-mini")
    elif summary_provider == "gemini":
        provider_info["summary_model"] = config_dict.get("gemini_summary_model", "gemini-1.5-pro")
    elif summary_provider == "anthropic":
        provider_info["summary_model"] = config_dict.get(
            "anthropic_summary_model", "claude-3-5-haiku-latest"
        )
    elif summary_provider == "mistral":
        provider_info["summary_model"] = config_dict.get(
            "mistral_summary_model", "mistral-large-latest"
        )
    elif summary_provider == "grok":
        provider_info["summary_model"] = config_dict.get("grok_summary_model", "grok-beta")
    elif summary_provider == "deepseek":
        provider_info["summary_model"] = config_dict.get("deepseek_summary_model", "deepseek-chat")
    elif summary_provider == "ollama":
        provider_info["summary_model"] = config_dict.get("ollama_summary_model", "llama3.1:8b")

    return provider_info


def _extract_episodes_from_index_json(index_json_path: Path) -> int:
    """Extract episode count from index.json.

    Args:
        index_json_path: Path to index.json file

    Returns:
        Number of episodes processed, or 0 if not found
    """
    if not index_json_path.exists():
        return 0

    try:
        with open(index_json_path, "r") as f:
            index_data = json.load(f)
            episodes = index_data.get("episodes", [])
            if isinstance(episodes, list):
                return len(episodes)
            elif isinstance(episodes, dict):
                return len(episodes.get("items", []))
    except Exception as e:
        logger.debug(f"Failed to read episode count from index.json: {e}")
    return 0


def _execute_process_with_streaming(
    cmd: list[str], stdout_path: Path, stderr_path: Path, config_name: str
) -> tuple[int, Dict[str, Any]]:
    """Execute process with real-time log streaming.

    Args:
        cmd: Command to execute
        stdout_path: Path to save stdout
        stderr_path: Path to save stderr
        config_name: Config name for log prefix

    Returns:
        Tuple of (exit_code, resource_usage_dict)
    """
    stdout_file = open(stdout_path, "w", buffering=1)  # Line buffered
    stderr_file = open(stderr_path, "w", buffering=1)  # Line buffered

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        text=True,
        bufsize=1,  # Line buffered
    )

    # Stream output in real-time
    def stream_output(pipe, file, prefix=""):
        """Stream output from pipe to both file and console."""
        for line in iter(pipe.readline, ""):
            if not line:
                break
            # Write to file
            file.write(line)
            file.flush()
            # Also print to console with prefix
            if prefix:
                print(f"{prefix}{line.rstrip()}", flush=True)
            else:
                print(line.rstrip(), flush=True)

    # Start threads to stream stdout and stderr
    stdout_thread = threading.Thread(
        target=stream_output,
        args=(process.stdout, stdout_file, f"[{config_name}] "),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=stream_output,
        args=(process.stderr, stderr_file, f"[{config_name}] "),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    # Monitor resources continuously in background thread
    resource_result = {"resource_usage": None}

    def monitor_resources():
        resource_result["resource_usage"] = monitor_process_resources(process)

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    # Wait for completion
    exit_code = process.wait()

    # Wait for monitoring to finish
    monitor_thread.join(timeout=2.0)
    resource_usage = resource_result["resource_usage"] or {
        "peak_memory_mb": None,
        "cpu_time_seconds": None,
        "cpu_percent": None,
    }

    # Wait for output threads to finish
    stdout_thread.join(timeout=1.0)
    stderr_thread.join(timeout=1.0)

    # Close files
    stdout_file.close()
    stderr_file.close()

    return exit_code, resource_usage


def _execute_process_without_streaming(
    cmd: list[str], stdout_path: Path, stderr_path: Path
) -> tuple[int, Dict[str, Any]]:
    """Execute process without log streaming (only save to files).

    Args:
        cmd: Command to execute
        stdout_path: Path to save stdout
        stderr_path: Path to save stderr

    Returns:
        Tuple of (exit_code, resource_usage_dict)
    """
    with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
        process = subprocess.Popen(
            cmd,
            stdout=stdout_file,
            stderr=stderr_file,
            env=os.environ.copy(),
        )

        # Monitor resources continuously
        resource_usage = monitor_process_resources(process)

        # Wait for completion
        exit_code = process.wait()

    return exit_code, resource_usage


def run_config(
    config_path: Path,
    e2e_server: Optional[E2EHTTPServer],
    output_dir: Path,
    run_id: str,
    use_fixtures: bool = True,
    show_logs: bool = True,
) -> Dict[str, Any]:
    """Run a single config and collect data.

    Args:
        config_path: Path to config file
        e2e_server: E2E server instance (None if using real feeds/APIs)
        output_dir: Base output directory
        run_id: Unique run identifier
        use_fixtures: If True, use E2E server fixtures; if False, use real RSS/APIs
        show_logs: If True, stream logs to console in real-time; if False, only save to files

    Returns:
        Dict with run data
    """
    config_name = config_path.stem
    logger.info(f"Running config: {config_name}")

    # Create timestamped run directory (not based on config name since feeds may differ)
    # output_dir is now runs_dir (session_dir / "runs"), so create run folder directly
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[
        :-3
    ]  # Include milliseconds for uniqueness
    run_dir_name = f"run_{run_timestamp}"
    run_output_dir = output_dir / run_dir_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original config to run folder for reference
    original_config_copy = run_output_dir / "config.original.yaml"
    shutil.copy2(config_path, original_config_copy)

    # Modify config (with optional fixture support)
    # Pass session_dir and run_output_dir so it can set the output_dir in the config
    # Need to get session_dir from output_dir (which is now runs_dir)
    session_dir = output_dir.parent  # runs_dir.parent is session_dir
    modified_config = modify_config_for_fixtures(
        config_path, e2e_server, session_dir, run_output_dir, use_fixtures=use_fixtures
    )

    # Get config hash (from original config)
    config_hash = get_config_hash(config_path)

    # Run the service
    start_time = time.time()
    start_timestamp = datetime.utcnow().isoformat() + "Z"

    # Build command
    python_cmd = sys.executable
    cmd = [
        python_cmd,
        "-m",
        "podcast_scraper.service",
        "--config",
        str(modified_config),
    ]

    # Capture stdout/stderr to files (and optionally stream to console)
    stdout_path = run_output_dir / "stdout.log"
    stderr_path = run_output_dir / "stderr.log"

    try:
        if show_logs:
            exit_code, resource_usage = _execute_process_with_streaming(
                cmd, stdout_path, stderr_path, config_name
            )
        else:
            exit_code, resource_usage = _execute_process_without_streaming(
                cmd, stdout_path, stderr_path
            )
    except Exception as e:
        logger.error(f"Failed to run config {config_name}: {e}", exc_info=True)
        exit_code = 1
        resource_usage = {
            "peak_memory_mb": None,
            "cpu_time_seconds": None,
            "cpu_percent": None,
        }

    end_time = time.time()
    end_timestamp = datetime.utcnow().isoformat() + "Z"
    duration_seconds = end_time - start_time

    # Detect dry-run mode EARLY (before collecting outputs)
    stdout_path = run_output_dir / "stdout.log"
    configs_to_check = [original_config_copy]
    if modified_config.exists():
        configs_to_check.append(modified_config)

    is_dry_run = _detect_dry_run_from_config(configs_to_check)
    if not is_dry_run:
        is_dry_run = _detect_dry_run_from_stdout(stdout_path)

    # Read stdout content for episode extraction (if needed)
    stdout_content = ""
    if stdout_path.exists():
        try:
            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
                stdout_content = f.read()
        except Exception:
            pass  # Ignore read errors

    # Copy service output files (metadata, transcripts, summaries) to run directory
    # The service writes to run_output_dir (or a subdirectory within it)
    # We copy those files directly to run_output_dir (flattened structure)
    # Metadata and transcripts folders are kept, but run.json, index.json, etc. are placed flat
    # Skip for dry-run mode (no files are created)
    if not is_dry_run:
        copy_service_outputs(run_output_dir, run_output_dir)

    # Collect logs
    logs = collect_logs_from_output(run_output_dir)

    # Collect outputs (skip for dry-run mode, or set to 0)
    if is_dry_run:
        outputs = {"transcripts": 0, "metadata": 0, "summaries": 0}
    else:
        outputs = collect_outputs(run_output_dir)

    # Determine episodes processed from service output files (run.json, index.json, or metrics.json)
    # These files contain the authoritative episode count from the service
    episodes_processed = 0

    if is_dry_run:
        episodes_processed = _extract_episodes_from_dry_run(stdout_content)
    else:
        # Try to read from run.json first (most reliable)
        run_json_path = run_output_dir / "run.json"
        episodes_processed = _extract_episodes_from_run_json(run_json_path)

        # Fallback: try index.json
        if episodes_processed == 0:
            index_json_path = run_output_dir / "index.json"
            episodes_processed = _extract_episodes_from_index_json(index_json_path)

        # Final fallback: count transcript files (less accurate, may include duplicates)
        # Note: This already excludes .cleaned.txt files in collect_outputs()
        if episodes_processed == 0:
            episodes_processed = outputs.get("transcripts", 0)
            logger.debug(f"Using transcript file count as fallback: {episodes_processed}")

    # Log dry-run detection result for debugging
    if is_dry_run:
        logger.info(
            f"✓ Dry-run mode detected for {config_name} (episodes_planned={episodes_processed})"
        )

    # Extract provider/model information from config
    provider_info = _extract_provider_info(original_config_copy)

    # Build run data
    run_data = {
        "run_id": run_id,
        "config_file": str(config_path),
        "config_name": config_name,
        "config_hash": config_hash,
        "start_time": start_timestamp,
        "end_time": end_timestamp,
        "duration_seconds": round(duration_seconds, 2),
        "exit_code": exit_code,
        "episodes_processed": episodes_processed,
        "is_dry_run": is_dry_run,  # Flag indicating dry-run mode
        "output_dir": str(run_output_dir),
        "logs": logs,
        "outputs": outputs,
        "resource_usage": resource_usage,
        "provider_info": provider_info,  # Provider/model information for benchmarking
    }

    # Save run data
    run_data_path = run_output_dir / "run_data.json"
    with open(run_data_path, "w") as f:
        json.dump(run_data, f, indent=2)

    logger.info(
        f"Completed {config_name}: exit_code={exit_code}, "
        f"duration={duration_seconds:.1f}s, episodes={episodes_processed}"
    )

    return run_data


def save_baseline(baseline_id: str, runs_data: List[Dict[str, Any]], output_dir: Path) -> None:
    """Save current runs as a baseline.

    Copies all run data into the baseline folder for easy access and comparison.

    Args:
        baseline_id: Baseline identifier
        runs_data: List of run data dicts
        output_dir: Base output directory
    """
    baseline_dir = output_dir / "baselines" / baseline_id
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Copy run data directories into baseline folder
    for run_data in runs_data:
        run_output_dir = Path(run_data.get("output_dir", ""))
        if run_output_dir.exists():
            # Copy the entire run directory to baseline
            run_name = run_output_dir.name
            baseline_run_dir = baseline_dir / run_name
            if baseline_run_dir.exists():
                # Remove existing if it exists (shouldn't happen, but be safe)
                shutil.rmtree(baseline_run_dir)
            shutil.copytree(run_output_dir, baseline_run_dir)
            logger.debug(f"Copied run data: {run_name} -> {baseline_run_dir}")

    # Save baseline metadata
    baseline_metadata = {
        "baseline_id": baseline_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "runs": runs_data,
        "total_runs": len(runs_data),
        "successful_runs": sum(1 for r in runs_data if r.get("exit_code", 1) == 0),
        "failed_runs": sum(1 for r in runs_data if r.get("exit_code", 1) != 0),
    }

    baseline_path = baseline_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_metadata, f, indent=2)

    logger.info(f"Saved baseline: {baseline_id} ({len(runs_data)} runs)")
    logger.info(f"  Baseline directory: {baseline_dir.absolute()}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run E2E acceptance tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help=(
            "Config file pattern (e.g., 'config/examples/config.example.yaml' "
            "or 'config/acceptance/*.yaml')"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".test_outputs/acceptance",
        help="Output directory for results (default: .test_outputs/acceptance)",
    )
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Baseline ID to compare current runs against (optional)",
    )
    parser.add_argument(
        "--save-as-baseline",
        type=str,
        default=None,
        help="Save current runs as baseline with this ID (optional)",
    )
    parser.add_argument(
        "--use-fixtures",
        action="store_true",
        default=False,
        help="Use E2E server fixtures (test feeds and mock APIs). "
        "If not set, uses real RSS feeds and real API providers from config/environment.",
    )
    parser.add_argument(
        "--show-logs",
        action="store_true",
        default=True,
        help="Stream service logs to console in real-time (default: True)",
    )
    parser.add_argument(
        "--no-show-logs",
        dest="show_logs",
        action="store_false",
        help="Disable streaming service logs to console (only save to files)",
    )
    parser.add_argument(
        "--no-auto-analyze",
        dest="auto_analyze",
        action="store_false",
        help="Disable automatic analysis after session completes (default: enabled)",
    )
    parser.set_defaults(auto_analyze=True)
    parser.add_argument(
        "--no-auto-benchmark",
        dest="auto_benchmark",
        action="store_false",
        help=(
            "Disable automatic performance benchmark report generation "
            "after session completes (default: enabled)"
        ),
    )
    parser.set_defaults(auto_benchmark=True)
    parser.add_argument(
        "--analyze-mode",
        type=str,
        default="basic",
        choices=["basic", "comprehensive"],
        help="Analysis mode for auto-analysis (default: basic)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Check E2E server availability (only needed if using fixtures)
    use_fixtures = args.use_fixtures
    e2e_server = None

    if use_fixtures:
        if E2EHTTPServer is None:
            logger.error(
                "E2E server not available. Cannot use fixtures without E2E server. "
                "Either install test dependencies or use --use-fixtures=false to use real feeds."
            )
            sys.exit(1)

        # Start E2E server
        logger.info("Starting E2E server for fixture feeds and mock APIs...")
        e2e_server = E2EHTTPServer()
        e2e_server.start()
        logger.info(f"E2E server started at {e2e_server.base_url}")
    else:
        logger.info("Using real RSS feeds and real API providers")
        logger.info(
            "Make sure your config files have valid RSS URLs and API keys are set in environment"
        )

    # Find config files
    config_files = find_config_files(args.configs)
    if not config_files:
        logger.error("No config files found")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create session folder
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = output_dir / "sessions" / f"session_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = session_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Confirm output directory at start
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"  - Session folder: {session_dir}")
    logger.info(f"  - Run data: {runs_dir}")
    logger.info(f"  - Baselines: {output_dir / 'baselines'}")
    logger.info("=" * 70)
    logger.info("")

    try:
        # Run all configs sequentially
        runs_data = []

        for i, config_file in enumerate(config_files, 1):
            run_id = f"{config_file.stem}_{session_id}"
            run_data = run_config(
                config_file,
                e2e_server,
                runs_dir,  # Pass runs_dir instead of output_dir
                run_id,
                use_fixtures=use_fixtures,
                show_logs=args.show_logs,
            )
            runs_data.append(run_data)

        # Save session summary in session folder
        session_summary = {
            "session_id": session_id,
            "start_time": runs_data[0]["start_time"] if runs_data else None,
            "end_time": runs_data[-1]["end_time"] if runs_data else None,
            "total_runs": len(runs_data),
            "successful_runs": sum(1 for r in runs_data if r["exit_code"] == 0),
            "failed_runs": sum(1 for r in runs_data if r["exit_code"] != 0),
            "total_duration_seconds": sum(r["duration_seconds"] for r in runs_data),
            "config_files": [str(cf) for cf in config_files],  # Store list of config files used
            "runs": runs_data,
        }

        session_summary_path = session_dir / "session.json"
        with open(session_summary_path, "w") as f:
            json.dump(session_summary, f, indent=2)

        logger.info(f"Session summary saved: {session_summary_path}")

        # Run automatic analysis if enabled (default)
        logger.debug(f"Auto-analyze setting: {args.auto_analyze}")
        if args.auto_analyze:
            logger.info("")
            logger.info("Running automatic analysis...")
            try:
                # Use absolute path to the analysis script
                analyze_script = Path(__file__).parent / "analyze_bulk_runs.py"
                if not analyze_script.exists():
                    raise FileNotFoundError(f"Analysis script not found: {analyze_script}")

                analyze_cmd = [
                    sys.executable,
                    str(analyze_script.absolute()),
                    "--session-id",
                    session_id,
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    args.analyze_mode,
                    "--output-format",
                    "both",
                    "--log-level",
                    "INFO",
                ]
                if args.compare_baseline:
                    analyze_cmd.extend(["--compare-baseline", args.compare_baseline])

                logger.debug(f"Running analysis command: {' '.join(analyze_cmd)}")
                result = subprocess.run(
                    analyze_cmd,
                    capture_output=False,
                )
                if result.returncode == 0:
                    logger.info("✓ Automatic analysis completed")
                else:
                    logger.warning(f"Analysis completed with exit code {result.returncode}")

                # Also generate performance benchmark report (if enabled)
                if args.auto_benchmark:
                    benchmark_script = Path(__file__).parent / "generate_performance_benchmark.py"
                    if benchmark_script.exists():
                        logger.info("")
                        logger.info("Generating performance benchmark report...")
                        benchmark_cmd = [
                            sys.executable,
                            str(benchmark_script.absolute()),
                            "--session-id",
                            session_id,
                            "--output-dir",
                            str(output_dir),
                            "--output-format",
                            "both",
                            "--log-level",
                            "INFO",
                        ]
                        if args.compare_baseline:
                            benchmark_cmd.extend(["--compare-baseline", args.compare_baseline])
                        logger.debug(f"Running benchmark command: {' '.join(benchmark_cmd)}")
                        benchmark_result = subprocess.run(
                            benchmark_cmd,
                            capture_output=False,
                        )
                        if benchmark_result.returncode == 0:
                            logger.info("✓ Performance benchmark report generated")
                        else:
                            logger.warning(
                                f"Benchmark generation completed with exit code "
                                f"{benchmark_result.returncode}"
                            )
                    else:
                        logger.warning(
                            "Benchmark script not found, skipping performance "
                            "benchmark report generation"
                        )
                else:
                    logger.debug(
                        "Auto-benchmark disabled, skipping performance "
                        "benchmark report generation"
                    )
            except Exception as e:
                logger.error(f"Failed to run automatic analysis: {e}", exc_info=True)
                logger.info(
                    "You can run analysis manually with: "
                    f"make analyze-acceptance SESSION_ID={session_id}"
                )

        # Save as baseline if requested
        if args.save_as_baseline:
            save_baseline(args.save_as_baseline, runs_data, output_dir)

        # Compare with baseline if requested
        if args.compare_baseline:
            baseline_path = output_dir / "baselines" / args.compare_baseline / "baseline.json"
            if baseline_path.exists():
                logger.info(f"Baseline comparison requested: {args.compare_baseline}")
                # Comparison will be done by analysis script
                logger.info(
                    "Use scripts/acceptance/analyze_bulk_runs.py to generate comparison report"
                )
            else:
                logger.warning(f"Baseline not found: {baseline_path}. " "Skipping comparison.")

        logger.info("Acceptance tests completed")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_dir.absolute()}")
        logger.info(f"  - Session folder: {session_dir.absolute()}")
        logger.info(f"  - Session summary: {session_summary_path.absolute()}")
        logger.info(f"  - Run data: {runs_dir.absolute()}")
        if args.auto_analyze:
            logger.info(
                f"  - Analysis reports: {session_dir / f'report_{session_id}.md'} and .json"
            )
        logger.info(f"  - Baselines: {(output_dir / 'baselines').absolute()}")
        logger.info("=" * 70)

    finally:
        # Stop E2E server (if it was started)
        if e2e_server:
            logger.info("Stopping E2E server...")
            e2e_server.stop()


if __name__ == "__main__":
    main()
