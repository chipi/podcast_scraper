"""Service API for programmatic use of podcast_scraper.

This module provides a clean, programmatic interface optimized for non-interactive use,
such as running as a daemon or service (e.g., with supervisor, systemd, etc.).

The service API is designed to:
- Work exclusively with configuration files (no CLI arguments)
- Provide clear return values and error handling
- Be suitable for process management tools
- Maintain clean separation from CLI concerns
- Expose only necessary public API surface

Example:
    >>> from podcast_scraper import service, config
    >>>
    >>> # Load config file and create Config object
    >>> config_dict = config.load_config_file("config.yaml")
    >>> cfg = config.Config(**config_dict)
    >>> result = service.run(cfg)
    >>> print(f"Processed {result.episodes_processed} episodes")
    >>> print(f"Summary: {result.summary}")

For daemon/service usage:
    # supervisor config
    [program:podcast_scraper]
    command=python -m podcast_scraper.service --config /path/to/config.yaml
    autostart=true
    autorestart=true
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__, config
from .utils import filesystem
from .utils.log_redaction import redact_for_log
from .workflow import orchestration as workflow
from .workflow.corpus_operations import (
    finalize_multi_feed_batch,
    MultiFeedFeedResult,
    utc_iso_now,
)
from .workflow.stages import setup

logger = logging.getLogger(__name__)


@dataclass
class ServiceResult:
    """Result of a service run.

    Attributes:
        episodes_processed: Number of episodes processed (transcripts saved/planned)
        summary: Human-readable summary message
        success: Whether the run completed successfully
        error: Error message if success is False, None otherwise
        multi_feed_summary: When ``run`` used multi-feed mode (two or more ``rss_urls``),
            the same JSON-shaped dict as ``corpus_run_summary.json`` at the corpus parent
            (GitHub #506). ``None`` for single-feed runs and for top-level failures before
            multi-feed finalize.
    """

    episodes_processed: int
    summary: str
    success: bool = True
    error: Optional[str] = None
    multi_feed_summary: Optional[Dict[str, Any]] = None


def _run_multi_feed(cfg: config.Config, feed_urls: List[str]) -> ServiceResult:
    """Run one pipeline per feed under ``output_dir/feeds/…`` (GitHub #440).

    Per-feed failures are isolated: other feeds still run; ``success`` is False if any fail.
    After the batch, writes manifest/summary (#506) and builds a parent vector index when
    ``vector_search`` and FAISS are enabled (#505).
    """
    from podcast_scraper.utils.corpus_lock import corpus_parent_lock

    total = 0
    ok_parts: List[str] = []
    err_parts: List[str] = []
    parent = cfg.output_dir or ""
    batch: List[MultiFeedFeedResult] = []
    skip_auto = cfg.vector_search is True and getattr(cfg, "vector_backend", "faiss") == "faiss"

    try:
        with corpus_parent_lock(Path(parent), logger=logger):
            for url in feed_urls:
                child_dir = filesystem.corpus_feed_output_dir(parent, url)
                sub_cfg = cfg.model_copy(
                    update={
                        "rss_url": url,
                        "output_dir": child_dir,
                        "rss_urls": None,
                        "skip_auto_vector_index": skip_auto,
                    },
                )
                try:
                    count, summary = workflow.run_pipeline(sub_cfg)
                    total += count
                    ok_parts.append(f"{url}: {summary}")
                    batch.append(
                        MultiFeedFeedResult(url, True, None, int(count), finished_at=utc_iso_now())
                    )
                except Exception as exc:
                    msg = redact_for_log(str(exc))
                    logger.error("Multi-feed: feed failed url=%s err=%s", url, msg, exc_info=True)
                    err_parts.append(f"{url}: {msg}")
                    batch.append(MultiFeedFeedResult(url, False, msg, 0, finished_at=utc_iso_now()))

            template = cfg.model_copy(
                update={
                    "output_dir": parent,
                    "rss_url": feed_urls[0],
                    "rss_urls": None,
                    "skip_auto_vector_index": False,
                }
            )
            summary_doc = finalize_multi_feed_batch(parent, template, batch)
    except RuntimeError as exc:
        return ServiceResult(
            episodes_processed=0,
            summary=str(exc),
            success=False,
            error=str(exc),
            multi_feed_summary=None,
        )

    if err_parts:
        summary_text = "\n".join(ok_parts + ["", "Failures:", *err_parts])
        return ServiceResult(
            episodes_processed=total,
            summary=summary_text,
            success=False,
            error="; ".join(err_parts),
            multi_feed_summary=summary_doc,
        )
    return ServiceResult(
        episodes_processed=total,
        summary="\n".join(ok_parts),
        success=True,
        error=None,
        multi_feed_summary=summary_doc,
    )


def run(cfg: config.Config) -> ServiceResult:
    """Run the podcast scraping pipeline with the given configuration.

    This is the main entry point for programmatic use. It executes the full pipeline
    and returns a structured result suitable for service/daemon use.

    When ``cfg.rss_urls`` contains two or more URLs (e.g. from YAML ``feeds:``), runs one
    pipeline per feed under ``output_dir/feeds/<stable_name>/`` (GitHub #440), same layout as
    the multi-feed CLI.

    Args:
        cfg: Configuration object (can be created from Config() or Config(**load_config_file()))

    Returns:
        ServiceResult with processing results

    Example:
        >>> from podcast_scraper import service, config
        >>> cfg = config.Config(rss_url="https://example.com/feed.xml")
        >>> result = service.run(cfg)
        >>> if result.success:
        ...     print(f"Success: {result.summary}")
        ... else:
        ...     print(f"Error: {result.error}")
    """
    try:
        # Apply logging configuration if specified
        if cfg.log_file or cfg.log_level:
            resolved_log = workflow.resolve_log_file_path(cfg.log_file, cfg.output_dir)
            workflow.apply_log_level(
                level=cfg.log_level or "INFO",
                log_file=resolved_log,
            )

        multi_urls = list(cfg.rss_urls or [])
        if len(multi_urls) >= 2:
            return _run_multi_feed(cfg, multi_urls)

        count, summary = workflow.run_pipeline(cfg)

        return ServiceResult(
            episodes_processed=count,
            summary=summary,
            success=True,
            error=None,
        )
    except Exception as e:
        error_safe = redact_for_log(str(e))
        logger.error("Pipeline execution failed: %s", error_safe, exc_info=True)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_safe,
        )


def run_from_config_file(config_path: str | Path) -> ServiceResult:
    """Run the pipeline from a configuration file.

    Convenience function that loads a config file and runs the pipeline.
    This is the recommended entry point for service/daemon usage.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        ServiceResult with processing results

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid

    Example:
        >>> from podcast_scraper import service
        >>> result = service.run_from_config_file("config.yaml")
        >>> if not result.success:
        ...     sys.exit(1)
    """
    try:
        config_dict = config.load_config_file(str(config_path))
        cfg = config.Config(**config_dict)
    except FileNotFoundError:
        error_msg = f"Configuration file not found: {config_path}"
        error_safe = redact_for_log(error_msg)
        logger.error("%s", error_safe)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_safe,
        )
    except Exception as exc:
        error_safe = redact_for_log(f"Failed to load configuration file: {exc}")
        logger.error("%s", error_safe)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_safe,
        )

    from .monitor.memray_util import maybe_reexec_memray_service

    memray_err = maybe_reexec_memray_service(
        memray=bool(cfg.memray),
        output_dir=cfg.output_dir,
        memray_output=cfg.memray_output,
    )
    if memray_err:
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=memray_err,
        )

    return run(cfg)


def main() -> int:
    """Main entry point for service mode (CLI-like but config-file only).

    This function is designed to be called as a script entry point:
    python -m podcast_scraper.service --config config.yaml

    It accepts a --config argument (optional if PODCAST_SCRAPER_CONFIG env var is set)
    and is optimized for non-interactive use.

    Config file resolution order:
    1. --config argument (if provided)
    2. PODCAST_SCRAPER_CONFIG environment variable
    3. Default: /app/config.yaml (for Docker/service usage)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    import os

    # Initialize ML environment variables early (before any ML imports)
    setup.initialize_ml_environment()

    # Default config path (for Docker/service usage)
    default_config = os.getenv("PODCAST_SCRAPER_CONFIG", "/app/config.yaml")

    parser = argparse.ArgumentParser(
        description="Podcast Scraper Service - Run pipeline from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m podcast_scraper.service --config config.yaml

  # Run with environment variable
  PODCAST_SCRAPER_CONFIG=/path/to/config.yaml python -m podcast_scraper.service

  # Run with default path (Docker/service mode)
  python -m podcast_scraper.service

  # For supervisor/systemd usage
  [program:podcast_scraper]
  command=python -m podcast_scraper.service --config /path/to/config.yaml
  autostart=true
  autorestart=true
        """,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to configuration file (JSON or YAML). "
            "If not provided, uses PODCAST_SCRAPER_CONFIG environment variable "
            f"or default: {default_config}"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"podcast_scraper {__version__}",
    )

    args = parser.parse_args()

    # Resolve config file path
    config_path = args.config or default_config

    # Run the service
    result = run_from_config_file(config_path)

    # Print results
    if result.success:
        print(result.summary)
        return 0
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
