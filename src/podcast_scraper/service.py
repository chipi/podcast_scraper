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
from typing import Optional

from . import __version__, config, workflow

logger = logging.getLogger(__name__)


@dataclass
class ServiceResult:
    """Result of a service run.

    Attributes:
        episodes_processed: Number of episodes processed (transcripts saved/planned)
        summary: Human-readable summary message
        success: Whether the run completed successfully
        error: Error message if success is False, None otherwise
    """

    episodes_processed: int
    summary: str
    success: bool = True
    error: Optional[str] = None


def run(cfg: config.Config) -> ServiceResult:
    """Run the podcast scraping pipeline with the given configuration.

    This is the main entry point for programmatic use. It executes the full pipeline
    and returns a structured result suitable for service/daemon use.

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
            workflow.apply_log_level(
                level=cfg.log_level or "INFO",
                log_file=cfg.log_file,
            )

        # Run the pipeline
        count, summary = workflow.run_pipeline(cfg)

        return ServiceResult(
            episodes_processed=count,
            summary=summary,
            success=True,
            error=None,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Pipeline execution failed: {error_msg}", exc_info=True)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_msg,
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
        logger.error(error_msg)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_msg,
        )
    except Exception as exc:
        error_msg = f"Failed to load configuration file: {exc}"
        logger.error(error_msg)
        return ServiceResult(
            episodes_processed=0,
            summary="",
            success=False,
            error=error_msg,
        )

    return run(cfg)


def main() -> int:
    """Main entry point for service mode (CLI-like but config-file only).

    This function is designed to be called as a script entry point:
    python -m podcast_scraper.service --config config.yaml

    It only accepts a --config argument and is optimized for non-interactive use.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse

    from . import workflow

    # Initialize ML environment variables early (before any ML imports)
    workflow._initialize_ml_environment()

    parser = argparse.ArgumentParser(
        description="Podcast Scraper Service - Run pipeline from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m podcast_scraper.service --config config.yaml

  # For supervisor/systemd usage
  [program:podcast_scraper]
  command=python -m podcast_scraper.service --config /path/to/config.yaml
  autostart=true
  autorestart=true
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file (JSON or YAML)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"podcast_scraper {__version__}",
    )

    args = parser.parse_args()

    # Run the service
    result = run_from_config_file(args.config)

    # Print results
    if result.success:
        print(result.summary)
        return 0
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
