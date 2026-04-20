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
from .utils.corpus_incidents import append_corpus_incident
from .utils.log_redaction import redact_for_log
from .workflow import orchestration as workflow
from .workflow.corpus_operations import (
    classify_multi_feed_feed_exception,
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
        soft_failures: When ``success`` is True but some feeds failed with **soft**
            classifications only (GitHub #559; default ``multi_feed_strict`` is False),
            holds the same aggregated detail that would have been in ``error`` under **strict**
            mode (``multi_feed_strict=True``). ``None`` when there were no soft-only failures.
    """

    episodes_processed: int
    summary: str
    success: bool = True
    error: Optional[str] = None
    multi_feed_summary: Optional[Dict[str, Any]] = None
    soft_failures: Optional[str] = None


def _run_multi_feed(cfg: config.Config, feed_urls: List[str]) -> ServiceResult:
    """Run one pipeline per feed under ``output_dir/feeds/…`` (GitHub #440).

    Per-feed failures are isolated: other feeds still run. When ``cfg.multi_feed_strict`` is
    False (default) and **every** failed feed is classified as **soft** (see
    ``classify_multi_feed_feed_exception``), ``success`` stays True with details on
    ``ServiceResult.soft_failures``. If ``multi_feed_strict`` is True (strict CI), or any
    failure is **hard**, ``success`` is False if any feed fails.
    After the batch, writes manifest/summary (#506) and builds a parent vector
    index when ``vector_search`` and FAISS are enabled (#505).
    """
    from podcast_scraper.utils.corpus_lock import corpus_parent_lock

    total = 0
    ok_parts: List[str] = []
    err_parts: List[str] = []
    parent = cfg.output_dir or ""
    batch: List[MultiFeedFeedResult] = []
    skip_auto = cfg.vector_search is True and getattr(cfg, "vector_backend", "faiss") == "faiss"
    incident_log_default = str(Path(parent) / "corpus_incidents.jsonl")
    incident_log_start = (
        Path(incident_log_default).stat().st_size if Path(incident_log_default).is_file() else 0
    )

    summary_doc: Optional[Dict[str, Any]] = None
    workflow.begin_multi_feed_ml_batch()
    try:
        try:
            with corpus_parent_lock(Path(parent), logger=logger):
                n_feeds = len(feed_urls)
                for idx, url in enumerate(feed_urls):
                    child_dir = filesystem.corpus_feed_output_dir(parent, url)
                    incident_log = (cfg.incident_log_path or "").strip()
                    if not incident_log:
                        incident_log = str(Path(parent) / "corpus_incidents.jsonl")
                    sub_cfg = cfg.model_copy(
                        update={
                            "rss_url": url,
                            "output_dir": child_dir,
                            "rss_urls": None,
                            "skip_auto_vector_index": skip_auto,
                            "incident_log_path": incident_log,
                        },
                    )
                    try:
                        count, summary = workflow.run_pipeline(sub_cfg)
                        total += count
                        ok_parts.append(f"{url}: {summary}")
                        batch.append(
                            MultiFeedFeedResult(
                                url, True, None, int(count), finished_at=utc_iso_now()
                            )
                        )
                    except Exception as exc:
                        msg = redact_for_log(str(exc))
                        kind = classify_multi_feed_feed_exception(exc)
                        logger.error(
                            "Multi-feed: feed failed url=%s err=%s", url, msg, exc_info=True
                        )
                        append_corpus_incident(
                            sub_cfg.incident_log_path,
                            scope="feed",
                            category=kind,
                            message=msg,
                            exception_type=type(exc).__name__,
                            stage="pipeline",
                            feed_url=url,
                        )
                        err_parts.append(f"{url}: {msg}")
                        batch.append(
                            MultiFeedFeedResult(
                                url,
                                False,
                                msg,
                                0,
                                finished_at=utc_iso_now(),
                                failure_kind=kind,
                            )
                        )
                    finally:
                        # Drop HF QA pipeline cache between feeds so later feeds do not inherit
                        # meta-tensor / lazy-init state from feed 1 (GitHub #539).
                        if n_feeds > 1 and idx < n_feeds - 1:
                            try:
                                from podcast_scraper.providers.ml import extractive_qa

                                extractive_qa.clear_qa_pipeline_cache()
                            except ImportError:
                                pass

                template = cfg.model_copy(
                    update={
                        "output_dir": parent,
                        "rss_url": feed_urls[0],
                        "rss_urls": None,
                        "skip_auto_vector_index": False,
                    }
                )
                summary_doc = finalize_multi_feed_batch(
                    parent,
                    template,
                    batch,
                    incident_log_path=incident_log_default,
                    incident_log_start_offset=incident_log_start,
                )
        except RuntimeError as exc:
            return ServiceResult(
                episodes_processed=0,
                summary=str(exc),
                success=False,
                error=str(exc),
                multi_feed_summary=None,
            )
    finally:
        workflow.end_multi_feed_ml_batch()

    if err_parts:
        summary_text = "\n".join(ok_parts + ["", "Failures:", *err_parts])
        joined = "; ".join(err_parts)
        soft_only = bool(
            (not cfg.multi_feed_strict)
            and batch
            and all((fr.ok or fr.failure_kind == "soft") for fr in batch)
        )
        if soft_only:
            logger.warning(
                "Multi-feed: %d feed(s) failed with soft-classified errors; "
                "treating run as success (multi_feed_strict=False).",
                sum(1 for fr in batch if not fr.ok),
            )
            return ServiceResult(
                episodes_processed=total,
                summary=summary_text,
                success=True,
                error=None,
                multi_feed_summary=summary_doc,
                soft_failures=joined,
            )
        return ServiceResult(
            episodes_processed=total,
            summary=summary_text,
            success=False,
            error=joined,
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

        # Single-feed path: optionally wrap output in feeds/<slug>/ so the
        # output layout matches the multi-feed corpus shape (#644). Opt-in via
        # Config.single_feed_uses_corpus_layout (default False keeps backwards
        # compatibility with legacy single-feed corpora written as
        # <output_dir>/run_<id>/...).
        #
        # Explicit ``is True`` check so that test doubles (MagicMock) whose
        # attribute access returns a truthy Mock don't accidentally trip the
        # wrapping path.
        effective_cfg = cfg
        use_corpus_layout = getattr(cfg, "single_feed_uses_corpus_layout", False) is True
        if (
            use_corpus_layout
            and isinstance(getattr(cfg, "rss_url", None), str)
            and cfg.rss_url
            and isinstance(getattr(cfg, "output_dir", None), str)
            and cfg.output_dir
        ):
            feed_dir = filesystem.corpus_feed_output_dir(cfg.output_dir, cfg.rss_url)
            effective_cfg = cfg.model_copy(update={"output_dir": feed_dir})
            logger.info(
                "Single-feed run using corpus layout: %s -> %s",
                cfg.output_dir,
                feed_dir,
            )

        count, summary = workflow.run_pipeline(effective_cfg)

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
