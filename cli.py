"""Command-line interface helpers for podcast_scraper."""

from __future__ import annotations

import argparse
import logging
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import urlparse

from pydantic import ValidationError

from . import config, filesystem, progress, workflow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tqdm

__version__ = "2.0.0"

_LOGGER = logging.getLogger(__name__)


class _TqdmProgress:
    """Simple adapter that exposes tqdm's update interface."""

    def __init__(self, bar: "tqdm.tqdm") -> None:
        self._bar = bar

    def update(self, advance: int) -> None:
        self._bar.update(advance)


@contextmanager
def _tqdm_progress(total: Optional[int], description: str) -> Iterator[_TqdmProgress]:
    """Create a tqdm progress context matching the shared progress API."""
    from tqdm import tqdm

    kwargs: Dict[str, Any] = {"desc": description}
    if total is None:
        kwargs.update(
            total=None,
            unit="",
            leave=False,
            miniters=1,
            mininterval=0.5,
            bar_format="{desc}: {elapsed}",
            ncols=80,
            dynamic_ncols=False,
        )
    else:
        kwargs.update(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        )

    with tqdm(**kwargs) as bar:
        yield _TqdmProgress(bar)


def validate_args(args: argparse.Namespace) -> None:  # noqa: C901 - CLI validation is consolidated
    """Validate parsed CLI arguments and raise ValueError when invalid."""
    errors: List[str] = []

    rss_value = (args.rss or "").strip()
    if not rss_value:
        errors.append("RSS URL is required")
    else:
        parsed_obj = urlparse(rss_value)
        if parsed_obj.scheme not in ("http", "https"):
            errors.append(f"RSS URL must be http or https: {rss_value}")
        if not parsed_obj.netloc:
            errors.append(f"RSS URL must have a valid hostname: {rss_value}")

    if args.max_episodes is not None and args.max_episodes <= 0:
        errors.append(f"--max-episodes must be positive, got: {args.max_episodes}")

    if args.timeout <= 0:
        errors.append(f"--timeout must be positive, got: {args.timeout}")

    if args.delay_ms < 0:
        errors.append(f"--delay-ms must be non-negative, got: {args.delay_ms}")

    valid_models = (
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "large.en",
    )
    if args.transcribe_missing and args.whisper_model not in valid_models:
        errors.append(f"--whisper-model must be one of {valid_models}, got: {args.whisper_model}")

    if args.screenplay and args.num_speakers < config.MIN_NUM_SPEAKERS:
        errors.append(
            f"--num-speakers must be at least {config.MIN_NUM_SPEAKERS}, got: {args.num_speakers}"
        )

    if args.speaker_names:
        names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
        if len(names) < config.MIN_NUM_SPEAKERS:
            errors.append("At least two speaker names required when specifying --speaker-names")

    if args.workers < 1:
        errors.append("--workers must be at least 1")

    if args.output_dir:
        try:
            filesystem.validate_and_normalize_output_dir(args.output_dir)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError("Invalid input parameters:\n  " + "\n  ".join(errors))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments, optionally merging configuration file defaults."""
    parser = argparse.ArgumentParser(
        description="Download podcast episode transcripts from an RSS feed."
    )
    parser.add_argument("--config", default=None, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("rss", nargs="?", default=None, help="Podcast RSS feed URL")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: output_rss_<host>_<hash>)"
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Maximum number of episodes to process"
    )
    parser.add_argument(
        "--prefer-type",
        action="append",
        default=[],
        help="Preferred transcript types or extensions (repeatable)",
    )
    parser.add_argument("--user-agent", default=config.DEFAULT_USER_AGENT, help="User-Agent header")
    parser.add_argument(
        "--timeout",
        type=int,
        default=config.DEFAULT_TIMEOUT_SECONDS,
        help="Request timeout in seconds",
    )
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between requests (ms)")
    parser.add_argument(
        "--transcribe-missing",
        action="store_true",
        help="Use Whisper when no transcript is provided",
    )
    parser.add_argument("--whisper-model", default="base", help="Whisper model to use")
    parser.add_argument(
        "--screenplay", action="store_true", help="Format Whisper transcript as screenplay"
    )
    parser.add_argument(
        "--screenplay-gap",
        type=float,
        default=config.DEFAULT_SCREENPLAY_GAP_SECONDS,
        help="Gap (seconds) to trigger speaker change",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=config.DEFAULT_NUM_SPEAKERS,
        help="Number of speakers to alternate between",
    )
    parser.add_argument(
        "--speaker-names",
        default="",
        help="Comma-separated speaker names to use instead of SPEAKER 1..N",
    )
    parser.add_argument(
        "--run-id", default=None, help="Optional run identifier; use 'auto' for timestamp"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip episodes whose output already exists"
    )
    parser.add_argument(
        "--clean-output", action="store_true", help="Remove the output directory before processing"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show planned work without saving files"
    )
    parser.add_argument("--version", action="store_true", help="Show program version and exit")
    parser.add_argument(
        "--log-level",
        default=config.DEFAULT_LOG_LEVEL,
        type=str.upper,
        help="Logging level (e.g., DEBUG, INFO)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config.DEFAULT_WORKERS,
        help="Number of concurrent download workers",
    )

    initial_args, _ = parser.parse_known_args(argv)

    if initial_args.version:
        print(f"podcast_scraper {__version__}")
        raise SystemExit(0)

    if initial_args.config:
        config_data = config.load_config_file(initial_args.config)
        valid_dests = {action.dest for action in parser._actions if action.dest}
        unknown_keys = [key for key in config_data.keys() if key not in valid_dests]
        if unknown_keys:
            raise ValueError("Unknown config option(s): " + ", ".join(sorted(unknown_keys)))

        try:
            config_model = config.Config.model_validate(config_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc

        defaults_updates: Dict[str, Any] = config_model.model_dump(
            exclude_none=True,
            by_alias=True,
        )

        speaker_list = defaults_updates.get("speaker_names")
        if isinstance(speaker_list, list):
            defaults_updates["speaker_names"] = ",".join(speaker_list)

        parser.set_defaults(**defaults_updates)
        args = parser.parse_args(argv)
        if not args.rss:
            raise ValueError("RSS URL is required (provide in config as 'rss' or via CLI)")
    else:
        args = parser.parse_args(argv)

    validate_args(args)
    return args


def _build_config(args: argparse.Namespace) -> config.Config:
    """Materialize a Config object from already-validated CLI arguments."""
    speaker_names_list = [s.strip() for s in (args.speaker_names or "").split(",") if s.strip()]
    payload: Dict[str, Any] = {
        "rss_url": args.rss,
        "output_dir": filesystem.derive_output_dir(args.rss, args.output_dir),
        "max_episodes": args.max_episodes,
        "user_agent": args.user_agent,
        "timeout": args.timeout,
        "delay_ms": args.delay_ms,
        "prefer_types": args.prefer_type,
        "transcribe_missing": args.transcribe_missing,
        "whisper_model": args.whisper_model,
        "screenplay": args.screenplay,
        "screenplay_gap_s": args.screenplay_gap,
        "screenplay_num_speakers": args.num_speakers,
        "screenplay_speaker_names": speaker_names_list,
        "run_id": args.run_id,
        "log_level": args.log_level,
        "workers": args.workers,
        "skip_existing": args.skip_existing,
        "clean_output": args.clean_output,
        "dry_run": args.dry_run,
    }
    return config.Config.model_validate(payload)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    apply_log_level_fn: Optional[Callable[[str], None]] = None,
    run_pipeline_fn: Optional[Callable[[config.Config], Tuple[int, str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Entry point for the CLI; returns an exit status code."""
    progress.set_progress_factory(_tqdm_progress)
    log = logger or _LOGGER
    if apply_log_level_fn is None:
        apply_log_level_fn = workflow.apply_log_level
    if run_pipeline_fn is None:
        run_pipeline_fn = workflow.run_pipeline

    try:
        args = parse_args(argv)
    except ValueError as exc:
        log.error(f"Error: {exc}")
        return 1

    try:
        cfg = _build_config(args)
    except ValidationError as exc:
        log.error(f"Invalid configuration: {exc}")
        return 1

    apply_log_level_fn(cfg.log_level)

    log.info("Starting podcast transcript scrape")
    log.info(f"  rss: {cfg.rss_url}")
    log.info(f"  output_dir: {cfg.output_dir}")
    log.info(f"  max_episodes: {cfg.max_episodes or 'all'}")
    log.info(f"  log_level: {cfg.log_level}")
    log.info(f"  workers: {cfg.workers}")
    log.info(f"  skip_existing: {cfg.skip_existing}")
    log.info(f"  clean_output: {cfg.clean_output}")
    log.info(f"  dry_run: {cfg.dry_run}")

    try:
        _, summary = run_pipeline_fn(cfg)
    except Exception as exc:  # pragma: no cover - defensive
        log.error(f"Unexpected failure: {exc}")
        return 1

    log.info(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
