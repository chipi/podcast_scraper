"""CLI for ``podcast serve``."""

from __future__ import annotations

import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, cast, Sequence


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _merged_bool_flag(cli: bool, env_name: str) -> bool:
    return bool(cli) or _env_truthy(env_name)


def _sync_reload_environ(args: Namespace, output_dir: Path) -> None:
    """Persist ``podcast serve`` flags for uvicorn --reload and reload-time app kwargs."""
    os.environ["PODCAST_SERVE_OUTPUT_DIR"] = str(output_dir)
    feeds_on = _merged_bool_flag(
        getattr(args, "enable_feeds_api", False), "PODCAST_SERVE_ENABLE_FEEDS_API"
    )
    op_on = _merged_bool_flag(
        getattr(args, "enable_operator_config_api", False),
        "PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API",
    )
    jobs_on = _merged_bool_flag(
        getattr(args, "enable_jobs_api", False), "PODCAST_SERVE_ENABLE_JOBS_API"
    )
    if feeds_on:
        os.environ["PODCAST_SERVE_ENABLE_FEEDS_API"] = "1"
    else:
        os.environ.pop("PODCAST_SERVE_ENABLE_FEEDS_API", None)
    if op_on:
        os.environ["PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API"] = "1"
    else:
        os.environ.pop("PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API", None)
    if jobs_on:
        os.environ["PODCAST_SERVE_ENABLE_JOBS_API"] = "1"
    else:
        os.environ.pop("PODCAST_SERVE_ENABLE_JOBS_API", None)
    if getattr(args, "config_file", None):
        os.environ["PODCAST_SERVE_CONFIG_FILE"] = str(Path(args.config_file).expanduser().resolve())


def _load_create_app() -> Callable[..., Any]:
    """Return the app factory lazily (unit tests patch this; avoids importing FastAPI early)."""
    from podcast_scraper.server.app import create_app

    return create_app


def parse_serve_argv(argv: Sequence[str]) -> Namespace:
    """Parse arguments after the ``serve`` token."""
    parser = argparse.ArgumentParser(
        prog="podcast serve",
        description="Run the GI/KG viewer HTTP API (install .[server] if needed).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Corpus output directory (metadata with .gi.json / .kg.json).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Dev only: restart workers when Python files change.",
    )
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Do not mount built SPA assets from web/gi-kg-viewer/dist even if present.",
    )
    parser.add_argument(
        "--enable-feeds-api",
        action="store_true",
        help="Expose GET/PUT /api/feeds (feeds.spec.yaml under corpus root).",
    )
    parser.add_argument(
        "--enable-operator-config-api",
        action="store_true",
        help="Expose GET/PUT /api/operator-config (non-secret YAML only).",
    )
    parser.add_argument(
        "--enable-jobs-api",
        action="store_true",
        help="Expose POST/GET /api/jobs pipeline subprocess jobs (RFC-077 Phase 2).",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        metavar="PATH",
        help=(
            "Operator YAML path when operator config API is enabled "
            "(default: <output-dir>/viewer_operator.yaml)."
        ),
    )
    args = parser.parse_args(list(argv))
    args.command = "serve"
    return args


def run_serve(args: Namespace, log: logging.Logger) -> int:
    """Start uvicorn with :func:`podcast_scraper.server.app.create_app`."""
    try:
        import uvicorn
    except ImportError:
        log.error(
            "Missing server dependencies. Install with: pip install -e '.[server]'",
        )
        return 1

    out = Path(args.output_dir).expanduser().resolve()
    if not out.is_dir():
        log.error("Output directory does not exist or is not a directory: %s", out)
        return 2

    _sync_reload_environ(args, out)
    static_kw: bool | None = False if getattr(args, "no_static", False) else None

    if args.reload:
        uvicorn.run(
            "podcast_scraper.server.app:create_app_for_uvicorn",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
        )
        return 0

    create_app = _load_create_app()
    from podcast_scraper.server.app import serve_feature_kwargs_from_environ

    kw = serve_feature_kwargs_from_environ()
    app = create_app(
        out,
        static_dir=static_kw,
        enable_feeds_api=bool(kw["enable_feeds_api"]),
        enable_operator_config_api=bool(kw["enable_operator_config_api"]),
        enable_jobs_api=bool(kw["enable_jobs_api"]),
        operator_config_file=cast(str | os.PathLike[str] | None, kw["operator_config_file"]),
    )
    # _load_create_app is typed as Callable[..., Any]; uvicorn expects an ASGI callable.
    uvicorn.run(cast(Any, app), host=args.host, port=args.port, reload=False)
    return 0
