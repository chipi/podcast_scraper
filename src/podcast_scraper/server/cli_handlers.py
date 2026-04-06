"""CLI for ``podcast serve`` (RFC-062)."""

from __future__ import annotations

import argparse
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Sequence


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

    os.environ["PODCAST_SERVE_OUTPUT_DIR"] = str(out)
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

    from podcast_scraper.server.app import create_app

    app = create_app(out, static_dir=static_kw)
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
    return 0
