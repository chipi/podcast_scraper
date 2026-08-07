"""CLI glue for ``podcast mcp`` (RFC-095 slice 1)."""

from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from typing import Sequence


def parse_mcp_argv(argv: Sequence[str]) -> Namespace:
    """Parse arguments after the ``mcp`` token."""
    parser = argparse.ArgumentParser(
        prog="podcast mcp",
        description=(
            "Run the generic MCP server (PRD-034 / RFC-095) over a corpus. Transports: stdio "
            "(local) or http (Streamable HTTP, auth-gated per RFC-112). "
            "Install with .[dev,search] (the MCP SDK ships in [dev])."
        ),
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Corpus output directory (the read context; metadata with .gi.json / .kg.json).",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio (local, no auth) or http (remote Streamable HTTP, bearer-token auth).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host (transport=http).")
    parser.add_argument("--port", type=int, default=8009, help="HTTP bind port (transport=http).")
    args = parser.parse_args(list(argv))
    args.command = "mcp"
    return args


def run_mcp(args: Namespace, log: logging.Logger) -> int:
    """Run the MCP server for ``args.corpus`` over the chosen transport."""
    from .server import run_server

    transport = getattr(args, "transport", "stdio")
    if transport == "http":
        log.info(
            "Starting MCP server (http) for corpus %s on %s:%s",
            args.corpus,
            args.host,
            args.port,
        )
    else:
        log.info("Starting MCP server (stdio) for corpus: %s", args.corpus)
    run_server(
        args.corpus,
        transport=transport,
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8009),
    )
    return 0
