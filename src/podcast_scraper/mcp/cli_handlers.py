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
            "Run the generic MCP server (PRD-034 / RFC-095) over a corpus, stdio transport. "
            "Requires the [mcp] extra (plus [search] + [dev] for the retrieval tools)."
        ),
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Corpus output directory (the read context; metadata with .gi.json / .kg.json).",
    )
    args = parser.parse_args(list(argv))
    args.command = "mcp"
    return args


def run_mcp(args: Namespace, log: logging.Logger) -> int:
    """Run the stdio MCP server for ``args.corpus``."""
    from .server import run_stdio

    log.info("Starting MCP server (stdio) for corpus: %s", args.corpus)
    run_stdio(args.corpus)
    return 0
