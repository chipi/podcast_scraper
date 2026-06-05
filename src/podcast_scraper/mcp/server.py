"""FastMCP server construction (RFC-095 slice 1).

Registers the plain ``tools/`` functions as MCP tools and runs over stdio. The MCP SDK is
imported inside :func:`build_server` so the rest of the package (and its tests) import
without the ``[mcp]`` extra installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .context import CorpusContext
from .tools.resolve import resolve_entity as _resolve_entity
from .tools.search import search_corpus as _search_corpus


def build_server(corpus_dir: Path | str) -> Any:
    """Build a FastMCP server bound to *corpus_dir* with the slice-1 tools registered."""
    from mcp.server.fastmcp import FastMCP

    ctx = CorpusContext.from_path(corpus_dir)
    server = FastMCP("podcast-scraper")

    @server.tool()
    def resolve_entity(name: str, kind: Optional[str] = None) -> dict:
        """Resolve a freeform name to a canonical corpus entity id.

        Use this FIRST when a user names a person, organization, or topic
        ("Sam Altman", "OpenAI", "inflation") — the relational and intelligence tools take
        canonical ids (``person:…`` / ``org:…`` / ``topic:…``), not names. Returns the best
        match with its kind, display name, score, and method (or no candidates if unknown).
        """
        return _resolve_entity(ctx, name, kind)

    @server.tool()
    def search_corpus(
        query: str,
        tier: str = "both",
        grounded_only: bool = False,
        feed: Optional[str] = None,
        since: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """Search the corpus with hybrid two-tier retrieval and get grounded evidence.

        ``tier``: "insight" (synthesized claims), "segment" (raw transcript quotes), or
        "both". ``grounded_only`` keeps only insights backed by a supporting quote. Each
        result carries ``source_tier``, a relevance ``score``, and provenance
        (``metadata`` with episode/feed ids); the response carries the detected
        ``query_type``. For exact quotes use ``tier="segment"``; for positions/claims use
        ``tier="insight"``.
        """
        return _search_corpus(
            ctx,
            query,
            tier=tier,
            grounded_only=grounded_only,
            feed=feed,
            since=since,
            top_k=top_k,
        )

    return server


def run_stdio(corpus_dir: Path | str) -> None:
    """Build and run the MCP server over stdio (the default agent-client transport)."""
    build_server(corpus_dir).run()
