"""Generic MCP server (PRD-034 / RFC-095).

Exposes the platform's read capabilities — entity resolution, hybrid search, the RFC-094
relational layer, CIL, catalog — as composable, read-only MCP tools for agentic clients.
Opt-in (the ``[mcp]`` extra); the corpus directory is the read context. The MCP SDK is
imported lazily in :mod:`server` so this package imports without it.
"""
