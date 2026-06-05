"""Generic MCP server (PRD-034 / RFC-095).

Exposes the platform's read capabilities — entity resolution, hybrid search, the RFC-094
relational layer, CIL, catalog — as composable, read-only MCP tools for agentic clients.
The MCP SDK ships in the ``[dev]`` extra (a core dev/server capability); the corpus
directory is the read context. The SDK is imported lazily in :mod:`server`, so this
package imports without it.
"""
