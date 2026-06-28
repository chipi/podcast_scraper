"""Enrichment layer — the 4th artifact tier on top of GIL / KG / bridge.

See RFC-088 (Active during chunk 1 implementation; Epic #1101). The
framework lands in chunk 1; concrete enrichers land in chunks 2–5.

Public re-exports kept minimal to avoid import cycles — consumers
should import from the specific submodule (``enrichment.protocol``,
``enrichment.registry``, ``enrichment.executor``, etc.).
"""

from __future__ import annotations
