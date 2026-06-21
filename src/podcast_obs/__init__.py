"""podcast_obs — a light, standalone prod observability control plane (#803).

Intentionally decoupled from the heavy ``podcast_scraper`` package: importing this
package must NOT pull in the pipeline (torch/spacy/providers). It depends only on the
standard library plus ``httpx`` (lazy) and ``PyYAML`` (lazy, config only), so it runs
cheaply anywhere — a plain local process or a small Docker container on the tailnet.

Layers:
- core "basics" — :mod:`podcast_obs.sources` probe/aggregate functions + a plain CLI.
- MCP server (added in a later slice) wraps the same core for agent clients.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .result import err, ok

__all__ = ["__version__", "ok", "err"]
