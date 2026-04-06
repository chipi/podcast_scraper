"""HTTP API for the GI/KG viewer (RFC-062).

Install optional dependencies: ``pip install -e '.[server]'``.
"""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from podcast_scraper.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
