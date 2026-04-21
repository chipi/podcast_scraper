"""HTTP API for the GI/KG viewer.

Install optional dependencies: ``pip install -e '.[server]'``.
"""

from __future__ import annotations

import importlib

__all__ = ["create_app"]


def __getattr__(name: str):
    if name == "create_app":
        from podcast_scraper.server.app import create_app

        return create_app
    if name == "app":
        # Lazy submodule so ``podcast_scraper.server.app`` matches normal packages
        # (e.g. ``pkgutil.resolve_name`` / ``unittest.mock.patch`` targets).
        return importlib.import_module(f"{__name__}.app")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
