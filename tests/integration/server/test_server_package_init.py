"""Integration tests for lazy ``create_app`` export on ``podcast_scraper.server``.

Requires ``fastapi`` (``pip install -e '.[server]'``).
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_server_lazy_create_app() -> None:
    mod = importlib.import_module("podcast_scraper.server")
    fn = getattr(mod, "create_app")
    assert callable(fn)
    assert fn.__module__ == "podcast_scraper.server.app"


def test_server_lazy_app_submodule() -> None:
    """``getattr(server, 'app')`` must work for ``patch('podcast_scraper.server.app.*')``."""
    mod = importlib.import_module("podcast_scraper.server")
    app_mod = getattr(mod, "app")
    assert app_mod.__name__ == "podcast_scraper.server.app"
    assert callable(getattr(app_mod, "create_app"))


def test_server_getattr_unknown_raises() -> None:
    mod = importlib.import_module("podcast_scraper.server")
    with pytest.raises(AttributeError, match="nosuch"):
        _ = mod.nosuch  # type: ignore[attr-defined]
