"""Tests for lazy ``create_app`` export on ``podcast_scraper.server``."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")


@pytest.mark.unit
def test_server_lazy_create_app() -> None:
    mod = importlib.import_module("podcast_scraper.server")
    fn = getattr(mod, "create_app")
    assert callable(fn)
    assert fn.__module__ == "podcast_scraper.server.app"


@pytest.mark.unit
def test_server_getattr_unknown_raises() -> None:
    mod = importlib.import_module("podcast_scraper.server")
    with pytest.raises(AttributeError, match="nosuch"):
        _ = mod.nosuch  # type: ignore[attr-defined]
