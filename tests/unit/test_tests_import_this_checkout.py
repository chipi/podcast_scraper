"""The test suite must exercise the code of the checkout it lives in, not whatever the venv's
editable install points at (a git worktree shares the main checkout's venv)."""

from __future__ import annotations

from pathlib import Path

import pytest

import podcast_scraper

pytestmark = pytest.mark.unit


def test_podcast_scraper_is_imported_from_this_checkout() -> None:
    src = (Path(__file__).resolve().parents[2] / "src").resolve()
    assert Path(podcast_scraper.__file__).resolve().is_relative_to(src)
