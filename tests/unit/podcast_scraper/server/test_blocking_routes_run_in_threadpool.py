"""Routes whose work is synchronous file IO must be plain ``def`` so FastAPI runs them in its
threadpool. As ``async def`` with no ``await`` they run ON the event loop and stall every other
request for their duration (prod, 2026-10-03: the corpus digest's cold build froze the api;
/api/usage held /api/health at 1.08 s instead of 0.008 s).
"""

from __future__ import annotations

import inspect

import pytest

from podcast_scraper.server.routes.corpus_digest import corpus_digest
from podcast_scraper.server.routes.usage import usage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("handler", [corpus_digest, usage], ids=["corpus_digest", "usage"])
def test_a_blocking_route_is_not_a_coroutine(handler) -> None:
    assert not inspect.iscoroutinefunction(handler)
