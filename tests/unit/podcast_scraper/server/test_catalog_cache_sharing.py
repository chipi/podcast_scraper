"""The two cached catalogs share equal rows instead of holding the same text twice, and serve the
same rows as an uncached build."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import catalog_cache
from podcast_scraper.server.corpus_catalog import build_catalog_rows, build_catalog_rows_cumulative

pytestmark = [pytest.mark.unit]

CORPUS = Path(__file__).resolve().parents[4] / "tests/fixtures/viewer-validation-corpus/v3"


@pytest.fixture(autouse=True)
def _empty_caches() -> None:
    perf_cache.clear()
    yield
    perf_cache.clear()


@pytest.mark.parametrize("first", ["cumulative", "last_run"])
def test_both_orders_share_rows_and_serve_what_a_fresh_build_serves(first: str) -> None:
    if first == "cumulative":
        cum = catalog_cache.cached_catalog(CORPUS)
        last = catalog_cache.cached_catalog_last_run(CORPUS)
    else:
        last = catalog_cache.cached_catalog_last_run(CORPUS)
        cum = catalog_cache.cached_catalog(CORPUS)
    assert cum == build_catalog_rows_cumulative(CORPUS)
    assert last == build_catalog_rows(CORPUS)
    ids = {id(r) for r in cum}
    shared = [r for r in last if id(r) in ids]
    assert last and len(shared) == len(last)


def test_the_last_run_catalog_alone_builds_no_cumulative_one() -> None:
    catalog_cache.cached_catalog_last_run(CORPUS)
    assert perf_cache.stats().get(catalog_cache._CATALOG_NS, {}).get("entries", 0) == 0


def test_a_row_whose_file_differs_keeps_its_own() -> None:
    rows = build_catalog_rows(CORPUS)
    changed = [rows[0].__class__(**{**rows[0].__dict__, "episode_title": "different"}), *rows[1:]]
    out = catalog_cache._sharing_rows(rows, changed)
    assert out[0] is rows[0]
    assert all(a is b for a, b in zip(out[1:], changed[1:]))
