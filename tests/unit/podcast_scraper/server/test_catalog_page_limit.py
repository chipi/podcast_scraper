"""``/api/corpus/episodes`` must honour its documented limit (#1654).

``slice_page`` capped at 200 while the route declared ``le=1000`` and its own docstring said
"limit raised from 200 → 1000". A caller asking for the full operator-facing set in one
request got 200 rows, no error, and nothing indicating truncation had occurred — it looked
like the corpus simply had 200 episodes.

Silently reducing a documented limit is worse than rejecting it: the caller believes it has
everything. This is how the corpus-wide baseline capture had to be written against cursor
paging after ``offset`` and ``limit`` both appeared to work and neither did.
"""

from __future__ import annotations

import pytest

from podcast_scraper.server.corpus_catalog import (
    CatalogEpisodeRow,
    MAX_CATALOG_PAGE_SIZE,
    slice_page,
)

pytestmark = [pytest.mark.unit]


def _rows(n: int) -> list[CatalogEpisodeRow]:
    return [
        CatalogEpisodeRow(
            metadata_relative_path=f"feeds/f/run_a/metadata/{i:04d}.metadata.json",
            feed_id="f",
            feed_title="Feed",
            episode_id=f"ep-{i}",
            episode_title=f"Episode {i}",
            publish_date="2026-08-14",
            summary_title=None,
            summary_bullets=(),
            summary_text=None,
            gi_relative_path="",
            kg_relative_path="",
            bridge_relative_path="",
            has_gi=True,
            has_kg=True,
            has_bridge=False,
        )
        for i in range(n)
    ]


class TestSlicePageHonoursItsLimit:
    def test_a_limit_above_the_old_200_cap_is_honoured(self) -> None:
        """The regression: 678 episodes, limit=1000, and only 200 came back."""
        page, next_cursor = slice_page(_rows(678), 0, 1000)
        assert len(page) == 678
        assert next_cursor is None

    def test_the_documented_maximum_is_reachable(self) -> None:
        page, _ = slice_page(_rows(1500), 0, MAX_CATALOG_PAGE_SIZE)
        assert len(page) == MAX_CATALOG_PAGE_SIZE

    def test_beyond_the_maximum_is_clamped_not_unbounded(self) -> None:
        page, _ = slice_page(_rows(1500), 0, 99_999)
        assert len(page) == MAX_CATALOG_PAGE_SIZE

    def test_small_limits_still_paginate(self) -> None:
        page, next_cursor = slice_page(_rows(10), 0, 3)
        assert len(page) == 3
        assert next_cursor is not None

    def test_next_cursor_is_none_on_the_last_page(self) -> None:
        _, next_cursor = slice_page(_rows(10), 8, 5)
        assert next_cursor is None

    def test_a_zero_or_negative_limit_still_returns_a_row(self) -> None:
        assert len(slice_page(_rows(5), 0, 0)[0]) == 1
        assert len(slice_page(_rows(5), 0, -3)[0]) == 1

    def test_negative_offset_is_treated_as_the_start(self) -> None:
        page, _ = slice_page(_rows(5), -10, 2)
        assert len(page) == 2


def test_the_cap_matches_what_the_route_declares() -> None:
    """Guards the drift that caused this: helper cap vs route ``le=`` diverging silently."""
    import inspect

    from podcast_scraper.server.routes import corpus_library

    source = inspect.getsource(corpus_library.corpus_episodes)
    assert f"le={MAX_CATALOG_PAGE_SIZE}" in source, (
        "the route's declared maximum no longer matches MAX_CATALOG_PAGE_SIZE — one of them "
        "was changed without the other, which is exactly how the 200/1000 mismatch arose"
    )
