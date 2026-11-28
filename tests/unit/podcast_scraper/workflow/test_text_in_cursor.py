"""Where a text source's cursor goes after a run (``text_in._next_cursor``) and the cap on what a
run fetches (``text_in._jobs_for_items``): an item that did not write is never passed, and an item
already written never counts toward the cap (2026-11-28: 67 news articles whose summary failed
were passed by their sources' cursors and never listed again)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.workflow import text_in

pytestmark = [pytest.mark.unit]

TEXT = ct.ContentType(
    name="test_text_item",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="test_text_item", stages=frozenset({ct.SUMMARY})),
    default_transport="fixture_feed",
    source_kind="test",
    card_kind="article",
)


def _item(name: str, cursor: Optional[str]) -> ct.SourceItem:
    return ct.SourceItem(item_id=name, text=f"{name}'s body.", cursor=cursor)


def _iso(day: int) -> ct.SourceItem:
    return _item(f"d{day}", f"2026-11-{day:02d}T00:00:00Z")


def _as_is(value: str) -> str:
    return value


def test_with_every_item_written_the_cursor_goes_where_the_transport_put_it() -> None:
    items = [_iso(3), _iso(2), _iso(1)]
    got = text_in._next_cursor("2026-10-01T00:00:00Z", "2026-11-03T00:00:00Z", items, [], _as_is)
    assert got == "2026-11-03T00:00:00Z"


def test_a_failed_item_in_the_middle_holds_the_cursor_at_the_newest_older_one() -> None:
    items = [_iso(4), _iso(3), _iso(2), _iso(1)]
    got = text_in._next_cursor(None, "2026-11-04T00:00:00Z", items, [items[2]], _as_is)
    assert got == "2026-11-01T00:00:00Z"


def test_with_no_older_item_written_the_cursor_stays_where_it_was() -> None:
    items = [_iso(2), _iso(1)]
    old = "2026-10-01T00:00:00Z"
    assert text_in._next_cursor(old, "2026-11-02T00:00:00Z", items, list(items), _as_is) == old
    assert text_in._next_cursor(None, "2026-11-02T00:00:00Z", items, [items[1]], _as_is) is None


def test_an_item_sharing_the_failed_ones_time_does_not_move_the_cursor_to_it() -> None:
    """The transports skip what is at or before the cursor, so the cursor stops strictly below."""
    failed, twin, older = _iso(2), _item("twin", "2026-11-02T00:00:00Z"), _iso(1)
    items = [twin, failed, older]
    got = text_in._next_cursor(None, "2026-11-02T00:00:00Z", items, [failed], _as_is)
    assert got == "2026-11-01T00:00:00Z"


def test_an_undated_item_never_holds_the_cursor() -> None:
    undated = _item("undated", None)
    items = [_iso(2), undated]
    got = text_in._next_cursor(None, "2026-11-02T00:00:00Z", items, [undated], _as_is)
    assert got == "2026-11-02T00:00:00Z"


def test_cursor_values_compare_by_the_transports_key_not_as_text() -> None:
    """hn's epoch seconds: "9" is older than "10", though it sorts after it as text."""
    items = [_item("ten", "10"), _item("nine", "9"), _item("eight", "8")]
    got = text_in._next_cursor(None, "10", items, [items[0]], int)
    assert got == "9"
    # As text "10" sorts first, so nothing looks older and the cursor would never move.
    assert text_in._next_cursor(None, "10", items, [items[0]], _as_is) is None


def _jobs(
    tmp_path: Path, items: List[ct.SourceItem], cap: Optional[int], written: List[str]
) -> tuple:
    ids = {ct.item_id(TEXT, i): [tmp_path / f"{i.item_id}.metadata.json"] for i in items}
    on_disk = {k: v for k, v in ids.items() if k.rsplit(":", 1)[-1] in written}
    return text_in._jobs_for_items(
        items,
        TEXT,
        str(tmp_path),
        None,
        source_id="s",
        transport="fixture_feed",
        written=on_disk,
        cap=cap,
    )


def test_the_cap_counts_only_items_not_yet_written(tmp_path: Path) -> None:
    items = [_iso(4), _iso(3), _iso(2), _iso(1)]
    _episodes, jobs, waiting = _jobs(tmp_path, items, cap=1, written=["d4"])
    assert [j.item.item_id for j in jobs] == [ct.item_id(TEXT, items[1])]
    assert [(listed.item_id, minted) for listed, minted in waiting] == [
        ("d3", ct.item_id(TEXT, items[1])),
        ("d2", None),
        ("d1", None),
    ]


def test_without_a_cap_every_new_item_is_a_job_and_waits_on_its_write(tmp_path: Path) -> None:
    items = [_iso(2), _iso(1)]
    _episodes, jobs, waiting = _jobs(tmp_path, items, cap=None, written=[])
    assert len(jobs) == 2
    assert [minted for _listed, minted in waiting] == [ct.item_id(TEXT, i) for i in items]
