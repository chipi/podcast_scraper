"""The pure parts of a board's Brief and of board search (RFC-119 amendment 2026-10-12)."""

from __future__ import annotations

from podcast_scraper.server.app_board_view import (
    _contains,
    _holds,
    Board,
    board_episode_slugs,
    query_words,
)


def _board(items: list[dict], highlights: dict[str, dict] | None = None) -> Board:
    return Board(meta={"id": "col_a", "name": "A"}, items=items, highlights=highlights or {})


def test_the_boards_episodes_are_its_saved_ones_and_its_moments_in_board_order_once() -> None:
    board = _board(
        [
            {"kind": "highlight", "ref": "h1"},
            {"kind": "episode", "ref": "ep-b"},
            {"kind": "show", "ref": "feed-1"},  # a show is not expanded into its episodes
            {"kind": "episode", "ref": "ep-a"},
            {"kind": "highlight", "ref": "h2"},  # same episode as a saved one: listed once
            {"kind": "topic", "ref": "topic:x"},
        ],
        {"h1": {"episode_slug": "ep-c"}, "h2": {"episode_slug": "ep-a"}},
    )
    assert board_episode_slugs(board) == ["ep-c", "ep-b", "ep-a"]


def test_what_a_board_holds_counts_subjects_together() -> None:
    board = _board(
        [
            {"kind": "topic", "ref": "topic:x"},
            {"kind": "person", "ref": "person:y"},
            {"kind": "storyline", "ref": "topic:z"},
            {"kind": "theme", "ref": "tc:w"},
            {"kind": "link", "ref": "https://x"},
        ]
    )
    holds = _holds(board, episodes=0)
    assert (holds.subjects, holds.links, holds.episodes, holds.moments) == (4, 1, 0, 0)


def test_a_query_matches_when_every_word_is_there_in_any_order() -> None:
    words = query_words("Gauges, LIE!")
    assert words == ["gauges", "lie"]
    assert _contains("Pressure gauges never lie", words)
    assert not _contains("Pressure gauges", words)
    assert not _contains("anything", query_words("  ,  "))
