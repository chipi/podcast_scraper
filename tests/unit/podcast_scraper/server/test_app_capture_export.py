"""Unit tests for the Markdown highlight export renderer (#1115, pure / no IO)."""

from __future__ import annotations

from podcast_scraper.server.app_capture_export import (
    _timecode,
    EpisodeHighlights,
    HighlightLine,
    render_highlights_markdown,
)


def test_timecode_formats() -> None:
    assert _timecode(None) == ""
    assert _timecode(0) == "0:00"
    assert _timecode(6_000) == "0:06"
    assert _timecode(90_000) == "1:30"
    assert _timecode(3_661_000) == "1:01:01"


def test_empty_export_is_a_friendly_placeholder() -> None:
    md = render_highlights_markdown([])
    assert "# My Highlights" in md
    assert "_No highlights captured yet._" in md


def test_span_highlight_renders_quote_speaker_color_and_timecode() -> None:
    md = render_highlights_markdown(
        [
            EpisodeHighlights(
                slug="show-ep01",
                title="How Sleep Works",
                show="Long Horizon Notes",
                highlights=[
                    HighlightLine(
                        kind="span",
                        start_ms=90_000,
                        quote_text="deep sleep consolidates memory",
                        speaker="Guest",
                        color="amber",
                    )
                ],
            )
        ]
    )
    assert "## How Sleep Works — Long Horizon Notes" in md
    assert "<!-- show-ep01 -->" in md
    assert '- [1:30] "deep sleep consolidates memory" — Guest _amber_' in md


def test_moment_and_insight_and_notes_and_drift() -> None:
    md = render_highlights_markdown(
        [
            EpisodeHighlights(
                slug="show-ep02",
                highlights=[
                    HighlightLine(kind="moment", start_ms=6_000, notes=["circle back to this"]),
                    HighlightLine(
                        kind="insight",
                        quote_text="A grounded claim",
                        anchor_status="drifted",
                    ),
                ],
            )
        ]
    )
    # falls back to the slug as the heading when no title
    assert "## show-ep02" in md
    assert "- [0:06] Marked moment" in md
    assert "  - _note:_ circle back to this" in md
    assert "A grounded claim" in md
    assert "⚠ anchor drifted" in md


def test_span_without_quote_degrades_cleanly() -> None:
    md = render_highlights_markdown(
        [EpisodeHighlights(slug="s", highlights=[HighlightLine(kind="span")])]
    )
    assert "- Highlighted span" in md
    assert md.endswith("\n")
