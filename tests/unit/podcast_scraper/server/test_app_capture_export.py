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
    assert '- **Quote** [1:30] "deep sleep consolidates memory" — Guest _amber_' in md


def test_moment_and_insight_and_notes() -> None:
    md = render_highlights_markdown(
        [
            EpisodeHighlights(
                slug="show-ep02",
                highlights=[
                    HighlightLine(kind="moment", start_ms=6_000, notes=["circle back to this"]),
                    HighlightLine(kind="insight", quote_text="A grounded claim"),
                ],
            )
        ]
    )
    # falls back to the slug as the heading when no title
    assert "## show-ep02" in md
    # No placeholder body: the kind label already leads the line, so the old "Marked moment"
    # fallback just said it twice.
    assert "- **Marked moment** [0:06] —" in md or "- **Marked moment** [0:06]" in md
    assert "[0:06] Marked moment" not in md
    assert "  - _note:_ circle back to this" in md
    assert "A grounded claim" in md
    # Drift deliberately does NOT travel (operator 2026-09-18): it exists because the app can JUMP
    # to a timestamp, and an export is a record of what was said, not a playback cursor.
    assert "drift" not in md.lower()


def test_span_without_quote_degrades_cleanly() -> None:
    md = render_highlights_markdown(
        [EpisodeHighlights(slug="s", highlights=[HighlightLine(kind="span")])]
    )
    assert md.strip().endswith("- **Quote**")
    assert md.endswith("\n")


def test_a_note_on_the_episode_itself_is_exported() -> None:
    """The export matched notes ONLY by highlight id, so episode notes were silently absent.

    The endpoint described itself as exporting highlights "with attached notes" while dropping a
    whole class of the user's writing.
    """
    md = render_highlights_markdown(
        [
            EpisodeHighlights(
                slug="ep-1",
                title="An Episode",
                highlights=[HighlightLine(kind="moment", start_ms=1000)],
                episode_notes=["a thought about the whole episode"],
            )
        ]
    )
    assert "a thought about the whole episode" in md
    assert "## An Episode" in md


def test_notes_with_nowhere_tidy_to_go_get_a_section_rather_than_the_bin() -> None:
    """A note on a saved insight has no episode heading to sit under — it must still be exported."""
    md = render_highlights_markdown([], ["a note on a saved insight"])
    assert "## Other notes" in md
    assert "a note on a saved insight" in md
    assert "_No highlights captured yet._" not in md, "there IS something to show"


def test_genuinely_empty_still_says_so() -> None:
    assert "_No highlights captured yet._" in render_highlights_markdown([], [])


# --- what an export carries is CONTENT, not app state (operator 2026-09-18) --------------------


def test_capture_date_kind_and_entities_travel() -> None:
    """The three fields added after auditing the export against the app.

    `kind` because a quotation, a bookmarked instant and a saved insight rendered identically;
    `created_at` because a year of captures otherwise has no chronology once it leaves the app
    (`start_ms` is a position INSIDE an episode, a different question); `entities` because the
    Obsidian vault was built around them and this format dropped them entirely.
    """
    md = render_highlights_markdown(
        [
            EpisodeHighlights(
                slug="s",
                highlights=[
                    HighlightLine(
                        kind="moment",
                        quote_text="a line",
                        created_at=1_757_000_000,  # 2025-09-04 UTC
                        entities=["Nora", "personal finance"],
                        notes=["my own words"],
                    )
                ],
            )
        ]
    )
    assert "**Marked moment**" in md
    assert "· captured 2025-09-04" in md
    assert "  - _about:_ Nora · personal finance" in md
    assert "  - _note:_ my own words" in md


def test_an_unusable_created_at_is_omitted_rather_than_fatal() -> None:
    """`created_at` comes off a per-user JSON file, so it can be absent or the wrong type."""
    # Deliberately the WRONG types: the annotation says `int | None`, but the value is read back
    # from a per-user JSON file that is hand-editable and may have been written by an older build,
    # so the runtime sees what it sees. The ignore marks the lie as intentional.
    for bad in (None, "yesterday", float("nan")):
        md = render_highlights_markdown(
            [
                EpisodeHighlights(
                    slug="s",
                    highlights=[
                        HighlightLine(kind="span", created_at=bad),  # type: ignore[arg-type]
                    ],
                )
            ]
        )
        assert "captured" not in md


def test_app_scheduling_state_is_not_exportable_at_all() -> None:
    """`retired` / resurfacing counts have no field here, by design and not by omission.

    They describe how this product nags you, which means nothing in a document read years later in
    another tool. Pinned so a future "export everything" pass has to argue with this rather than
    quietly reverse it.
    """
    assert not hasattr(HighlightLine("span"), "retired")
    assert not hasattr(HighlightLine("span"), "anchor_status")
