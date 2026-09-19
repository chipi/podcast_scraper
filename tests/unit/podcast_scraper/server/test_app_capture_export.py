"""Unit tests for the Markdown highlight export renderer (#1115, pure / no IO)."""

from __future__ import annotations

from podcast_scraper.server.app_capture_export import (
    _timecode,
    EpisodeHighlights,
    format_duration,
    HighlightLine,
    render_highlights_html,
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


# --- printable HTML: the PDF path (operator 2026-09-18) -----------------------------------------


def _doc() -> list[EpisodeHighlights]:
    return [
        EpisodeHighlights(
            slug="s",
            title="How Sleep Works",
            show="Long Horizon Notes",
            url="https://x.test/episode/s",
            publish_date="2024-01-02",
            duration_seconds=416,
            summary_title="A headline",
            summary_text="The prose the Summary button renders.",
            summary_bullets=["first bullet", "second bullet"],
            highlights=[
                HighlightLine(
                    kind="span",
                    start_ms=90_000,
                    quote_text="deep sleep consolidates memory",
                    speaker="Guest",
                    color="amber",
                    created_at=1_757_000_000,
                    entities=["Nora"],
                    jump_url="https://x.test/episode/s?t=90",
                    notes=["(2025-09-04) my own words"],
                )
            ],
        )
    ]


def test_printable_html_carries_the_same_document_as_markdown() -> None:
    """Both formats render the SAME structure, so neither may quietly hold less than the other."""
    html_out = render_highlights_html(_doc())
    md = render_highlights_markdown(_doc())
    for fragment in (
        "How Sleep Works",
        "deep sleep consolidates memory",
        "Guest",
        "A headline",
        "The prose the Summary button renders.",
        "first bullet",
        "my own words",
        "Nora",
    ):
        assert fragment in html_out, f"the printable HTML dropped {fragment!r}"
        assert fragment in md, f"the markdown dropped {fragment!r}"
    # The jump link is what makes an exported timestamp worth anything.
    assert 'href="https://x.test/episode/s?t=90"' in html_out


def test_printable_html_escapes_everything_it_renders() -> None:
    """Quote text, speaker and titles are user- or feed-supplied and reach the page verbatim.

    A capture whose text contains markup would otherwise inject it into the printable document.
    """
    doc = [
        EpisodeHighlights(
            slug="s",
            title="<script>alert(1)</script>",
            highlights=[
                HighlightLine(
                    kind="span",
                    quote_text="<img src=x onerror=alert(1)>",
                    speaker='" onmouseover="evil()',
                )
            ],
        )
    ]
    out = render_highlights_html(doc)
    assert "<script>alert(1)</script>" not in out
    assert "<img src=x" not in out
    assert 'onmouseover="evil()' not in out
    assert "&lt;script&gt;" in out


def test_printable_html_has_a_print_stylesheet() -> None:
    """The whole approach is "the browser is the PDF renderer" — without @media print it is a
    web page that happens to be printable, which is what the CSS is doing the work to avoid."""
    out = render_highlights_html(_doc())
    assert "@media print" in out
    assert "@page" in out
    # A heading stranded at the foot of a page, and a capture split across two, are the two things
    # a printed document gets wrong by default.
    assert "break-after: avoid" in out
    assert "break-inside: avoid" in out


def test_printable_html_is_honest_when_empty() -> None:
    assert "No highlights captured yet" in render_highlights_html([])


def test_duration_renders_as_a_length_not_a_count_of_seconds() -> None:
    assert format_duration(416) == "6 min"
    assert format_duration(3920) == "1 h 5 min"
    assert format_duration(3600) == "1 h"
    assert format_duration(30) == "under a minute"
    # Off a JSON file: absent, the wrong type, or nonsense must not raise.
    for bad in (None, "", "twelve", -5, True):
        assert format_duration(bad) == ""


class TestEveryPrintableDocumentIsBranded:
    """A loose PDF has to say where it came from (operator 2026-09-19).

    The complaint was that the export came out "white and black" — an anonymous sheet of text with
    nothing on it identifying the app. These pin the chrome to the DOCUMENTS, not to
    ``brand_header``/``brand_footer``, because the way this regresses is not someone deleting the
    helpers: it is a new export path added without calling them, or an existing one losing the call
    in a refactor. A test that only asserted the helpers return a string would pass in both cases.
    """

    def _docs(self) -> dict[str, str]:
        from podcast_scraper.server.app_episode_notes import (
            EpisodeNotes,
            render_episode_notes_html,
        )

        return {
            "highlights (with rows)": render_highlights_html(_doc()),
            "highlights (empty state)": render_highlights_html([]),
            "episode notes": render_episode_notes_html(
                EpisodeNotes(slug="ep", title="An Episode", show="A Show")
            ),
        }

    def test_each_document_carries_the_brand_header_and_footer(self) -> None:
        for name, doc in self._docs().items():
            assert 'class="brandbar"' in doc, f"{name} has no brand header"
            assert 'class="brandfoot"' in doc, f"{name} has no brand footer"
            assert "Close Listening" in doc, f"{name} never names the app"

    def test_the_footer_links_home_rather_than_just_naming_the_app(self) -> None:
        # The point of the footer is that a reader can GET BACK. A footer that named the app
        # without a link would satisfy the check above and be useless on paper.
        from podcast_scraper.server.app_capture_export import public_origin

        origin = public_origin()
        for name, doc in self._docs().items():
            assert f'<a href="{origin}"' in doc, f"{name}'s footer has no link home"

    def test_the_printed_page_body_stays_light(self) -> None:
        # Deliberate: the app is dark, an A4 page is not. Printing the app's navy canvas would
        # empty a cartridge rendering a background nobody asked for.
        from podcast_scraper.server.app_capture_export import _PRINT_CSS

        body = _PRINT_CSS[_PRINT_CSS.index("body") :].split("}")[0]
        assert "#080d1b" not in body, "the print body took on the app's dark canvas"
