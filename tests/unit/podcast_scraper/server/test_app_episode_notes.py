"""Unit tests for the episode-notes document (operator 2026-09-18, pure / no IO)."""

from __future__ import annotations

from podcast_scraper.server.app_episode_notes import (
    EpisodeNotes,
    NoteCapture,
    NoteInsight,
    NoteQuote,
    render_episode_notes_html,
    render_episode_notes_markdown,
)


def _doc(**over) -> EpisodeNotes:
    """A fully-populated document; keyword overrides replace one field at a time.

    Constructed directly rather than through a dict splat so the dataclass's types are checked —
    a `**dict[str, object]` splat erases them and the fixture could drift from the real shape.
    """
    doc = EpisodeNotes(
        slug="ep1",
        title="How Sleep Works",
        show="Long Horizon Notes",
        publish_date="2024-01-02",
        duration_seconds=4810,
        url="https://x.test/episode/ep1",
        summary_title="A headline",
        summary_text="The prose the Summary button renders.",
        summary_bullets=["first point", "second point"],
        people=["Nora", "Daniel Cho"],
        topics=["sleep", "memory"],
        insights=[
            NoteInsight(
                text="Deep sleep is when memory consolidates.",
                insight_type="observation",
                quotes=[
                    NoteQuote(
                        text="the consolidation happens in slow-wave sleep",
                        speaker="Daniel Cho",
                        start_ms=90_000,
                        jump_url="https://x.test/episode/ep1?t=90",
                    )
                ],
            )
        ],
        captures=[
            NoteCapture(
                kind="span",
                quote_text="deep sleep consolidates memory",
                speaker="Daniel Cho",
                color="amber",
                start_ms=90_000,
                created_at=1_757_000_000,
                jump_url="https://x.test/episode/ep1?t=90",
                notes=["(2025-09-04) my own words"],
            )
        ],
        episode_notes=["a thought about the whole episode"],
    )
    for key, value in over.items():
        setattr(doc, key, value)
    return doc


def test_the_document_is_the_whole_episode_not_just_my_captures() -> None:
    """The point of this artifact, and what separates it from the highlights export.

    That one answers "what did I save, across everything"; this answers "what was this episode".
    If any of these sections went missing it would silently become the other document.
    """
    md = render_episode_notes_markdown(_doc())
    for fragment in (
        "# How Sleep Works",
        "Long Horizon Notes · 2024-01-02 · 1 h 20 min",
        "## Summary",
        "A headline",
        "The prose the Summary button renders.",
        "## Key points",
        "- first point",
        "## Topics & people",
        "Nora · Daniel Cho",
        "sleep · memory",
        "## What was said",
        "Deep sleep is when memory consolidates.",
        "the consolidation happens in slow-wave sleep",
        "## What I saved",
        "deep sleep consolidates memory",
        "my own words",
        "## My notes on this episode",
        "a thought about the whole episode",
    ):
        assert fragment in md, f"the episode notes dropped {fragment!r}"


def test_both_the_claim_and_its_evidence_travel() -> None:
    """An insight is a distilled claim PLUS the lines supporting it; they do different jobs.

    Rendering only the claim loses the words actually spoken; only the quotes loses the point.
    """
    md = render_episode_notes_markdown(_doc())
    assert "**Deep sleep is when memory consolidates.**" in md
    assert "> the consolidation happens in slow-wave sleep" in md
    assert "> — Daniel Cho, [1:30](https://x.test/episode/ep1?t=90)" in md


def test_nothing_is_capped() -> None:
    """The operator's call: printed notes are complete or they are a teaser.

    Pinned because a cap is exactly the kind of thing a later "tidy up the export" pass adds.
    """
    many = [NoteInsight(text=f"insight {i}") for i in range(120)]
    md = render_episode_notes_markdown(_doc(insights=many))
    assert "insight 0" in md and "insight 119" in md
    assert "_120 grounded insights._" in md


def test_every_section_is_independently_optional() -> None:
    """No KG, no GI, no captures: still produces notes, exactly as the panel still renders."""
    md = render_episode_notes_markdown(EpisodeNotes(slug="bare", title="Bare Episode"))
    assert "# Bare Episode" in md
    for absent in (
        "## Summary",
        "## Key points",
        "## Topics",
        "## What was said",
        "## What I saved",
    ):
        assert absent not in md


def test_html_carries_the_same_document_and_escapes_it() -> None:
    doc = _doc(
        title="<script>alert(1)</script>",
        insights=[NoteInsight(text="<img src=x onerror=alert(1)>")],
    )
    out = render_episode_notes_html(doc)
    assert "<script>alert(1)</script>" not in out
    assert "<img src=x" not in out
    assert "&lt;script&gt;" in out
    # Still the print-to-PDF path, so it must carry the stylesheet.
    assert "@media print" in out and "@page" in out


def test_html_and_markdown_agree_on_content() -> None:
    """Neither format may quietly hold less than the other."""
    doc = _doc()
    md, html_out = render_episode_notes_markdown(doc), render_episode_notes_html(doc)
    for fragment in (
        "A headline",
        "first point",
        "Deep sleep is when memory consolidates.",
        "the consolidation happens in slow-wave sleep",
        "deep sleep consolidates memory",
        "a thought about the whole episode",
    ):
        assert fragment in md and fragment in html_out, f"formats disagree on {fragment!r}"
