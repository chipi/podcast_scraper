"""D-36 slice b, operator 2026-10-13: every COPY of a translated quote says it is a translation.

A capture, an episode-notes document, a note-app export and a highlight card all carry a quote out
of the app, where the player's own "Translated from …" label cannot follow it. The user's own notes
are never labelled: they are the user's words, not a translation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from podcast_scraper.languages import translated_from_label
from podcast_scraper.server.app_episode_notes import (
    EpisodeNotes,
    NoteCapture,
    NoteInsight,
    NoteQuote,
    render_episode_notes_html,
    render_episode_notes_markdown,
)
from podcast_scraper.server.schemas import HighlightCreate

pytestmark = pytest.mark.unit

_CORPUS = (
    Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
)


class TestLabel:
    def test_names_the_language(self) -> None:
        assert translated_from_label("es") == "Translated from Spanish"
        assert translated_from_label("pt-BR") == "Translated from Portuguese"

    @pytest.mark.parametrize("code", [None, ""])
    def test_no_language_no_label(self, code) -> None:
        assert translated_from_label(code) is None


class TestCaptureCarriesTheLanguage:
    def _create(self, **kw) -> HighlightCreate:
        return HighlightCreate(episode_slug="ep", kind="span", quote_text="x", **kw)

    def test_normalised(self) -> None:
        assert self._create(translated_from="es-ES").translated_from == "es"

    def test_english_is_not_a_translation(self) -> None:
        assert self._create(translated_from="en").translated_from is None

    def test_absent_stays_absent(self) -> None:
        assert self._create().translated_from is None

    def test_garbage_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            self._create(translated_from="not a language!!")


def _notes() -> EpisodeNotes:
    return EpisodeNotes(
        slug="ep1",
        title="Senderos",
        show="Sesiones de Sendero",
        publish_date="2026-01-02",
        duration_seconds=600,
        url="https://x.test/episode/ep1",
        insights=[
            NoteInsight(
                text="Drainage decides how long a trail lasts.",
                quotes=[NoteQuote(text="drainage is the design decision", translated_from="es")],
            )
        ],
        captures=[
            NoteCapture(
                kind="span",
                quote_text="drainage is the design decision",
                translated_from="es",
                notes=["(2026-10-13) my own words"],
            )
        ],
        episode_notes=["my thought about the episode"],
    )


class TestEpisodeNotes:
    def test_markdown_labels_the_podcast_quotes_and_not_my_notes(self) -> None:
        md = render_episode_notes_markdown(_notes())
        assert md.count("Translated from Spanish") == 2  # the GI quote and the captured quote
        assert "- **Quote**" in md  # the capture's kind label is not displaced by the language
        for line in md.splitlines():
            if "my own words" in line or "my thought about the episode" in line:
                assert "Translated" not in line

    def test_html_labels_the_podcast_quotes(self) -> None:
        assert render_episode_notes_html(_notes()).count("Translated from Spanish") == 2

    def test_a_native_quote_carries_no_label(self) -> None:
        doc = _notes()
        doc.insights[0].quotes[0].translated_from = None
        doc.captures[0].translated_from = None
        assert "Translated" not in render_episode_notes_markdown(doc)


class TestHighlightCard:
    def test_a_translated_capture_says_so_on_its_card(self) -> None:
        from podcast_scraper.server.app_catalog_cache import cached_catalog
        from podcast_scraper.server.app_slugs import slug_for_row
        from podcast_scraper.server.og.build import build_highlight_card

        row = next(r for r in cached_catalog(_CORPUS) if r.feed_id == "p10")
        hl = {"episode_slug": slug_for_row(row), "quote_text": "drainage", "start_ms": 1000}
        card = build_highlight_card(_CORPUS, {**hl, "translated_from": "es"}, with_art=False)
        assert card is not None and card.translated_from == "es"
        native = build_highlight_card(_CORPUS, hl, with_art=False)
        assert native is not None and native.translated_from is None
