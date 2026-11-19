"""A summary's line breaks survive sanitizing, so a one-bullet-per-line reply parses as bullets.

2026-11-18, the news airgapped run: SummLlama replied with 6-8 bullets, one per line, each
starting with ``- ``. ``sanitize_summary`` collapsed every whitespace run, newlines included, to a
space, so the parser saw one line starting with ``- `` and stored the whole reply as one bullet
(78 of 78 items; reply 1,692 characters, stored bullet 1,690).
"""

from __future__ import annotations

import pytest

from podcast_scraper.preprocessing.core import sanitize_summary
from podcast_scraper.schemas.summary_schema import parse_summary_output

pytestmark = pytest.mark.unit

_REPLY = (
    "- Microsoft made Word, Excel and PowerPoint available inside Copilot.\n"
    "- Copilot Autopilot is a cloud agent billed through Copilot Credits.\n"
    "- Microsoft IQ houses Work IQ, a context layer for agents."
)


def test_a_bullet_per_line_reply_parses_as_separate_bullets() -> None:
    result = parse_summary_output(sanitize_summary(_REPLY), provider=None)

    assert result.schema is not None
    assert result.schema.bullets == [
        "Microsoft made Word, Excel and PowerPoint available inside Copilot.",
        "Copilot Autopilot is a cloud agent billed through Copilot Credits.",
        "Microsoft IQ houses Work IQ, a context layer for agents.",
    ]


def test_line_and_paragraph_breaks_are_kept_spaces_inside_a_line_collapse() -> None:
    text = "First  line \t here.  \n\n\n\n  Second   paragraph.\r\nThird line."

    assert sanitize_summary(text) == "First line here.\n\nSecond paragraph.\nThird line."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Summary text. Continue Reading BelowWalsh", "Summary text."),
        ("Content. Read more about this topic.", "Content."),
        ("One.. two . three", "One. two. three"),
    ],
)
def test_page_furniture_and_stray_periods_still_go(text: str, expected: str) -> None:
    assert sanitize_summary(text) == expected
