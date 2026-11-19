"""A bullet that opens with a quoted term no longer costs the item its summary.

Two production payloads, same shape: the model quoted a term at the start of a bullet and broke
the JSON string around it, so the summary was dropped after the ADR-148 re-roll (temperature 0
repeats the same output).

- 2026-11-08, podcast (prod VictoriaLogs, "Unparsable summary payload (len=1521)"):
  ``"bullets": [“Are We Cooking?” rebrand reflects ...`` (no opening ASCII quote).
- 2026-11-18, news all8, The Verge: ``"bullets": ["AGI-pilled" describes a mindset ...``
  (the term's closing quote ends the JSON string early).
"""

from __future__ import annotations

import pytest

from podcast_scraper.schemas.summary_schema import parse_summary_output

pytestmark = pytest.mark.unit

_TAIL = '"The second bullet stays as it was."]}'


@pytest.mark.parametrize(
    ("payload", "first"),
    [
        (
            '{"title": "Nadella on AGI", "bullets": ["AGI-pilled" describes a mindset where the '
            'economy must reorganize around AGI.", ' + _TAIL,
            '"AGI-pilled" describes a mindset where the economy must reorganize around AGI.',
        ),
        (
            '{"title": "Midterm Elections", "bullets": [“Are We Cooking?” rebrand reflects a '
            'shift toward politics.", ' + _TAIL,
            "“Are We Cooking?” rebrand reflects a shift toward politics.",
        ),
        (
            '{"title": "T", "bullets": ["The so-called "soft landing" was declared early.", '
            + _TAIL,
            'The so-called "soft landing" was declared early.',
        ),
    ],
)
def test_a_quoted_term_in_a_bullet_parses(payload: str, first: str) -> None:
    result = parse_summary_output(payload, provider=None)
    assert result.success and result.schema is not None
    assert result.schema.bullets == [first, "The second bullet stays as it was."]
    assert result.repair_attempted


def test_valid_json_is_untouched() -> None:
    payload = '{"title": "T", "bullets": ["He said \\"no\\" twice.", "Two."]}'
    result = parse_summary_output(payload, provider=None)
    assert result.schema is not None
    assert result.schema.bullets == ['He said "no" twice.', "Two."]
    assert not result.repair_attempted
