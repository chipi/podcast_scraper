"""A PERSON entity must name one person (#2197).

Every string below was measured on the prod corpus on 2026-09-30 as a Person node served to the
app — a pair, a list, a year or a diarization label, each minted as a followable "person". The fix
lives at extraction (``parse_kg_graph_response``), so a re-derived episode cannot mint them again.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.kg.llm_extract import parse_kg_graph_response, person_entity_names

pytestmark = pytest.mark.unit


def _people(*names: str, kind: str = "person") -> list[str]:
    raw = json.dumps(
        {
            "topics": [{"label": "behavioural economics"}],
            "entities": [{"name": n, "entity_kind": kind, "description": "d"} for n in names],
        }
    )
    out = parse_kg_graph_response(raw)
    assert out is not None
    return [e["name"] for e in out["entities"] if e["entity_kind"] == kind]


@pytest.mark.parametrize(
    "raw, want",
    [
        ("Kahneman and Tversky", ["Kahneman", "Tversky"]),
        ("Nixon and Kissinger", ["Nixon", "Kissinger"]),
        ("Himmelstein, Wohlhandler, and Warren", ["Himmelstein", "Wohlhandler", "Warren"]),
        ("Michael and Hannah", ["Michael", "Hannah"]),
        ("Romer and Romer", ["Romer"]),
        (
            "Brandon Anderson, RJ Honicky, and Latent.Space",
            ["Brandon Anderson", "RJ Honicky"],
        ),
    ],
)
def test_a_pair_or_list_becomes_one_person_each(raw: str, want: list[str]) -> None:
    assert _people(raw) == want


@pytest.mark.parametrize("raw", ["2017", "2018", "SPEAKER_01", "speaker_15", "unknown_guest_1"])
def test_a_year_or_a_diarization_label_is_not_a_person(raw: str) -> None:
    assert _people(raw) == []


@pytest.mark.parametrize(
    "raw",
    [
        "Mary, Queen of Scots",
        "Thomas Howard, 4th Duke of Norfolk",
        "Henry I, Duke of Guise",
        "Empress Elisabeth (Sisi)",
        "Patio11",
        "Luiz Inácio Lula da Silva",
    ],
)
def test_real_names_with_commas_titles_or_digits_stay_whole(raw: str) -> None:
    """A bare comma is how titles are written; only a conjunction means 'several people'."""
    assert _people(raw) == [raw]


def test_a_suffix_stays_with_its_name() -> None:
    assert person_entity_names("Martin Luther King, Jr. and Coretta Scott King") == [
        "Martin Luther King, Jr.",
        "Coretta Scott King",
    ]


def test_only_person_entities_are_touched() -> None:
    """An organization called 'Simon & Schuster' is one organization, not two people."""
    assert _people("Simon & Schuster", kind="organization") == ["Simon & Schuster"]
