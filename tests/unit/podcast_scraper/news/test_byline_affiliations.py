"""Bylines as the real news sources write them (2026-11-18 ingestion): a name followed by its role
and institution is one person; a list of names is several. Cases are the real bylines."""

from __future__ import annotations

from typing import List

import pytest

from podcast_scraper.news import byline

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "raw, names",
    [
        # The Conversation: name, role, institution.
        (
            "Alexander Korolev, Associate Professor in Politics and International Relations, UNSW",
            ["Alexander Korolev"],
        ),
        (
            "Vanessa Bohns, Braunstein Family Professor and Chair of Organizational Behavior, "
            "Cornell University",
            ["Vanessa Bohns"],
        ),
        (
            "Jennifer Morgan, Senior Fellow, Center for International Environment and Resource "
            "Policy and Climate Policy Lab, Tufts University",
            ["Jennifer Morgan"],
        ),
        (
            "César Pablo Domínguez Prieto, Catedrático de Literatura Comparada, Universidade de "
            "Santiago de Compostela; University of Glasgow",
            ["César Pablo Domínguez Prieto"],
        ),
        (
            "Linggong Kong, Ph.D. Candidate in Political Science, Auburn University",
            ["Linggong Kong"],
        ),
        (
            "Owen Exeter, Postdoctoral Researcher, Marine Ecology, University of Exeter",
            ["Owen Exeter"],
        ),
        ("Marianne Dahl, Deputy Director", ["Marianne Dahl"]),
        ("Bob Berwyn, Inside Climate News", ["Bob Berwyn"]),
        # Lists of names, however they are joined.
        (
            "Autumn Fiester, Aliza Narva,and Stacy C. Smith",
            ["Autumn Fiester", "Aliza Narva", "Stacy C. Smith"],
        ),
        ("Brian Rhoads,Raymond Ma", ["Brian Rhoads", "Raymond Ma"]),
        (
            "Olivia Ireland; Sarah Rainsford; Ilya Abishev",
            ["Olivia Ireland", "Sarah Rainsford", "Ilya Abishev"],
        ),
        (
            "Jahan Taganova, Saleem H. Ali, and Ambika Vishwanath",
            ["Jahan Taganova", "Saleem H. Ali", "Ambika Vishwanath"],
        ),
        (
            "Yasemin Saplakoglu, Philip Ball, Perri Thaler and Jake Buehler",
            ["Yasemin Saplakoglu", "Philip Ball", "Perri Thaler", "Jake Buehler"],
        ),
    ],
)
def test_a_real_byline_gives_its_people(raw: str, names: List[str]) -> None:
    assert byline.parse(raw) == names


@pytest.mark.parametrize("raw", ["AFP; Reuters", "AFP; AP", "Reuters", "Associated Press"])
def test_a_news_agency_credit_is_not_a_person(raw: str) -> None:
    """Al Jazeera's wire bylines (2026-11-18): an agency is an organisation, never an author."""
    assert byline.parse(raw) == []


def test_an_agency_beside_a_reporter_leaves_the_reporter() -> None:
    assert byline.parse("Ana Example; Reuters") == ["Ana Example"]
