"""Every remaining name surface shows a joined person's one spelling (RFC-131 S6d).

Invented names only."""

from __future__ import annotations

import pytest

from podcast_scraper.identity.display import PersonDisplay
from podcast_scraper.search.corpus_graph import CorpusGraph
from podcast_scraper.server.kg_view import entities_from_kg

pytestmark = pytest.mark.unit

VIEW = PersonDisplay({"person:korin-vaal": "Korin Vael"})
KG = {
    "nodes": [
        {"id": "person:korin-vaal", "type": "Person", "properties": {"name": "Korin Vaal"}},
        {"id": "person:odile-brand", "type": "Person", "properties": {"name": "Odile Brand"}},
        {"id": "org:quorland", "type": "Organization", "properties": {"name": "Quorland"}},
    ]
}


def test_people_chips_show_the_group_spelling() -> None:
    persons, orgs, _ = entities_from_kg(KG, people=VIEW)
    assert {p.id: p.name for p in persons} == {
        "person:korin-vaal": "Korin Vael",
        "person:odile-brand": "Odile Brand",
    }
    assert [o.name for o in orgs] == ["Quorland"]
    persons, _, _ = entities_from_kg(KG)
    assert {p.name for p in persons} == {"Korin Vaal", "Odile Brand"}  # no view: as before


def test_the_corpus_graph_spells_person_nodes() -> None:
    graph = CorpusGraph(people=VIEW)
    graph._ingest(KG, "kg")
    graph._spell_people()
    assert graph._nodes["person:korin-vaal"].payload["name"] == "Korin Vael"
    assert graph._nodes["person:odile-brand"].payload["name"] == "Odile Brand"
    plain = CorpusGraph()
    plain._ingest(KG, "kg")
    plain._spell_people()
    assert plain._nodes["person:korin-vaal"].payload["name"] == "Korin Vaal"


COAPPEARANCE = {
    "pairs": [
        {
            "person_a_id": "person:korin-vaal",
            "person_b_id": "person:odile-brand",
            "person_a_name": "Korin Vaal",
            "person_b_name": "Odile Brand",
            "episode_count": 4,
        }
    ],
    "communities": [
        {
            "community_id": "pco:korin-vaal",
            "community_label": "Korin Vaal",
            "member_ids": ["person:korin-vaal", "person:odile-brand"],
            "member_count": 2,
        }
    ],
}


def test_coappearance_names_follow_the_ledger() -> None:
    from podcast_scraper.identity.display import spell_enrichment_people

    out = spell_enrichment_people("guest_coappearance", COAPPEARANCE, VIEW)
    assert out["pairs"][0]["person_a_name"] == "Korin Vael"
    assert out["pairs"][0]["person_b_name"] == "Odile Brand"
    assert out["communities"][0]["community_label"] == "Korin Vael"
    assert out["communities"][0]["community_id"] == "pco:korin-vaal"  # ids never move
    assert COAPPEARANCE["pairs"][0]["person_a_name"] == "Korin Vaal"  # the cached copy is untouched
    assert spell_enrichment_people("guest_coappearance", COAPPEARANCE, None) is COAPPEARANCE
    assert spell_enrichment_people("topic_similarity", COAPPEARANCE, VIEW) is COAPPEARANCE


def test_consensus_rows_follow_the_ledger() -> None:
    from podcast_scraper.identity.display import spell_enrichment_people

    data = {
        "consensus": [
            {
                "person_a_id": "person:odile-brand",
                "person_a_name": "Odile Brand",
                "person_b_id": "person:korin-vaal",
                "person_b_name": "Korin Vaal",
            }
        ]
    }
    row = spell_enrichment_people("topic_consensus", data, VIEW)["consensus"][0]
    assert (row["person_a_name"], row["person_b_name"]) == ("Odile Brand", "Korin Vael")


def test_a_community_named_by_two_ids_of_one_person_is_spelled() -> None:
    from podcast_scraper.identity.display import spell_enrichment_people

    view = PersonDisplay({"person:k1": "Korin Vael", "person:k2": "Korin Vael"})
    data = {
        "pairs": [
            {
                "person_a_id": "person:k1",
                "person_b_id": "person:o",
                "person_a_name": "Korin Vaal",
                "person_b_name": "Odile Brand",
                "episode_count": 2,
            },
            {
                "person_a_id": "person:k2",
                "person_b_id": "person:o",
                "person_a_name": "Korin Vaal",
                "person_b_name": "Odile Brand",
                "episode_count": 2,
            },
        ],
        "communities": [
            {"community_label": "Korin Vaal", "member_ids": ["person:k1", "person:k2", "person:o"]}
        ],
    }
    out = spell_enrichment_people("guest_coappearance", data, view)
    assert out["communities"][0]["community_label"] == "Korin Vael"
