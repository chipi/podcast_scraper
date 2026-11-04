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


def test_the_person_card_signals_follow_the_ledger(tmp_path, monkeypatch) -> None:
    from podcast_scraper.server import corpus_signals as E

    monkeypatch.setattr(
        E, "corpus_signals", lambda _root, _wanted: {"guest_coappearance": COAPPEARANCE}
    )
    monkeypatch.setattr(E, "person_display", lambda _root: VIEW)
    monkeypatch.setattr(E, "served_id_map", lambda _root: {})
    out = E.filtered_entity_signals(tmp_path, "person", "person:odile-brand")
    assert out["guest_coappearance"]["pairs"][0]["person_a_name"] == "Korin Vael"


def test_the_person_card_joins_a_persons_ids(tmp_path, monkeypatch) -> None:
    """A partner under two ids is listed once, a person never co-appears with themselves, and the
    card of either id sees the group's rows, turned to the id it asked for."""
    from podcast_scraper.server import corpus_signals as E

    def pair(a: str, b: str, n: int) -> dict:
        names = {"k1": "Korin Vaal", "k2": "Korin Vael", "o": "Odile Brand", "m": "Mara Tull"}
        return {
            "person_a_id": f"person:{a}",
            "person_b_id": f"person:{b}",
            "person_a_name": names[a],
            "person_b_name": names[b],
            "episode_count": n,
        }

    signals = {
        "guest_coappearance": {
            "pairs": [
                pair("k1", "o", 3),
                pair("o", "k2", 2),
                pair("k1", "k2", 1),
                pair("m", "k2", 1),
            ]
        },
        "topic_consensus": {"consensus": [dict(pair("k2", "o", 0), topic_id="topic:t")]},
    }
    joined = {"person:k1": "person:k1", "person:k2": "person:k1"}
    monkeypatch.setattr(E, "corpus_signals", lambda _root, _wanted: signals)
    monkeypatch.setattr(E, "served_id_map", lambda _root: joined)
    monkeypatch.setattr(
        E,
        "person_display",
        lambda _root: PersonDisplay({"person:k1": "Korin Vael", "person:k2": "Korin Vael"}),
    )
    card = E.filtered_entity_signals(tmp_path, "person", "person:o")["guest_coappearance"]["pairs"]
    assert [(r["person_a_id"], r["person_b_id"], r["episode_count"]) for r in card] == [
        ("person:k1", "person:o", 5)
    ]
    assert card[0]["person_a_name"] == "Korin Vael"
    card = E.filtered_entity_signals(tmp_path, "person", "person:k2")
    rows = card["guest_coappearance"]["pairs"]
    partners = {
        r["person_b_id"] if r["person_a_id"] == "person:k2" else r["person_a_id"] for r in rows
    }
    assert partners == {"person:o", "person:m"}  # the k1-k2 self pair is gone
    assert {
        r["episode_count"] for r in rows if "person:o" in (r["person_a_id"], r["person_b_id"])
    } == {5}
    row = card["topic_consensus"]["consensus"][0]
    assert (row["person_a_id"], row["person_b_id"]) == ("person:k2", "person:o")


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
