"""A name the corpus's KG extraction decisively calls an organisation is not a speaker (#2220).

Every count here is the measured prod shape (2,002 served episodes, 2026-10-01).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from podcast_scraper.speaker_detectors.entity_kind_votes import (
    corpus_kind_votes,
    kind_key,
    KindVotes,
    votes_from_kg_payloads,
)
from podcast_scraper.speaker_detectors.hosts import drop_non_person_names

pytestmark = [pytest.mark.unit]


def _kg(nodes: List[Tuple[str, str, str]]) -> Dict:
    """``[(type, name, role)]`` -> a KG payload."""
    return {
        "nodes": [
            {"id": f"{t.lower()}:{i}", "type": t, "properties": {"name": n, "role": r}}
            for i, (t, n, r) in enumerate(nodes)
        ]
    }


def _votes(**names: Tuple[int, int]) -> KindVotes:
    """``name=(organization_votes, person_votes)`` -> votes built through the real counter."""
    payloads = []
    for name, (org, person) in names.items():
        label = name.replace("_", " ")
        payloads += [_kg([("Organization", label, "mentioned")])] * org
        payloads += [_kg([("Person", label, "mentioned")])] * person
    return votes_from_kg_payloads(payloads)


class TestTheRule:
    def test_a_name_extraction_calls_an_organisation_is_dropped(self) -> None:
        votes = _votes(The_Brazilian_Report=(23, 0), Americas_Online=(24, 0))
        assert drop_non_person_names(
            ["The Brazilian Report", "Americas Online", "Fernanda Cavalcanti"],
            "Explaining Brazil",
            votes,
        ) == ["Fernanda Cavalcanti"]

    def test_a_mostly_organisation_name_is_dropped(self) -> None:
        """Carnegie India: 10 Organization to 2 Person — 5:1 clears the 4:1 bar."""
        assert _votes(Carnegie_India=(10, 2)).calls_organisation("Carnegie India")

    def test_a_contested_name_is_left_to_the_existing_rules(self) -> None:
        """Trivium China: 10 : 10. Votes cannot decide; the show-name rule still does."""
        votes = _votes(Trivium_China=(10, 10))
        assert not votes.calls_organisation("Trivium China")
        assert drop_non_person_names(["Trivium China"], "The Trivium China Podcast", votes) == []

    def test_a_stray_vote_never_costs_a_person_their_name(self) -> None:
        """Measured: "Jonas" with ONE Organization vote, "Felix" with two."""
        votes = _votes(Jonas=(1, 0), Felix=(2, 1))
        assert drop_non_person_names(["Jonas", "Felix"], "Some Show", votes) == ["Jonas", "Felix"]

    def test_a_person_is_kept(self) -> None:
        votes = _votes(Peter_Attia=(0, 39))
        assert drop_non_person_names(["Ezra Example"], "Show", votes) == ["Ezra Example"]
        assert not votes.calls_organisation("Peter Attia")

    def test_no_votes_changes_nothing(self) -> None:
        names = ["The Brazilian Report", "Fernanda Cavalcanti"]
        assert drop_non_person_names(names, "Explaining Brazil", None) == names


class TestWhoVotes:
    def test_roster_host_and_guest_nodes_do_not_vote(self) -> None:
        """Andreessen Horowitz is Person/host ×53 by the ROSTER — the thing being judged."""
        payload = _kg([("Person", "Andreessen Horowitz", "host")] * 53)
        assert votes_from_kg_payloads([payload]).counts == {}

    def test_a_demoted_roster_node_does_not_vote(self) -> None:
        """m0009 demoted show names to `mentioned`; they kept their HOSTS edge. Measured: all 31
        Person votes for "Machine Learning Street" were these — the show voting itself a person."""
        node = {
            "id": "person:machine-learning-street",
            "type": "Person",
            "properties": {"name": "Machine Learning Street", "role": "mentioned"},
        }
        edge = {"type": "HOSTS", "from": "person:machine-learning-street", "to": "podcast:mlst"}
        org_props = {"name": "Machine Learning Street", "role": "mentioned"}
        org = {"type": "Organization", "properties": org_props}
        votes = votes_from_kg_payloads(
            [{"nodes": [node], "edges": [edge]}] * 31 + [{"nodes": [org], "edges": []}] * 17
        )
        assert votes.counts[kind_key("Machine Learning Street")] == (17, 0)
        assert votes.calls_organisation("Machine Learning Street")

    def test_object_nodes_do_not_vote(self) -> None:
        payload = _kg([("Object", "Claude Code", "mentioned")] * 6)
        assert votes_from_kg_payloads([payload]).counts == {}

    def test_spellings_vote_together(self) -> None:
        """Keyed by the canonical spelling: `Acme Labs)` and `acme labs` are one name."""
        votes = votes_from_kg_payloads(
            [
                _kg([("Organization", "Acme Labs)", "mentioned")]),
                _kg([("Organization", "acme labs", "mentioned")]),
                _kg([("Organization", "Acme  Labs", "mentioned")]),
            ]
        )
        assert votes.counts[kind_key("Acme Labs")] == (3, 0)
        assert votes.calls_organisation("ACME LABS")


class TestCorpusVotes:
    def test_reads_the_served_kgs_of_a_corpus(self, tmp_path: Path) -> None:
        corpus_kind_votes.cache_clear()
        for run in ("run_a", "run_b", "run_c"):
            meta = tmp_path / "feeds" / "f1" / run / "metadata"
            meta.mkdir(parents=True)
            stem = f"ep_{run}"
            (meta / f"{stem}.metadata.json").write_text(
                json.dumps({"episode": {"episode_id": stem}, "feed": {"feed_id": "f1"}})
            )
            (meta / f"{stem}.kg.json").write_text(
                json.dumps(_kg([("Organization", "World Bank", "mentioned")]))
            )
        votes = corpus_kind_votes(str(tmp_path))
        assert votes.counts[kind_key("World Bank")] == (3, 0)
        assert votes.calls_organisation("World Bank")

    def test_no_corpus_is_no_votes(self, tmp_path: Path) -> None:
        corpus_kind_votes.cache_clear()
        assert corpus_kind_votes(str(tmp_path / "missing")).counts == {}
