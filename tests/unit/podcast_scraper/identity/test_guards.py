"""Identity guards inside the join (ADR-167, RFC-131 S4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity import guards as G
from podcast_scraper.kg.entity_clusters import build_entity_canonical_map, EntityCandidate


def _cand(cid: str, name: str, episodes, show: str = "pod:a") -> EntityCandidate:
    return EntityCandidate(id=cid, kind="person", name=name, episodes=set(episodes), shows={show})


def _ev(people, roster=(), publisher=()):
    return G.episode_evidence(people, roster, publisher)


A, B = "person:jorna-velt", "person:jorna-vellt"
PEOPLE = [(A, "Jorna Velt"), (B, "Jorna Vellt")]


class TestEpisodeEvidence:
    def test_roster_voices_land_on_the_node_by_name_or_minted_id(self):
        ev = _ev(PEOPLE, [("jorna  velt", ["S0"]), ("Jorna Vellt", ["S1"])])
        assert ev.voice_of == {A: frozenset({"S0"}), B: frozenset({"S1"})}
        assert ev.mention_only == frozenset()

    def test_unplaced_and_unlisted_are_mention_only(self):
        ev = _ev(PEOPLE, [("Jorna Velt", [])])
        assert ev.mention_only == {A, B}

    def test_a_name_that_mints_no_id_matches_by_name(self):
        ev = _ev([("person:x-1", "张川红")], [("张川红", ["S0"])], ["张川红"])
        assert ev.voice_of == {"person:x-1": frozenset({"S0"})}
        assert ev.publisher_named == {"person:x-1"}

    def test_publisher_names_map_to_nodes(self):
        assert _ev(PEOPLE, publisher=["Jorna Velt", "Someone Else"]).publisher_named == {A}


class TestGuardFires:
    def test_co_voice_different_voices(self):
        ev = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"]), ("Jorna Vellt", ["S1"])])}
        assert G.guard_fires(A, B, ["e1"], ev) == (G.GUARD_CO_VOICE, "e1")

    def test_one_voice_two_names_is_not_co_voice(self):
        ev = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"]), ("Jorna Vellt", ["S0"])])}
        assert G.guard_fires(A, B, ["e1"], ev) is None

    def test_publisher_list_names_both(self):
        ev = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"])], ["Jorna Velt", "Jorna Vellt"])}
        assert G.guard_fires(A, B, ["e1"], ev) == (G.GUARD_PUBLISHER_LIST, "e1")

    def test_publisher_list_naming_one_does_not_fire(self):
        ev = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"]), ("Jorna Vellt", ["S0"])], ["Jorna Velt"])}
        assert G.guard_fires(A, B, ["e1"], ev) is None

    def test_mention_cooccurrence_only_when_neither_is_placed(self):
        both_mentions = {"e1": _ev(PEOPLE)}
        assert G.guard_fires(A, B, ["e1"], both_mentions) == (G.GUARD_MENTION_COOCCURRENCE, "e1")
        voice_plus_mention = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"])])}
        assert G.guard_fires(A, B, ["e1"], voice_plus_mention) is None

    def test_one_name_on_two_ids_is_no_evidence_of_two_people(self):
        same = [(A, "Jorna Velt"), (B, "jorna  velt")]
        assert G.guard_fires(A, B, ["e1"], {"e1": _ev(same, publisher=["Jorna Velt"])}) is None
        assert G.guard_fires(A, B, ["e1"], {"e1": _ev(same)}) is None

    def test_only_episodes_where_both_are_present(self):
        ev = {"e1": _ev([(A, "Jorna Velt")]), "e2": _ev([(B, "Jorna Vellt")])}
        assert G.guard_fires(A, B, ["e1", "e2"], ev) is None

    def test_disabled_guard_does_not_fire(self):
        ev = {"e1": _ev(PEOPLE)}
        on = (G.GUARD_CO_VOICE, G.GUARD_PUBLISHER_LIST)
        assert G.guard_fires(A, B, ["e1"], ev, enabled=on) is None


class TestJoinTest:
    def test_without_guard_the_build_is_unchanged(self):
        cands = {
            c.id: c for c in (_cand(A, "Jorna Velt", ["e1", "e2"]), _cand(B, "Jorna Vellt", ["e1"]))
        }
        payload, id_map = build_entity_canonical_map(cands)
        assert id_map == {B: A}
        assert "blocked_joins" not in payload

    def test_guard_refuses_and_records(self):
        cands = {
            c.id: c for c in (_cand(A, "Jorna Velt", ["e1", "e2"]), _cand(B, "Jorna Vellt", ["e1"]))
        }
        ev = {"e1": _ev(PEOPLE, [("Jorna Velt", ["S0"]), ("Jorna Vellt", ["S1"])])}
        payload, id_map = build_entity_canonical_map(cands, guard=G.candidate_guard(ev))
        assert id_map == {}
        assert payload["blocked_joins"] == [[B, A, G.GUARD_CO_VOICE, "e1"]]

    def test_guard_against_any_member_blocks_the_cluster(self):
        c = "person:jorna-veltt"
        cands = {
            x.id: x
            for x in (
                _cand(A, "Jorna Velt", ["e1", "e2", "e3"]),
                _cand(c, "Jorna Veltt", ["e2", "e4"]),
                _cand(B, "Jorna Vellt", ["e4"]),
            )
        }
        ev = {
            "e4": _ev(
                [(c, "Jorna Veltt"), (B, "Jorna Vellt")],
                [("Jorna Veltt", ["S0"]), ("Jorna Vellt", ["S1"])],
            )
        }
        _payload, id_map = build_entity_canonical_map(cands, guard=G.candidate_guard(ev))
        assert id_map == {c: A}

    def test_orgs_are_not_guarded(self):
        def org(i: str, n: str) -> EntityCandidate:
            return EntityCandidate(id=i, kind="org", name=n, episodes={"e1"}, shows={"pod:a"})

        cands = {
            "org:data-bricks": org("org:data-bricks", "Data Bricks"),
            "org:databricks": org("org:databricks", "Databricks"),
        }
        called = []
        _p, id_map = build_entity_canonical_map(
            cands, guard=lambda c, m: called.append(1) or ("x", "e1")
        )
        assert len(id_map) == 1 and not called


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    run = root / "feeds" / "feed_a" / "run_x_20261001-000000"

    def seg(v: str, label: str) -> dict:
        return {"start": 0, "end": 1, "text": "w", "speaker": v, "speaker_label": label}

    for stem, eid, people, speakers, segments, tried in (
        (
            "ep1",
            "a1",
            PEOPLE,
            [{"name": "Jorna Velt", "voices": ["S0"]}, {"name": "Jorna Vellt", "voices": ["S1"]}],
            [],
            {"known_hosts": ["Jorna Velt"]},
        ),
        (
            "ep2",
            "a2",
            [(A, "Jorna Velt")],
            [{"name": "Jorna Velt"}],
            [seg("S3", "Jorna Velt")],
            None,
        ),
    ):
        _write(
            run / "metadata" / f"{stem}.metadata.json",
            {
                "episode": {"episode_id": eid},
                "feed": {
                    "language": "en"
                },  # an English show (S5b: untagged joins by equality only)
                "content": {
                    "speakers": speakers,
                    "transcript_file_path": f"transcripts/{stem}.txt",
                },
            },
        )
        _write(run / "transcripts" / f"{stem}.segments.json", segments)
        if tried:
            _write(run / "transcripts" / f"{stem}.speakers.diagnostics.json", {"tried": tried})
        nodes = [{"id": "episode:" + eid, "type": "Episode", "properties": {"podcast_id": "pod:a"}}]
        nodes += [{"id": i, "type": "Person", "properties": {"name": n}} for i, n in people]
        _write(run / "metadata" / f"{stem}.kg.json", {"episode_id": eid, "nodes": nodes})
    return root


def test_collect_episode_evidence_reads_the_corpus(corpus):
    ev = G.collect_episode_evidence(corpus)
    assert ev["a1"].voice_of == {A: frozenset({"S0"}), B: frozenset({"S1"})}
    assert ev["a1"].publisher_named == {A}
    assert ev["a2"].voice_of == {A: frozenset({"S3"})}  # old record: voices from segment labels


def test_ledger_records_the_blocked_join(corpus):
    from podcast_scraper.identity.ledger import build_ledger, ledger_id_map

    ledger = build_ledger(corpus)
    assert ledger_id_map(ledger) == {}
    assert ledger["blocked_joins"] == [[B, A, G.GUARD_CO_VOICE, "a1"]]


class TestPersonRule:
    """Mode ``cluster_person`` and the no-one-word-bridge linkage (RFC-131 S4, N12)."""

    @pytest.mark.parametrize(
        "a,b,same",
        [
            ("Marvelin", "Garvelin", False),  # one edit apart, two names
            ("Kestrovan", "Estrovan", False),
            ("Teshun", "Teshan", False),
            ("Oğuzan", "Oguzan", True),  # accents folded
            ("tolvane", "Tolvane", True),
            ("Patio11", "Patio 11", True),
            ("Tolvane", "Mara Tolvane", True),
            ("Tracy Alloway", "Tracey Alloway", True),
        ],
    )
    def test_cluster_person_mode(self, a, b, same):
        from podcast_scraper.identity.person_match import same_person

        assert same_person(a, b, mode="cluster_person") is same
        assert same_person(b, a, mode="cluster_person") is same

    def test_mode_cluster_is_unchanged(self):
        from podcast_scraper.identity.person_match import same_person

        assert same_person("Marvelin", "Garvelin", mode="cluster") is True

    def _build(self, linkage):
        cands = {
            c.id: c
            for c in (
                _cand("person:mara-tolvane", "Mara Tolvane", ["e1", "e2", "e3"]),
                _cand("person:tolvane", "Tolvane", ["e4", "e5"]),
                _cand("person:ilse-tolvane", "Ilse Tolvane", ["e6"]),
                _cand("person:brannick-holt", "Brannick Holt", ["e7", "e8", "e9", "e10"]),
                _cand("person:brannick-hollt", "Brannick Hollt", ["e11"]),
                _cand("person:president-corvel", "President Corvel", ["e12", "e13"]),
                _cand("person:arno-corvel", "Arno Corvel", ["e14", "e15", "e16"]),
                _cand("person:wilm-corvel", "Wilm Corvel", ["e17"]),
            )
        }
        return build_entity_canonical_map(cands, person_mode="cluster_person", linkage=linkage)[1]

    def test_single_linkage_bridges_through_a_surname(self):
        id_map = self._build("single")
        assert id_map["person:ilse-tolvane"] == "person:mara-tolvane"
        assert id_map["person:wilm-corvel"] == "person:arno-corvel"

    def test_no_bridge_keeps_full_names_apart_and_drift_joined(self):
        id_map = self._build("no_one_word_bridge")
        assert "person:ilse-tolvane" not in id_map
        assert "person:wilm-corvel" not in id_map  # "President Corvel" is one word after the title
        assert id_map["person:tolvane"] == "person:mara-tolvane"
        assert id_map["person:brannick-hollt"] == "person:brannick-holt"

    def test_unknown_linkage_is_refused(self):
        with pytest.raises(ValueError):
            build_entity_canonical_map({}, linkage="chain")


class TestReviewedPairRule:
    """Record s5d: a near spelling of two one-episode mentions with no voice and no publisher text
    is not evidence of one person, unless a reviewed group lists both; a reviewed two-people pair is
    never joined. Synthetic names only."""

    C, D = "person:tavik-orlen", "person:tavik-orlan"

    def _build(self, cands, evidence, reviewed):
        rule = G.reviewed_pair_rule(evidence, reviewed)
        payload, id_map = build_entity_canonical_map(
            {c.id: c for c in cands},
            guard=G.candidate_guard(evidence),
            person_mode="cluster_person",
            linkage="no_one_word_bridge",
            pair_rule=rule,
        )
        return payload, id_map

    def _one_episode_each(self, roster=(), publisher=()):
        cands = [_cand(self.C, "Tavik Orlen", {"e1"}), _cand(self.D, "Tavik Orlan", {"e2"})]
        evidence = {
            "e1": _ev([(self.C, "Tavik Orlen")], roster, publisher),
            "e2": _ev([(self.D, "Tavik Orlan")]),
        }
        return cands, evidence

    def test_two_one_episode_mentions_are_not_joined_on_a_near_spelling(self):
        cands, ev = self._one_episode_each()
        payload, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert id_map == {}
        assert payload["blocked_joins"] == []

    def test_a_reviewed_group_lets_the_pair_join(self):
        cands, ev = self._one_episode_each()
        _p, id_map = self._build(cands, ev, {"groups": [[self.C, self.D]], "apart": []})
        assert len(id_map) == 1

    def test_a_second_episode_lets_the_pair_join(self):
        cands, ev = self._one_episode_each()
        cands[0].episodes.add("e3")
        ev["e3"] = _ev([(self.C, "Tavik Orlen")])
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert id_map == {self.D: self.C}

    def test_a_voice_lets_the_pair_join(self):
        cands, ev = self._one_episode_each(roster=[("Tavik Orlen", ["S0"])])
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert len(id_map) == 1

    def test_publisher_text_lets_the_pair_join(self):
        cands, ev = self._one_episode_each(publisher=["Tavik Orlen"])
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert len(id_map) == 1

    def test_an_equal_name_is_never_ignored(self):
        cands = [_cand(self.C, "Tavik Orlen", {"e1"}), _cand(self.D, "tavik  orlen", {"e2"})]
        ev = {"e1": _ev([(self.C, "Tavik Orlen")]), "e2": _ev([(self.D, "tavik  orlen")])}
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert len(id_map) == 1

    def test_a_middle_initial_variant_is_never_ignored(self):
        cands = [_cand(self.C, "Tavik R. Orlen", {"e1"}), _cand(self.D, "Tavik Orlen", {"e2"})]
        ev = {"e1": _ev([(self.C, "Tavik R. Orlen")]), "e2": _ev([(self.D, "Tavik Orlen")])}
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert len(id_map) == 1

    def test_an_id_with_no_episode_counts_as_one_episode(self):
        cands = [_cand(self.C, "Tavik Orlen", set()), _cand(self.D, "Tavik Orlan", {"e2"})]
        ev = {"e2": _ev([(self.D, "Tavik Orlan")])}
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert id_map == {}

    def test_any_edge_that_counts_admits_the_candidate(self):
        """A third member with two episodes matches the candidate: that edge stands."""
        e = "person:tavik-orlenn"
        cands, ev = self._one_episode_each()
        cands.append(_cand(e, "Tavik Orlenn", {"e4", "e5"}))
        ev["e4"], ev["e5"] = _ev([(e, "Tavik Orlenn")]), _ev([(e, "Tavik Orlenn")])
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": []})
        assert id_map == {self.C: e, self.D: e}

    def test_a_reviewed_apart_pair_is_refused_and_recorded(self):
        cands, ev = self._one_episode_each()
        for c in cands:  # with two episodes each, rule 1 no longer applies; rule 2 still does
            c.episodes.add(c.id + "-x")
            ev[c.id + "-x"] = _ev([(c.id, c.name)])
        payload, id_map = self._build(cands, ev, {"groups": [], "apart": [[self.C, self.D, "r"]]})
        assert id_map == {}
        assert [r[:3] for r in payload["blocked_joins"]] == [
            [self.C, self.D, G.GUARD_REVIEWED_APART]
        ]

    def test_an_apart_pair_is_refused_even_through_a_matching_third_member(self):
        e = "person:tavik-orlenn"
        cands, ev = self._one_episode_each()
        cands.append(_cand(e, "Tavik Orlenn", {"e4", "e5"}))
        ev["e4"], ev["e5"] = _ev([(e, "Tavik Orlenn")]), _ev([(e, "Tavik Orlenn")])
        _p, id_map = self._build(cands, ev, {"groups": [], "apart": [[self.C, self.D, "r"]]})
        # The second of the pair visited (C) may not join the group its partner joined.
        assert id_map == {self.D: e}

    def test_without_a_pair_rule_the_build_is_unchanged(self):
        cands, ev = self._one_episode_each()
        _p, id_map = build_entity_canonical_map(
            {c.id: c for c in cands},
            guard=G.candidate_guard(ev),
            person_mode="cluster_person",
            linkage="no_one_word_bridge",
        )
        assert len(id_map) == 1

    def test_the_shipped_list_loads(self):
        reviewed = G.load_reviewed()
        assert len(reviewed["groups"]) == 65 and len(reviewed["apart"]) == 10
        ids = [i for g in reviewed["groups"] for i in g]
        assert len(ids) == len(set(ids))
        assert not {i for p in reviewed["apart"] for i in p[:2]} & set(ids)
