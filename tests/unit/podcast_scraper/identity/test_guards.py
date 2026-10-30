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
