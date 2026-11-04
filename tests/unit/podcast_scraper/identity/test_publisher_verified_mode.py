"""The ``publisher_verified`` evidence mode (record l_publisher_verified_mode): V1 verifies the
extracted names against the publisher's own text, V2 finds the text's own spelling, V3 joins a
show's host-seat ASR spelling to its verified host. Shipped switched off."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity import guards as G, ledger as L, publisher_text as PT
from podcast_scraper.kg.entity_clusters import build_entity_canonical_map, EntityCandidate

pytestmark = pytest.mark.unit


def test_the_shipped_mode_is_todays() -> None:
    assert L.EVIDENCE_MODE == PT.EXTRACTED
    assert L.RULE_VERSION_BY_EVIDENCE[PT.EXTRACTED] == L.RULE_VERSION
    assert L.RULE_VERSION_BY_EVIDENCE[PT.PUBLISHER_VERIFIED] != L.RULE_VERSION


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        G.collect_episode_evidence(tmp_path, mode="guess")
    with pytest.raises(ValueError):
        L.build_ledger(tmp_path, evidence_mode="guess")


def test_spans_take_the_publisher_spelling_without_a_possessive() -> None:
    text = "Host Sarah Gonzalez talks to Cyril Ramaphosa’s envoy and sarah lower."
    assert PT.text_spans(text, "sarah") == ["Sarah Gonzalez"]  # a lowercase word ends a span
    assert PT.text_spans(text, "cyril") == ["Cyril Ramaphosa"]


def test_v1_keeps_only_names_the_text_contains_and_v2_adds_its_spelling() -> None:
    meta = {"feed": {"title": "Curious Show", "description": "With Dara Ó Briain."}, "episode": {}}
    by_key = {"known_hosts": ["Dara Ó Briain", "Dara Obrey"], "metadata_named": []}
    out, pairs = PT.verify_and_find(meta, by_key, roster=["Dara Obrey"])
    assert out["known_hosts"] == ["Dara Ó Briain"]
    assert "Dara Ó Briain" in out["metadata_named"]
    assert ("dara o briain", "dara obrey") in pairs


def test_v2_finds_nothing_when_the_text_holds_two_fitting_spellings() -> None:
    meta = {"episode": {"description": "Sarah Gonzalez and Sarah Gonzales disagree."}}
    out, pairs = PT.verify_and_find(meta, {"metadata_named": []}, roster=["Sarah Gonzalis"])
    assert out["metadata_named"] == [] and pairs == frozenset()


@pytest.mark.parametrize(
    ("host", "other", "by_first", "route"),
    [
        ("Erika Beras", "Erika Barris", {"erika": ["Erika Beras"]}, "A"),
        # Surnames neither Soundex-equal nor at ratio 0.6: no join; V2 fixes its spelling instead.
        ("Dara Ó Briain", "Dara Obrey", {"dara": ["Dara Ó Briain"]}, None),
        ("Alexandra Karppi", "Alexander Carpi", {"alexandra": ["Alexandra Karppi"]}, "B"),
        ("Jonquilyn Hill", "John Glyn Hill", {"jonquilyn": ["Jonquilyn Hill"]}, "C"),
        # The measured failures (prototype runs 1-3), each two people:
        ("Katie Martin", "Tim Martin", {"katie": ["Katie Martin"]}, None),
        ("Nicholas Fandos", "Nicholas Casey", {"nicholas": ["Nicholas Fandos"]}, None),
        ("David George", "David Booth", {"david": ["David George", "David Clark"]}, None),
        ("Patrick O'Shaughnessy", "Patrick McKenzie", {"patrick": ["Patrick O'Shaughnessy"]}, None),
        ("Lane Florsheim", "Dan Frosch", {"lane": ["Lane Florsheim"]}, None),
        ("Fei-Fei Li", "Jennifer Li", {"fei": ["Fei-Fei Li"]}, None),
    ],
)
def test_host_routes(host: str, other: str, by_first: dict, route) -> None:
    assert PT.host_route(host, other, by_first) == route


def _cand(cid: str, name: str, episodes, show: str = "pod:a") -> EntityCandidate:
    return EntityCandidate(id=cid, kind="person", name=name, episodes=set(episodes), shows={show})


def test_a_join_answer_joins_and_the_guards_still_decide() -> None:
    a, b = _cand("person:dara-o-briain", "Dara Ó Briain", ["e1"]), _cand(
        "person:dara-obrey", "Dara Obrey", ["e2"]
    )
    join = lambda c, m: "join" if {c.id, m.id} == {a.id, b.id} else None  # noqa: E731
    _payload, id_map = build_entity_canonical_map(
        {a.id: a, b.id: b}, person_mode="cluster_person", pair_rule=join
    )
    assert id_map.get(b.id, b.id) == id_map.get(a.id, a.id)
    payload, id_map = build_entity_canonical_map(
        {a.id: a, b.id: b},
        person_mode="cluster_person",
        pair_rule=join,
        guard=lambda c, m: ("co_voice", "e1"),
    )
    assert id_map.get(b.id, b.id) != id_map.get(a.id, a.id)
    assert payload["blocked_joins"]


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def show(tmp_path: Path) -> Path:
    """One show whose publisher text names its host; the diagnostics' known_hosts also carry an ASR
    spelling, and two episodes seat that ASR spelling as host."""
    root = tmp_path / "corpus"
    run = root / "feeds" / "feed_z" / "run_x_20261001-000000"
    feed = {
        "feed_id": "feed:z",
        "title": "Curious Show",
        "description": "Hosted by Erika Beras.",
        "language": "en",
    }
    episodes = [
        ("e1", "Erika Beras", "person:erika-beras"),
        ("e2", "Erika Barris", "person:erika-barris"),
        ("e3", "Erika Barris", "person:erika-barris"),
    ]
    for eid, name, pid in episodes:
        _write(
            run / "metadata" / f"{eid}.metadata.json",
            {
                "episode": {"episode_id": eid, "title": f"Episode {eid}"},
                "feed": feed,
                "content": {
                    "speakers": [{"name": name, "voices": ["S0"]}],
                    "transcript_file_path": f"transcripts/{eid}.txt",
                },
            },
        )
        _write(run / "transcripts" / f"{eid}.segments.json", [])
        _write(
            run / "transcripts" / f"{eid}.speakers.diagnostics.json",
            {"tried": {"known_hosts": ["Erika Beras", "Erika Barris"]}},
        )
        _write(
            run / "metadata" / f"{eid}.kg.json",
            {
                "episode_id": eid,
                "nodes": [
                    {
                        "id": "episode:" + eid,
                        "type": "Episode",
                        "properties": {"feed_id": "feed:z"},
                    },
                    {"id": pid, "type": "Person", "properties": {"name": name, "role": "host"}},
                ],
            },
        )
    return root


def test_todays_mode_takes_the_asr_known_host_as_the_publishers(show: Path) -> None:
    ev = G.collect_episode_evidence(show)
    assert "Erika Barris" in ev["e2"].publisher_names
    assert ev["e2"].publisher_pairs == frozenset() and ev["e2"].host_seated == frozenset()


def test_the_verified_mode_reads_the_text_and_records_the_seats(show: Path) -> None:
    ev = G.collect_episode_evidence(show, mode=PT.PUBLISHER_VERIFIED)
    assert "Erika Barris" not in ev["e2"].publisher_names
    assert ev["e2"].host_seated == {"person:erika-barris"} and ev["e2"].show == "feed:z"
    assert "Erika Beras" in ev["e2"].show_names


def test_the_verified_mode_joins_the_asr_host_and_shows_the_publishers_spelling(show: Path) -> None:
    today = L.build_ledger(show)
    verified = L.build_ledger(show, evidence_mode=PT.PUBLISHER_VERIFIED)
    assert today["rule_version"] == L.RULE_VERSION
    assert verified["rule_version"] == L.RULE_VERSION_BY_EVIDENCE[PT.PUBLISHER_VERIFIED]
    assert L.ledger_id_map(today).get("person:erika-barris") is None
    ids = L.ledger_id_map(verified)
    assert (
        ids.get("person:erika-barris") == "person:erika-beras"
        or ids.get("person:erika-beras") == "person:erika-barris"
    )
    shown = {
        c["canonical_id"]: c.get("display_name") for c in verified["clusters_payload"]["clusters"]
    }
    assert "Erika Beras" in shown.values()


def test_the_default_build_is_the_extracted_build(show: Path) -> None:
    a, b = L.build_ledger(show), L.build_ledger(show, evidence_mode=PT.EXTRACTED)
    for led in (a, b):
        led.pop("built_at")
    assert a == b


def test_the_publishers_spelling_reaches_a_group_only_v2_pairs_with_it(tmp_path: Path) -> None:
    """The Dara shape: the publisher's spelling differs from the ASR one too much for the roster's
    own same-person tests, and their surnames too much for a V3 join; only V2's pair shows it."""
    root = tmp_path / "corpus"
    run = root / "feeds" / "feed_q" / "run_x_20261001-000000"
    feed = {
        "feed_id": "feed:q",
        "title": "Curious Show",
        "description": "With Dara Ó Briain.",
        "language": "en",
    }
    for eid in ("q1", "q2"):
        _write(
            run / "metadata" / f"{eid}.metadata.json",
            {
                "episode": {"episode_id": eid, "title": f"Episode {eid}"},
                "feed": feed,
                "content": {
                    "speakers": [{"name": "Dara Obrey", "voices": ["S0"]}],
                    "transcript_file_path": f"transcripts/{eid}.txt",
                },
            },
        )
        _write(run / "transcripts" / f"{eid}.segments.json", [])
        _write(run / "transcripts" / f"{eid}.speakers.diagnostics.json", {"tried": {}})
        _write(
            run / "metadata" / f"{eid}.kg.json",
            {
                "episode_id": eid,
                "nodes": [
                    {
                        "id": "episode:" + eid,
                        "type": "Episode",
                        "properties": {"feed_id": "feed:q"},
                    },
                    {
                        "id": "person:dara-obrey",
                        "type": "Person",
                        "properties": {"name": "Dara Obrey"},
                    },
                    {
                        "id": "person:dara-obrie",
                        "type": "Person",
                        "properties": {"name": "Dara Obrie"},
                    },
                ],
            },
        )
    shown = lambda led: {  # noqa: E731
        c["canonical_id"]: c.get("display_name") for c in led["clusters_payload"]["clusters"]
    }
    today = shown(L.build_ledger(root))
    verified = shown(L.build_ledger(root, evidence_mode=PT.PUBLISHER_VERIFIED))
    assert "Dara Ó Briain" not in today.values()
    assert "Dara Ó Briain" in verified.values()
