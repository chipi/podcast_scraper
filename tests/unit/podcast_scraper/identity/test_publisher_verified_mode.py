"""The ``publisher_verified`` evidence mode (record l_publisher_verified_mode): V1 verifies the
extracted names against the publisher's own text, V2 finds the text's own spelling, V3 joins a
show's host-seat ASR spelling to its verified host. Switched on 2026-11-06."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.identity import guards as G, ledger as L, publisher_text as PT
from podcast_scraper.kg.entity_clusters import build_entity_canonical_map, EntityCandidate

pytestmark = pytest.mark.unit


def test_the_shipped_mode_is_publisher_verified_and_still_serves_the_extracted_ledger() -> None:
    assert L.EVIDENCE_MODE == PT.PUBLISHER_VERIFIED
    assert L.RULE_VERSION_BY_EVIDENCE[PT.PUBLISHER_VERIFIED] == L.RULE_VERSION
    assert L.RULE_VERSION_BY_EVIDENCE[PT.EXTRACTED] == L.PREVIOUS_RULE_VERSION


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        G.collect_episode_evidence(tmp_path, mode="guess")
    with pytest.raises(ValueError):
        L.build_ledger(tmp_path, evidence_mode="guess")


def test_spans_take_the_publisher_spelling_without_a_possessive() -> None:
    text = "Host Hanne Vasquez talks to Tobin Ashcombe’s envoy and sarah lower."
    assert PT.text_spans(text, "hanne") == ["Hanne Vasquez"]  # a lowercase word ends a span
    assert PT.text_spans(text, "tobin") == ["Tobin Ashcombe"]


def test_v1_keeps_only_names_the_text_contains_and_v2_adds_its_spelling() -> None:
    meta = {
        "feed": {"title": "Curious Show", "description": "With Siun Ó Fearaigh."},
        "episode": {},
    }
    by_key = {"known_hosts": ["Siun Ó Fearaigh", "Siun Ofary"], "metadata_named": []}
    out, pairs = PT.verify_and_find(meta, by_key, roster=["Siun Ofary"])
    assert out["known_hosts"] == ["Siun Ó Fearaigh"]
    assert "Siun Ó Fearaigh" in out["metadata_named"]
    assert ("siun o fearaigh", "siun ofary") in pairs


def test_v2_finds_nothing_when_the_text_holds_two_fitting_spellings() -> None:
    meta = {"episode": {"description": "Hanne Vasquez and Hanne Vasques disagree."}}
    out, pairs = PT.verify_and_find(meta, {"metadata_named": []}, roster=["Hanne Vasquis"])
    assert out["metadata_named"] == [] and pairs == frozenset()


@pytest.mark.parametrize(
    ("host", "other", "by_first", "route"),
    [
        ("Elsa Mordas", "Elsa Martiz", {"elsa": ["Elsa Mordas"]}, "A"),
        # Surnames neither Soundex-equal nor at ratio 0.6: no join; V2 fixes its spelling instead.
        ("Siun Ó Fearaigh", "Siun Ofary", {"siun": ["Siun Ó Fearaigh"]}, None),
        ("Rosalinda Varppi", "Rosalinde Carpi", {"rosalinda": ["Rosalinda Varppi"]}, "B"),
        ("Jonquilda Brook", "John Gilda Brook", {"jonquilda": ["Jonquilda Brook"]}, "C"),
        # The measured failures (prototype runs 1-3), each two people:
        ("Katia Verlund", "Tomas Verlund", {"katia": ["Katia Verlund"]}, None),
        ("Nikolai Brandt", "Nikolai Coyle", {"nikolai": ["Nikolai Brandt"]}, None),
        ("Davin Holt", "Davin Marsh", {"davin": ["Davin Holt", "Davin Pryor"]}, None),
        ("Pieter Okonkwo", "Pieter McAllister", {"pieter": ["Pieter Okonkwo"]}, None),
        ("Lane Ferrimore", "Dan Frossel", {"lane": ["Lane Ferrimore"]}, None),
        ("Mei-Mei Lo", "Jennette Lo", {"mei": ["Mei-Mei Lo"]}, None),
    ],
)
def test_host_routes(host: str, other: str, by_first: dict, route) -> None:
    assert PT.host_route(host, other, by_first) == route


def _cand(cid: str, name: str, episodes, show: str = "pod:a") -> EntityCandidate:
    return EntityCandidate(id=cid, kind="person", name=name, episodes=set(episodes), shows={show})


def test_a_join_answer_joins_and_the_guards_still_decide() -> None:
    a, b = _cand("person:siun-o-fearaigh", "Siun Ó Fearaigh", ["e1"]), _cand(
        "person:siun-ofary", "Siun Ofary", ["e2"]
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
        "description": "Hosted by Elsa Mordas.",
        "language": "en",
    }
    episodes = [
        ("e1", "Elsa Mordas", "person:elsa-mordas"),
        ("e2", "Elsa Martiz", "person:elsa-martiz"),
        ("e3", "Elsa Martiz", "person:elsa-martiz"),
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
            {"tried": {"known_hosts": ["Elsa Mordas", "Elsa Martiz"]}},
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
    assert "Elsa Martiz" in ev["e2"].publisher_names
    assert ev["e2"].publisher_pairs == frozenset() and ev["e2"].host_seated == frozenset()


def test_the_verified_mode_reads_the_text_and_records_the_seats(show: Path) -> None:
    ev = G.collect_episode_evidence(show, mode=PT.PUBLISHER_VERIFIED)
    assert "Elsa Martiz" not in ev["e2"].publisher_names
    assert ev["e2"].host_seated == {"person:elsa-martiz"} and ev["e2"].show == "feed:z"
    assert "Elsa Mordas" in ev["e2"].show_names


def test_the_verified_mode_joins_the_asr_host_and_shows_the_publishers_spelling(show: Path) -> None:
    today = L.build_ledger(show, evidence_mode=PT.EXTRACTED)
    verified = L.build_ledger(show, evidence_mode=PT.PUBLISHER_VERIFIED)
    assert today["rule_version"] == L.RULE_VERSION_BY_EVIDENCE[PT.EXTRACTED]
    assert verified["rule_version"] == L.RULE_VERSION
    old_ids = L.ledger_id_map(today)
    assert "person:elsa-martiz" not in old_ids and "person:elsa-mordas" not in old_ids
    assert old_ids != L.ledger_id_map(verified)
    ids = L.ledger_id_map(verified)
    assert (
        ids.get("person:elsa-martiz") == "person:elsa-mordas"
        or ids.get("person:elsa-mordas") == "person:elsa-martiz"
    )
    shown = {
        c["canonical_id"]: c.get("display_name") for c in verified["clusters_payload"]["clusters"]
    }
    assert "Elsa Mordas" in shown.values()


def test_the_default_build_is_the_publisher_verified_build(show: Path) -> None:
    a, b = L.build_ledger(show), L.build_ledger(show, evidence_mode=PT.PUBLISHER_VERIFIED)
    for led in (a, b):
        led.pop("built_at")
    assert a == b


def test_the_publishers_spelling_reaches_a_group_only_v2_pairs_with_it(tmp_path: Path) -> None:
    """The heavy-ASR shape: the publisher's spelling differs from the ASR one too much for the
    roster's own same-person tests, and their surnames too much for a V3 join; only V2's pair
    shows it."""
    root = tmp_path / "corpus"
    run = root / "feeds" / "feed_q" / "run_x_20261001-000000"
    feed = {
        "feed_id": "feed:q",
        "title": "Curious Show",
        "description": "With Siun Ó Fearaigh.",
        "language": "en",
    }
    for eid in ("q1", "q2"):
        _write(
            run / "metadata" / f"{eid}.metadata.json",
            {
                "episode": {"episode_id": eid, "title": f"Episode {eid}"},
                "feed": feed,
                "content": {
                    "speakers": [{"name": "Siun Ofary", "voices": ["S0"]}],
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
                        "id": "person:siun-ofary",
                        "type": "Person",
                        "properties": {"name": "Siun Ofary"},
                    },
                    {
                        "id": "person:siun-ofarie",
                        "type": "Person",
                        "properties": {"name": "Siun Ofarie"},
                    },
                ],
            },
        )
    shown = lambda led: {  # noqa: E731
        c["canonical_id"]: c.get("display_name") for c in led["clusters_payload"]["clusters"]
    }
    today = shown(L.build_ledger(root, evidence_mode=PT.EXTRACTED))
    verified = shown(L.build_ledger(root, evidence_mode=PT.PUBLISHER_VERIFIED))
    assert "Siun Ó Fearaigh" not in today.values()
    assert "Siun Ó Fearaigh" in verified.values()


def test_a_reviewed_apart_pair_is_never_joined_by_v3() -> None:
    """Review 2, 2026-11-04: V3's "join" answered before the reviewed list's "apart", so a pair a
    human read as two people was joined with no blocked_joins record."""
    a = _cand("person:mira-tolbec", "Mira Tolbec", ["e1"])
    b = _cand("person:mira-tolbek", "Mira Tolbek", ["e2"])
    base = lambda c, m: "apart" if {c.id, m.id} == {a.id, b.id} else None  # noqa: E731
    rule = G.host_join_pair_rule({}, base)
    rule.joins.add(frozenset((a.id, b.id)))
    assert rule(a, b) == "apart"
    payload, id_map = build_entity_canonical_map(
        {a.id: a, b.id: b}, person_mode="cluster_person", pair_rule=rule
    )
    assert id_map.get(a.id, a.id) != id_map.get(b.id, b.id)
    assert payload["blocked_joins"]


def test_v1_compares_a_non_latin_name_as_written() -> None:
    """Review 2: folding to ASCII left a Cyrillic or CJK name empty, so V1 dropped every one."""
    meta = {"feed": {"title": "Иван Петров и гости"}, "episode": {"title": "王小明 談話"}}
    by_key = {"known_hosts": ["Иван Петров", "王小明", "Ольга Смирнова"]}
    out, _pairs = PT.verify_and_find(meta, by_key, roster=[])
    assert out["known_hosts"] == ["Иван Петров", "王小明"]


def test_each_build_is_stamped_with_its_own_modes_version(show: Path) -> None:
    """Review 2: the stamp came from RULE_VERSION on the default path whatever mode shipped."""
    for mode in PT.EVIDENCE_MODES:
        assert (
            L.build_ledger(show, evidence_mode=mode)["rule_version"]
            == L.RULE_VERSION_BY_EVIDENCE[mode]
        )
    assert L.build_ledger(show)["rule_version"] == L.RULE_VERSION_BY_EVIDENCE[L.EVIDENCE_MODE]
