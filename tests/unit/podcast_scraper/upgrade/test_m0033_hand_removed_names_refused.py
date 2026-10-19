"""m0033: the voice a reprocess published under a name 0015 removed by hand, named as the review
read it, through 0031's rewrite. Synthetic episodes from m0031's test fixtures (m0025's corpus)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from podcast_scraper.speaker_detectors import hosts
from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0015_unpublishable_speaker_names_removed as m15,
    m0019_descriptor_speaker_names_removed as m19,
    m0031_named_voices_shown as m31,
    m0033_hand_removed_names_refused as m33,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0031_tests", Path(__file__).with_name("test_m0031_named_voices_shown.py")
)
assert _spec and _spec.loader
t31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t31)
fx = t31.fx

ENTRY = {("ep", "SPEAKER_01"): ("Generation", "guest", "Steve McConnell", "guest", "the guest")}


def _review(monkeypatch, tmp_path: Path, verdict: str = "CORRECT") -> None:
    d = tmp_path / "_verdicts33"
    d.mkdir(exist_ok=True)
    names = m31._names(m33.VOICES)
    rows = [
        "\t".join((eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read"))
        for (eid, voice), name in sorted(names.items())
    ]
    head = (
        f"# migration: {m33.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(names)}\n# reviewer: test\n# date: 2026-10-19\n"
    )
    (d / f"{m33.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m33.HandRemovedNamesRefusedMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_registered_right_after_0031_with_its_own_receipts() -> None:
    migs = get_migrations()
    ids = [m.id for m in migs]
    at = ids.index(m33.MIGRATION_ID)
    assert ids[at - 1] == m31.MIGRATION_ID and migs[at].to_version == "2.7.27"
    assert len({m33.BACKUP_TAG, m31.BACKUP_TAG, "0015", "0019"}) == 4
    assert m33.RECEIPTS_FILE not in (m31.RECEIPTS_FILE, m15.RECEIPTS_FILE, m19.RECEIPTS_FILE)


def test_the_shipped_review_names_the_voice_and_matches_the_list() -> None:
    mig = m33.HandRemovedNamesRefusedMigration()
    verdicts = mig._verdicts()
    [(key, v)] = verdicts.rows.items()
    assert key in m33.VOICES and (v.name, v.verdict) == ("David Roblin", "CORRECT")
    assert m33.VOICES[key][0] in hosts.HAND_REMOVED_SPEAKER_NAMES


def test_the_voice_under_a_hand_removed_name_takes_the_reviewed_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    mig = m33.HandRemovedNamesRefusedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    text = p["txt"].read_text(encoding="utf-8")
    assert "\nSteve McConnell: Thanks for having me." in text and "Generation:" not in text
    guest = t31._guest(p)
    assert guest["speaker"] == "Steve McConnell" and "surfaceable" not in guest
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert "Generation" not in names and "Steve McConnell" in names
    assert mig.verify(_ctx(tmp_path))[0]
    assert m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert (tmp_path / m33.RECEIPTS_FILE).is_file()
    assert not (tmp_path / m31.RECEIPTS_FILE).exists()
    restored, refused = m33.undo(tmp_path)
    assert restored and not refused
    assert "Generation: Thanks for having me." in p["txt"].read_text(encoding="utf-8")


def test_without_a_verdict_the_episode_is_refused_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    d = tmp_path / "_none"
    monkeypatch.setattr(m33.HandRemovedNamesRefusedMigration, "_verdicts_dir", d)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    before = p["seg"].read_text(encoding="utf-8")
    ok, message = m33.HandRemovedNamesRefusedMigration().verify(_ctx(tmp_path))
    assert not ok and message.startswith("verdicts file unusable")
    with pytest.raises(rv.VerdictError):
        m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    assert p["seg"].read_text(encoding="utf-8") == before


def test_0031_still_checks_its_own_list_and_receipts(tmp_path: Path, monkeypatch) -> None:
    """The list and receipts became parameters of 0031's outcome; 0031's defaults are its own."""
    assert m31.NamedVoicesShownMigration().listed() is m31.m29.NAMING_ARC_VOICES
    assert m33.HandRemovedNamesRefusedMigration().listed() is m33.VOICES
    assert m31._written_metas(tmp_path) == set()


def test_0015_and_0019_read_the_gates_list() -> None:
    assert m15.REMOVE is hosts.HAND_REMOVED_SPEAKER_NAMES
    assert m19.REMOVE is hosts.HAND_REMOVED_SPEAKER_NAMES


MINTED = "person:unresolved-generation-ep"


def _as_prod(p) -> None:
    """E's shape on prod: the old person under a MINTED id in GI and KG (with its role and mention
    edges), the new person already in the KG as mentioned, the stated guest unplaced, the record's
    heard entities, and the entry's evidence a self-introduction."""
    gi = fx._r(p["gi"])
    for n in gi["nodes"]:
        if n["id"] == "person:generation":
            n["id"] = MINTED
        props = n.get("properties") or {}
        if props.get("speaker_id") == "person:generation":
            props["speaker_id"] = MINTED
    for e in gi["edges"]:
        if e.get("to") == "person:generation":
            e["to"] = MINTED
    fx._w(p["gi"], gi)
    kg = fx._r(p["kg"])
    for n in kg["nodes"]:
        if n["id"] == "person:generation":
            n["id"] = MINTED
    kg["nodes"].append(
        {
            "id": "person:steve-mcconnell",
            "type": "Person",
            "properties": {
                "name": "Steve McConnell",
                "label": "Steve McConnell",
                "role": "mentioned",
            },
        }
    )
    kg["edges"] = [
        {"type": "GUESTS_ON", "from": MINTED, "to": "podcast:x"},
        {"type": "MENTIONS", "from": MINTED, "to": "episode:ep"},
        {"type": "MENTIONS", "from": "person:steve-mcconnell", "to": "episode:ep"},
        {"type": "HOSTS", "from": "person:gergely-orosz", "to": "podcast:x"},
    ]
    fx._w(p["kg"], kg)
    meta = fx._r(p["meta"])
    content = meta["content"]
    content["speakers"][1]["source"] = "self_intro"
    content["speakers"].append(
        {
            "id": "unplaced_1",
            "name": "Steve McConnell",
            "role": "guest",
            "placed": False,
            "voices": [],
            "source": "episode_metadata",
        }
    )
    content["normalized_entities"] = [
        {
            "canonical": "gergely orosz",
            "aliases": ["gergely orosz"],
            "original": "Gergely Orosz",
            "provenance": "transcript",
        },
        {
            "canonical": "generation",
            "aliases": ["generation"],
            "original": "Generation",
            "provenance": "transcript",
        },
    ]
    fx._w(p["meta"], meta)


def test_a_rename_into_a_person_the_graph_holds_merges_into_them(
    tmp_path: Path, monkeypatch
) -> None:
    """Prod, 10-16 snapshot: the rewrite left E's "Generation" Person node (a minted id is never
    ``person_id`` of the name), and dropped its ``GUESTS_ON`` edge with the node (as in 7 of
    0029's 10 renames). The voice is the same speaker under a new name: its edges move."""
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    res = m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    gi = fx._r(p["gi"])
    people = [n["id"] for n in gi["nodes"] if n["type"] == "Person"]
    assert MINTED not in people and "person:steve-mcconnell" in people
    kg = fx._r(p["kg"])
    assert [n["id"] for n in kg["nodes"]].count("person:steve-mcconnell") == 1
    assert MINTED not in [n["id"] for n in kg["nodes"]]
    steve = next(n for n in kg["nodes"] if n["id"] == "person:steve-mcconnell")
    assert steve["properties"]["role"] == "guest"
    edges = sorted((e["type"], e["from"]) for e in kg["edges"])
    assert edges == [
        ("GUESTS_ON", "person:steve-mcconnell"),
        ("HOSTS", "person:gergely-orosz"),
        ("MENTIONS", "person:steve-mcconnell"),
    ]
    content = fx._r(p["meta"])["content"]
    speakers = [(s["name"], s.get("placed"), s.get("source")) for s in content["speakers"]]
    assert ("Steve McConnell", True, "roster") in speakers
    assert all(name != "Generation" for name, _p, _s in speakers)
    assert [s["name"] for s in content["speakers"]].count("Steve McConnell") == 1
    canon = [r["canonical"] for r in content["normalized_entities"]]
    assert "generation" not in canon and "steve mcconnell" in canon and "gergely orosz" in canon


def _mention(p, pid: str) -> None:
    """Another insight MENTIONS ``pid`` (a Person the rename would otherwise remove)."""
    gi = fx._r(p["gi"])
    gi["nodes"].append(
        {"id": "insight:other", "type": "Insight", "properties": {"text": "x", "tier": 2}}
    )
    gi["edges"].append({"type": "MENTIONS_PERSON", "from": "insight:other", "to": pid})
    fx._w(p["gi"], gi)


def test_a_person_another_insight_mentions_is_kept_and_its_mentions_stay(
    tmp_path: Path, monkeypatch
) -> None:
    """Revision B (B5): moving a mention to the new person would claim the insight names them."""
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    _mention(p, MINTED)
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    gi = fx._r(p["gi"])
    ids = {n["id"] for n in gi["nodes"]}
    kept = next(n for n in gi["nodes"] if n["id"] == MINTED)
    assert kept["properties"]["name"] == "Generation"
    assert [e for e in gi["edges"] if e["type"] == "MENTIONS_PERSON"] == [
        {"type": "MENTIONS_PERSON", "from": "insight:other", "to": MINTED}
    ]
    assert not [e for e in gi["edges"] if e["type"] == "SPOKEN_BY" and e["to"] == MINTED]
    assert all(e["from"] in ids and e["to"] in ids for e in gi["edges"])
    assert len({json.dumps(e, sort_keys=True) for e in gi["edges"]}) == len(gi["edges"])
    # Still published as a person, so 0015's verify says so (its check deletes a Person by name).
    (tmp_path / m15.RECEIPTS_FILE).write_text(
        json.dumps({"kind": "header", "migration": m15.MIGRATION_ID, "names": {"Generation": 1}})
        + "\n",
        encoding="utf-8",
    )
    assert not m15.UnpublishableSpeakerNamesRemovedMigration().verify(_ctx(tmp_path))[0]


def test_an_unmentioned_old_person_still_goes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    _mention(p, "person:gergely-orosz")
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    assert MINTED not in {n["id"] for n in fx._r(p["gi"])["nodes"]}


def test_the_diagnostics_source_follows_the_entry(tmp_path: Path, monkeypatch) -> None:
    """Revision B (B6): the voice's diagnostics no longer claim it introduced itself as the new
    name; a respelling of the same person keeps its source (and so does its roster entry)."""
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    diag = fx._r(p["diag"])
    for v in diag.get("voices") or []:
        if v.get("voice") == "SPEAKER_01":
            v.update(source="self_intro", resolved_name="Generation")
    fx._w(p["diag"], diag)
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    v1 = next(v for v in fx._r(p["diag"])["voices"] if v.get("voice") == "SPEAKER_01")
    assert (v1["resolved_name"], v1["source"]) == ("Steve McConnell", "roster")


def test_a_respelling_keeps_its_source(tmp_path: Path, monkeypatch) -> None:
    entry = {("ep", "SPEAKER_01"): ("Steve Mcconnell", "guest", "Steve McConnell", "guest", "x")}
    monkeypatch.setattr(m33, "VOICES", entry)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Steve Mcconnell", "guest", typed=False)
    meta = fx._r(p["meta"])
    meta["content"]["speakers"][1]["source"] = "self_intro"
    fx._w(p["meta"], meta)
    diag = fx._r(p["diag"])
    for v in diag.get("voices") or []:
        if v.get("voice") == "SPEAKER_01":
            v.update(source="self_intro", resolved_name="Steve Mcconnell")
    fx._w(p["diag"], diag)
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    entry1 = fx._r(p["meta"])["content"]["speakers"][1]
    assert (entry1["name"], entry1["source"]) == ("Steve McConnell", "self_intro")
    v1 = next(v for v in fx._r(p["diag"])["voices"] if v.get("voice") == "SPEAKER_01")
    assert v1["source"] == "self_intro"


@pytest.mark.parametrize(("role", "edge"), [("guest", "GUESTS_ON"), ("host", "HOSTS")])
def test_the_moved_role_edge_is_typed_by_the_voices_role_now(
    tmp_path: Path, monkeypatch, role: str, edge: str
) -> None:
    """The old person HOSTED; the voice is now a guest (or stays a host): the edge says so."""
    entry = {("ep", "SPEAKER_01"): ("Generation", "guest", "Steve McConnell", role, "x")}
    monkeypatch.setattr(m33, "VOICES", entry)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    kg = fx._r(p["kg"])
    for e in kg["edges"]:
        if e["from"] == MINTED and e["type"] == "GUESTS_ON":
            e["type"] = "HOSTS" if role == "guest" else "GUESTS_ON"
    fx._w(p["kg"], kg)
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    steve = {e["type"] for e in fx._r(p["kg"])["edges"] if e["from"] == "person:steve-mcconnell"}
    assert edge in steve and ({"HOSTS", "GUESTS_ON"} - {edge}).isdisjoint(steve)


def test_a_voice_left_unnamed_takes_its_old_persons_edges_with_it(
    tmp_path: Path, monkeypatch
) -> None:
    """A stale person whose voice is unnamed has nobody to move to: its edges go, none dangles."""
    entry = {("ep", "SPEAKER_01"): ("Generation", "guest", None, None, "x")}
    monkeypatch.setattr(m33, "VOICES", entry)
    monkeypatch.setattr(m33.HandRemovedNamesRefusedMigration, "review_names", lambda self: {})
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    ep = m33._Episode(p["meta"], tmp_path)
    ep.voice_names = {"SPEAKER_01": None}
    ep._old_label_by_voice = {"SPEAKER_01": "Generation"}
    ep._stale_names = {MINTED: "Generation"}
    kg = fx._r(p["kg"])
    out = ep._edges_moved_off(kg["edges"], kg["nodes"], {MINTED})
    assert all(MINTED not in (e["from"], e["to"]) for e in out)
    assert ("HOSTS", "person:gergely-orosz") in {(e["type"], e["from"]) for e in out}


def test_a_non_person_node_of_the_same_name_keeps_its_edges(tmp_path: Path) -> None:
    """Revision B (B7): only a Person is the person the voice now carries (the Topic listed
    last, so a name map over every node would take it)."""
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    ep = m33._Episode(p["meta"], tmp_path)
    ep.voice_names = {"SPEAKER_01": "Steve McConnell"}
    ep.voice_roles = {"SPEAKER_01": "guest"}
    ep._old_label_by_voice = {"SPEAKER_01": "Generation"}
    ep._stale_names = {MINTED: "Generation"}
    nodes = [
        {
            "id": "person:steve-mcconnell",
            "type": "Person",
            "properties": {"name": "Steve McConnell"},
        },
        {"id": "topic:steve", "type": "Topic", "properties": {"name": "Steve McConnell"}},
    ]
    out = ep._edges_moved_off(
        [{"type": "GUESTS_ON", "from": MINTED, "to": "podcast:x"}], nodes, {MINTED}
    )
    assert out == [{"type": "GUESTS_ON", "from": "person:steve-mcconnell", "to": "podcast:x"}]


def test_heard_entities_drop_a_summary_union_and_skip_the_unplaced(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    _as_prod(p)
    meta = fx._r(p["meta"])
    rows = meta["content"]["normalized_entities"]
    rows[1]["provenance"] = "both"
    meta["content"]["speakers"].append(
        {"id": "u2", "name": "Nobody Placed", "role": "guest", "placed": False, "voices": []}
    )
    fx._w(p["meta"], meta)
    m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    by = {
        r["canonical"]: r["provenance"] for r in fx._r(p["meta"])["content"]["normalized_entities"]
    }
    assert by["generation"] == "summary"
    assert "nobody placed" not in by and by["steve mcconnell"] == "transcript"


def test_verify_checks_this_migrations_list_and_receipts(tmp_path: Path, monkeypatch) -> None:
    """0031's outcome check takes the list and receipts as arguments; 0033 passes its own."""
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    monkeypatch.setattr(m31.m29, "NAMING_ARC_VOICES", {})
    _review(monkeypatch, tmp_path)
    t31._episode(tmp_path, "Generation", "guest", typed=False)
    mig = m33.HandRemovedNamesRefusedMigration()
    mig.apply(_ctx(tmp_path))
    ok, message = mig.verify(_ctx(tmp_path))
    assert ok and "served_listed=1" in message and "insights_checked=1" in message
    # Its own receipts say it wrote the episode: with the verdict gone, verify must say so.
    d = tmp_path / "_verdicts33" / f"{m33.MIGRATION_ID}.tsv"
    head = [line for line in d.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    d.write_text("\n".join(head) + "\n" + "\t".join(rv.COLUMNS) + "\n", encoding="utf-8")
    ok, message = m33.HandRemovedNamesRefusedMigration().verify(_ctx(tmp_path))
    assert not ok and "written by this migration, and has no verdict" in message
