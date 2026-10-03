"""m0017: a published name renamed to its clean form, its person id moving with it.

Shapes from prod (2026-10-02): "Your Host Luisa Leni" / "Deputy Editor Eilish Hart" / "Planet
Money's Kenny Malone" carry the job or the show with the person; "Professor Hannah Fry" minted
``person:professor-hannah-fry`` beside ``person:hannah-fry`` (7 such splits, ~45 more titled ids).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0017_speaker_names_canonicalised import (
    display_target,
    dry_run_report,
    RECEIPTS_FILE,
    SpeakerNamesCanonicalisedMigration,
    title_only,
    undo,
)

pytestmark = [pytest.mark.unit]

PREFIXED = "Your Host Luisa Leni"
PREFIXED_ID = "person:your-host-luisa-leni"
CLEAN = "Luisa Leni"
CLEAN_ID = "person:luisa-leni"
TITLED = "Professor Hannah Fry"
TITLED_ID = "person:professor-hannah-fry"
PLAIN = "Hannah Fry"
PLAIN_ID = "person:hannah-fry"
LONE = "Dr. Jeff Beck"
LONE_ID = "person:dr-jeff-beck"
LONE_NEW_ID = "person:jeff-beck"
# Entities the KG typed as Person that are not speakers; the stated-name cleaner would mangle them.
NON_SPEAKERS = {
    "person:moores-law": "Moore's Law",
    "person:the-economist": "The Economist",
    "person:cosimo-de-medici": "Cosimo de' Medici",
    "person:lorenzo-de-medici": "Lorenzo de' Medici",
    "person:elon-musks-mother": "Elon Musk's mother",
    "person:alzheimers-disease": "Alzheimer's disease",
    "person:the-flip": "The Flip",
}
GUNTER_ID = "person:dr-jen-gunter"
REAL = "Christopher Guest"
REAL_ID = "person:christopher-guest"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _person(pid: str, name: str, **props: Any) -> Dict[str, Any]:
    return {"id": pid, "type": "Person", "properties": {"name": name, **props}}


def _episode(root: Path) -> Dict[str, Path]:
    run = root / "feeds" / "f" / "run_1"
    meta = run / "metadata"
    paths = {
        "meta": meta / "ep1.metadata.json",
        "kg": meta / "ep1.kg.json",
        "gi": meta / "ep1.gi.json",
        "bridge": meta / "ep1.bridge.json",
        "seg": run / "transcripts" / "ep1.segments.json",
        "adfree": run / "transcripts" / "ep1.adfree.segments.json",
    }
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": "ep1"},
            "feed": {"title": "Some Show"},
            "content": {
                "transcript_file_path": "transcripts/ep1.txt",
                "speakers": [
                    {"id": "host", "name": PREFIXED, "role": "host", "placed": True},
                    {"id": "guest", "name": TITLED, "role": "guest", "placed": True},
                    {"id": "guest_1", "name": REAL, "role": "guest", "placed": True},
                ],
                "detected_hosts": [PREFIXED],
                "detected_guests": [TITLED, REAL],
            },
        },
    )
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": PREFIXED},
        {"start": 5.0, "speaker": "SPEAKER_01", "speaker_label": TITLED},
        {"start": 9.0, "speaker": "SPEAKER_02", "speaker_label": REAL},
    ]
    _w(paths["seg"], segs)
    _w(paths["adfree"], segs)
    _w(
        paths["kg"],
        {
            "nodes": [
                _person(TITLED_ID, TITLED, role="mentioned"),
                _person(PLAIN_ID, PLAIN, role="guest", bio="kept"),
                _person(PREFIXED_ID, PREFIXED, role="host"),
                _person(LONE_ID, LONE, role="mentioned"),
                _person(REAL_ID, REAL, role="guest"),
                _person(GUNTER_ID, "Dr. Jen Gunter", role="mentioned"),
                *[_person(i, n, role="mentioned") for i, n in NON_SPEAKERS.items()],
            ],
            "edges": [
                *[{"type": "MENTIONS", "from": "episode:ep1", "to": i} for i in NON_SPEAKERS],
                {"type": "GUESTS_ON", "from": TITLED_ID, "to": "podcast:f"},
                {"type": "GUESTS_ON", "from": PLAIN_ID, "to": "podcast:f"},
                {"type": "HOSTS", "from": PREFIXED_ID, "to": "podcast:f"},
                {"type": "MENTIONS", "from": "episode:ep1", "to": LONE_ID},
            ],
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                _person(TITLED_ID, TITLED),
                _person(PLAIN_ID, PLAIN, role="guest"),
                _person(PREFIXED_ID, PREFIXED),
                _person(REAL_ID, REAL),
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": PREFIXED_ID}},
                {
                    "id": "quote:2",
                    "type": "Quote",
                    "properties": {"speaker_id": TITLED_ID, "speaker_name": TITLED},
                },
                {
                    "id": "quote:3",
                    "type": "Quote",
                    "properties": {"speaker_id": PREFIXED_ID, "speaker_name": PREFIXED},
                },
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "speaker": PREFIXED,
                        "tier": 3,
                        "surfaceable": True,
                        "routing_tag": "surface",
                    },
                },
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "quote:1", "to": PREFIXED_ID},
                {"type": "SPOKEN_BY", "from": "quote:2", "to": TITLED_ID},
                {"type": "SPOKEN_BY", "from": "quote:3", "to": PREFIXED_ID},
            ],
        },
    )
    _w(
        paths["bridge"],
        {
            "episode_id": "ep1",
            "identities": [
                {
                    "id": TITLED_ID,
                    "type": "person",
                    "display_name": TITLED,
                    "aliases": [],
                    "sources": {"gi": True, "kg": False},
                },
                {
                    "id": PLAIN_ID,
                    "type": "person",
                    "display_name": PLAIN,
                    "aliases": [],
                    "sources": {"gi": False, "kg": True},
                },
                {
                    "id": PREFIXED_ID,
                    "type": "person",
                    "display_name": PREFIXED,
                    "aliases": [PREFIXED],
                    "sources": {"gi": True, "kg": True},
                },
                {
                    "id": REAL_ID,
                    "type": "person",
                    "display_name": REAL,
                    "aliases": [],
                    "sources": {"gi": True, "kg": True},
                },
            ],
        },
    )
    return paths


@pytest.fixture
def corpus(tmp_path: Path) -> Dict[str, Path]:
    paths = _episode(tmp_path)
    paths["root"] = tmp_path
    return paths


def _snapshot(corpus: Dict[str, Path]) -> Dict[str, bytes]:
    return {k: p.read_bytes() for k, p in corpus.items() if k != "root"}


def _run(root: Path, dry: bool = False):
    return SpeakerNamesCanonicalisedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry)
    )


def _ids(payload: Dict[str, Any]) -> List[str]:
    return [n["id"] for n in payload["nodes"]]


def test_prefix_rename_on_all_five_surfaces(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    content = _r(corpus["meta"])["content"]
    assert content["speakers"][0]["name"] == CLEAN
    assert content["detected_hosts"] == [CLEAN]
    for key in ("seg", "adfree"):
        assert _r(corpus[key])[0]["speaker_label"] == CLEAN
    kg = {n["id"]: n for n in _r(corpus["kg"])["nodes"]}
    assert PREFIXED_ID not in kg and kg[CLEAN_ID]["properties"] == {"name": CLEAN, "role": "host"}
    assert {"type": "HOSTS", "from": CLEAN_ID, "to": "podcast:f"} in _r(corpus["kg"])["edges"]
    gi = _r(corpus["gi"])
    nodes = {n["id"]: n for n in gi["nodes"]}
    assert PREFIXED_ID not in nodes and nodes[CLEAN_ID]["properties"]["name"] == CLEAN
    assert nodes["quote:1"]["properties"]["speaker_id"] == CLEAN_ID
    assert nodes["quote:3"]["properties"] == {"speaker_id": CLEAN_ID, "speaker_name": CLEAN}
    insight = nodes["insight:1"]["properties"]
    assert insight["speaker"] == CLEAN
    assert insight["surfaceable"] is True and insight["routing_tag"] == "surface"
    assert {e["to"] for e in gi["edges"] if e["from"] in ("quote:1", "quote:3")} == {CLEAN_ID}
    bridge = {i["id"]: i for i in _r(corpus["bridge"])["identities"]}
    assert PREFIXED_ID not in bridge
    assert bridge[CLEAN_ID]["display_name"] == CLEAN and bridge[CLEAN_ID]["aliases"] == [CLEAN]


def test_titled_id_merges_into_the_plain_id_already_in_the_episode(
    corpus: Dict[str, Path],
) -> None:
    _run(corpus["root"])
    kg = _r(corpus["kg"])
    assert _ids(kg).count(PLAIN_ID) == 1 and TITLED_ID not in _ids(kg)
    survivor = next(n for n in kg["nodes"] if n["id"] == PLAIN_ID)
    # The EXISTING node's properties win; a stated role beats the `mentioned` it folded in.
    assert survivor["properties"] == {"name": PLAIN, "role": "guest", "bio": "kept"}
    guests_on = [e for e in kg["edges"] if e["type"] == "GUESTS_ON"]
    assert guests_on == [{"type": "GUESTS_ON", "from": PLAIN_ID, "to": "podcast:f"}]
    gi = _r(corpus["gi"])
    assert _ids(gi).count(PLAIN_ID) == 1 and TITLED_ID not in _ids(gi)
    quote = next(n for n in gi["nodes"] if n["id"] == "quote:2")
    assert quote["properties"]["speaker_id"] == PLAIN_ID
    assert {"type": "SPOKEN_BY", "from": "quote:2", "to": PLAIN_ID} in gi["edges"]
    identities = _r(corpus["bridge"])["identities"]
    assert [i["id"] for i in identities].count(PLAIN_ID) == 1
    merged = next(i for i in identities if i["id"] == PLAIN_ID)
    assert merged["display_name"] == PLAIN and merged["sources"] == {"gi": True, "kg": True}


def test_titled_display_name_is_unchanged_everywhere(corpus: Dict[str, Path]) -> None:
    """A title leaves the id, not the display: "Dr. Adam Rodman" is published as stated."""
    _run(corpus["root"])
    content = _r(corpus["meta"])["content"]
    assert content["speakers"][1]["name"] == TITLED
    assert _r(corpus["seg"])[1]["speaker_label"] == TITLED
    gi = _r(corpus["gi"])
    quote = next(n for n in gi["nodes"] if n["id"] == "quote:2")
    assert quote["properties"]["speaker_name"] == TITLED


def test_titled_id_without_a_plain_twin_is_reminted(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    kg = _r(corpus["kg"])
    assert LONE_ID not in _ids(kg)
    node = next(n for n in kg["nodes"] if n["id"] == LONE_NEW_ID)
    assert node["properties"] == {"name": LONE, "role": "mentioned"}
    assert {"type": "MENTIONS", "from": "episode:ep1", "to": LONE_NEW_ID} in kg["edges"]


def test_receipt_freezes_the_maps_and_details_list_the_changed_ids(
    corpus: Dict[str, Path],
) -> None:
    result = _run(corpus["root"])
    header = json.loads((corpus["root"] / RECEIPTS_FILE).read_text().splitlines()[0])
    assert header["kind"] == "header"
    assert header["names"] == {PREFIXED: CLEAN}
    assert header["ids"] == {
        PREFIXED_ID: CLEAN_ID,
        TITLED_ID: PLAIN_ID,
        LONE_ID: LONE_NEW_ID,
        GUNTER_ID: "person:jen-gunter",
    }
    assert result.details["person_ids_changed"] == header["ids"]


def test_second_apply_is_a_no_op(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    after = _snapshot(corpus)
    receipts = (corpus["root"] / RECEIPTS_FILE).read_bytes()
    result = _run(corpus["root"])
    assert result.details["episodes"] == 0 and result.details["files_written"] == 0
    assert _snapshot(corpus) == after
    assert (corpus["root"] / RECEIPTS_FILE).read_bytes() == receipts


def test_dry_run_writes_nothing(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    result = _run(corpus["root"], dry=True)
    assert result.details["episodes"] == 1 and result.details["person_ids_changed"]
    assert _snapshot(corpus) == before
    assert not (corpus["root"] / RECEIPTS_FILE).exists()
    assert not (corpus["root"] / ".podcast_scraper").exists()


def test_verify_fails_before_apply_and_passes_after(corpus: Dict[str, Path]) -> None:
    ctx = MigrationContext(corpus_root=corpus["root"])
    ok, _msg = SpeakerNamesCanonicalisedMigration().verify(ctx)
    assert not ok
    _run(corpus["root"])
    ok, msg = SpeakerNamesCanonicalisedMigration().verify(ctx)
    assert ok, msg


def test_verify_judges_against_the_frozen_map(corpus: Dict[str, Path]) -> None:
    """An old name that comes back is caught even though a live plan would call it clean."""
    _run(corpus["root"])
    meta = _r(corpus["meta"])
    meta["content"]["detected_hosts"] = [PREFIXED]
    _w(corpus["meta"], meta)
    ok, msg = SpeakerNamesCanonicalisedMigration().verify(
        MigrationContext(corpus_root=corpus["root"])
    )
    assert not ok and "ep1.metadata.json" in msg


def test_undo_restores_byte_for_byte(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    _run(corpus["root"])
    restored, refused = undo(corpus["root"])
    assert refused == [] and restored == 6
    assert _snapshot(corpus) == before


def test_a_real_name_that_only_looks_titled_is_not_renamed(corpus: Dict[str, Path]) -> None:
    """ "Guest" is a surname, "Lord" and "Dr." lead a two-word name that has no other identifier.

    The pipeline's own publish gate accepts all three and ``person_identity_name`` keeps a title
    unless at least two words remain, so neither the display nor the id moves.
    """
    for name in (REAL, "Lord Kinnock", "Dr. Dre", "Hannah Fry"):
        assert display_target(name) is None
    _run(corpus["root"])
    assert REAL_ID in _ids(_r(corpus["kg"])) and REAL_ID in _ids(_r(corpus["gi"]))
    assert _r(corpus["meta"])["content"]["speakers"][2]["name"] == REAL
    assert _r(corpus["seg"])[2]["speaker_label"] == REAL
    header = json.loads((corpus["root"] / RECEIPTS_FILE).read_text().splitlines()[0])
    assert REAL_ID not in header["ids"] and REAL not in header["names"]


@pytest.mark.parametrize(
    "name,target",
    [
        ("Your Host Luisa Leni", "Luisa Leni"),
        ("Deputy Editor Eilish Hart", "Eilish Hart"),
        ("Planet Money's Kenny Malone", "Kenny Malone"),
        ("Celestin Ntawirema CEO", "Celestin Ntawirema"),
        ("Professor Hannah Fry", None),
        ("Dr. Adam Rodman", None),
        ("Sir Christopher Hatton", None),
    ],
)
def test_display_target(name: str, target: Any) -> None:
    assert display_target(name) == target


def test_non_speakers_are_never_renamed_or_reminted(corpus: Dict[str, Path]) -> None:
    """Prod dry run: "Moore's Law" -> "Law", both de' Medici -> "Medici" (two people merged)."""
    before = _r(corpus["kg"])
    _run(corpus["root"])
    kg = {n["id"]: n for n in _r(corpus["kg"])["nodes"]}
    for nid, name in NON_SPEAKERS.items():
        assert kg[nid]["properties"]["name"] == name
    assert "person:medici" not in kg and "person:law" not in kg and "person:mother" not in kg
    edges = _r(corpus["kg"])["edges"]
    assert [e for e in edges if e["type"] == "MENTIONS" and e["to"] in NON_SPEAKERS] == [
        e for e in before["edges"] if e["type"] == "MENTIONS" and e["to"] in NON_SPEAKERS
    ]
    header = json.loads((corpus["root"] / RECEIPTS_FILE).read_text().splitlines()[0])
    assert not set(NON_SPEAKERS) & set(header["ids"])
    assert not set(NON_SPEAKERS.values()) & set(header["names"])


def test_a_mentioned_person_still_loses_only_a_title_from_the_id(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    kg = {n["id"]: n for n in _r(corpus["kg"])["nodes"]}
    assert GUNTER_ID not in kg
    assert kg["person:jen-gunter"]["properties"]["name"] == "Dr. Jen Gunter"


def test_two_different_people_do_not_merge(tmp_path: Path) -> None:
    """The de' Medici pair, even when both are published as speakers, stays two people."""
    paths = _episode(tmp_path)
    meta = _r(paths["meta"])
    meta["content"]["speakers"] = [
        {"id": "host", "name": "Cosimo de' Medici", "role": "host", "placed": True},
        {"id": "guest", "name": "Lorenzo de' Medici", "role": "guest", "placed": True},
    ]
    _w(paths["meta"], meta)
    kg = _r(paths["kg"])
    for node in kg["nodes"]:
        if node["id"] in ("person:cosimo-de-medici", "person:lorenzo-de-medici"):
            node["properties"]["role"] = "host" if "cosimo" in node["id"] else "guest"
    _w(paths["kg"], kg)
    _run(tmp_path)
    ids = _ids(_r(paths["kg"]))
    assert "person:cosimo-de-medici" in ids and "person:lorenzo-de-medici" in ids
    assert "person:medici" not in ids
    speakers = [s["name"] for s in _r(paths["meta"])["content"]["speakers"]]
    assert speakers == ["Cosimo de' Medici", "Lorenzo de' Medici"]


@pytest.mark.parametrize(
    "name", ["Moore's Law", "The Economist", "Cosimo de' Medici", "Elon Musk's mother", "The Flip"]
)
def test_the_stated_name_cleaner_never_reduces_a_name_to_one_word(name: str) -> None:
    assert display_target(name) is None and not title_only(name)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Dr. Jen Gunter", True),
        ("Sir John Bell", True),
        ("Professor Peng Zhongchao", True),
        ("Lord Kinnock", False),
        ("Dr. Dre", False),
        ("Hannah Fry", False),
        ("Cosimo de' Medici", False),
    ],
)
def test_title_only(name: str, expected: bool) -> None:
    assert title_only(name) is expected


def test_dry_run_report_is_read_only_and_complete(corpus: Dict[str, Path]) -> None:
    before = _snapshot(corpus)
    report = dry_run_report(corpus["root"])
    assert report["names"] == {PREFIXED: CLEAN}
    assert report["ids"][TITLED_ID] == PLAIN_ID and report["conflicts"] == []
    assert report["episodes"] == 1
    assert _snapshot(corpus) == before and not (corpus["root"] / RECEIPTS_FILE).exists()
