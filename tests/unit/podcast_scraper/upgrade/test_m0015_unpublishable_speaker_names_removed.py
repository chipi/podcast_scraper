"""m0015: a published name today's gate refuses leaves every speaker surface; a cleanable one stays.

Shapes from prod (2026-10-02): "Host" seated as a speaker, the show's own name on a voice
("Machine Learning Street"), and "Your Host Luisa Leni" — a person behind a role prefix, which
m0017 renames instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0015_unpublishable_speaker_names_removed import (
    refused,
    rename_target,
    undo,
    UnpublishableSpeakerNamesRemovedMigration,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

JUNK = "Host"
JUNK_ID = "person:host"
REAL = "Brian Winter"
PREFIXED = "Your Host Luisa Leni"
SHOW = "Peter Attia"
FEED = "The Peter Attia Drive"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode(root: Path) -> Dict[str, Path]:
    run = root / "feeds" / "mlst" / "run_1"
    meta = run / "metadata"
    stem = "ep1"
    paths = {
        "meta": meta / f"{stem}.metadata.json",
        "kg": meta / f"{stem}.kg.json",
        "gi": meta / f"{stem}.gi.json",
        "bridge": meta / f"{stem}.bridge.json",
        "seg": run / "transcripts" / f"{stem}.segments.json",
        "adfree": run / "transcripts" / f"{stem}.adfree.segments.json",
    }
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": "ep1"},
            "feed": {"title": FEED},
            "content": {
                "transcript_file_path": f"transcripts/{stem}.txt",
                "speakers": [
                    {"id": "host", "name": JUNK, "role": "host", "placed": True},
                    {"id": "guest", "name": REAL, "role": "guest", "placed": True},
                    {"id": "show", "name": SHOW, "role": "host", "placed": True},
                    {"id": "co", "name": PREFIXED, "role": "host", "placed": True},
                ],
                "detected_hosts": [JUNK, SHOW, PREFIXED],
                "detected_guests": [REAL],
            },
        },
    )
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": JUNK, "speaker_role": "host"},
        {"start": 9.0, "speaker": "SPEAKER_01", "speaker_label": REAL, "speaker_role": "guest"},
        {"start": 19.0, "speaker": "SPEAKER_02", "speaker_label": SHOW, "speaker_role": "host"},
        {"start": 29.0, "speaker": "SPEAKER_03", "speaker_label": PREFIXED, "speaker_role": "host"},
    ]
    _w(paths["seg"], segs)
    _w(paths["adfree"], segs)
    _w(
        paths["kg"],
        {
            "nodes": [
                {"id": JUNK_ID, "type": "Person", "properties": {"name": JUNK, "role": "host"}},
                {"id": "person:brian-winter", "type": "Person", "properties": {"name": REAL}},
            ],
            "edges": [{"type": "HOSTS", "from": JUNK_ID, "to": "podcast:mlst"}],
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                {"id": JUNK_ID, "type": "Person", "properties": {"name": JUNK}},
                {"id": "person:brian-winter", "type": "Person", "properties": {"name": REAL}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": JUNK_ID}},
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "speaker": JUNK,
                        "tier": 3,
                        "grounded": True,
                        "surfaceable": True,
                        "routing_tag": "surface",
                    },
                },
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "quote:1", "to": JUNK_ID},
                {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
            ],
        },
    )
    _w(
        paths["bridge"],
        {
            "identities": [
                {"id": JUNK_ID, "type": "person", "display_name": JUNK},
                {"id": "person:brian-winter", "type": "person", "display_name": REAL},
            ]
        },
    )
    return paths


@pytest.fixture
def corpus(tmp_path: Path) -> Dict[str, Path]:
    paths = _episode(tmp_path)
    paths["root"] = tmp_path
    return paths


def _run(root: Path, dry: bool = False):
    return UnpublishableSpeakerNamesRemovedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry)
    )


def test_the_predicate_matches_todays_gate() -> None:
    assert refused(JUNK, FEED) and refused("OK", None) and refused("Roblox CEO", None)
    # The eponymous host is not "the show" (prod dry run: 43 episodes of The Peter Attia Drive).
    assert not refused(SHOW, FEED)
    assert not refused(REAL, FEED) and not refused("Christopher Guest", None)
    # Real names whose parts are ordinary words are not removed (prod dry run, 2026-10-03).
    for real in ("Ethan He", "Henry He", "Michael I. Jordan"):
        assert not refused(real, None), real
    assert rename_target("Ben Fritz's") == "Ben Fritz"
    # Kept by hand: a real person the abbreviation rule refuses (prod dry run, 2026-10-03).
    assert not refused("RJ", None) and refused("GE", None)
    # Removed by hand: junk the safe rules miss (operator-approved list, 2026-10-03).
    for junk in ("Before Gene", "As Colin", "Super Willing To Be", "Moral", "Generation"):
        assert refused(junk, None), junk
    assert not refused(PREFIXED, FEED) and rename_target(PREFIXED) == "Luisa Leni"


def test_refused_names_leave_every_surface(corpus: Dict[str, Path]) -> None:
    result = _run(corpus["root"])
    assert result.details["names"] == {JUNK: 1}
    content = _r(corpus["meta"])["content"]
    assert [s["name"] for s in content["speakers"]] == [REAL, SHOW, PREFIXED]
    assert content["detected_hosts"] == [SHOW, PREFIXED]
    for key in ("seg", "adfree"):
        rows = _r(corpus[key])
        assert "speaker_label" not in rows[0] and rows[0]["voice_type"] == "unknown"
        assert rows[2]["speaker_label"] == SHOW
    kg = _r(corpus["kg"])
    assert JUNK_ID not in {n["id"] for n in kg["nodes"]} and kg["edges"] == []
    gi = {n["id"]: n for n in _r(corpus["gi"])["nodes"]}
    assert JUNK_ID not in gi and gi["quote:1"]["properties"]["speaker_id"] is None
    ins = gi["insight:1"]["properties"]
    assert "speaker" not in ins and ins["surfaceable"] is False
    assert [i["id"] for i in _r(corpus["bridge"])["identities"]] == ["person:brian-winter"]


def test_a_person_behind_a_prefix_is_left_for_the_rename(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    rows = _r(corpus["seg"])
    assert rows[3]["speaker_label"] == PREFIXED and rows[1]["speaker_label"] == REAL


def test_dry_run_writes_nothing(corpus: Dict[str, Path]) -> None:
    before = {k: p.read_bytes() for k, p in corpus.items() if k != "root"}
    result = _run(corpus["root"], dry=True)
    assert result.details["totals"]["roster_entries"] == 1
    assert {k: p.read_bytes() for k, p in corpus.items() if k != "root"} == before


def test_second_run_is_a_no_op(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    after = {k: p.read_bytes() for k, p in corpus.items() if k != "root"}
    assert _run(corpus["root"]).details["episodes"] == 0
    assert {k: p.read_bytes() for k, p in corpus.items() if k != "root"} == after


def test_verify_against_the_frozen_set(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    ok, msg = UnpublishableSpeakerNamesRemovedMigration().verify(
        MigrationContext(corpus_root=corpus["root"])
    )
    assert ok, msg


def test_undo_restores_byte_for_byte_and_verify_then_fails(corpus: Dict[str, Path]) -> None:
    before = {k: p.read_bytes() for k, p in corpus.items() if k != "root"}
    _run(corpus["root"])
    restored, refused_files = undo(corpus["root"])
    assert refused_files == [] and restored == 6
    assert {k: p.read_bytes() for k, p in corpus.items() if k != "root"} == before
    ok, _msg = UnpublishableSpeakerNamesRemovedMigration().verify(
        MigrationContext(corpus_root=corpus["root"])
    )
    assert not ok


def test_registered_after_0014() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0015_unpublishable_speaker_names_removed") == (
        ids.index("0014_eponymous_hosts_restored") + 1
    )
