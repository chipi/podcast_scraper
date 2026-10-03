"""m0019: a published name the gate refuses since m0015 ran is removed from every surface.

Prod shape (Freakonomics, 2026-10-03): "Pulitzer Prize-winning" placed as a guest on a voice,
with the real guest Jennifer Egan unplaced beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0019_descriptor_speaker_names_removed import (
    DescriptorSpeakerNamesRemovedMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

JUNK = "Pulitzer Prize-winning"
JUNK_ID = "person:pulitzer-prize-winning"
HOST = "Stephen J. Dubner"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus(root: Path, *, junk: bool = True) -> Dict[str, Path]:
    meta = root / "feeds" / "f" / "run_1" / "metadata"
    paths = {"meta": meta / "ep.metadata.json", "gi": meta / "ep.gi.json"}
    speakers = [{"id": "host", "name": HOST, "role": "host", "placed": True}]
    if junk:
        speakers.append({"id": "guest", "name": JUNK, "role": "guest", "placed": True})
    speakers.append({"id": "u1", "name": "Jennifer Egan", "role": "guest", "placed": False})
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": "ep"},
            "feed": {"title": "Freakonomics Radio"},
            "content": {"speakers": speakers},
        },
    )
    nodes = [{"id": "person:stephen-j-dubner", "type": "Person", "properties": {"name": HOST}}]
    edges = []
    if junk:
        nodes += [
            {"id": JUNK_ID, "type": "Person", "properties": {"name": JUNK}},
            {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": JUNK_ID}},
        ]
        edges.append({"type": "SPOKEN_BY", "from": "quote:1", "to": JUNK_ID})
    _w(paths["gi"], {"nodes": nodes, "edges": edges})
    paths["root"] = root
    return paths


def _run(root: Path, dry: bool = False):
    return DescriptorSpeakerNamesRemovedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry)
    )


def test_the_descriptor_leaves_roster_and_gi_and_the_people_stay(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    result = _run(tmp_path)
    assert result.details["names"] == {JUNK: 1}
    assert len(result.details["episodes"]) == 1
    names = [s["name"] for s in _r(c["meta"])["content"]["speakers"]]
    assert JUNK not in names and HOST in names and "Jennifer Egan" in names
    gi = {n["id"]: n for n in _r(c["gi"])["nodes"]}
    assert JUNK_ID not in gi and "person:stephen-j-dubner" in gi
    assert gi["quote:1"]["properties"]["speaker_id"] is None
    ok, msg = DescriptorSpeakerNamesRemovedMigration().verify(
        MigrationContext(corpus_root=tmp_path)
    )
    assert ok, msg


def test_dry_run_writes_nothing_and_second_run_is_a_no_op(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    before = c["meta"].read_bytes()
    assert len(_run(tmp_path, dry=True).details["episodes"]) == 1
    assert c["meta"].read_bytes() == before
    _run(tmp_path)
    after = c["meta"].read_bytes()
    assert _run(tmp_path).details["episodes"] == []
    assert c["meta"].read_bytes() == after


def test_a_clean_corpus_is_a_no_op(tmp_path: Path) -> None:
    c = _corpus(tmp_path, junk=False)
    before = c["meta"].read_bytes()
    assert _run(tmp_path).details["episodes"] == []
    assert c["meta"].read_bytes() == before
    ok, _ = DescriptorSpeakerNamesRemovedMigration().verify(MigrationContext(corpus_root=tmp_path))
    assert ok


def test_undo_restores(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    before = c["meta"].read_bytes()
    _run(tmp_path)
    restored, refused = undo(tmp_path)
    assert refused == [] and restored >= 1
    assert c["meta"].read_bytes() == before


def test_registered_after_0018() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0019_descriptor_speaker_names_removed") == (
        ids.index("0018_org_speakers_removed_residue") + 1
    )
