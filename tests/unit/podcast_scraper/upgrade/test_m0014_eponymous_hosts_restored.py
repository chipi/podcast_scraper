"""m0014: the host a show is named after gets their seat back, from their own self-introduction.

Prod shape (2026-10-01), The Peter Attia Drive: diagnostics say SPEAKER_00 introduced itself as
"Peter Attia" (source self_intro); the roster kept only an unplaced "Peter Attia" from the feed; KG
had person:peter-attia `mentioned`; GI attributed his quotes to an episode-scoped
"Unidentified speaker"; the bridge carried that scoped identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0014_eponymous_hosts_restored import (
    EponymousHostsRestoredMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

SHOW = "The Peter Attia Drive"
HOST = "Peter Attia"
PID = "person:peter-attia"
EP = "ep-1"
SCOPED = "person:unresolved-peter-attia-ep-1"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode(root: Path, label: str = HOST, source: str = "self_intro", show: str = SHOW) -> Dict:
    run = root / "feeds" / "attia" / "run_1"
    meta = run / "metadata"
    paths = {
        "meta": meta / "ep.metadata.json",
        "kg": meta / "ep.kg.json",
        "gi": meta / "ep.gi.json",
        "bridge": meta / "ep.bridge.json",
        "diag": run / "transcripts" / "ep.speakers.diagnostics.json",
    }
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": EP},
            "feed": {"title": show},
            "content": {
                "transcript_file_path": "transcripts/ep.txt",
                "speakers": [
                    {"id": "guest", "name": "Rhonda Patrick", "role": "guest", "placed": True},
                    {"id": "unplaced_1", "name": label, "role": "host", "placed": False},
                ],
            },
        },
    )
    _w(
        paths["diag"],
        {
            "voices": [
                {"voice": "SPEAKER_00", "resolved_name": label, "role": "host", "source": source},
                {"voice": "SPEAKER_01", "resolved_name": "Rhonda Patrick", "source": "x"},
            ]
        },
    )
    _w(
        paths["kg"],
        {
            "nodes": [
                {"id": "episode:ep-1", "type": "Episode", "properties": {}},
                {"id": "podcast:attia", "type": "Podcast", "properties": {}},
                {"id": PID, "type": "Person", "properties": {"name": label, "role": "mentioned"}},
            ],
            "edges": [],
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                {"id": SCOPED, "type": "Person", "properties": {"name": "Unidentified speaker"}},
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {"speaker_id": SCOPED, "speaker_name": "Unidentified speaker"},
                },
            ],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:1", "to": SCOPED}],
        },
    )
    _w(paths["bridge"], {"identities": [{"id": SCOPED, "type": "person", "display_name": "x"}]})
    return paths


def _run(root: Path, dry: bool = False):
    return EponymousHostsRestoredMigration().apply(MigrationContext(corpus_root=root, dry_run=dry))


def test_every_surface_gets_the_host_back(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    assert _run(tmp_path).details["names"] == [HOST]
    speakers = _r(p["meta"])["content"]["speakers"]
    assert speakers[0] == {
        "id": "host",
        "name": HOST,
        "role": "host",
        "placed": True,
        "voices": ["SPEAKER_00"],
        "source": "self_intro",
    }
    assert not [s for s in speakers if s.get("placed") is False and s["name"] == HOST]
    kg = _r(p["kg"])
    assert next(n for n in kg["nodes"] if n["id"] == PID)["properties"]["role"] == "host"
    assert {"from": PID, "to": "podcast:attia", "type": "HOSTS", "properties": {}} in kg["edges"]
    gi = _r(p["gi"])
    ids = {n["id"]: n for n in gi["nodes"]}
    assert SCOPED not in ids and ids[PID]["properties"]["name"] == HOST
    quote = ids["quote:1"]["properties"]
    assert quote["speaker_id"] == PID and "speaker_name" not in quote
    assert _r(p["bridge"])["identities"][0]["id"] == PID


def test_a_show_name_nobody_said_is_left_alone(tmp_path: Path) -> None:
    p = _episode(
        tmp_path,
        label="Machine Learning Street",
        source="known_hosts",
        show="Machine Learning Street Talk (MLST)",
    )
    before = {k: v.read_bytes() for k, v in p.items()}
    assert _run(tmp_path).details["episodes"] == 0
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_dry_run_writes_nothing_and_second_run_is_a_no_op(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    assert _run(tmp_path, dry=True).details["episodes"] == 1
    assert {k: v.read_bytes() for k, v in p.items()} == before
    _run(tmp_path)
    after = {k: v.read_bytes() for k, v in p.items()}
    assert _run(tmp_path).details["episodes"] == 0
    assert {k: v.read_bytes() for k, v in p.items()} == after


def test_verify_and_undo(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    m = EponymousHostsRestoredMigration()
    ctx = MigrationContext(corpus_root=tmp_path)
    assert not m.verify(ctx)[0]
    _run(tmp_path)
    assert m.verify(ctx)[0]
    restored, refused = undo(tmp_path)
    assert refused == [] and restored == 4
    assert {k: v.read_bytes() for k, v in p.items()} == before
    assert not m.verify(ctx)[0]


def test_registered_last() -> None:
    assert [m.id for m in get_migrations()][-1] == "0014_eponymous_hosts_restored"
