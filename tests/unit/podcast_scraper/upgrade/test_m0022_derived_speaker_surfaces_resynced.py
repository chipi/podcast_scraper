"""m0022: context.json and the speaker diagnostics follow what the name repairs rewrote.

Prod shape (2026-10-06): after m0012 removed "China Daily" from metadata, segments, KG, GI and
bridge, the episode's context.json still listed it as a guest and its diagnostics still named the
voice — 436 served episodes carried a removed name that way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0022_derived_speaker_surfaces_resynced import (
    DerivedSpeakerSurfacesResyncedMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ORG = "China Daily"
HOST = "Steve Hatherly"
TX = "transcripts/ep.txt"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus(root: Path) -> Dict[str, Path]:
    run = root / "feeds" / "f" / "run_1"
    meta = run / "metadata"
    p = {
        "meta": meta / "ep.metadata.json",
        "gi": meta / "ep.gi.json",
        "kg": meta / "ep.kg.json",
        "ctx": meta / "ep.context.json",
        "seg": run / "transcripts" / "ep.segments.json",
        "diag": run / "transcripts" / "ep.speakers.diagnostics.json",
    }
    _w(
        p["meta"],
        {
            "episode": {"episode_id": "ep", "title": "T"},
            "feed": {"title": "Round Table"},
            "content": {
                "transcript_file_path": TX,
                "speakers": [{"id": "host", "name": HOST, "role": "host", "placed": True}],
            },
        },
    )
    _w(
        p["kg"],
        {
            "nodes": [
                {
                    "id": "person:steve",
                    "type": "Person",
                    "properties": {"name": HOST, "role": "host"},
                },
                {"id": "org:cd", "type": "Organization", "properties": {"name": ORG}},
            ],
            "edges": [],
        },
    )
    _w(p["gi"], {"nodes": [], "edges": []})
    # The repairs already ran: the voice lost its segment label. The two derived files did not.
    _w(
        p["seg"],
        [
            {"speaker": "SPEAKER_04", "speaker_label": HOST, "text": "Welcome."},
            {"speaker": "SPEAKER_03", "text": "In June, China Daily reported on it."},
        ],
    )
    _w(
        p["diag"],
        {
            "summary": {"named": 2, "unresolved": 0, "exposed": {"named": 2}},
            "voices": [
                {
                    "voice": "SPEAKER_04",
                    "resolved_name": HOST,
                    "named": True,
                    "source": "self_intro",
                },
                {
                    "voice": "SPEAKER_03",
                    "resolved_name": ORG,
                    "named": True,
                    "source": "self_intro",
                    "voice_type": "person",
                },
            ],
        },
    )
    _w(
        p["ctx"],
        {
            "episode_id": "ep",
            "basic": {"title": "T", "hosts": [HOST], "guests": [ORG]},
            "people": [ORG, HOST],
            "companies": [ORG],
            "summary": "kept as is",
        },
    )
    return p


def test_registered() -> None:
    assert "0022_derived_speaker_surfaces_resynced" in [m.id for m in get_migrations()]


def test_a_removed_name_leaves_context_and_diagnostics(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    m = DerivedSpeakerSurfacesResyncedMigration()
    ok, _ = m.verify(MigrationContext(corpus_root=tmp_path))
    assert not ok

    m.apply(MigrationContext(corpus_root=tmp_path))

    ctx = _r(p["ctx"])
    assert ctx["basic"]["guests"] == []
    assert ORG not in ctx["people"] and HOST in ctx["people"]
    # Only the speaker fields are rebuilt; the organisation is still a company, the rest as was.
    assert ctx["companies"] == [ORG] and ctx["summary"] == "kept as is"
    voices = {v["voice"]: v for v in _r(p["diag"])["voices"]}
    assert voices["SPEAKER_03"]["resolved_name"] == "SPEAKER_03"
    assert voices["SPEAKER_03"]["named"] is False and voices["SPEAKER_03"]["source"] == "raw"
    assert voices["SPEAKER_03"]["voice_type"] == "unknown"
    assert voices["SPEAKER_04"]["resolved_name"] == HOST  # a correct name is never touched
    summary = _r(p["diag"])["summary"]
    assert (summary["named"], summary["unresolved"], summary["exposed"]["named"]) == (1, 1, 1)
    assert m.verify(MigrationContext(corpus_root=tmp_path))[0]


def test_dry_run_writes_nothing_and_undo_restores(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    before = {k: v.read_bytes() for k, v in p.items()}
    m = DerivedSpeakerSurfacesResyncedMigration()
    m.apply(MigrationContext(corpus_root=tmp_path, dry_run=True))
    assert {k: v.read_bytes() for k, v in p.items()} == before

    m.apply(MigrationContext(corpus_root=tmp_path))
    assert p["ctx"].read_bytes() != before["ctx"]
    restored, refused = undo(tmp_path)
    assert restored == 2 and not refused
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_consistent_episode_is_left_alone(tmp_path: Path) -> None:
    p = _corpus(tmp_path)
    m = DerivedSpeakerSurfacesResyncedMigration()
    m.apply(MigrationContext(corpus_root=tmp_path))
    after = {k: v.read_bytes() for k, v in p.items()}
    m.apply(MigrationContext(corpus_root=tmp_path))
    assert {k: v.read_bytes() for k, v in p.items()} == after
