"""m0012: a name the corpus's extraction calls an organisation leaves every speaker surface.

Shape from prod (2026-10-01): "Americas Online" seated as host on Latin America in Focus, the host
voice labelled with it in the segments, a host Person in KG, the speaker of quotes in GI, an
identity in the bridge — while extraction calls it an Organization 24 times and a Person never.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0012_org_speakers_removed import (
    OrgSpeakersRemovedMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ORG = "Americas Online"
ORG_ID = "person:americas-online"
HOST = "Brian Winter"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _voter(root: Path, i: int) -> None:
    """An unrelated served episode whose EXTRACTION calls the org an Organization."""
    meta = root / "feeds" / "other" / f"run_v{i}" / "metadata"
    _w(meta / f"v{i}.metadata.json", {"episode": {"episode_id": f"v{i}"}, "feed": {"title": "X"}})
    props = {"name": ORG, "role": "mentioned"}
    node = {"id": f"org:americas-online-{i}", "type": "Organization", "properties": props}
    _w(meta / f"v{i}.kg.json", {"nodes": [node], "edges": []})


def _episode(root: Path) -> Dict[str, Path]:
    run = root / "feeds" / "lai" / "run_1"
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
            "feed": {"title": "Latin America in Focus"},
            "content": {
                "transcript_file_path": f"transcripts/{stem}.txt",
                "speakers": [
                    {"id": "host", "name": ORG, "role": "host", "placed": True},
                    {"id": "guest", "name": HOST, "role": "guest", "placed": True},
                ],
                "detected_hosts": [ORG],
                "detected_guests": [HOST],
            },
        },
    )
    segs: List[Dict[str, Any]] = [
        {"start": 0.0, "speaker": "SPEAKER_00", "speaker_label": ORG, "speaker_role": "host"},
        {"start": 9.0, "speaker": "SPEAKER_01", "speaker_label": HOST, "speaker_role": "guest"},
    ]
    _w(paths["seg"], segs)
    _w(paths["adfree"], segs)
    _w(
        paths["kg"],
        {
            "nodes": [
                {"id": ORG_ID, "type": "Person", "properties": {"name": ORG, "role": "host"}},
                {"id": "person:brian-winter", "type": "Person", "properties": {"name": HOST}},
            ],
            "edges": [{"type": "HOSTS", "from": ORG_ID, "to": "podcast:lai"}],
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                {"id": ORG_ID, "type": "Person", "properties": {"name": ORG}},
                {"id": "person:brian-winter", "type": "Person", "properties": {"name": HOST}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": ORG_ID}},
                {
                    "id": "quote:2",
                    "type": "Quote",
                    "properties": {"speaker_id": "person:brian-winter"},
                },
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {
                        "speaker": ORG,
                        "tier": 3,
                        "grounded": True,
                        "surfaceable": True,
                        "routing_tag": "surface",
                    },
                },
                {
                    "id": "insight:2",
                    "type": "Insight",
                    "properties": {"speaker": HOST, "tier": 3, "routing_tag": "surface"},
                },
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": "quote:1", "to": ORG_ID},
                {"type": "SPOKEN_BY", "from": "quote:2", "to": "person:brian-winter"},
                {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"},
                {"type": "SUPPORTED_BY", "from": "insight:2", "to": "quote:2"},
            ],
        },
    )
    _w(
        paths["bridge"],
        {
            "identities": [
                {"id": ORG_ID, "type": "person", "display_name": ORG},
                {"id": "person:brian-winter", "type": "person", "display_name": HOST},
            ]
        },
    )
    return paths


@pytest.fixture
def corpus(tmp_path: Path) -> Dict[str, Path]:
    for i in range(3):
        _voter(tmp_path, i)
    paths = _episode(tmp_path)
    paths["root"] = tmp_path
    return paths


def _run(root: Path, dry: bool = False):
    return OrgSpeakersRemovedMigration().apply(MigrationContext(corpus_root=root, dry_run=dry))


def test_every_surface_stops_naming_the_org(corpus: Dict[str, Path]) -> None:
    result = _run(corpus["root"])
    assert result.details["episodes"] == 1
    content = _r(corpus["meta"])["content"]
    assert [s["name"] for s in content["speakers"]] == [HOST]
    assert content["detected_hosts"] == []
    for key in ("seg", "adfree"):
        org_voice = _r(corpus[key])[0]
        assert "speaker_label" not in org_voice and org_voice["voice_type"] == "unknown"
        assert org_voice["speaker"] == "SPEAKER_00"  # the voice is real; only its name is gone
    kg = _r(corpus["kg"])
    assert ORG_ID not in {n["id"] for n in kg["nodes"]} and kg["edges"] == []
    gi = _r(corpus["gi"])
    nodes = {n["id"]: n for n in gi["nodes"]}
    assert ORG_ID not in nodes
    assert nodes["quote:1"]["properties"]["speaker_id"] is None
    assert nodes["quote:1"]["properties"]["speaker_voice_type"] == "unknown"
    ins = nodes["insight:1"]["properties"]
    assert "speaker" not in ins and ins["surfaceable"] is False and ins["routing_tag"] == "connect"
    assert {e["to"] for e in gi["edges"] if e["type"] == "SPOKEN_BY"} == {"person:brian-winter"}
    assert [i["id"] for i in _r(corpus["bridge"])["identities"]] == ["person:brian-winter"]


def test_the_real_person_is_untouched(corpus: Dict[str, Path]) -> None:
    _run(corpus["root"])
    gi = {n["id"]: n for n in _r(corpus["gi"])["nodes"]}
    assert gi["insight:2"]["properties"]["speaker"] == HOST
    assert gi["insight:2"]["properties"]["routing_tag"] == "surface"
    assert _r(corpus["seg"])[1]["speaker_label"] == HOST


def test_the_org_is_deleted_not_demoted_so_it_never_votes_person(corpus: Dict[str, Path]) -> None:
    """A demoted `mentioned` Person would vote Person and flip the verdict on the next run."""
    _run(corpus["root"])
    kg = _r(corpus["kg"])
    assert not [n for n in kg["nodes"] if (n.get("properties") or {}).get("name") == ORG]


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
    ctx = MigrationContext(corpus_root=corpus["root"])
    _run(corpus["root"])
    ok, msg = OrgSpeakersRemovedMigration().verify(ctx)
    assert ok, msg


def test_undo_restores_byte_for_byte_and_verify_then_fails(corpus: Dict[str, Path]) -> None:
    before = {k: p.read_bytes() for k, p in corpus.items() if k != "root"}
    _run(corpus["root"])
    restored, refused = undo(corpus["root"])
    assert refused == [] and restored == 6
    assert {k: p.read_bytes() for k, p in corpus.items() if k != "root"} == before
    ok, _msg = OrgSpeakersRemovedMigration().verify(MigrationContext(corpus_root=corpus["root"]))
    assert not ok


def test_below_the_threshold_nothing_is_touched(tmp_path: Path) -> None:
    """Two votes is not decisive — a stray label must never cost a name."""
    for i in range(2):
        _voter(tmp_path, i)
    paths = _episode(tmp_path)
    before = paths["meta"].read_bytes()
    assert _run(tmp_path).details["episodes"] == 0
    assert paths["meta"].read_bytes() == before


def test_registered_last() -> None:
    assert [m.id for m in get_migrations()][-1] == "0012_org_speakers_removed"
