"""m0018: finish m0012 on what it left — an m0012-frozen org held only as a GI Person node.

Prod shape (2026-10-03 verify): "Africa Tech Summit" as a GI Person on two episodes, no roster
entry, while m0012's receipt froze it as an organisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.speaker_detectors.entity_kind_votes import kind_key
from podcast_scraper.upgrade.file_rewrite import append_receipts
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0012_org_speakers_removed import (
    RECEIPTS_FILE as M0012_RECEIPTS,
)
from podcast_scraper.upgrade.migrations.m0018_org_speakers_removed_residue import (
    OrgSpeakersRemovedResidueMigration,
    undo,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ORG = "Africa Tech Summit"
ORG_ID = "person:africa-tech-summit"
REAL = "Ada Brook"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _corpus(root: Path, *, frozen: bool = True) -> Dict[str, Path]:
    meta = root / "feeds" / "f" / "run_1" / "metadata"
    paths = {"meta": meta / "ep.metadata.json", "gi": meta / "ep.gi.json"}
    _w(
        paths["meta"],
        {
            "episode": {"episode_id": "ep"},
            "feed": {"title": "Made In Africa"},
            "content": {"speakers": [{"id": "h", "name": REAL, "role": "host", "placed": True}]},
        },
    )
    _w(
        paths["gi"],
        {
            "nodes": [
                {"id": ORG_ID, "type": "Person", "properties": {"name": ORG}},
                {"id": "person:ada-brook", "type": "Person", "properties": {"name": REAL}},
                {"id": "quote:1", "type": "Quote", "properties": {"speaker_id": ORG_ID}},
            ],
            "edges": [{"type": "SPOKEN_BY", "from": "quote:1", "to": ORG_ID}],
        },
    )
    if frozen:
        # append_receipts writes nothing without a receipt row; one stands in for m0012's apply.
        row = {"path": "feeds/x/run/metadata/old.gi.json", "sha256": "0" * 64}
        append_receipts(root, M0012_RECEIPTS, {"orgs": {kind_key(ORG): [3, 0]}}, [row])
    paths["root"] = root
    return paths


def _run(root: Path, dry: bool = False):
    return OrgSpeakersRemovedResidueMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry)
    )


def test_the_residue_org_leaves_gi_and_the_real_person_stays(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    ok, _ = OrgSpeakersRemovedResidueMigration().verify(MigrationContext(corpus_root=tmp_path))
    assert not ok
    result = _run(tmp_path)
    assert len(result.details["episodes"]) == 1
    gi = {n["id"]: n for n in _r(c["gi"])["nodes"]}
    assert ORG_ID not in gi and "person:ada-brook" in gi
    assert gi["quote:1"]["properties"]["speaker_id"] is None
    ok, msg = OrgSpeakersRemovedResidueMigration().verify(MigrationContext(corpus_root=tmp_path))
    assert ok, msg


def test_dry_run_writes_nothing_and_second_run_is_a_no_op(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    before = c["gi"].read_bytes()
    assert len(_run(tmp_path, dry=True).details["episodes"]) == 1
    assert c["gi"].read_bytes() == before
    _run(tmp_path)
    after = c["gi"].read_bytes()
    assert _run(tmp_path).details["episodes"] == []
    assert c["gi"].read_bytes() == after


def test_without_m0012_receipts_nothing_is_decided(tmp_path: Path) -> None:
    c = _corpus(tmp_path, frozen=False)
    before = c["gi"].read_bytes()
    assert _run(tmp_path).details["episodes"] == []
    assert c["gi"].read_bytes() == before


def test_undo_restores(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    before = c["gi"].read_bytes()
    _run(tmp_path)
    restored, refused = undo(tmp_path)
    assert refused == [] and restored == 1
    assert c["gi"].read_bytes() == before


def test_registered_after_0017() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0018_org_speakers_removed_residue") == (
        ids.index("0017_speaker_names_canonicalised") + 1
    )
