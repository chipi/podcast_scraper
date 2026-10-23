"""m0034: the talk floor on served episodes, from a frozen, reviewed list, through 0031's rewrite.

Synthetic episodes from m0025's corpus fixture (each turn one second).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0029_naming_arc_voices_repaired as m29,
    m0034_talk_floor_names_removed as m34,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


def _episode(tmp_path: Path) -> Dict[str, Any]:
    """Host and guest ten turns each; an ad read named as a person, one turn (4.8% of the talk)."""
    turns = []
    for i in range(10):
        turns.append(("SPEAKER_00", "Gergely Orosz", "host", f"Question {i} about the book."))
        turns.append(("SPEAKER_01", "Steve McConnell", "guest", f"Answer {i} about the book."))
    turns.append(("SPEAKER_02", "Daniel Atkinson", "guest", "This episode is brought to you by."))
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Gergely Orosz", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Steve McConnell", "guest", ["SPEAKER_01"], "roster", "guest"),
            fx._speaker("Daniel Atkinson", "guest", ["SPEAKER_02"], "roster", "guest"),
        ],
        turns=turns,
        known_hosts=["Gergely Orosz"],
    )
    return p


def _review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: Dict, verdict: str = "WRONG_PERSON"
) -> None:
    d = tmp_path / "_verdicts34"
    d.mkdir(exist_ok=True)
    names = {k: str(e[0]) for k, e in entries.items()}
    rows = [
        "\t".join((eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read"))
        for (eid, voice), name in sorted(names.items())
    ]
    head = (
        f"# migration: {m34.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(names)}\n# reviewer: test\n# date: 2026-10-23\n"
    )
    (d / f"{m34.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m34.TalkFloorNamesRemovedMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_the_floor_finds_only_the_named_voice_that_barely_speaks() -> None:
    rows = [
        {
            "speaker": "SPEAKER_00",
            "speaker_label": "Ann Lee",
            "speaker_role": "host",
            "start": 0,
            "end": 60,
        },
        {
            "speaker": "SPEAKER_01",
            "speaker_label": "Bo Diaz",
            "speaker_role": "guest",
            "start": 60,
            "end": 120,
        },
        {
            "speaker": "SPEAKER_02",
            "speaker_label": "Ad Reader",
            "speaker_role": "guest",
            "start": 120,
            "end": 123,
        },
        {"speaker": "SPEAKER_03", "speaker_label": "SPEAKER_03", "start": 123, "end": 125},
    ]
    out = m34.floor_changes(rows)
    assert set(out) == {"SPEAKER_02"}  # SPEAKER_03 is under the floor but carries no name
    old_name, old_role, new_name, new_role, why = out["SPEAKER_02"]
    assert (old_name, old_role, new_name, new_role) == ("Ad Reader", "guest", None, None)
    assert why.startswith("talk share 2.4%")


def test_the_list_freezes_and_loads_back(tmp_path: Path) -> None:
    changes = {
        ("ep", "SPEAKER_02"): ("Daniel Atkinson", "guest", None, None, "talk share 4.8% < 5%")
    }
    path = tmp_path / "list.json"
    m34.write_list(path, changes)
    assert m34.load_list(path) == changes
    assert m34.load_list(tmp_path / "missing.json") == {}


def test_the_detector_reads_the_served_corpus(tmp_path: Path) -> None:
    _episode(tmp_path)
    changes = m34.find_changes(tmp_path)
    assert list(changes) == [("ep", "SPEAKER_02")]
    assert changes[("ep", "SPEAKER_02")][:4] == ("Daniel Atkinson", "guest", None, None)


def test_the_reviewed_list_unnames_the_voice_on_every_surface_and_undoes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    mig = m34.TalkFloorNamesRemovedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    text = p["txt"].read_text(encoding="utf-8")
    assert "Daniel Atkinson:" not in text and "Gergely Orosz:" in text
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert "Daniel Atkinson" not in names and "Steve McConnell" in names
    assert mig.verify(_ctx(tmp_path))[0]
    assert m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert (tmp_path / m34.RECEIPTS_FILE).is_file()
    assert not (tmp_path / m29.RECEIPTS_FILE).exists()
    restored, refused = m34.undo(tmp_path)
    assert restored and not refused
    assert "Daniel Atkinson:" in p["txt"].read_text(encoding="utf-8")


def test_without_a_verdict_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    monkeypatch.setattr(m34.TalkFloorNamesRemovedMigration, "_verdicts_dir", tmp_path / "_none")
    before = p["seg"].read_text(encoding="utf-8")
    ok, message = m34.TalkFloorNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok and message.startswith("verdicts file unusable")
    with pytest.raises(rv.VerdictError):
        m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    assert p["seg"].read_text(encoding="utf-8") == before


def test_a_voice_whose_stored_name_moved_on_refuses_its_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    stale = {("ep", "SPEAKER_02"): ("Someone Else", "guest", None, None, "talk share 4.8% < 5%")}
    monkeypatch.setattr(m34, "load_list", lambda path=None: stale)
    _review(monkeypatch, tmp_path, stale)
    before = p["seg"].read_text(encoding="utf-8")
    res = m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == [] and len(res.details["refused"]) == 1
    assert p["seg"].read_text(encoding="utf-8") == before


def test_not_registered_until_its_list_and_verdicts_exist() -> None:
    assert m34.MIGRATION_ID not in [m.id for m in get_migrations()]
    assert not m34.LIST_FILE.is_file()


def test_a_name_the_review_reads_as_the_voices_own_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A host's short intro is under the floor and is the host: CORRECT keeps it on the voice.
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, verdict="CORRECT")
    before = p["txt"].read_text(encoding="utf-8")
    mig = m34.TalkFloorNamesRemovedMigration()
    assert mig.apply(_ctx(tmp_path)).details["episodes"] == []
    assert p["txt"].read_text(encoding="utf-8") == before
    assert mig.verify(_ctx(tmp_path))[0]


def test_a_listed_voice_without_a_verdict_fails_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, {})
    ok, message = m34.TalkFloorNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok
