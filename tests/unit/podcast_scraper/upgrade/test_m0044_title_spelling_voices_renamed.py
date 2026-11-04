"""m0044: stored voices take the spelling their own episode title states, from a frozen, reviewed
list (B5's stored episodes, #2363).

Record: docs/wip/changes/0044_title_spelling_voices_renamed.md. The shapes on record: a guest's
voice carrying the heard spelling of the titled guest, and a guest's voice carrying the show host's
name. Built on m0025's corpus fixture; fictional names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0044_title_spelling_voices_renamed as m44
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)
_spec34 = importlib.util.spec_from_file_location(
    "m0034_tests", Path(__file__).with_name("test_m0034_talk_floor_names_removed.py")
)
assert _spec34 and _spec34.loader
t34 = importlib.util.module_from_spec(_spec34)
_spec34.loader.exec_module(t34)

HOST = "Ann Lindqvist"
HEARD = "Bo Fereyra"
TITLED = "Bo Ferreira"
HEARD_ENTRY = {("ep", "SPEAKER_01"): (HEARD, "guest", TITLED, "guest", "L6: title; segment 1")}
HOST_ON_GUEST = {("ep", "SPEAKER_01"): (HOST, "host", TITLED, "guest", "L6: title; segment 1")}


def _episode(tmp_path: Path, guest_voice_name: str, guest_role: str) -> Dict[str, Any]:
    turns: List[Tuple[str, str, str, str]] = []
    for i in range(10):
        turns.append(("SPEAKER_00", HOST, "host", f"Question {i} about the harbour."))
        turns.append(("SPEAKER_01", guest_voice_name, guest_role, f"Answer {i} about the harbour."))
    speakers = [fx._speaker(HOST, "host", ["SPEAKER_00"], "roster", "host")]
    if guest_voice_name != HOST:
        speakers.append(
            fx._speaker(guest_voice_name, guest_role, ["SPEAKER_01"], "roster", guest_role)
        )
    else:
        speakers[0]["voices"] = ["SPEAKER_00", "SPEAKER_01"]
    return dict(fx._corpus(tmp_path, speakers=speakers, turns=turns, known_hosts=[HOST]))


def _review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: Dict, verdict: str) -> None:
    d = tmp_path / "_verdicts44"
    d.mkdir(exist_ok=True)
    keep = verdict != "WRONG_PERSON"
    rows = [
        "\t".join(
            (eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, str(e[2] or e[0]), verdict)
            + (
                ("read", "addressed", "1", rv.line_sha256("read"))
                if keep
                else ("read", "another_person", "1", rv.line_sha256("read"))
            )
        )
        for (eid, voice), e in sorted(entries.items())
    ]
    head = (
        f"# migration: {m44.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-11-04\n"
    )
    (d / f"{m44.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m44.TitleSpellingVoicesRenamedMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def _by_voice(p: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    out: Dict[str, set] = {}
    for r in t34._rows(fx._r(p["seg"])):
        out.setdefault(r["speaker"], set()).add((r.get("speaker_label"), r.get("speaker_role")))
    assert all(len(v) == 1 for v in out.values()), out
    return {v: next(iter(s)) for v, s in out.items()}


def test_a_heard_spelling_takes_the_titles_on_every_surface_and_undoes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, HEARD, "guest")
    monkeypatch.setattr(m44, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "CORRECT")
    mig = m44.TitleSpellingVoicesRenamedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert _by_voice(p) == {"SPEAKER_00": (HOST, "host"), "SPEAKER_01": (TITLED, "guest")}
    names = {s["name"] for s in fx._r(p["meta"])["content"]["speakers"]}
    assert TITLED in names and HEARD not in names
    gi_people = {n["properties"]["name"] for n in fx._r(p["gi"])["nodes"] if n["type"] == "Person"}
    assert TITLED in gi_people and HEARD not in gi_people
    assert f"{TITLED}: Answer 0" in p["txt"].read_text(encoding="utf-8")
    assert mig.verify(_ctx(tmp_path))[0]
    restored, refused = m44.undo(tmp_path)
    assert restored and not refused
    assert _by_voice(p)["SPEAKER_01"] == (HEARD, "guest")


def test_the_hosts_name_on_the_titled_guests_voice_takes_the_guest_and_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, HOST, "host")
    monkeypatch.setattr(m44, "load_list", lambda path=None: HOST_ON_GUEST)
    _review(monkeypatch, tmp_path, HOST_ON_GUEST, "CORRECT")
    res = m44.TitleSpellingVoicesRenamedMigration().apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert _by_voice(p) == {"SPEAKER_00": (HOST, "host"), "SPEAKER_01": (TITLED, "guest")}


def test_a_rename_the_review_rejects_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, HEARD, "guest")
    monkeypatch.setattr(m44, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "WRONG_PERSON")
    before = p["seg"].read_text(encoding="utf-8")
    res = m44.TitleSpellingVoicesRenamedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == []
    assert p["seg"].read_text(encoding="utf-8") == before


def test_an_episode_whose_stored_name_moved_on_is_refused_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, "Someone Else", "guest")
    monkeypatch.setattr(m44, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "CORRECT")
    before = p["seg"].read_text(encoding="utf-8")
    res = m44.TitleSpellingVoicesRenamedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == [] and len(res.details["refused"]) == 1
    assert p["seg"].read_text(encoding="utf-8") == before


def test_the_shipped_list_holds_the_recorded_voices() -> None:
    """The record's step 4: 20 renames in 20 episodes, each naming its deciding segment."""
    listed = m44.load_list()
    assert len(listed) == 20 and len({eid for eid, _ in listed}) == 20
    assert all(e[0] and e[2] and e[0] != e[2] for e in listed.values())


def test_registered_after_0043_with_its_own_backups_receipts_and_verdicts() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index(m44.MIGRATION_ID) == ids.index("0043_non_participant_names_removed") + 1
    assert m44.BACKUP_TAG == "0044" and m44.RECEIPTS_FILE == "title_spelling_voices_renamed.jsonl"
    entries = m44.load_list()
    names = {k: str(e[2]) for k, e in entries.items()}
    from podcast_scraper.upgrade.migrations import m0035_voice_identity_names as m35

    parsed = rv.load(m44.MIGRATION_ID, names, listed=entries, require_basis=True, bases=m35.BASES)
    assert sorted(parsed.rows) == sorted(entries)
