"""m0045: one episode's guest-named voices take their own persons' names, from a frozen, reviewed
list (found by 0044's review).

Record: docs/wip/changes/0045_episode_voices_corrected.md. The shapes on record: the show's host's
voice carrying a guest's heard spelling (renamed, role to host), and a rename the review refuses.
Built on m0025's corpus fixture; fictional names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0045_episode_voices_corrected as m45
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
SECOND_HOST = "Cy Ostrander"
GUEST_ON_HOST = {
    ("ep", "SPEAKER_01"): (HEARD, "guest", SECOND_HOST, "host", "self intro; segment 1")
}


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
    d = tmp_path / "_verdicts45"
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
        f"# migration: {m45.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-11-04\n"
    )
    (d / f"{m45.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m45.EpisodeVoicesCorrectedMigration, "_verdicts_dir", d)


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
    monkeypatch.setattr(m45, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "CORRECT")
    mig = m45.EpisodeVoicesCorrectedMigration()
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
    restored, refused = m45.undo(tmp_path)
    assert restored and not refused
    assert _by_voice(p)["SPEAKER_01"] == (HEARD, "guest")


def test_a_guests_name_on_the_hosts_voice_takes_the_host_and_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, HEARD, "guest")
    monkeypatch.setattr(m45, "load_list", lambda path=None: GUEST_ON_HOST)
    _review(monkeypatch, tmp_path, GUEST_ON_HOST, "CORRECT")
    res = m45.EpisodeVoicesCorrectedMigration().apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert _by_voice(p) == {"SPEAKER_00": (HOST, "host"), "SPEAKER_01": (SECOND_HOST, "host")}
    roles = {s["name"]: s["role"] for s in fx._r(p["meta"])["content"]["speakers"]}
    assert roles.get(SECOND_HOST) == "host" and HEARD not in roles


def test_a_rename_the_review_rejects_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, HEARD, "guest")
    monkeypatch.setattr(m45, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "WRONG_PERSON")
    before = p["seg"].read_text(encoding="utf-8")
    res = m45.EpisodeVoicesCorrectedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == []
    assert p["seg"].read_text(encoding="utf-8") == before


def test_an_episode_whose_stored_name_moved_on_is_refused_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path, "Someone Else", "guest")
    monkeypatch.setattr(m45, "load_list", lambda path=None: HEARD_ENTRY)
    _review(monkeypatch, tmp_path, HEARD_ENTRY, "CORRECT")
    before = p["seg"].read_text(encoding="utf-8")
    res = m45.EpisodeVoicesCorrectedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == [] and len(res.details["refused"]) == 1
    assert p["seg"].read_text(encoding="utf-8") == before


def test_the_shipped_list_holds_the_recorded_voices() -> None:
    """The record's step 4: 2 renames in 1 episode, each naming its deciding segment."""
    listed = m45.load_list()
    assert len(listed) == 2 and len({eid for eid, _ in listed}) == 1
    assert all(e[0] and e[2] and e[0] != e[2] for e in listed.values())


def test_registered_after_0044_with_its_own_backups_receipts_and_verdicts() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index(m45.MIGRATION_ID) == ids.index("0044_title_spelling_voices_renamed") + 1
    assert m45.BACKUP_TAG == "0045" and m45.RECEIPTS_FILE == "episode_voices_corrected.jsonl"
    entries = m45.load_list()
    names = {k: str(e[2]) for k, e in entries.items()}
    from podcast_scraper.upgrade.migrations import m0035_voice_identity_names as m35

    parsed = rv.load(m45.MIGRATION_ID, names, listed=entries, require_basis=True, bases=m35.BASES)
    assert sorted(parsed.rows) == sorted(entries)


def test_the_shipped_review_applies_the_host_rename_and_refuses_the_other() -> None:
    entries = m45.load_list()
    names = {k: str(e[2]) for k, e in entries.items()}
    from podcast_scraper.upgrade.migrations import m0035_voice_identity_names as m35

    parsed = rv.load(m45.MIGRATION_ID, names, listed=entries, require_basis=True, bases=m35.BASES)
    by_role = {entries[k][3]: row.verdict for k, row in parsed.rows.items()}
    assert by_role == {"host": "CORRECT", "guest": "WRONG_PERSON"}
