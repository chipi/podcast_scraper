"""m0037: served voices holding another person's name get their own, from a frozen, reviewed list.

Record: docs/wip/changes/0037_named_voices_corrected.md. The shapes on record: two voices of one
episode holding each other's names (a swap, host and guest roles with them), and one voice holding
a name that belongs to nobody else in the episode. Built on m0025's corpus fixture.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0037_named_voices_corrected as m37

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
GUEST = "Bo Ferreira"
SWAP = {
    ("ep", "SPEAKER_00"): (GUEST, "guest", HOST, "host", "readers: addressed, segment 0"),
    ("ep", "SPEAKER_01"): (HOST, "host", GUEST, "guest", "readers: addressed, segment 1"),
}


def _swapped(tmp_path: Path) -> Dict[str, Any]:
    """The host's voice published as the guest and the guest's as the host."""
    turns: List[Tuple[str, str, str, str]] = []
    for i in range(10):
        turns.append(("SPEAKER_00", GUEST, "guest", f"Question {i} about the harbour."))
        turns.append(("SPEAKER_01", HOST, "host", f"Answer {i} about the harbour."))
    return dict(
        fx._corpus(
            tmp_path,
            speakers=[
                fx._speaker(GUEST, "guest", ["SPEAKER_00"], "roster", "guest"),
                fx._speaker(HOST, "host", ["SPEAKER_01"], "roster", "host"),
            ],
            turns=turns,
            known_hosts=[HOST],
        )
    )


def _review(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entries: Dict, verdict: str) -> None:
    d = tmp_path / "_verdicts37"
    d.mkdir(exist_ok=True)
    rows = [
        "\t".join(
            (eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, str(e[2] or e[0]), verdict)
            + ("read", "addressed", "0", rv.line_sha256("read"))
        )
        for (eid, voice), e in sorted(entries.items())
    ]
    head = (
        f"# migration: {m37.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-27\n"
    )
    (d / f"{m37.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m37.NamedVoicesCorrectedMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def _by_voice(p: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """``{voice: (label, role)}`` from the segments, every row of a voice agreeing."""
    out: Dict[str, set] = {}
    for r in t34._rows(fx._r(p["seg"])):
        out.setdefault(r["speaker"], set()).add((r.get("speaker_label"), r.get("speaker_role")))
    assert all(len(v) == 1 for v in out.values()), out
    return {v: next(iter(s)) for v, s in out.items()}


def test_a_reviewed_swap_gives_each_voice_its_own_name_and_role_on_every_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _swapped(tmp_path)
    monkeypatch.setattr(m37, "load_list", lambda path=None: SWAP)
    _review(monkeypatch, tmp_path, SWAP, "CORRECT")
    mig = m37.NamedVoicesCorrectedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert _by_voice(p) == {"SPEAKER_00": (HOST, "host"), "SPEAKER_01": (GUEST, "guest")}
    diag = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}
    assert (diag["SPEAKER_00"]["resolved_name"], diag["SPEAKER_01"]["resolved_name"]) == (
        HOST,
        GUEST,
    )
    # A reviewed correction is not the voice identity rung's name, and must not vote as one.
    assert {diag[v]["source"] for v in ("SPEAKER_00", "SPEAKER_01")} == {"roster"}
    speakers = {s["name"]: s for s in fx._r(p["meta"])["content"]["speakers"]}
    assert speakers[HOST]["voices"] == ["SPEAKER_00"] and speakers[HOST]["role"] == "host"
    assert speakers[GUEST]["voices"] == ["SPEAKER_01"] and speakers[GUEST]["role"] == "guest"
    gi_people = [n["properties"]["name"] for n in fx._r(p["gi"])["nodes"] if n["type"] == "Person"]
    assert sorted(gi_people) == sorted({HOST, GUEST}), "one Person per name, none duplicated"
    text = p["txt"].read_text(encoding="utf-8")
    assert f"{HOST}: Question 0" in text and f"{GUEST}: Answer 0" in text
    assert mig.verify(_ctx(tmp_path))[0]
    restored, refused = m37.undo(tmp_path)
    assert restored and not refused
    assert _by_voice(p) == {"SPEAKER_00": (GUEST, "guest"), "SPEAKER_01": (HOST, "host")}


def test_a_swap_the_review_rejects_on_one_voice_writes_nothing_of_the_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half a swap would put one name on two voices: a pair is applied whole or not at all."""
    p = _swapped(tmp_path)
    monkeypatch.setattr(m37, "load_list", lambda path=None: SWAP)
    _review(monkeypatch, tmp_path, SWAP, "CORRECT")
    tsv = tmp_path / "_verdicts37" / f"{m37.MIGRATION_ID}.tsv"
    lines = tsv.read_text(encoding="utf-8").splitlines()
    lines[-1] = (
        lines[-1]
        .replace("\tCORRECT\t", "\tWRONG_PERSON\t")
        .replace("\taddressed\t", "\tundecidable\t")
    ).replace(f"\tundecidable\t0\t{rv.line_sha256('read')}", "\tundecidable\t\t")
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    before = p["seg"].read_text(encoding="utf-8")
    res = m37.NamedVoicesCorrectedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == []
    assert p["seg"].read_text(encoding="utf-8") == before


def test_an_episode_whose_stored_name_moved_on_is_refused_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _swapped(tmp_path)
    moved = dict(SWAP)
    moved[("ep", "SPEAKER_00")] = ("Someone Else", "guest", HOST, "host", "stale")
    monkeypatch.setattr(m37, "load_list", lambda path=None: moved)
    _review(monkeypatch, tmp_path, moved, "CORRECT")
    before = p["seg"].read_text(encoding="utf-8")
    res = m37.NamedVoicesCorrectedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == [] and len(res.details["refused"]) == 1
    assert p["seg"].read_text(encoding="utf-8") == before


def test_the_shipped_list_holds_the_recorded_voices() -> None:
    """The record's step 4: 6 renames in 4 episodes, no unnaming."""
    listed = m37.load_list()
    assert len(listed) == 6
    assert len({eid for eid, _ in listed}) == 4
    assert all(e[0] and e[2] for e in listed.values())


def test_the_backups_and_receipts_are_its_own() -> None:
    assert m37.BACKUP_TAG == "0037" and m37.RECEIPTS_FILE == "named_voices_corrected.jsonl"
    assert m37.NamedVoicesCorrectedMigration.id == m37.MIGRATION_ID


def test_an_entry_without_its_deciding_segment_or_a_name_is_refused(tmp_path: Path) -> None:
    for row in (
        ["ep", "SPEAKER_00", GUEST, "guest", HOST, "host", "readers agree"],
        ["ep", "SPEAKER_00", None, None, HOST, "host", "addressed, segment 3"],
        ["ep", "SPEAKER_00", GUEST, "guest", None, None, "addressed, segment 3"],
    ):
        path = tmp_path / "list.json"
        path.write_text(json.dumps({"migration": m37.MIGRATION_ID, "voices": [row]}))
        with pytest.raises(m37.ListError):
            m37.load_list(path)
