"""m0034: the talk floor on served episodes, from a frozen, reviewed list, through 0031's rewrite.

Synthetic episodes from m0025's corpus fixture (each turn one second).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


def _rows(payload: Any) -> list:
    return payload if isinstance(payload, list) else payload["segments"]


def fx_surfaces(p: Dict[str, Any]) -> Dict[str, set]:
    """The names every derived surface the fixture writes now carries."""
    gi = fx._r(p["gi"])
    people = {n["id"]: n["properties"]["name"] for n in gi["nodes"] if n["type"] == "Person"}
    out = {
        "metadata speakers": {s["name"] for s in fx._r(p["meta"])["content"]["speakers"]},
        "segments": {r.get("speaker_label") for r in _rows(fx._r(p["seg"]))},
        "gi persons": {n["properties"]["name"] for n in gi["nodes"] if n["type"] == "Person"},
        # Readers resolve a quote's speaker from its speaker_id through the Person node
        # (gi/load.py, server/app_gi_view.py); the rewrite clears the authored speaker_name.
        "gi quotes": {
            people.get(n["properties"].get("speaker_id"))
            for n in gi["nodes"]
            if n["type"] == "Quote"
        },
        "kg persons": {n["properties"]["name"] for n in fx._r(p["kg"])["nodes"]},
        "diagnostics": {v["resolved_name"] for v in fx._r(p["diag"])["voices"] if v.get("named")},
    }
    if p.get("turns") is not None and p["turns"].is_file():
        out["turns"] = {t.get("speaker_label") for t in fx._r(p["turns"])["turns"]}
    return out


def _episode(tmp_path: Path) -> Dict[str, Any]:
    """Host and guest ten turns each; an ad read named as a person, one turn (4.8% of the talk)."""
    turns = []
    for i in range(10):
        turns.append(("SPEAKER_00", "Gerald Orban", "host", f"Question {i} about the book."))
        turns.append(("SPEAKER_01", "Stan McCormack", "guest", f"Answer {i} about the book."))
    turns.append(("SPEAKER_02", "Dennis Aitken", "guest", "This episode is brought to you by."))
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Gerald Orban", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Stan McCormack", "guest", ["SPEAKER_01"], "roster", "guest"),
            fx._speaker("Dennis Aitken", "guest", ["SPEAKER_02"], "roster", "guest"),
        ],
        turns=turns,
        known_hosts=["Gerald Orban"],
    )
    return p


def _review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entries: Dict,
    verdict: str = "WRONG_PERSON",
    with_rows: bool = True,
) -> None:
    d = tmp_path / "_verdicts34"
    d.mkdir(exist_ok=True)
    names = {k: str(e[0]) for k, e in entries.items()}
    # 0034's verdicts state a basis and the deciding line (4d): a removal needs text showing
    # someone else, so the fixture's removals cite one; its keeps cite a self-introduction.
    basis = "another_person" if verdict == "WRONG_PERSON" else "self_intro"
    line = rv.line_sha256("read")
    rows = [
        "\t".join(
            (eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read")
            + (basis, "0", line)
        )
        for (eid, voice), name in sorted(names.items())
        if with_rows
    ]
    # The header binds the WHOLE list (every field of every entry), not only the judged names.
    head = (
        f"# migration: {m34.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-23\n"
    )
    (d / f"{m34.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
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
    changes = {("ep", "SPEAKER_02"): ("Dennis Aitken", "guest", None, None, "talk share 4.8% < 5%")}
    path = tmp_path / "list.json"
    m34.write_list(path, changes)
    assert m34.load_list(path) == changes
    assert m34.load_list(tmp_path / "missing.json") == {}


def test_the_detector_reads_the_served_corpus(tmp_path: Path) -> None:
    _episode(tmp_path)
    changes = m34.find_changes(tmp_path)
    assert list(changes) == [("ep", "SPEAKER_02")]
    assert changes[("ep", "SPEAKER_02")][:4] == ("Dennis Aitken", "guest", None, None)


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
    assert "Dennis Aitken:" not in text and "Gerald Orban:" in text
    for surface, names in fx_surfaces(p).items():
        assert "Dennis Aitken" not in names, surface
        assert "Stan McCormack" in names, surface
    diag = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}
    assert not diag["SPEAKER_02"]["named"] and diag["SPEAKER_02"]["source"] == "raw"
    assert mig.verify(_ctx(tmp_path))[0]
    header = json.loads((tmp_path / m34.RECEIPTS_FILE).read_text().splitlines()[0])
    assert header["list_sha256"] == rv.list_sha256(entries)
    verdicts_text = (tmp_path / "_verdicts34" / f"{m34.MIGRATION_ID}.tsv").read_text(
        encoding="utf-8"
    )
    assert header["verdicts_sha256"] == hashlib.sha256(verdicts_text.encode("utf-8")).hexdigest()
    assert m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert (tmp_path / m34.RECEIPTS_FILE).is_file()
    assert not (tmp_path / m29.RECEIPTS_FILE).exists()
    restored, refused = m34.undo(tmp_path)
    assert restored and not refused
    assert "Dennis Aitken:" in p["txt"].read_text(encoding="utf-8")


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


def test_registered_with_its_list_and_verdicts_shipped() -> None:
    # Registered 2026-10-26 once the list (snapshot-prod-20261025) and its reviewed verdicts
    # existed: both files ship beside the code, and the verdicts load against the list.
    ids = [m.id for m in get_migrations()]
    assert m34.MIGRATION_ID in ids
    assert ids.index(m34.MIGRATION_ID) == ids.index("0033_hand_removed_names_refused") + 1
    entries = m34.load_list(m34.LIST_FILE)
    assert entries
    names = {k: str(e[0]) for k, e in entries.items()}
    rv.load(m34.MIGRATION_ID, names, listed=entries)


@pytest.mark.parametrize("verdict", ["CORRECT", "CORRECT_MISSPELLED"])
def test_a_name_the_review_reads_as_the_voices_own_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    # A host's short intro is under the floor and is the host: CORRECT keeps it on the voice.
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, verdict=verdict)
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
    _review(monkeypatch, tmp_path, entries, with_rows=False)
    ok, message = m34.TalkFloorNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok and "without a verdict" in message


def test_an_entry_edited_after_the_review_refuses_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review M3: the verdicts header binds every field, so a role flipped after review fails.
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    _review(monkeypatch, tmp_path, entries)
    edited = {k: (e[0], "host", *e[2:]) for k, e in entries.items()}
    monkeypatch.setattr(m34, "load_list", lambda path=None: edited)
    before = p["seg"].read_text(encoding="utf-8")
    ok, message = m34.TalkFloorNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok and "list changed since it was reviewed" in message
    with pytest.raises(rv.VerdictError):
        m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    assert p["seg"].read_text(encoding="utf-8") == before


def test_plan_without_a_verdicts_file_is_a_refusal_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    monkeypatch.setattr(m34.TalkFloorNamesRemovedMigration, "_verdicts_dir", tmp_path / "_none")
    line = m34.TalkFloorNamesRemovedMigration().plan(_ctx(tmp_path))
    assert line.startswith(f"{m34.MIGRATION_ID}: refused, verdicts file unusable")


def test_the_frozen_list_is_part_of_the_migrations_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.upgrade import outcome_hook

    path = tmp_path / "list.json"
    monkeypatch.setattr(m34, "LIST_FILE", path)
    m34.write_list(path, {("ep", "SPEAKER_02"): ("A B", "guest", None, None, "x")})
    first = outcome_hook.migration_digest(m34.TalkFloorNamesRemovedMigration())
    m34.write_list(path, {("ep", "SPEAKER_02"): ("A B", "host", None, None, "x")})
    assert outcome_hook.migration_digest(m34.TalkFloorNamesRemovedMigration()) != first


@pytest.mark.parametrize("label", ["SPEAKER_03", "Speaker", "Speaker 2", "Unknown Speaker"])
def test_a_placeholder_label_is_no_name_to_take_off(label: str) -> None:
    # Second review m7: a provider's "Speaker 2" is not a name the review should judge.
    rows = [
        {"speaker": "SPEAKER_00", "speaker_label": "Ann Lee", "start": 0, "end": 100},
        {"speaker": "SPEAKER_03", "speaker_label": label, "start": 100, "end": 101},
    ]
    assert m34.served_by_voice(rows)["SPEAKER_03"][0] is None
    assert m34.floor_changes(rows) == {}


def test_the_dry_run_rows_carry_counts_and_input_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The outcome hook diffs prod's per-episode plan against the clone's: each row needs the
    # episode's counters and the sha256 of every file it read, and the rows sum to plan()'s totals.
    _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    rows = list(m34.dry_run_report(tmp_path))
    assert rows and all(r["counts"] and r["inputs"] for r in rows)
    # a file the plan looked for and did not find is recorded "absent", which the diff also compares
    assert all(len(h) == 64 or h == "absent" for r in rows for h in r["inputs"].values())
    assert any(len(h) == 64 for r in rows for h in r["inputs"].values())
    plan = m34.TalkFloorNamesRemovedMigration().plan(_ctx(tmp_path))
    for key, n in {k: sum(r["counts"].get(k, 0) for r in rows) for k in rows[0]["counts"]}.items():
        assert f"{key}={n}" in plan


# --- mutation-killing tests (step 8) -----------------------------------------------------------


def _row(voice: str, label: str, start: float, end: float, role: str = "guest") -> dict:
    return {
        "speaker": voice,
        "speaker_label": label,
        "speaker_role": role,
        "start": start,
        "end": end,
    }


def test_the_floor_is_the_pipelines_floor_on_prod() -> None:
    # The value comes from the profile the pipeline runs on, not from this module's own opinion.
    profile = Path(__file__).resolve().parents[4] / "config" / "profiles" / "prod_dgx_full.yaml"
    found = re.search(
        r"^speaker_name_min_talk_share:\s*([0-9.]+)\s*$",
        profile.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert found, "prod_dgx_full.yaml has no speaker_name_min_talk_share"
    assert m34.FLOOR == float(found.group(1))


def test_a_voice_at_exactly_the_floor_keeps_its_name() -> None:
    # The pipeline floors with a strict "<" (providers/ml/diarization/pipeline.py
    # _apply_talk_floor: ``if role.named and share < floor``), so a voice at exactly 5% is kept.
    rows = [_row("SPEAKER_00", "Ann Lee", 0, 95, "host"), _row("SPEAKER_01", "Bo Diaz", 95, 100)]
    assert m34.floor_changes(rows) == {}
    just_under = [
        _row("SPEAKER_00", "Ann Lee", 0, 95.5, "host"),
        _row("SPEAKER_01", "Bo", 95.5, 100),
    ]
    assert set(m34.floor_changes(just_under)) == {"SPEAKER_01"}


def test_a_voice_between_five_and_six_percent_keeps_its_name() -> None:
    # Pins the constant itself: 5.66% is over a 5% floor and would be under a 6% one.
    rows = [_row("SPEAKER_00", "Ann Lee", 0, 100, "host"), _row("SPEAKER_01", "Bo Diaz", 100, 106)]
    assert m34.floor_changes(rows) == {}


def test_an_unnamed_voice_still_counts_in_the_episodes_talk() -> None:
    # 8s of 208s is 3.8%: the 100s unnamed voice is part of the episode's talk.
    rows = [
        _row("SPEAKER_00", "Ann Lee", 0, 100, "host"),
        _row("SPEAKER_01", "SPEAKER_01", 100, 200),
        _row("SPEAKER_02", "Ad Reader", 200, 208),
    ]
    assert set(m34.floor_changes(rows)) == {"SPEAKER_02"}


def test_segments_without_a_speaker_are_no_voices_talk() -> None:
    # 6s of the 106s the voices speak is 5.7%; counting a 100s unattributed stretch would make 2.9%.
    rows = [
        _row("SPEAKER_00", "Ann Lee", 0, 100, "host"),
        {"start": 100, "end": 200, "text": "applause"},
        _row("SPEAKER_01", "Bo Diaz", 200, 206),
    ]
    assert m34.talk_by_voice(rows) == {"SPEAKER_00": 100.0, "SPEAKER_01": 6.0}
    assert m34.floor_changes(rows) == {}


def test_a_segment_that_ends_before_it_starts_is_no_negative_talk() -> None:
    # Clamped to 0: with the -60 counted the total would be 44s and 4s of it 9%, not 3.8% of 104s.
    rows = [
        _row("SPEAKER_00", "Ann Lee", 0, 100, "host"),
        _row("SPEAKER_00", "Ann Lee", 160, 100, "host"),
        _row("SPEAKER_01", "Bo Diaz", 160, 164),
    ]
    assert m34.talk_by_voice(rows)["SPEAKER_00"] == 100.0
    assert set(m34.floor_changes(rows)) == {"SPEAKER_01"}


def test_a_voice_is_served_under_the_name_of_its_first_segment() -> None:
    rows = [
        _row("SPEAKER_00", "Ann Lee", 0, 5, "host"),
        _row("SPEAKER_00", "Someone Else", 5, 10, "guest"),
    ]
    assert m34.served_by_voice(rows) == {"SPEAKER_00": ("Ann Lee", "host")}


def test_a_listed_voice_the_review_confirms_wrong_is_removed_and_a_refused_episode_fails_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    stale = {("ep", "SPEAKER_02"): ("Someone Else", "guest", None, None, "talk share 4.8% < 5%")}
    monkeypatch.setattr(m34, "load_list", lambda path=None: stale)
    _review(monkeypatch, tmp_path, stale)
    before = p["seg"].read_text(encoding="utf-8")
    ok, message = m34.TalkFloorNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok and "1 refused" in message
    assert p["seg"].read_text(encoding="utf-8") == before


def _unserved_older_run(tmp_path: Path) -> None:
    """A superseded run of the same episode that still holds a different voice under the floor."""
    old = tmp_path / "feeds" / "f" / "run_0"
    shutil.copytree(tmp_path / "feeds" / "f" / "run_1", old)
    seg = old / "transcripts" / "ep.segments.json"
    rows = json.loads(seg.read_text(encoding="utf-8"))
    rows.append(_row("SPEAKER_09", "Old Ghost", 100, 101))
    seg.write_text(json.dumps(rows), encoding="utf-8")


def test_the_detector_ignores_a_run_the_corpus_does_not_serve(tmp_path: Path) -> None:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    _episode(tmp_path)
    _unserved_older_run(tmp_path)
    served, superseded = select_served_artifacts(tmp_path, ".metadata.json")
    assert [q.parts[-3] for q in served] == ["run_1"]
    assert [q.parts[-3] for q in superseded] == ["run_0"]
    assert list(m34.find_changes(tmp_path)) == [("ep", "SPEAKER_02")]


def test_the_detector_reads_only_text_transcripts(tmp_path: Path) -> None:
    p = _episode(tmp_path)
    meta = json.loads(p["meta"].read_text(encoding="utf-8"))
    # A path that is not a .txt transcript, pointing at a file that holds segment rows.
    meta["content"]["transcript_file_path"] = "transcripts/ep.segments.json"
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    assert m34.find_changes(tmp_path) == {}


def test_the_backup_tag_is_this_migrations_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.upgrade.migrations import m0035_voice_identity_names as m35

    assert m34.BACKUP_TAG == "0034" and m34.BACKUP_TAG != m35.BACKUP_TAG
    assert m34.TalkFloorNamesRemovedMigration._backup_tag == m34.BACKUP_TAG
    p = _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    backups = tmp_path / ".podcast_scraper" / "upgrade-backups"
    assert [d.name for d in backups.iterdir()] == ["0034"]
    assert p["txt"].is_file()


def test_write_list_sorts_its_rows(tmp_path: Path) -> None:
    # Sorted rows make the frozen list byte-stable whatever order the detector found them in; an
    # unsorted dump would change the list's file (and so the migration's digest) between runs.
    entry = ("A B", "guest", None, None, "x")
    changes = {
        ("ep2", "SPEAKER_01"): entry,
        ("ep1", "SPEAKER_02"): entry,
        ("ep1", "SPEAKER_00"): entry,
    }
    path = tmp_path / "list.json"
    m34.write_list(path, changes)
    keys = [(r[0], r[1]) for r in json.loads(path.read_text(encoding="utf-8"))["voices"]]
    assert keys == [("ep1", "SPEAKER_00"), ("ep1", "SPEAKER_02"), ("ep2", "SPEAKER_01")]


def _with_twin_and_insight(tmp_path: Path, insight_type: Any = None) -> Dict[str, Any]:
    """The ad read ("Dennis Aitken", one turn) plus an insight authored by that label."""
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    props: Dict[str, Any] = {"text": "Ad.", "speaker": "Dennis Aitken", "grounded": True, "tier": 2}
    if insight_type:
        props.update(speaker_voice_type=insight_type, surfaceable=False)
    gi["nodes"].append({"id": "insight:ad", "type": "Insight", "properties": props})
    fx._w(p["gi"], gi)
    return p


def _reviewed_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    return m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))


def _insight(p: Dict[str, Any]) -> Dict[str, Any]:
    props: Dict[str, Any] = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "insight:ad")[
        "properties"
    ]
    return props


def test_an_insight_of_the_unnamed_voice_is_typed_unknown_and_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _with_twin_and_insight(tmp_path)
    assert len(_reviewed_apply(tmp_path, monkeypatch).details["episodes"]) == 1
    props = _insight(p)
    assert "speaker" not in props
    assert props["speaker_voice_type"] == "unknown" and props["surfaceable"] is False
    # re-routed on its tier as the pipeline does for an unattributed stance: connect, never surface
    assert props["routing_tag"] == "connect" and props["salience"] == 0.6667


def test_an_insight_typed_commercial_stays_commercial_when_its_voice_loses_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Criterion 12: the shared unnaming (m0025) set "unknown" unconditionally, turning an ad
    # insight into an unknown-voice one. A commercial type is a correct non-person class; it stays.
    p = _with_twin_and_insight(tmp_path, insight_type="commercial")
    assert len(_reviewed_apply(tmp_path, monkeypatch).details["episodes"]) == 1
    assert _insight(p)["speaker_voice_type"] == "commercial"


def test_a_not_a_render_transcript_with_a_shared_label_refuses_the_episode_untouched(
    tmp_path: Path,
) -> None:
    # A forced twin carries the same label as the voice losing its name, and the transcript is not
    # a pure render of the segments: the in-place fallback cannot tell the two apart, so refuse.
    turns = []
    for i in range(10):
        turns.append(("SPEAKER_00", "Gerald Orban", "host", f"Question {i} about the book."))
        turns.append(("SPEAKER_01", "Dennis Aitken", "guest", f"Answer {i} about the book."))
    turns.append(("SPEAKER_02", "Dennis Aitken", "guest", "This episode is brought to you by."))
    p = fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Gerald Orban", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Dennis Aitken", "guest", ["SPEAKER_01", "SPEAKER_02"], "roster", "guest"),
        ],
        turns=turns,
        known_hosts=["Gerald Orban"],
    )
    p["txt"].write_text(p["txt"].read_text(encoding="utf-8") + "\n[music]\n", encoding="utf-8")
    # No quotes: with them the episode is refused earlier, for a quote's voice, not for the render.
    gi = fx._r(p["gi"])
    gi["nodes"] = [n for n in gi["nodes"] if n["type"] != "Quote"]
    gi["edges"] = []
    fx._w(p["gi"], gi)
    before = {k: v.read_text(encoding="utf-8") for k, v in p.items() if v.is_file()}
    entries: Dict[
        Tuple[str, str], Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]
    ] = {("ep", "SPEAKER_02"): ("Dennis Aitken", "guest", None, None, "talk share 4.8% < 5%")}
    mig = m34.TalkFloorNamesRemovedMigration()
    mig._apply = entries
    write, refused, _totals = mig._scan(tmp_path)
    assert write == [] and len(refused) == 1
    assert refused[0].refused == "transcripts: not a render, and a rewritten label is shared"
    assert {k: v.read_text(encoding="utf-8") for k, v in p.items() if v.is_file()} == before


def test_a_bridge_identity_kept_by_an_alias_survives_the_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Found on the 0034 clone (snapshot-prod-20261025): a bridge identity whose id is an older
    # unresolved form, with the person's GI / KG id only among its aliases, was dropped although
    # GI and KG still hold the person. Only the removed voice's person may leave the bridge.
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    pid = {n["properties"]["name"]: n["id"] for n in gi["nodes"] if n["type"] == "Person"}
    bridge_path = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "episode_id": "ep",
                "identities": [
                    {
                        "id": "person:unresolved-stan-ep",
                        "type": "person",
                        "display_name": "Stan McCormack",
                        "aliases": ["Stan McCormack", pid["Stan McCormack"]],
                        "sources": {"gi": True, "kg": True},
                    },
                    {
                        "id": pid["Dennis Aitken"],
                        "type": "person",
                        "display_name": "Dennis Aitken",
                        "aliases": [],
                        "sources": {"gi": True, "kg": True},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    assert len(m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path)).details["episodes"]) == 1
    left = {i["display_name"] for i in fx._r(bridge_path)["identities"]}
    assert "Stan McCormack" in left
    assert "Dennis Aitken" not in left
    stan = next(
        i for i in fx._r(bridge_path)["identities"] if i["display_name"] == "Stan McCormack"
    )
    assert stan["sources"] == {
        "gi": True,
        "kg": True,
    }  # from the alias, the id itself is in neither


@pytest.mark.parametrize("alias_of", ["kept", "self"])
def test_a_removed_persons_bridge_identity_goes_even_when_an_alias_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_of: str
) -> None:
    # The alias rule must not keep the person the rewrite takes off: an alias pointing at a kept
    # person (or at its own id) would otherwise leave the removed name on the bridge.
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    pid = {n["properties"]["name"]: n["id"] for n in gi["nodes"] if n["type"] == "Person"}
    alias = pid["Stan McCormack"] if alias_of == "kept" else pid["Dennis Aitken"]
    bridge_path = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    ident = {"type": "person", "sources": {"gi": True, "kg": True}}
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "episode_id": "ep",
                "identities": [
                    {
                        **ident,
                        "id": pid["Stan McCormack"],
                        "display_name": "Stan McCormack",
                        "aliases": [],
                    },
                    {
                        **ident,
                        "id": pid["Dennis Aitken"],
                        "display_name": "Dennis Aitken",
                        "aliases": [alias],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    res = m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    left = {i["display_name"] for i in fx._r(bridge_path)["identities"]}
    assert left == {"Stan McCormack"}
    assert "bridge_identities_removed=1" in res.message


def test_a_non_person_alias_never_keeps_a_bridge_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only person: ids are identities. A bare label among the aliases (here equal to a stray KG
    # Person node id written without the prefix) must not keep the removed voice's person.
    p = _episode(tmp_path)
    kg = fx._r(p["kg"])
    kg["nodes"].append({"id": "Stan", "type": "Person", "properties": {"name": "Stan"}})
    p["kg"].write_text(json.dumps(kg), encoding="utf-8")
    bridge_path = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "episode_id": "ep",
                "identities": [
                    {
                        "id": "person:unresolved-dennis-ep",
                        "type": "person",
                        "display_name": "Dennis Aitken",
                        "aliases": ["Stan"],
                        "sources": {"gi": True, "kg": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    assert [i["display_name"] for i in fx._r(bridge_path)["identities"]] == []


def test_a_bridge_identity_stays_while_gi_keeps_its_person_through_a_mention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The person the rewrite takes off every voice may stay in GI because an insight mentions
    # them; their bridge identity then stays too (found on the 9a7118eb4 clone: 14 episodes).
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    pid = {n["properties"]["name"]: n["id"] for n in gi["nodes"] if n["type"] == "Person"}
    gi["edges"].append(
        {"type": "MENTIONS_PERSON", "from": "insight:other", "to": pid["Dennis Aitken"]}
    )
    p["gi"].write_text(json.dumps(gi), encoding="utf-8")
    bridge_path = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "episode_id": "ep",
                "identities": [
                    {
                        "id": pid["Dennis Aitken"],
                        "type": "person",
                        "display_name": "Dennis Aitken",
                        "aliases": [],
                        "sources": {"gi": True, "kg": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    after = fx._r(p["gi"])
    assert pid["Dennis Aitken"] in {n["id"] for n in after["nodes"] if n["type"] == "Person"}
    assert [i["display_name"] for i in fx._r(bridge_path)["identities"]] == ["Dennis Aitken"]


def test_the_receipts_file_has_its_own_literal_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The name is a contract (undo and the runner read it by name); a test that follows the
    # constant would pass whatever the constant became.
    assert m34.RECEIPTS_FILE == "talk_floor_names_removed.jsonl"
    assert m34.RECEIPTS_FILE != m29.RECEIPTS_FILE
    _episode(tmp_path)
    assert len(_reviewed_apply(tmp_path, monkeypatch).details["episodes"]) == 1
    assert (tmp_path / "talk_floor_names_removed.jsonl").is_file()
    assert m34.TalkFloorNamesRemovedMigration._receipts_file == "talk_floor_names_removed.jsonl"


def test_a_verdicts_file_without_the_basis_columns_is_unusable_for_this_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0034 requires a basis and a deciding line on every row (4d); a pre-basis file must not read.
    _episode(tmp_path)
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    d = tmp_path / "_verdicts_nobasis"
    d.mkdir()
    head = (
        f"# migration: {m34.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-23\n"
    )
    rows = [
        "\t".join((eid, "feeds/f/ep.metadata.json", voice, str(e[0]), "WRONG_PERSON", "read"))
        for (eid, voice), e in sorted(entries.items())
    ]
    (d / f"{m34.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m34.TalkFloorNamesRemovedMigration, "_verdicts_dir", d)
    mig = m34.TalkFloorNamesRemovedMigration()
    ok, msg = mig.verify(_ctx(tmp_path))
    assert not ok and "verdicts file unusable" in msg and "columns are" in msg
    with pytest.raises(rv.VerdictError, match="columns are"):
        mig.apply(_ctx(tmp_path))


def _ad_quote(p: Dict[str, Any]) -> Dict[str, Any]:
    gi = fx._r(p["gi"])
    pid = next(
        n["id"]
        for n in gi["nodes"]
        if n["type"] == "Person" and n["properties"]["name"] == "Dennis Aitken"
    )
    return next(
        n["properties"]
        for n in fx._r(p["gi"])["nodes"]
        if n["type"] == "Quote" and n["properties"].get("speaker_id") == pid
    )


def _type_the_ad_quote(p: Dict[str, Any], voice_type: str) -> None:
    gi = fx._r(p["gi"])
    pid = next(
        n["id"]
        for n in gi["nodes"]
        if n["type"] == "Person" and n["properties"]["name"] == "Dennis Aitken"
    )
    for n in gi["nodes"]:
        if n["type"] == "Quote" and n["properties"].get("speaker_id") == pid:
            n["properties"]["speaker_voice_type"] = voice_type
    fx._w(p["gi"], gi)


def test_a_quote_typed_commercial_stays_commercial_when_its_voice_loses_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    _type_the_ad_quote(p, "commercial")
    assert len(_reviewed_apply(tmp_path, monkeypatch).details["episodes"]) == 1
    q = _ad_quote_after(p)
    assert q["speaker_voice_type"] == "commercial"
    assert q.get("speaker_id") is None


def test_a_quote_of_a_voice_that_loses_the_name_is_typed_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    assert "speaker_voice_type" not in _ad_quote(p)
    assert len(_reviewed_apply(tmp_path, monkeypatch).details["episodes"]) == 1
    q = _ad_quote_after(p)
    assert q["speaker_voice_type"] == "unknown"
    assert q.get("speaker_id") is None


def _ad_quote_after(p: Dict[str, Any]) -> Dict[str, Any]:
    # The rewritten quote has no speaker_id any more: it is the one whose text is the ad read.
    return next(
        n["properties"]
        for n in fx._r(p["gi"])["nodes"]
        if n["type"] == "Quote" and "brought to you by" in n["properties"]["text"]
    )


def test_an_alias_kept_bridge_identity_has_its_sources_set_from_the_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The identity's own id is in neither GI nor KG; its alias is in both. Whatever sources it
    # arrived with, the rewrite sets them from what GI and KG hold now.
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    pid = {n["properties"]["name"]: n["id"] for n in gi["nodes"] if n["type"] == "Person"}
    bridge_path = p["meta"].with_name(p["meta"].name.replace(".metadata.json", ".bridge.json"))
    bridge_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "episode_id": "ep",
                "identities": [
                    {
                        "id": "person:unresolved-stan-ep",
                        "type": "person",
                        "display_name": "Stan McCormack",
                        "aliases": [pid["Stan McCormack"]],
                        "sources": {"gi": False, "kg": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    entries = m34.find_changes(tmp_path)
    monkeypatch.setattr(m34, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries)
    m34.TalkFloorNamesRemovedMigration().apply(_ctx(tmp_path))
    (ident,) = fx._r(bridge_path)["identities"]
    assert ident["display_name"] == "Stan McCormack"
    assert ident["sources"] == {"gi": True, "kg": True}
