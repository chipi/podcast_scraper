"""m0043: a name off a voice that only reads an ad, a promo or a shout-out, typed commercial.

Record: docs/wip/changes/0043_non_participant_names_removed.md. Built on 0034's fixture: a host
and a guest, and a third voice whose only line is an ad, seated as a guest under its reader's name.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0043_non_participant_names_removed as m43
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0034_tests", Path(__file__).with_name("test_m0034_talk_floor_names_removed.py")
)
assert _spec and _spec.loader
t34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t34)
fx = t34.fx

READER = "Dennis Aitken"
AD = ("ep", "SPEAKER_02")
ENTRY = {AD: (READER, "guest", None, None, "class B: a promo; segment 20")}


def _episode(tmp_path: Path) -> Dict[str, Any]:
    p: Dict[str, Any] = t34._episode(tmp_path)
    meta = fx._r(p["meta"])
    meta["content"]["detected_guests"] = ["Stan McCormack", READER]  # the pre-1.2.0 projection
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    diag = fx._r(p["diag"])
    for v in diag["voices"]:
        v["voice_type"] = "person"
        v["talk_time_s"] = 1.0
    diag["summary"] = {
        "by_voice_type": {"person": 3},
        "voice_census": {"person": {"count": 3, "talk_s": 21.0, "talk_share": 1.0}},
    }
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    return p


def _review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entries: Dict = ENTRY,
    verdict: str = "WRONG_PERSON",
    basis: str = "non_participant",
) -> None:
    d = tmp_path / "_verdicts43"
    d.mkdir(exist_ok=True)
    rows = [
        "\t".join(
            (eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, str(e[0]), verdict, "read")
            + (basis, "20", rv.line_sha256("read"))
        )
        for (eid, voice), e in sorted(entries.items())
    ]
    head = (
        f"# migration: {m43.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-11-04\n"
    )
    (d / f"{m43.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS_WITH_BASIS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m43.NonParticipantNamesRemovedMigration, "_verdicts_dir", d)
    monkeypatch.setattr(m43, "load_list", lambda path=None: entries)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_the_reader_loses_the_name_and_the_voice_is_commercial_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    _review(monkeypatch, tmp_path)
    mig = m43.NonParticipantNamesRemovedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    for surface, names in t34.fx_surfaces(p).items():
        assert READER not in names, surface
        assert "Stan McCormack" in names, surface
    rows = t34._rows(fx._r(p["seg"]))
    assert {r.get("voice_type") for r in rows if r["speaker"] == "SPEAKER_02"} == {"commercial"}
    assert all("voice_type" not in r for r in rows if r["speaker"] != "SPEAKER_02")
    # The pipeline's shape for a commercial voice: no role on its segments, role unknown in the
    # diagnostics (a commercial voice is nobody's seat; step 7, 2026-11-04).
    assert all("speaker_role" not in r for r in rows if r["speaker"] == "SPEAKER_02")
    assert {r.get("speaker_role") for r in rows if r["speaker"] == "SPEAKER_00"} == {"host"}
    diag = fx._r(p["diag"])
    assert {v["voice"]: v["role"] for v in diag["voices"]}["SPEAKER_02"] == "unknown"
    by_voice = {v["voice"]: v for v in diag["voices"]}
    assert by_voice["SPEAKER_02"]["voice_type"] == "commercial"
    assert by_voice["SPEAKER_00"]["voice_type"] == "person"
    assert diag["summary"]["by_voice_type"] == {"person": 2, "commercial": 1}
    census = diag["summary"]["voice_census"]
    assert census["commercial"]["count"] == 1 and census["person"]["count"] == 2
    assert abs(census["person"]["talk_share"] + census["commercial"]["talk_share"] - 1.0) < 1e-6
    meta = fx._r(p["meta"])["content"]
    assert meta["detected_guests"] == ["Stan McCormack"]
    gi = fx._r(p["gi"])
    ad_quotes = [
        n
        for n in gi["nodes"]
        if n["type"] == "Quote" and "brought to you" in n["properties"]["text"]
    ]
    assert ad_quotes and {q["properties"].get("speaker_voice_type") for q in ad_quotes} == {
        "commercial"
    }
    assert mig.verify(_ctx(tmp_path))[0]
    restored, refused = m43.undo(tmp_path)
    assert restored and not refused
    assert f"{READER}:" in p["txt"].read_text(encoding="utf-8")
    assert fx._r(p["meta"])["content"]["detected_guests"] == ["Stan McCormack", READER]


def test_a_voice_the_review_reads_as_a_participant_keeps_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    before = p["seg"].read_text(encoding="utf-8")
    _review(monkeypatch, tmp_path, verdict="CORRECT", basis="participant")
    res = m43.NonParticipantNamesRemovedMigration().apply(_ctx(tmp_path))
    assert res.details["episodes"] == []
    assert p["seg"].read_text(encoding="utf-8") == before


@pytest.mark.parametrize("basis", ["another_person", "self_intro", "undecidable"])
def test_a_verdict_on_another_question_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, basis: str
) -> None:
    """0043 asks "does this voice only read an ad, a promo or a shout-out?"; a basis from
    0034's question (is the name this voice's own) answers something else."""
    _episode(tmp_path)
    verdict = "CORRECT" if basis != "another_person" else "WRONG_PERSON"
    _review(monkeypatch, tmp_path, verdict=verdict, basis=basis)
    ok, msg = m43.NonParticipantNamesRemovedMigration().verify(_ctx(tmp_path))
    assert not ok and "verdicts file unusable" in msg


def test_without_verdicts_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    monkeypatch.setattr(m43, "load_list", lambda path=None: ENTRY)
    monkeypatch.setattr(m43.NonParticipantNamesRemovedMigration, "_verdicts_dir", tmp_path / "x")
    before = p["seg"].read_text(encoding="utf-8")
    mig = m43.NonParticipantNamesRemovedMigration()
    assert "refused" in mig.plan(_ctx(tmp_path))
    with pytest.raises(rv.VerdictError):
        mig.apply(_ctx(tmp_path))
    assert p["seg"].read_text(encoding="utf-8") == before


def test_a_name_on_another_voice_stays_in_detected_guests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name leaves detected_guests only when no roster entry keeps it."""
    p = _episode(tmp_path)
    meta = fx._r(p["meta"])
    meta["content"]["speakers"].append(fx._speaker(READER, "guest", [], "metadata", "other"))
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    _review(monkeypatch, tmp_path)
    m43.NonParticipantNamesRemovedMigration().apply(_ctx(tmp_path))
    content = fx._r(p["meta"])["content"]
    kept = [s for s in content["speakers"] if s["name"] == READER]
    assert kept and all(not s["voices"] for s in kept)  # the metadata's own unplaced statement
    assert READER in content["detected_guests"]


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ((READER, "guest", "Someone Else", "guest", "class B: x; segment 1"), "only takes a name"),
        ((None, None, None, None, "class B: x; segment 1"), "only takes a name"),
        ((READER, "guest", None, None, "a promo; segment 1"), "names no class"),
        ((READER, "guest", None, None, "class B: a promo"), "names no class"),
        ((READER, "guest", None, None, "class D: a promo; segment 1"), "names no class"),
    ],
)
def test_the_list_refuses_an_entry_it_cannot_apply(
    tmp_path: Path, entry: tuple, problem: str
) -> None:
    path = tmp_path / "list.json"
    m43.m34.write_list(path, {AD: entry}, m43.MIGRATION_ID)
    with pytest.raises(m43.ListError, match=problem):
        m43.load_list(path)


def test_the_backups_and_receipts_are_its_own() -> None:
    assert m43.BACKUP_TAG == "0043" and m43.RECEIPTS_FILE == "non_participant_names_removed.jsonl"
    mig = m43.NonParticipantNamesRemovedMigration()
    assert (mig._backup_tag, mig._receipts_file) == (m43.BACKUP_TAG, m43.RECEIPTS_FILE)


def test_registered_after_0042_with_its_list_and_verdicts_shipped() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids[ids.index(m43.MIGRATION_ID) - 1] == "0042_insight_cluster_quotes_resynced"
    assert m43.verdicts_file().is_file() and m43.list_file().is_file()
    listed = m43.load_list()
    mig = m43.NonParticipantNamesRemovedMigration()
    verdicts = mig._verdicts()
    assert len(listed) == 113
    assert all(verdicts.get(*key) is not None for key in listed)
    applied, missing = mig._reviewed()
    assert missing == [] and len(applied) == 111


def test_a_name_leaving_the_episode_leaves_detected_hosts_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 2, 2026-11-04: only detected_guests was tested."""
    p = _episode(tmp_path)
    meta = fx._r(p["meta"])
    meta["content"]["detected_hosts"] = ["Gerald Orban", READER]
    p["meta"].write_text(json.dumps(meta), encoding="utf-8")
    _review(monkeypatch, tmp_path)
    m43.NonParticipantNamesRemovedMigration().apply(_ctx(tmp_path))
    assert fx._r(p["meta"])["content"]["detected_hosts"] == ["Gerald Orban"]


def test_a_gi_node_already_unknown_stays_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _episode(tmp_path)
    gi = fx._r(p["gi"])
    other = next(
        n for n in gi["nodes"] if n["type"] == "Quote" and "Answer 0" in n["properties"]["text"]
    )
    other["properties"]["speaker_voice_type"] = "unknown"
    p["gi"].write_text(json.dumps(gi), encoding="utf-8")
    _review(monkeypatch, tmp_path)
    m43.NonParticipantNamesRemovedMigration().apply(_ctx(tmp_path))
    after = {n["id"]: n for n in fx._r(p["gi"])["nodes"]}
    assert after[other["id"]]["properties"]["speaker_voice_type"] == "unknown"


def test_only_rows_the_rewrite_typed_unknown_become_commercial(tmp_path: Path) -> None:
    ep = m43._Episode.__new__(m43._Episode)
    ep.counts = {}
    rows = [
        {"speaker": "SPEAKER_02", "voice_type": "unknown", "speaker_role": "guest"},
        {"speaker": "SPEAKER_02", "voice_type": "cameo", "speaker_role": "guest"},
        {"speaker": "SPEAKER_00", "voice_type": "unknown", "speaker_role": "host"},
    ]
    ep.json_files = {tmp_path / "ep.segments.json": rows}
    ep._type_segments({"SPEAKER_02"})
    assert rows[0] == {"speaker": "SPEAKER_02", "voice_type": "commercial"}
    assert rows[1]["voice_type"] == "cameo" and rows[2]["voice_type"] == "unknown"


@pytest.mark.parametrize("old", [None, "person", "unidentified"])
def test_the_census_moves_one_voice_to_commercial(old) -> None:
    """Revision 4d: the shares come from the total after the move, so they sum to 1 whether the
    voice leaves a counted type, had no type, or names a type the census has no bucket for."""
    summary: Dict[str, Any] = {
        "by_voice_type": {"person": 2},
        "voice_census": {"person": {"count": 2, "talk_s": 30.0, "talk_share": 1.0}},
    }
    m43._move_census(summary, old, 10.0)
    assert summary["by_voice_type"]["commercial"] == 1
    assert summary["by_voice_type"]["person"] == (1 if old == "person" else 2)
    census = summary["voice_census"]
    assert census["commercial"]["count"] == 1 and census["commercial"]["talk_s"] == 10.0
    total = sum(c["talk_s"] for c in census.values())
    assert total == (30.0 if old == "person" else 40.0)
    assert abs(sum(c["talk_share"] for c in census.values()) - 1.0) < 1e-3
    assert census["commercial"]["talk_share"] == round(10.0 / total, 4)
