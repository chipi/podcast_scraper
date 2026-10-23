"""m0035: the voice identity rung on served episodes with fingerprints, reviewed, rewritten.

Synthetic feed: two sibling episodes whose host voice is published as the host, and the episode
under test, where the host's voice was left unnamed. Built on m0025's corpus fixture.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import voice_identity as vi
from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0035_voice_identity_names as m35
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

MODEL = "pyannote/speaker-diarization-community-1"
HOST = [1.0, 0.0, 0.0]
GUEST = [0.0, 1.0, 0.0]


def _sibling(root: Path, run: str, eid: str, absolute: bool = False) -> None:
    base = root / "feeds" / "f" / run
    (base / "metadata").mkdir(parents=True)
    (base / "transcripts").mkdir()
    stored = (
        f"/app/output/feeds/f/{run}/transcripts/{eid}.txt" if absolute else f"transcripts/{eid}.txt"
    )
    (base / "metadata" / f"{eid}.metadata.json").write_text(
        json.dumps({"episode": {"episode_id": eid}, "content": {"transcript_file_path": stored}})
    )
    (base / "transcripts" / f"{eid}.speakers.diagnostics.json").write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "voice": "SPEAKER_00",
                        "resolved_name": "Gergely Orosz",
                        "named": True,
                        "role": "host",
                        "source": "self_intro",
                        "talk_time_s": 900,
                    },
                ],
                "voice_embeddings": {
                    "model_name": MODEL,
                    "dimension": 3,
                    "by_voice": {"SPEAKER_00": HOST},
                },
            }
        )
    )


def _feed(tmp_path: Path, ad_read: bool = False, absolute: bool = False) -> Dict[str, Any]:
    turns: List[Tuple[str, str, Optional[str], str]] = []
    for i in range(12):
        turns.append(("SPEAKER_00", "SPEAKER_00", None, f"Question {i} about the book."))
        turns.append(("SPEAKER_01", "Steve McConnell", "guest", f"Answer {i} about the book."))
    if ad_read:
        # The host reading an ad: the host's fingerprint, 1 of 25 turns (4%, under the floor).
        turns.append(("SPEAKER_02", "SPEAKER_02", None, "This episode is brought to you by."))
    p: Dict[str, Any] = fx._corpus(
        tmp_path,
        speakers=[fx._speaker("Steve McConnell", "guest", ["SPEAKER_01"], "roster", "guest")],
        turns=turns,
        known_hosts=["Gergely Orosz"],
    )
    diag = fx._r(p["diag"])
    for v in diag["voices"]:
        v["talk_time_s"] = 300
        if v["voice"] == "SPEAKER_00":
            v["named"] = False
    diag["voice_embeddings"] = {
        "model_name": MODEL,
        "dimension": 3,
        "by_voice": {"SPEAKER_00": HOST, "SPEAKER_01": GUEST, "SPEAKER_02": HOST},
    }
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    _sibling(tmp_path, "run_0", "e1", absolute)
    _sibling(tmp_path, "run_2", "e2", absolute)
    return p


def _review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entries: Dict,
    verdict: str,
    with_rows: bool = True,
) -> None:
    d = tmp_path / "_verdicts35"
    d.mkdir(exist_ok=True)
    names = {k: str(e[2] or e[0]) for k, e in entries.items()}
    rows = [
        "\t".join((eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read"))
        for (eid, voice), name in sorted(names.items())
        if with_rows
    ]
    head = (
        f"# migration: {m35.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(entries)}\n# reviewer: test\n# date: 2026-10-23\n"
    )
    (d / f"{m35.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m35.VoiceIdentityNamesMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_the_detector_names_the_hosts_unnamed_voice_from_the_feeds_fingerprints(
    tmp_path: Path,
) -> None:
    _feed(tmp_path)
    changes = m35.find_changes(tmp_path)
    assert list(changes) == [("ep", "SPEAKER_00")]
    old_name, old_role, new_name, new_role, why = changes[("ep", "SPEAKER_00")]
    assert (old_name, new_name, new_role) == (None, "Gergely Orosz", "host")
    assert "fingerprint group at 1.00" in why


def test_an_episode_without_fingerprints_is_not_touched(tmp_path: Path) -> None:
    p = _feed(tmp_path)
    diag = fx._r(p["diag"])
    del diag["voice_embeddings"]
    p["diag"].write_text(json.dumps(diag), encoding="utf-8")
    assert m35.find_changes(tmp_path) == {}


def test_a_reviewed_rename_is_written_on_every_surface_and_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    mig = m35.VoiceIdentityNamesMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert "\nGergely Orosz: " in "\n" + p["txt"].read_text(encoding="utf-8")
    for surface, names in t34.fx_surfaces(p).items():
        assert "Gergely Orosz" in names, surface
    host = {v["voice"]: v for v in fx._r(p["diag"])["voices"]}["SPEAKER_00"]
    assert host["named"] and host["resolved_name"] == "Gergely Orosz"
    # Review M4: the rung's own names never vote, and this is the rung's name.
    assert host["source"] == vi.SOURCE
    assert mig.verify(_ctx(tmp_path))[0]
    assert (tmp_path / m35.RECEIPTS_FILE).is_file()
    restored, refused = m35.undo(tmp_path)
    assert restored and not refused
    assert "Gergely Orosz:" not in p["txt"].read_text(encoding="utf-8")


def test_a_rename_the_review_reads_as_wrong_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "WRONG_PERSON")
    before = p["txt"].read_text(encoding="utf-8")
    assert m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert p["txt"].read_text(encoding="utf-8") == before


def test_review_names_a_rename_by_its_new_name_and_an_unnaming_by_its_old_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = {
        ("e", "SPEAKER_00"): (None, None, "Ann Lee", "host", "match"),
        ("e", "SPEAKER_01"): ("Bo Diaz", "guest", None, None, "clearly not"),
    }
    monkeypatch.setattr(m35, "load_list", lambda path=None: listed)
    assert m35.VoiceIdentityNamesMigration().review_names() == {
        ("e", "SPEAKER_00"): "Ann Lee",
        ("e", "SPEAKER_01"): "Bo Diaz",
    }


def test_not_registered_until_its_list_and_verdicts_exist() -> None:
    assert m35.MIGRATION_ID not in [m.id for m in get_migrations()]
    assert not m35.LIST_FILE.is_file()


def test_a_listed_voice_without_a_verdict_fails_verify_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT", with_rows=False)
    before = p["txt"].read_text(encoding="utf-8")
    ok, message = m35.VoiceIdentityNamesMigration().verify(_ctx(tmp_path))
    assert not ok and "without a verdict" in message
    assert m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert p["txt"].read_text(encoding="utf-8") == before


def test_a_voice_under_the_talk_floor_is_not_listed_even_when_it_is_the_hosts_voice(
    tmp_path: Path,
) -> None:
    # Review M2: the pipeline floors after the rung; a host-read ad must not get the name back.
    _feed(tmp_path, ad_read=True)
    assert list(m35.find_changes(tmp_path)) == [("ep", "SPEAKER_00")]


def test_siblings_whose_metadata_stores_an_absolute_path_still_vote(tmp_path: Path) -> None:
    # Review M1: a snapshot copy is not mounted at /app/output.
    _feed(tmp_path, absolute=True)
    assert list(m35.find_changes(tmp_path)) == [("ep", "SPEAKER_00")]


def test_a_name_this_migration_applied_does_not_vote_afterwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _feed(tmp_path)
    entries = m35.find_changes(tmp_path)
    monkeypatch.setattr(m35, "load_list", lambda path=None: entries)
    _review(monkeypatch, tmp_path, entries, "CORRECT")
    m35.VoiceIdentityNamesMigration().apply(_ctx(tmp_path))
    history = vi.feed_history(tmp_path / "feeds" / "f", "someone-else", MODEL)
    mine = [h for h in history if h.episode_id == "ep" and h.vec == vi._unit(HOST)]
    assert mine and all(h.name is None for h in mine)
