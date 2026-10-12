"""m0029: the voices #2304's naming ladder names differently, applied to the served records.

Synthetic episodes built with m0025's fixture (the same voice-by-voice rewrite); the prod list is
replaced per test with monkeypatch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0029_naming_arc_voices_repaired as m29
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

TURNS = [
    ("SPEAKER_00", "Gergely Orosz", "host", "So, Steve, welcome to the podcast."),
    ("SPEAKER_01", "SPEAKER_01", "unknown", "Thanks for having me. I wrote the book."),
    ("SPEAKER_00", "Gergely Orosz", "host", "Tell me about the book."),
]


def _episode(tmp_path: Path):
    return fx._corpus(
        tmp_path,
        speakers=[fx._speaker("Gergely Orosz", "host", ["SPEAKER_00"], "known_hosts", "host")],
        turns=TURNS,
        known_hosts=["Gergely Orosz"],
    )


def _apply(root: Path, dry_run: bool = False):
    return m29.NamingArcVoicesRepairedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _verify(root: Path):
    return m29.NamingArcVoicesRepairedMigration().verify(MigrationContext(corpus_root=root))


def test_registered_after_0028_and_the_prod_list_is_well_formed() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index(m29.MIGRATION_ID) == ids.index("0028_transcript_chunk_times") + 1
    assert len(m29.NAMING_ARC_VOICES) == 83
    for (eid, voice), (old, old_role, new, new_role, why) in m29.NAMING_ARC_VOICES.items():
        assert eid and voice.startswith("SPEAKER_") and why
        assert (old, old_role) != (new, new_role), (eid, voice)
        assert new is not None or old is not None  # a role change on an unnamed voice is not here


def test_a_listed_voice_takes_its_name_on_every_surface_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "the title's guest")},
    )
    p = _episode(tmp_path)
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert fx._r(p["seg"])[1]["speaker_label"] == "SPEAKER_01"  # a dry run writes nothing
    res = _apply(tmp_path)
    assert res.details["refused"] == []
    row = next(r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    assert (row["speaker_label"], row["speaker_role"]) == ("Steve McConnell", "guest")
    names = {s["name"]: s["voices"] for s in fx._r(p["meta"])["content"]["speakers"]}
    assert names == {"Gergely Orosz": ["SPEAKER_00"], "Steve McConnell": ["SPEAKER_01"]}
    assert "\nSteve McConnell: Thanks for having me." in p["txt"].read_text(encoding="utf-8")
    fx._quotes_hold(p)
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m29.undo(tmp_path)
    assert restored and not refused
    assert fx._r(p["seg"])[1]["speaker_label"] == "SPEAKER_01"


def test_a_record_that_moved_on_since_the_replay_is_refused_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): ("Somebody Else", "guest", "Steve McConnell", "guest", "x")},
    )
    p = _episode(tmp_path)
    before = p["seg"].read_text(encoding="utf-8")
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the replay read" in res.details["refused"][0]["why"]
    assert p["seg"].read_text(encoding="utf-8") == before
    assert not _verify(tmp_path)[0]  # a refusal never reads as done


def test_a_named_voice_can_be_unnamed_or_change_role(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_00"): ("Gergely Orosz", "host", "Gergely Orosz", "guest", "role")},
    )
    p = _episode(tmp_path)
    _apply(tmp_path)
    assert [s["role"] for s in fx._r(p["meta"])["content"]["speakers"]] == ["guest"]
    assert {r["speaker_role"] for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_00"} == {"guest"}
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_00"): ("Gergely Orosz", "guest", None, None, "unnamed")},
    )
    _apply(tmp_path)
    assert fx._r(p["meta"])["content"]["speakers"] == []
    assert all("speaker_label" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_00")
