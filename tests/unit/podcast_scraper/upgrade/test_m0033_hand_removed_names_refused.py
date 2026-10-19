"""m0033: the voice a reprocess published under a name 0015 removed by hand, named as the review
read it, through 0031's rewrite. Synthetic episodes from m0031's test fixtures (m0025's corpus)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from podcast_scraper.speaker_detectors import hosts
from podcast_scraper.upgrade import reviewer_verdicts as rv
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0015_unpublishable_speaker_names_removed as m15,
    m0019_descriptor_speaker_names_removed as m19,
    m0031_named_voices_shown as m31,
    m0033_hand_removed_names_refused as m33,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0031_tests", Path(__file__).with_name("test_m0031_named_voices_shown.py")
)
assert _spec and _spec.loader
t31 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t31)
fx = t31.fx

ENTRY = {("ep", "SPEAKER_01"): ("Generation", "guest", "Steve McConnell", "guest", "the guest")}


def _review(monkeypatch, tmp_path: Path, verdict: str = "CORRECT") -> None:
    d = tmp_path / "_verdicts33"
    d.mkdir(exist_ok=True)
    names = m31._names(m33.VOICES)
    rows = [
        "\t".join((eid, "feeds/f/run_1/metadata/ep.metadata.json", voice, name, verdict, "read"))
        for (eid, voice), name in sorted(names.items())
    ]
    head = (
        f"# migration: {m33.MIGRATION_ID}\n# reviewed_commit: x\n"
        f"# list_sha256: {rv.list_sha256(names)}\n# reviewer: test\n# date: 2026-10-19\n"
    )
    (d / f"{m33.MIGRATION_ID}.tsv").write_text(
        head + "\t".join(rv.COLUMNS) + "\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(m33.HandRemovedNamesRefusedMigration, "_verdicts_dir", d)


def _ctx(root: Path) -> MigrationContext:
    return MigrationContext(corpus_root=root)


def test_registered_right_after_0031_with_its_own_receipts() -> None:
    migs = get_migrations()
    ids = [m.id for m in migs]
    at = ids.index(m33.MIGRATION_ID)
    assert ids[at - 1] == m31.MIGRATION_ID and migs[at].to_version == "2.7.27"
    assert len({m33.BACKUP_TAG, m31.BACKUP_TAG, "0015", "0019"}) == 4
    assert m33.RECEIPTS_FILE not in (m31.RECEIPTS_FILE, m15.RECEIPTS_FILE, m19.RECEIPTS_FILE)


def test_the_shipped_review_names_the_voice_and_matches_the_list() -> None:
    mig = m33.HandRemovedNamesRefusedMigration()
    verdicts = mig._verdicts()
    [(key, v)] = verdicts.rows.items()
    assert key in m33.VOICES and (v.name, v.verdict) == ("David Roblin", "CORRECT")
    assert m33.VOICES[key][0] in hosts.HAND_REMOVED_SPEAKER_NAMES


def test_the_voice_under_a_hand_removed_name_takes_the_reviewed_name(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    _review(monkeypatch, tmp_path)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    mig = m33.HandRemovedNamesRefusedMigration()
    assert not mig.verify(_ctx(tmp_path))[0]
    res = mig.apply(_ctx(tmp_path))
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    text = p["txt"].read_text(encoding="utf-8")
    assert "\nSteve McConnell: Thanks for having me." in text and "Generation:" not in text
    guest = t31._guest(p)
    assert guest["speaker"] == "Steve McConnell" and "surfaceable" not in guest
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert "Generation" not in names and "Steve McConnell" in names
    assert mig.verify(_ctx(tmp_path))[0]
    assert m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path)).details["episodes"] == []
    assert (tmp_path / m33.RECEIPTS_FILE).is_file()
    assert not (tmp_path / m31.RECEIPTS_FILE).exists()
    restored, refused = m33.undo(tmp_path)
    assert restored and not refused
    assert "Generation: Thanks for having me." in p["txt"].read_text(encoding="utf-8")


def test_without_a_verdict_the_episode_is_refused_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m33, "VOICES", ENTRY)
    d = tmp_path / "_none"
    monkeypatch.setattr(m33.HandRemovedNamesRefusedMigration, "_verdicts_dir", d)
    p = t31._episode(tmp_path, "Generation", "guest", typed=False)
    before = p["seg"].read_text(encoding="utf-8")
    ok, message = m33.HandRemovedNamesRefusedMigration().verify(_ctx(tmp_path))
    assert not ok and message.startswith("verdicts file unusable")
    with pytest.raises(rv.VerdictError):
        m33.HandRemovedNamesRefusedMigration().apply(_ctx(tmp_path))
    assert p["seg"].read_text(encoding="utf-8") == before


def test_0031_still_checks_its_own_list_and_receipts(tmp_path: Path, monkeypatch) -> None:
    """The list and receipts became parameters of 0031's outcome; 0031's defaults are its own."""
    assert m31.NamedVoicesShownMigration().listed() is m31.m29.NAMING_ARC_VOICES
    assert m33.HandRemovedNamesRefusedMigration().listed() is m33.VOICES
    assert m31._written_metas(tmp_path) == set()


def test_0015_and_0019_read_the_gates_list() -> None:
    assert m15.REMOVE is hosts.HAND_REMOVED_SPEAKER_NAMES
    assert m19.REMOVE is hosts.HAND_REMOVED_SPEAKER_NAMES
