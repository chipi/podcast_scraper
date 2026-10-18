"""m0031: 0029's voices through the fixed rewrite — named where a reprocess unnamed them again,
shown where 0029 left them hidden under their name (#2380).

Synthetic episodes built with m0025's fixture; the prod list is replaced per test with monkeypatch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import (
    m0029_naming_arc_voices_repaired as m29,
    m0031_named_voices_shown as m31,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

_spec = importlib.util.spec_from_file_location(
    "m0025_fixture", Path(__file__).with_name("test_m0025_one_person_one_entry.py")
)
assert _spec and _spec.loader
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

ENTRY = {("ep", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "the guest")}


def _episode(tmp_path: Path, label: str, role: str, typed: bool):
    """Host + guest; the guest's voice carries ``label``/``role``, typed and hidden if ``typed``."""
    speakers = [fx._speaker("Gergely Orosz", "host", ["SPEAKER_00"], "known_hosts", "host")]
    if not label.startswith("SPEAKER_"):  # the record 0029 wrote carries the guest's entry
        speakers.append(fx._speaker(label, role, ["SPEAKER_01"], "roster", "guest"))
    p = fx._corpus(
        tmp_path,
        speakers=speakers,
        turns=[
            ("SPEAKER_00", "Gergely Orosz", "host", "So, Steve, welcome to the podcast."),
            ("SPEAKER_01", label, role, "Thanks for having me. I wrote the book."),
            ("SPEAKER_00", "Gergely Orosz", "host", "Tell me about the book."),
        ],
        known_hosts=["Gergely Orosz"],
    )
    rows = fx._r(p["seg"])
    gi = fx._r(p["gi"])
    hidden = {"speaker_voice_type": "unknown", "surfaceable": False} if typed else {}
    if typed:
        for r in rows:
            if r["speaker"] == "SPEAKER_01":
                r["voice_type"] = "unknown"
        for n in gi["nodes"]:
            if n["id"] == "quote:1":
                n["properties"]["speaker_voice_type"] = "unknown"
    gi["nodes"].append(
        {
            "id": "insight:guest",
            "type": "Insight",
            "properties": {
                "text": "The book matters.",
                "speaker": label,
                "grounded": True,
                "tier": 2,
                "routing_tag": "connect" if typed else "surface",
                "salience": 0.6667 if typed else 0.7667,
                **hidden,
            },
        }
    )
    gi["edges"].append({"type": "SUPPORTED_BY", "from": "insight:guest", "to": "quote:1"})
    fx._w(p["seg"], rows)
    fx._w(p["gi"], gi)
    return p


def _mig() -> m31.NamedVoicesShownMigration:
    return m31.NamedVoicesShownMigration()


def _apply(root: Path):
    return _mig().apply(MigrationContext(corpus_root=root))


def _guest(p) -> dict:
    props: dict = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "insight:guest")[
        "properties"
    ]
    return props


def test_registered_last_right_after_0030() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids[-1] == m31.MIGRATION_ID
    assert ids.index(m31.MIGRATION_ID) == ids.index("0030_adfree_cut_resynced") + 1
    versions = [m.to_version for m in get_migrations()][-2:]
    assert versions == ["2.7.24", "2.7.25"]
    assert m31.BACKUP_TAG != m29.BACKUP_TAG and m31.RECEIPTS_FILE != m29.RECEIPTS_FILE


def test_a_voice_0029_named_but_left_hidden_is_shown_and_keeps_every_label(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    text_before = p["txt"].read_text(encoding="utf-8")
    meta_before = fx._r(p["meta"])["content"]["speakers"]
    assert not _mig().verify(MigrationContext(corpus_root=tmp_path))[0]
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and len(res.details["episodes"]) == 1
    assert res.details["totals"]["voices_shown"] == 1
    assert all("voice_type" not in r for r in fx._r(p["seg"]))
    assert {r["speaker_label"] for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01"} == {
        "Steve McConnell"
    }
    assert p["txt"].read_text(encoding="utf-8") == text_before
    assert fx._r(p["meta"])["content"]["speakers"] == meta_before
    guest = _guest(p)
    assert guest["speaker"] == "Steve McConnell"
    assert "surfaceable" not in guest and "speaker_voice_type" not in guest
    assert guest["routing_tag"] == "surface"
    quote = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "quote:1")["properties"]
    assert "speaker_voice_type" not in quote
    fx._quotes_hold(p)
    assert _mig().verify(MigrationContext(corpus_root=tmp_path))[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m31.undo(tmp_path)
    assert restored and not refused
    assert _guest(p)["surfaceable"] is False


def test_a_voice_a_reprocess_unnamed_again_is_named_and_shown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    p = _episode(tmp_path, "SPEAKER_01", "unknown", typed=True)
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and res.details["totals"]["voices_named"] == 1
    assert "\nSteve McConnell: Thanks for having me." in p["txt"].read_text(encoding="utf-8")
    guest = _guest(p)
    assert (guest["speaker"], guest["routing_tag"]) == ("Steve McConnell", "surface")
    assert "surfaceable" not in guest


def test_a_named_untyped_voice_is_done_and_nothing_is_written(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    p = _episode(tmp_path, "Steve McConnell", "guest", typed=False)
    before = {k: p[k].read_bytes() for k in ("seg", "gi", "meta", "txt")}
    assert _apply(tmp_path).details["episodes"] == []
    assert {k: p[k].read_bytes() for k in before} == before
    assert _mig().verify(MigrationContext(corpus_root=tmp_path))[0]


def test_an_unnaming_entry_is_not_this_migrations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): ("Steve McConnell", "guest", None, None, "unnamed")},
    )
    _episode(tmp_path, "Steve McConnell", "guest", typed=True)
    assert _apply(tmp_path).details["episodes"] == []


def test_a_record_that_moved_on_is_refused_untouched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", ENTRY)
    p = _episode(tmp_path, "Somebody Else", "guest", typed=True)
    before = p["seg"].read_bytes()
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the replay read" in res.details["refused"][0]["why"]
    assert p["seg"].read_bytes() == before
    assert not _mig().verify(MigrationContext(corpus_root=tmp_path))[0]
