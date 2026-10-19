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


def _hidden_like_prod(p) -> None:
    """The stored state of an unnamed voice, as the pipeline writes it (prod, 2026-10-13 run):
    its segments typed ``unknown``, its quotes ``speaker_voice_type: unknown``, and every insight
    whose first quote it speaks hidden (``surfaceable: false``, routed to connect)."""
    rows = fx._r(p["seg"])
    for r in rows:
        if r["speaker"] == "SPEAKER_01":
            r["voice_type"] = "unknown"
    fx._w(p["seg"], rows)
    gi = fx._r(p["gi"])
    for n in gi["nodes"]:
        if n["type"] == "Quote" and n["id"] == "quote:1":
            n["properties"].update(speaker_id=None, speaker_voice_type="unknown")
            n["properties"].pop("speaker_name", None)
    gi["nodes"] += [
        {
            "id": "insight:guest",
            "type": "Insight",
            "properties": {
                "text": "The book matters.",
                "speaker": "SPEAKER_01",
                "speaker_voice_type": "unknown",
                "surfaceable": False,
                "grounded": True,
                "tier": 2,
                "routing_tag": "connect",
                "salience": 0.6667,
            },
        },
        {
            "id": "insight:host",
            "type": "Insight",
            "properties": {
                "text": "Books are worth asking about.",
                "speaker": "Gergely Orosz",
                "grounded": True,
                "tier": 2,
                "routing_tag": "surface",
                "salience": 0.7667,
            },
        },
    ]
    gi["edges"] = [e for e in gi["edges"] if e["from"] != "quote:1"] + [
        {"type": "SUPPORTED_BY", "from": "insight:guest", "to": "quote:1"},
        {"type": "SUPPORTED_BY", "from": "insight:host", "to": "quote:2"},
        {"type": "SUPPORTED_BY", "from": "insight:host", "to": "quote:1"},
    ]
    fx._w(p["gi"], gi)


def test_a_voice_that_gains_a_name_is_shown_as_a_fresh_run_would_show_it(
    tmp_path: Path, monkeypatch
) -> None:
    """#2380: naming a voice must also lift the hiding its unnamed state carried. On prod 0029 put
    the names on and left 1,779 insights named but hidden (S1n), their segments still typed
    ``unknown``. A named person carries no voice type anywhere, and its insights route on tier."""
    monkeypatch.setattr(
        m29,
        "NAMING_ARC_VOICES",
        {("ep", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "the guest")},
    )
    p = _episode(tmp_path)
    _hidden_like_prod(p)
    host_before = next(n for n in fx._r(p["gi"])["nodes"] if n["id"] == "insight:host")
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and res.details["episodes"]
    assert all("voice_type" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    nodes = {n["id"]: n for n in fx._r(p["gi"])["nodes"]}
    quote = nodes["quote:1"]["properties"]
    assert quote["speaker_id"] == "person:steve-mcconnell"
    assert "speaker_voice_type" not in quote
    guest = nodes["insight:guest"]["properties"]
    assert guest["speaker"] == "Steve McConnell"
    assert "speaker_voice_type" not in guest and "surfaceable" not in guest
    assert (guest["routing_tag"], guest["salience"]) == ("surface", 0.7667)
    assert nodes["insight:host"] == host_before  # its first quote is the host's: untouched
    assert _verify(tmp_path)[0]
    restored, refused = m29.undo(tmp_path)
    assert restored and not refused
    back = {n["id"]: n for n in fx._r(p["gi"])["nodes"]}["insight:guest"]["properties"]
    assert (back["surfaceable"], back["speaker_voice_type"]) == (False, "unknown")


def test_an_entry_the_review_judged_wrong_person_is_done_in_its_restored_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Prod, 2026-10-19: 0031 put the Espinoza voice back (no name) on the review's WRONG_PERSON
    verdict, and 0029's verify still wanted its list's "Justin Norman". Only verify reads it."""
    entry = {("ep", "SPEAKER_01"): (None, "unknown", "Steve McConnell", "guest", "x")}
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", entry)
    p = _episode(tmp_path)
    assert not _verify(tmp_path)[0]  # no verdict: the old state is still to write
    monkeypatch.setattr(m29, "restored_by_review", lambda: frozenset(entry))
    assert _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert len(_apply(tmp_path, dry_run=True).details["episodes"]) == 1  # plan and apply: as listed
    _apply(tmp_path)
    row = next(r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    assert row["speaker_label"] == "Steve McConnell"
    assert _verify(tmp_path)[0]  # 0029's own state passes too (before 0031 restores it)


def test_a_wrong_person_entry_in_neither_state_still_fails(tmp_path: Path, monkeypatch) -> None:
    entry = {("ep", "SPEAKER_01"): ("Somebody Else", "guest", "Steve McConnell", "guest", "x")}
    monkeypatch.setattr(m29, "NAMING_ARC_VOICES", entry)
    monkeypatch.setattr(m29, "restored_by_review", lambda: frozenset(entry))
    _episode(tmp_path)
    ok, message = _verify(tmp_path)
    assert not ok and "1 refused" in message


def test_the_shipped_review_restores_exactly_the_espinoza_voice() -> None:
    """Read from the shipped verdicts file against the shipped list: one WRONG_PERSON entry."""
    assert m29.restored_by_review() == {("5555e937-4e0a-4e17-abc3-68f5b77e5501", "SPEAKER_00")}
