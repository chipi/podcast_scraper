"""m0032 (0031 until 2026-10-18): the names the deployed naming code gives the flagged served
episodes, without a relabel.

The rewrite is m0029's (``m0029._Episode`` with this list); synthetic episodes come from m0025's
fixture and the prod list is replaced per test with monkeypatch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0032_naming_voices_rederived as m31
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
    ("SPEAKER_01", "Kara Swisher", "guest", "Hi, everyone. I'm Kara Swisher. Twice a week, Pivot."),
    ("SPEAKER_00", "Gergely Orosz", "host", "Tell me about the book."),
]


def _episode(tmp_path: Path):
    return fx._corpus(
        tmp_path,
        speakers=[
            fx._speaker("Gergely Orosz", "host", ["SPEAKER_00"], "known_hosts", "host"),
            fx._speaker("Kara Swisher", "guest", ["SPEAKER_01"], "llm", "guest"),
        ],
        turns=TURNS,
        known_hosts=["Gergely Orosz"],
    )


def _apply(root: Path, dry_run: bool = False):
    return m31.NamingVoicesRederivedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _verify(root: Path):
    return m31.NamingVoicesRederivedMigration().verify(MigrationContext(corpus_root=root))


def test_held_so_not_registered_and_the_prod_list_is_well_formed() -> None:
    ids = [m.id for m in get_migrations()]
    assert m31.MIGRATION_ID not in ids  # HELD (#2380): registered only once it passes its gate
    assert m31.MIGRATION_ID > ids[-1]  # and when it is, it sorts after everything registered
    assert len(m31.NAMING_VOICES) == 166
    assert len({eid for eid, _voice in m31.NAMING_VOICES}) == 126
    for (eid, voice), (old, old_role, new, new_role, why) in m31.NAMING_VOICES.items():
        assert eid and voice.startswith("SPEAKER_") and why
        assert (old, old_role) != (new, new_role), (eid, voice)
        assert new is not None or old is not None  # a role change on an unnamed voice is not here
        assert not (new and not new_role), (eid, voice)  # a name always carries its role


def test_a_promo_voice_loses_its_name_on_every_surface_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m31,
        "NAMING_VOICES",
        {("ep", "SPEAKER_01"): ("Kara Swisher", "guest", None, "unknown", "Pivot promo")},
    )
    p = _episode(tmp_path)
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert fx._r(p["seg"])[1]["speaker_label"] == "Kara Swisher"  # a dry run writes nothing
    res = _apply(tmp_path)
    assert res.details["refused"] == []
    assert all("speaker_label" not in r for r in fx._r(p["seg"]) if r["speaker"] == "SPEAKER_01")
    names = [s["name"] for s in fx._r(p["meta"])["content"]["speakers"]]
    assert names == ["Gergely Orosz"]
    assert "Kara Swisher:" not in p["txt"].read_text(encoding="utf-8")
    fx._quotes_hold(p)
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m31.undo(tmp_path)
    assert restored and not refused
    assert fx._r(p["seg"])[1]["speaker_label"] == "Kara Swisher"


def test_a_voice_that_moved_on_since_the_replay_is_refused_untouched(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        m31,
        "NAMING_VOICES",
        {("ep", "SPEAKER_01"): ("Scott Galloway", "guest", None, "unknown", "x")},
    )
    p = _episode(tmp_path)
    before = p["seg"].read_text(encoding="utf-8")
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and "the replay read" in res.details["refused"][0]["why"]
    assert p["seg"].read_text(encoding="utf-8") == before
    assert not _verify(tmp_path)[0]  # a refusal never reads as done


def test_a_host_seat_takes_its_name_and_a_respelling_its_spelling(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        m31,
        "NAMING_VOICES",
        {
            ("ep", "SPEAKER_00"): ("Gergely Orosz", "host", "Gergely Orosz", "guest", "role"),
            ("ep", "SPEAKER_01"): ("Kara Swisher", "guest", "Kara Swisher-Smith", "host", "spell"),
        },
    )
    p = _episode(tmp_path)
    _apply(tmp_path)
    by_name = {s["name"]: s["role"] for s in fx._r(p["meta"])["content"]["speakers"]}
    assert by_name == {"Gergely Orosz": "guest", "Kara Swisher-Smith": "host"}
    rows = {r["speaker"]: (r.get("speaker_label"), r["speaker_role"]) for r in fx._r(p["seg"])}
    assert rows == {
        "SPEAKER_00": ("Gergely Orosz", "guest"),
        "SPEAKER_01": ("Kara Swisher-Smith", "host"),
    }
    assert _verify(tmp_path)[0]


def test_the_ad_map_follows_the_rewritten_transcript(tmp_path: Path, monkeypatch) -> None:
    """Measured on the 2026-10-16 snapshot: 0030 then 0031 left 99 ad-maps pointing beside the
    ads they describe, because the rename changes the speaker prefixes they are offsets into."""
    monkeypatch.setattr(
        m31,
        "NAMING_VOICES",
        {("ep", "SPEAKER_01"): ("Kara Swisher", "guest", None, "unknown", "Pivot promo")},
    )
    p = _episode(tmp_path)
    old = p["txt"].read_text(encoding="utf-8")
    start = old.index("\nKara Swisher: ") + 1
    end = old.index("Pivot.") + len("Pivot.")
    admap_path = p["txt"].with_name("ep.adfree.admap.json")
    admap = {
        "preroll_cut_end": None,
        "postroll_cut_start": None,
        "chars_removed": end - start,
        "source_length": len(old),
        "excised_ranges": [[start, end]],
        "excised_texts": [old[start:end]],
    }
    fx._w(admap_path, admap)
    res = _apply(tmp_path)
    assert res.details["totals"].get("admaps_carried") == 1
    new = p["txt"].read_text(encoding="utf-8")
    assert new != old
    out = fx._r(admap_path)
    ((a, b),) = out["excised_ranges"]
    assert out["source_length"] == len(new)
    assert out["excised_texts"] == [new[a:b]]
    assert new[a:b].endswith("Hi, everyone. I'm Kara Swisher. Twice a week, Pivot.")
    assert new[a - 1] == "\n" and "Kara Swisher:" not in new[a:b]  # the line, its new label
    assert out["chars_removed"] == b - a
    restored, refused = m31.undo(tmp_path)
    assert restored and not refused and fx._r(admap_path) == admap


def test_the_metadata_counts_follow_the_rewritten_graph(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        m31,
        "NAMING_VOICES",
        {("ep", "SPEAKER_01"): ("Kara Swisher", "guest", None, "unknown", "Pivot promo")},
    )
    p = _episode(tmp_path)
    meta = fx._r(p["meta"])
    kg = fx._r(p["kg"])
    meta["knowledge_graph"] = {"node_count": len(kg["nodes"]), "edge_count": len(kg["edges"])}
    fx._w(p["meta"], meta)
    _apply(tmp_path)
    kg = fx._r(p["kg"])
    assert "Kara Swisher" not in {n["properties"].get("name") for n in kg["nodes"]}
    assert fx._r(p["meta"])["knowledge_graph"] == {
        "node_count": len(kg["nodes"]),
        "edge_count": len(kg["edges"]),
    }
