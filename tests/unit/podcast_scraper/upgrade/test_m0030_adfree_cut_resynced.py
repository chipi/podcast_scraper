"""m0030: served episodes carry today's ad cut, every offset into the ad-free text carried across.

Prod shape (#2360, 2026-10-15 snapshot): the stored ad-free text cut a pre-roll that today's code
keeps as content, or kept a mid-roll that today's code cuts, and GI quotes point into it by
character offset. Today's cut is stood in for here by a render of the segments that are not ads,
so the test pins what the migration does with a changed cut, not what the cut detects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0030_adfree_cut_resynced as m
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.workflow.adfree_transcript import AdfreeArtifacts
from podcast_scraper.workflow.turns_artifact import build_turns_document

pytestmark = [pytest.mark.unit]

TX = "transcripts/ep.txt"
INTRO = "Welcome back to the show, today we talk rockets."
AD = "This episode is brought to you by Acme mattresses."
QUOTE = "The engine burns methane and oxygen."
ROWS: List[Dict[str, Any]] = [
    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "speaker_label": "Ann", "text": INTRO},
    {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01", "speaker_label": "Bo", "text": "Thanks."},
    {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00", "speaker_label": "Ann", "text": AD},
    {"start": 12.0, "end": 16.0, "speaker": "SPEAKER_01", "speaker_label": "Bo", "text": QUOTE},
]


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), "utf-8")


def _r(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _todays_cut(text: str, segments: Any, **_kw: Any) -> AdfreeArtifacts:
    text, segs = format_diarized_screenplay_with_offsets([s for s in segments if s["text"] != AD])
    return AdfreeArtifacts(text=text, segments=segs, ad_map={"cut": "today"}, chars_removed=1)


@pytest.fixture(autouse=True)
def _cut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "build_adfree_artifacts", _todays_cut)


def _corpus(root: Path, stored_rows: List[Dict[str, Any]], quote: str = QUOTE) -> Dict[str, Path]:
    run = root / "feeds" / "f" / "run_1"
    p = {
        "meta": run / "metadata" / "ep.metadata.json",
        "gi": run / "metadata" / "ep.gi.json",
        "txt": run / "transcripts" / "ep.txt",
        "seg": run / "transcripts" / "ep.segments.json",
        "atxt": run / "transcripts" / "ep.adfree.txt",
        "aseg": run / "transcripts" / "ep.adfree.segments.json",
        "amap": run / "transcripts" / "ep.adfree.admap.json",
        "turns": run / "transcripts" / "ep.adfree.turns.json",
    }
    _w(p["meta"], {"episode": {"episode_id": "ep"}, "content": {"transcript_file_path": TX}})
    raw, _ = format_diarized_screenplay_with_offsets(ROWS)
    _w(p["txt"], raw)
    _w(p["seg"], ROWS)
    old, old_segs = format_diarized_screenplay_with_offsets(stored_rows)
    _w(p["atxt"], old)
    _w(p["aseg"], old_segs)
    _w(p["amap"], {"cut": "old"})
    _w(
        p["turns"],
        build_turns_document(
            old, old_segs, rel_transcript_path=TX, episode_slug="ep", language="en"
        ),
    )
    cs = old.index(quote)
    _w(
        p["gi"],
        {
            "nodes": [
                {"id": "insight:1", "type": "Insight", "properties": {"text": "x"}},
                {
                    "id": "quote:keep-me",
                    "type": "Quote",
                    "properties": {
                        "text": quote,
                        "char_start": cs,
                        "char_end": cs + len(quote),
                        "transcript_ref": "/app/output/feeds/f/run_1/" + TX[:-4] + ".adfree.txt",
                    },
                },
            ],
            "edges": [
                {"type": "HAS_INSIGHT", "from": "episode:ep", "to": "insight:1"},
                {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:keep-me"},
            ],
        },
    )
    return p


def _run(root: Path, dry: bool = False) -> Any:
    return m.AdfreeCutResyncedMigration().apply(MigrationContext(corpus_root=root, dry_run=dry))


def _quote(p: Dict[str, Path]) -> Dict[str, Any]:
    props: Dict[str, Any] = next(n for n in _r(p["gi"])["nodes"] if n["type"] == "Quote")[
        "properties"
    ]
    return props


def _assert_today(p: Dict[str, Path]) -> None:
    new, segs = format_diarized_screenplay_with_offsets([s for s in ROWS if s["text"] != AD])
    assert p["atxt"].read_text("utf-8") == new
    assert _r(p["aseg"]) == segs
    assert _r(p["amap"]) == {"cut": "today"}
    q = _quote(p)
    assert new[q["char_start"] : q["char_end"]] == QUOTE


def test_a_restored_intro_moves_every_quote_after_it(tmp_path: Path) -> None:
    """The #2360 shape: the old cut took the opening turn, today's keeps it."""
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    result = _run(tmp_path)
    assert result.details["totals"]["episodes_recut"] == 1
    assert result.details["totals"]["segments_restored"] == 1
    _assert_today(p)
    turns = _r(p["turns"])
    assert turns["source"]["segments_sha256"]
    assert turns["turns"][0]["char_start"] == len("Ann: ")


def test_a_newly_cut_mid_roll_moves_the_quote_after_it_back(tmp_path: Path) -> None:
    """The #2168 shape: the old cut kept an ad mid-episode, today's removes it."""
    p = _corpus(tmp_path, ROWS)
    assert _run(tmp_path).details["totals"]["segments_cut"] == 1
    _assert_today(p)


def test_the_quote_keeps_its_id(tmp_path: Path) -> None:
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    _run(tmp_path)
    assert [n["id"] for n in _r(p["gi"])["nodes"]] == ["insight:1", "quote:keep-me"]


def _shift_quote(p: Dict[str, Path], by: int, text: str = QUOTE) -> None:
    gi = _r(p["gi"])
    props = gi["nodes"][1]["properties"]
    props["char_start"] += by
    props["char_end"] += by
    props["text"] = text
    _w(p["gi"], gi)


def test_a_quote_already_off_its_text_is_placed_on_its_one_occurrence(tmp_path: Path) -> None:
    """Prod: one episode's 73 quotes all one character early, before any cut changed."""
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    _shift_quote(p, -1)
    gi = _r(p["gi"])
    gi["nodes"][1]["properties"].update(timestamp_start_ms=4000, timestamp_end_ms=8000)
    _w(p["gi"], gi)
    totals = _run(tmp_path).details["totals"]
    assert totals["quotes_reanchored"] == 1 and "refused" not in totals
    _assert_today(p)
    # Its times came from the same wrong offsets: recomputed from where its words are.
    assert (_quote(p)["timestamp_start_ms"], _quote(p)["timestamp_end_ms"]) == (12000, 16000)


def test_a_quote_off_its_text_that_is_not_there_once_moves_with_its_place(tmp_path: Path) -> None:
    """Prod: fragments like "the companies that" stored off their text and said 2-4 times. They
    must not hold the episode's cut: moved with their place, no worse than before, and counted."""
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    _shift_quote(p, -1, text="Words nobody said.")
    old = p["atxt"].read_text("utf-8")
    cs = _quote(p)["char_start"]
    totals = _run(tmp_path).details["totals"]
    assert totals["quotes_already_off_left"] == 1 and "refused" not in totals
    new = p["atxt"].read_text("utf-8")
    assert _quote(p)["char_start"] == cs + (len(new) - len(old))


def test_a_quote_off_its_text_in_text_the_cut_now_removes_goes(tmp_path: Path) -> None:
    p = _corpus(tmp_path, ROWS, quote=AD)
    _shift_quote(p, 2, text="Words nobody said.")
    totals = _run(tmp_path).details["totals"]
    assert totals["quotes_already_off_left"] == 1 and totals["quotes_dropped_as_ad_text"] == 1
    assert not [n for n in _r(p["gi"])["nodes"] if n["type"] == "Quote"]


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    before = {k: v.read_bytes() for k, v in p.items()}
    assert "would rewrite 1" in _run(tmp_path, dry=True).message
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_second_run_writes_nothing_and_verify_passes(tmp_path: Path) -> None:
    _corpus(tmp_path, [ROWS[1], ROWS[3]])
    ctx = MigrationContext(corpus_root=tmp_path)
    assert m.AdfreeCutResyncedMigration().verify(ctx)[0] is False
    _run(tmp_path)
    assert _run(tmp_path).details["files_written"] == 0
    assert m.AdfreeCutResyncedMigration().verify(ctx) == (
        True,
        "every served ad-free base is today's cut",
    )


def test_undo_restores_every_file(tmp_path: Path) -> None:
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    before = {k: v.read_bytes() for k, v in p.items()}
    _run(tmp_path)
    restored, refused = m.undo(tmp_path)
    assert restored == 5 and refused == []
    assert {k: v.read_bytes() for k, v in p.items()} == before


def test_a_quote_in_text_the_cut_now_removes_goes_and_so_does_an_insight_it_alone_held(
    tmp_path: Path,
) -> None:
    """The pipeline drops an ad-span quote; an insight with no other quote restated the ad."""
    p = _corpus(tmp_path, ROWS, quote=AD)
    result = _run(tmp_path)
    assert not result.details["refused"]
    totals = result.details["totals"]
    assert totals["quotes_dropped_as_ad_text"] == 1 and totals["insights_removed_as_ad_text"] == 1
    gi = _r(p["gi"])
    assert gi["nodes"] == [] and gi["edges"] == []
    assert m.AdfreeCutResyncedMigration().verify(MigrationContext(corpus_root=tmp_path))[0]


def _with_graph(p: Dict[str, Path], entities: List[Dict[str, Any]]) -> None:
    """Entities on GI and KG, a bridge row each, and the metadata counts and context they feed."""
    gi = _r(p["gi"])
    gi["nodes"] += [{k: v for k, v in e.items()} for e in entities]
    gi["edges"] += [{"type": "MENTIONS_ORG", "from": "insight:1", "to": e["id"]} for e in entities]
    _w(p["gi"], gi)
    kg_path = p["gi"].with_name("ep.kg.json")
    kg_nodes = [{"id": "episode:ep", "type": "Episode", "properties": {}}] + entities
    kg_edges = [{"type": "MENTIONS", "from": e["id"], "to": "episode:ep"} for e in entities]
    _w(kg_path, {"nodes": kg_nodes, "edges": kg_edges})
    _w(
        p["gi"].with_name("ep.bridge.json"),
        {"identities": [{"id": e["id"], "sources": {"gi": True, "kg": True}} for e in entities]},
    )
    meta = _r(p["meta"])
    meta["grounded_insights"] = {"insight_count": 1}
    meta["knowledge_graph"] = {"node_count": len(kg_nodes), "edge_count": len(kg_edges)}
    _w(p["meta"], meta)
    p["kg"], p["bridge"] = kg_path, p["gi"].with_name("ep.bridge.json")


def _org(name: str, **props: Any) -> Dict[str, Any]:
    return {
        "id": "org:" + name.lower(),
        "type": "Organization",
        "properties": {"name": name, **props},
    }


def test_an_entity_named_only_in_the_cut_ad_leaves_gi_kg_and_the_bridge(tmp_path: Path) -> None:
    p = _corpus(tmp_path, ROWS)
    _with_graph(p, [_org("Acme")])
    assert _run(tmp_path).details["totals"]["entities_removed_as_ad_text"] == 1
    assert [n["id"] for n in _r(p["kg"])["nodes"]] == ["episode:ep"] and _r(p["kg"])["edges"] == []
    assert "org:acme" not in {n["id"] for n in _r(p["gi"])["nodes"]}
    assert _r(p["bridge"])["identities"] == []
    assert _r(p["meta"])["knowledge_graph"] == {"node_count": 1, "edge_count": 0}
    assert _r(p["meta"])["grounded_insights"] == {"insight_count": 1}


def test_the_context_digest_drops_a_removed_sponsor_from_its_companies(tmp_path: Path) -> None:
    p = _corpus(tmp_path, ROWS)
    _with_graph(p, [_org("Acme")])
    ctx = p["gi"].with_name("ep.context.json")
    _w(ctx, {"episode_id": "ep", "companies": ["Acme"], "people": [], "basic": {}})
    _run(tmp_path)
    assert _r(ctx)["companies"] == []


def test_a_turns_invariant_error_refuses_the_episode_and_the_scan_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.providers.ml.diarization.turns import TurnInvariantError

    def broken(*_a: Any, **_k: Any) -> None:
        raise TurnInvariantError("turn_0003 has char_end < char_start")

    monkeypatch.setattr(m, "build_turns_document", broken)
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    before = p["atxt"].read_bytes()
    result = _run(tmp_path)
    assert result.details["refused"] == ["feeds/f/run_1/metadata/ep.metadata.json"]
    assert p["atxt"].read_bytes() == before


def test_an_entity_also_named_in_kept_text_stays(tmp_path: Path) -> None:
    """A name in the cut ad AND in kept talk ("to": "brought to you", "back to the show")."""
    assert " to " in AD and " to " in INTRO
    p = _corpus(tmp_path, ROWS)
    _with_graph(p, [_org("to")])
    before = (p["kg"].read_bytes(), p["bridge"].read_bytes())
    _run(tmp_path)
    assert (p["kg"].read_bytes(), p["bridge"].read_bytes()) == before


def test_a_host_or_guest_is_never_removed(tmp_path: Path) -> None:
    p = _corpus(tmp_path, ROWS)
    person = {
        "id": "person:acme",
        "type": "Person",
        "properties": {"name": "Acme", "role": "guest"},
    }
    _with_graph(p, [person])
    _run(tmp_path)
    assert "person:acme" in {n["id"] for n in _r(p["kg"])["nodes"]}


def test_an_insight_that_never_had_a_quote_is_not_this_migrations_to_remove(
    tmp_path: Path,
) -> None:
    p = _corpus(tmp_path, ROWS, quote=AD)
    gi = _r(p["gi"])
    gi["nodes"].append({"id": "insight:2", "type": "Insight", "properties": {"text": "y"}})
    _w(p["gi"], gi)
    _run(tmp_path)
    assert [n["id"] for n in _r(p["gi"])["nodes"]] == ["insight:2"]


def test_an_insight_with_a_quote_left_in_kept_text_stays(tmp_path: Path) -> None:
    p = _corpus(tmp_path, ROWS)
    _run(tmp_path)
    assert "insight:1" in {n["id"] for n in _r(p["gi"])["nodes"]}


def test_a_kept_segment_that_changed_more_than_its_place_refuses(tmp_path: Path) -> None:
    """A role on the stored ad-free row the raw row lacks: rebuilding would drop it."""
    p = _corpus(tmp_path, [ROWS[1], {**ROWS[3], "speaker_role": "guest"}])
    before = p["aseg"].read_bytes()
    assert _run(tmp_path).details["refused"]
    assert p["aseg"].read_bytes() == before


def test_a_value_the_stored_row_never_had_is_not_a_change(tmp_path: Path) -> None:
    """A plain transcript's row stored with no label; today's build carries the raw row's."""
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    _w(p["aseg"], [{**row, "speaker_label": None} for row in _r(p["aseg"])])
    assert not _run(tmp_path).details["refused"]
    _assert_today(p)


def test_an_absolute_transcript_path_is_read_in_this_corpus(tmp_path: Path) -> None:
    """Older runs store "/app/output/feeds/…"; off prod that path names nothing."""
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    _w(
        p["meta"],
        {
            "episode": {"episode_id": "ep"},
            "content": {"transcript_file_path": "/app/output/feeds/f/run_1/" + TX},
        },
    )
    assert _run(tmp_path).details["totals"]["episodes_recut"] == 1
    _assert_today(p)


def test_the_same_text_with_an_older_ad_map_rewrites_only_the_map(tmp_path: Path) -> None:
    p = _corpus(tmp_path, [r for r in ROWS if r["text"] != AD])
    gi_before = p["gi"].read_bytes()
    totals = _run(tmp_path).details["totals"]
    assert totals == {"admaps_rewritten": 1}
    assert _r(p["amap"]) == {"cut": "today"} and p["gi"].read_bytes() == gi_before


def test_an_episode_with_no_ad_free_base_is_left_alone(tmp_path: Path) -> None:
    p = _corpus(tmp_path, [ROWS[1], ROWS[3]])
    for key in ("atxt", "aseg", "amap", "turns"):
        p[key].unlink()
    assert _run(tmp_path).details["files_written"] == 0


def test_it_is_registered_after_0029() -> None:
    ids = [mig.id for mig in get_migrations()]
    assert ids.index(m.MIGRATION_ID) == ids.index("0029_naming_arc_voices_repaired") + 1
