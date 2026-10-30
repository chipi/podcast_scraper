"""m0038: a served episode's files stop crediting raw diarization labels as speakers (RFC-131 N11).

Written before the migration (change record 0038, step 6). One synthetic episode per path:
a quote on a raw-labelled untyped segment, on a typed segment, one that cannot be placed, a raw
quote that is not an insight's first, a named quote beside a raw one, the bridge identity, the
search of a second run, and the standing checks (second apply, verify, undo byte-identical).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.formatting import (
    format_diarized_screenplay_with_offsets,
)
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0038_raw_label_speakers_removed as m38
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

RAW = "person:speaker-ep-07"
NAMED = "person:jorna-velt"


def _w(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode(
    root: Path,
    turns: Sequence[Tuple[str, str, Optional[str], str]],
    insights: List[Tuple[str, List[int], Dict[str, Any]]],
    *,
    run: str = "run_1",
    eid: str = "ep",
    unplaced: Sequence[int] = (),
) -> Dict[str, Path]:
    """``turns`` = (voice, label, voice_type, text); one Quote per turn, credited by the label
    (a raw label credits ``RAW``); ``insights`` = (id, quote indexes in SUPPORTED_BY order,
    props). A quote in ``unplaced`` gets offsets past every segment."""
    base = root / "feeds" / "f" / run
    p = {
        "meta": base / "metadata" / "ep.metadata.json",
        "gi": base / "metadata" / "ep.gi.json",
        "bridge": base / "metadata" / "ep.bridge.json",
        "seg": base / "transcripts" / "ep.segments.json",
        "txt": base / "transcripts" / "ep.txt",
    }
    _w(
        p["meta"],
        {"episode": {"episode_id": eid}, "content": {"transcript_file_path": "transcripts/ep.txt"}},
    )
    rows = []
    for i, (v, lab, vt, t) in enumerate(turns):
        row: Dict[str, Any] = {
            "start": float(i),
            "end": i + 1.0,
            "speaker": v,
            "speaker_label": lab,
            "text": t,
        }
        if vt:
            row["voice_type"] = vt
        rows.append(row)
    _w(p["seg"], rows)
    text, offsets = format_diarized_screenplay_with_offsets(rows)
    for row, o in zip(rows, offsets):  # explicit offsets, as the ad-free base carries (#974)
        row["char_start"], row["char_end"] = o["char_start"], o["char_end"]
    _w(p["seg"], rows)
    p["txt"].write_text(text, encoding="utf-8")
    nodes: List[Dict[str, Any]] = [
        {"id": RAW, "type": "Person", "properties": {"name": "SPEAKER_07"}},
        {"id": NAMED, "type": "Person", "properties": {"name": "Jorna Velt"}},
    ]
    edges: List[Dict[str, Any]] = []
    for i, o in enumerate(offsets):
        pid = RAW if o["speaker_label"].startswith("SPEAKER_") else NAMED
        cs, ce = (
            (len(text) + 10, len(text) + 20) if i in unplaced else (o["char_start"], o["char_end"])
        )
        nodes.append(
            {
                "id": f"quote:{i}",
                "type": "Quote",
                "properties": {
                    "text": o["text"],
                    "char_start": cs,
                    "char_end": ce,
                    "transcript_ref": "transcripts/ep.txt",
                    "speaker_id": pid,
                },
            }
        )
        edges.append({"type": "SPOKEN_BY", "from": f"quote:{i}", "to": pid})
    for iid, qs, props in insights:
        nodes.append(
            {"id": iid, "type": "Insight", "properties": {"text": iid, "grounded": True, **props}}
        )
        edges.extend({"type": "SUPPORTED_BY", "from": iid, "to": f"quote:{q}"} for q in qs)
    _w(p["gi"], {"nodes": nodes, "edges": edges})
    _w(
        p["bridge"],
        {
            "schema_version": "1.0",
            "episode_id": eid,
            "identities": [
                {"id": RAW, "type": "person", "display_name": "SPEAKER_07"},
                {"id": NAMED, "type": "person", "display_name": "Jorna Velt"},
            ],
        },
    )
    return p


def _nodes(path: Path) -> Dict[str, dict]:
    return {n["id"]: n for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}


def _ctx(root: Path, dry_run: bool = False) -> MigrationContext:
    return MigrationContext(corpus_root=root, dry_run=dry_run)


def _tree(root: Path) -> Dict[str, str]:
    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in root.rglob("*")
        if f.is_file()
        and ".podcast_scraper" not in f.parts
        and not f.name.endswith(".jsonl")
        and not f.name.startswith("upgrade_ledger")  # the runner's bookkeeping, not the migration's
    }


TURNS = [
    ("SPEAKER_00", "Jorna Velt", None, "I think the rates will fall."),
    ("SPEAKER_07", "SPEAKER_07", None, "They will not fall this year."),
    ("SPEAKER_03", "SPEAKER_03", "unidentified", "Prices are up everywhere."),
]


@pytest.fixture
def ep(tmp_path: Path) -> Dict[str, Path]:
    tier_props = {"tier": 2, "surfaceable": True, "speaker": "SPEAKER_07"}
    return _episode(
        tmp_path / "corpus",
        TURNS + [("SPEAKER_07", "SPEAKER_07", None, "And that is the whole story.")],
        [
            ("insight:raw-first", [1, 0], dict(tier_props)),  # raw first quote, named second
            ("insight:typed-first", [2], {"tier": 2, "surfaceable": True}),
            (
                "insight:named-first",
                [0, 1],
                {"tier": 2, "surfaceable": True, "speaker": "Jorna Velt"},
            ),
            ("insight:unplaced-first", [3], {"tier": 1, "surfaceable": True}),
        ],
        unplaced=[3],
    )


def test_the_raw_person_its_edges_and_its_bridge_identity_go(ep):
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(ep["meta"].parents[4]))
    nodes = _nodes(ep["gi"])
    assert RAW not in nodes and NAMED in nodes
    gi = json.loads(ep["gi"].read_text(encoding="utf-8"))
    assert not [e for e in gi["edges"] if e.get("to") == RAW]
    bridge = json.loads(ep["bridge"].read_text(encoding="utf-8"))
    assert [i["id"] for i in bridge["identities"]] == [NAMED]


def test_quotes_lose_the_raw_speaker_and_take_their_segments_type(ep):
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(ep["meta"].parents[4]))
    nodes = _nodes(ep["gi"])
    assert "speaker_id" not in nodes["quote:1"]["properties"]
    assert nodes["quote:1"]["properties"]["speaker_voice_type"] == "unknown"  # raw, untyped
    assert nodes["quote:2"]["properties"]["speaker_voice_type"] == "unidentified"  # typed segment
    assert nodes["quote:3"]["properties"]["speaker_voice_type"] == "unknown"  # cannot be placed
    assert nodes["quote:0"]["properties"]["speaker_id"] == NAMED  # a named quote keeps its speaker
    assert "speaker_voice_type" not in nodes["quote:0"]["properties"]


def test_an_insight_follows_its_first_quote_and_is_rerouted(ep):
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(ep["meta"].parents[4]))
    nodes = _nodes(ep["gi"])
    raw_first = nodes["insight:raw-first"]["properties"]
    assert raw_first["speaker_voice_type"] == "unknown" and raw_first["surfaceable"] is False
    assert raw_first["speaker"] == "SPEAKER_07"  # the pipeline's own field: the voice id, kept
    assert raw_first["routing_tag"] == "connect"  # tier 2 but no longer surfaceable
    assert nodes["insight:typed-first"]["properties"]["speaker_voice_type"] == "unidentified"
    assert nodes["insight:unplaced-first"]["properties"]["speaker_voice_type"] == "unknown"
    named_first = nodes["insight:named-first"]["properties"]
    assert named_first == {
        "text": "insight:named-first",
        "grounded": True,
        "tier": 2,
        "surfaceable": True,
        "speaker": "Jorna Velt",
    }


def test_a_type_the_pipeline_surfaces_keeps_the_insight_surfaceable(tmp_path):
    p = _episode(
        tmp_path / "corpus",
        [
            ("SPEAKER_00", "Jorna Velt", None, "Hello."),
            ("SPEAKER_07", "SPEAKER_07", "cameo", "Hi."),
        ],
        [("insight:cameo", [1], {"tier": 2, "surfaceable": True})],
    )
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(tmp_path / "corpus"))
    props = _nodes(p["gi"])["insight:cameo"]["properties"]
    assert props["speaker_voice_type"] == "cameo" and props["surfaceable"] is True
    assert props["routing_tag"] == "surface"


def test_an_insight_speaker_field_alone_is_not_a_change(tmp_path):
    """The pipeline writes the voice id into ``speaker`` for every insight; that is not N11's."""
    p = _episode(
        tmp_path / "corpus",
        [("SPEAKER_03", "SPEAKER_03", "unknown", "Hello.")],
        [("insight:a", [0], {"tier": 2, "speaker": "SPEAKER_03", "surfaceable": False})],
    )
    gi = json.loads(p["gi"].read_text(encoding="utf-8"))
    gi["nodes"] = [n for n in gi["nodes"] if n["id"] != RAW]
    for n in gi["nodes"]:
        n.get("properties", {}).pop("speaker_id", None)
    gi["edges"] = [e for e in gi["edges"] if e["type"] != "SPOKEN_BY"]
    _w(p["gi"], gi)
    _w(p["bridge"], {"schema_version": "1.0", "episode_id": "ep", "identities": []})
    before = _tree(tmp_path / "corpus")
    result = m38.RawLabelSpeakersRemovedMigration().apply(_ctx(tmp_path / "corpus"))
    assert result.details["episodes_written"] == 0 and _tree(tmp_path / "corpus") == before


def test_an_episode_with_no_raw_label_is_not_written(tmp_path):
    p = _episode(
        tmp_path / "corpus",
        [("SPEAKER_00", "Jorna Velt", None, "Hello.")],
        [("insight:a", [0], {"tier": 2})],
    )
    gi = json.loads(p["gi"].read_text(encoding="utf-8"))
    gi["nodes"] = [n for n in gi["nodes"] if n["id"] != RAW]
    _w(p["gi"], gi)
    _w(p["bridge"], {"schema_version": "1.0", "episode_id": "ep", "identities": []})
    before = _tree(tmp_path / "corpus")
    result = m38.RawLabelSpeakersRemovedMigration().apply(_ctx(tmp_path / "corpus"))
    assert result.details["episodes_written"] == 0
    assert _tree(tmp_path / "corpus") == before


def test_a_superseded_run_is_not_read(tmp_path):
    root = tmp_path / "corpus"
    old = _episode(root, TURNS, [("insight:a", [1], {"tier": 2})], run="run_20260101-000000_a")
    new = _episode(root, TURNS, [("insight:a", [1], {"tier": 2})], run="run_20260201-000000_b")
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(root))
    assert RAW in _nodes(old["gi"]) and RAW not in _nodes(new["gi"])


def test_dry_run_writes_nothing_and_counts_what_apply_would(ep):
    root = ep["meta"].parents[4]
    before = _tree(root)
    m = m38.RawLabelSpeakersRemovedMigration()
    dry = m.apply(_ctx(root, dry_run=True))
    assert _tree(root) == before
    assert dry.details["persons_removed"] == 1 and dry.details["quotes_unattributed"] == 3
    assert dry.details["insights_retyped"] == 3 and dry.details["bridge_identities_removed"] == 1
    wet = m.apply(_ctx(root))
    assert {k: wet.details[k] for k in m38.COUNTERS} == {k: dry.details[k] for k in m38.COUNTERS}


def test_verify_second_apply_and_undo(ep):
    root = ep["meta"].parents[4]
    m = m38.RawLabelSpeakersRemovedMigration()
    before = _tree(root)
    ok, _msg = m.verify(_ctx(root))
    assert ok is False
    m.apply(_ctx(root))
    assert m.verify(_ctx(root))[0] is True
    after = _tree(root)
    again = m.apply(_ctx(root))
    assert again.details["episodes_written"] == 0 and _tree(root) == after
    restored, refused = m38.undo(root)
    assert (restored, refused) == (2, [])  # the gi and the bridge
    assert _tree(root) == before


def test_registered_after_0037():
    ids = [m.id for m in get_migrations()]
    assert (
        ids.index("0038_raw_label_speakers_removed") == ids.index("0037_named_voices_corrected") + 1
    )


def test_a_raw_quote_on_a_now_named_segment_refuses_the_episode(tmp_path):
    root = tmp_path / "corpus"
    p = _episode(
        root,
        [("SPEAKER_00", "Jorna Velt", None, "Hello."), ("SPEAKER_07", "SPEAKER_07", None, "Hi.")],
        [("insight:a", [1], {"tier": 2})],
    )
    gi = json.loads(p["gi"].read_text(encoding="utf-8"))
    for n in gi["nodes"]:
        if n["id"] == "quote:0":
            n["properties"]["speaker_id"] = RAW  # the GI credits the raw label where a name now is
    _w(p["gi"], gi)
    before = _tree(root)
    m = m38.RawLabelSpeakersRemovedMigration()
    result = m.apply(_ctx(root))
    assert result.details["episodes_written"] == 0
    assert result.details["refused"] == {
        "a raw-credited quote is on a named segment": ["feeds/f/run_1/metadata/ep.metadata.json"]
    }
    assert _tree(root) == before
    assert m.verify(_ctx(root))[0] is False


def test_a_raw_bridge_identity_goes_even_without_its_gi_person(tmp_path):
    p = _episode(
        tmp_path / "corpus",
        [("SPEAKER_00", "Jorna Velt", None, "Hello.")],
        [("insight:a", [0], {"tier": 2})],
    )
    gi = json.loads(p["gi"].read_text(encoding="utf-8"))
    gi["nodes"] = [n for n in gi["nodes"] if n["id"] != RAW]
    _w(p["gi"], gi)
    _w(
        p["bridge"],
        {
            "schema_version": "1.0",
            "episode_id": "ep",
            "identities": [
                {"id": "person:speaker-ep-02", "type": "person", "display_name": "SPEAKER_02"},
                {"id": NAMED, "type": "person", "display_name": "Jorna Velt"},
            ],
        },
    )
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(tmp_path / "corpus"))
    assert [i["id"] for i in json.loads(p["bridge"].read_text(encoding="utf-8"))["identities"]] == [
        NAMED
    ]


def test_a_quote_in_another_runs_transcript_is_placed_there(tmp_path):
    """``transcript_ref`` is relative to the run directory (the gold join's rule): a quote kept from
    an earlier run names that run's transcript, and takes the type of its segment there."""
    root = tmp_path / "corpus"
    other = _episode(
        root,
        [
            ("SPEAKER_00", "Jorna Velt", None, "Hello."),
            ("SPEAKER_07", "SPEAKER_07", "unidentified", "Hi there."),
        ],
        [],
        run="run_20260101-000000_a",
        eid="other",
    )
    p = _episode(
        root,
        [
            ("SPEAKER_00", "Jorna Velt", None, "Hello."),
            ("SPEAKER_07", "SPEAKER_07", None, "Hi there."),
        ],
        [("insight:a", [1], {"tier": 2})],
        run="run_20260201-000000_b",
    )
    gi = json.loads(p["gi"].read_text(encoding="utf-8"))
    for n in gi["nodes"]:
        if n["id"] == "quote:1":
            n["properties"]["transcript_ref"] = "../run_20260101-000000_a/transcripts/ep.txt"
    _w(p["gi"], gi)
    assert other["seg"].is_file()
    m38.RawLabelSpeakersRemovedMigration().apply(_ctx(root))
    assert _nodes(p["gi"])["quote:1"]["properties"]["speaker_voice_type"] == "unidentified"
