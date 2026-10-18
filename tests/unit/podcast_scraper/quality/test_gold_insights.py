"""S7 / S8: every insight placed by its first quote's raw voice, counted once, on the served run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from podcast_scraper.quality import gold_insights as gi

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[4]
# (voice, label, text): the transcript is "label: text" lines; spans are given explicitly.
TURNS = [
    ("SPEAKER_00", "Gergely Orosz", "Welcome to the show, Steve."),
    ("SPEAKER_01", "Steve McConnell", "Thanks for having me. I wrote the book."),
    ("SPEAKER_00", "Gergely Orosz", "Tell me about the book."),
]


def _w(p: Path, payload: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")


def _episode(
    root: Path,
    run: str = "run_1",
    eid: str = "ep",
    insights: Optional[List[Tuple[str, List[int], Dict[str, Any]]]] = None,
    turns=TURNS,
    language: str = "en",
    ref: Optional[str] = None,
) -> str:
    """One served episode; ``insights`` = (id, [quote indexes in SUPPORTED_BY order], props)."""
    base = root / "feeds" / "f" / run
    lines, segs, pos = [], [], 0
    for voice, label, text in turns:
        line = f"{label}: {text}"
        start = pos + len(label) + 2
        segs.append(
            {
                "speaker": voice,
                "speaker_label": label,
                "text": text,
                "char_start": start,
                "char_end": start + len(text),
            }
        )
        lines.append(line)
        pos += len(line) + 1
    _w(base / "transcripts" / "ep.txt", "\n".join(lines))
    _w(base / "transcripts" / "ep.segments.json", segs)
    _w(
        base / "metadata" / "ep.metadata.json",
        {
            "feed": {"id": "f", "title": "Show"},
            "episode": {"episode_id": eid, "title": "T", "language": language},
            "content": {"transcript_file_path": "transcripts/ep.txt"},
        },
    )
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    for i, s in enumerate(segs):
        nodes.append(
            {
                "id": f"quote:{i}",
                "type": "Quote",
                "properties": {
                    "char_start": s["char_start"],
                    "char_end": s["char_end"],
                    "transcript_ref": ref or "transcripts/ep.txt",
                },
            }
        )
    for iid, quotes, props in insights or []:
        nodes.append({"id": iid, "type": "Insight", "properties": props})
        edges += [{"type": "SUPPORTED_BY", "from": iid, "to": f"quote:{q}"} for q in quotes]
    _w(base / "metadata" / "ep.gi.json", {"nodes": nodes, "edges": edges})
    return f"feeds/f/{run}/metadata/ep.metadata.json"


def _gold(
    gold: Path,
    gset: str,
    case: str,
    meta: str,
    voices: Optional[List[Dict[str, Any]]],
    labels: str = "labels_v1",
) -> None:
    _w(gold / gset / "cases" / f"{case}.json", {"case_id": case, "meta_relpath": meta})
    if voices is not None:
        _w(gold / gset / labels / f"{case}.json", {"case_id": case, "voices": voices})


def _lab(
    voice: str, name: Optional[str], role: str = "guest", confidence: str = "high"
) -> Dict[str, Any]:
    return {"voice": voice, "name": name, "role": role, "confidence": confidence}


LABELS = [_lab("SPEAKER_00", "Gergely Orosz", "host"), _lab("SPEAKER_01", "Steve McConnell")]


def _run(root: Path, gold: Path, sets=("g",)) -> Dict[str, Any]:
    return gi.run(root, gold, list(sets), "snap-test")


def _buckets(report: Dict[str, Any], gset: str = "g") -> Dict[str, str]:
    return {r["insight"]: r["bucket"] for r in report["rows"] if r["set"] == gset}


def test_each_insight_is_judged_by_its_first_quotes_raw_voice(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root,
        insights=[
            ("i:steve", [1], {"speaker": "Steve McConnell"}),
            ("i:first", [1, 0], {"speaker": "Gergely Orosz"}),  # first quote is Steve's voice
            ("i:wrong", [0], {"speaker": "Steve McConnell"}),
            ("i:unnamed", [1], {"speaker": "SPEAKER_01", "surfaceable": False}),
            ("i:speakerless", [1], {}),
            ("i:none", [], {"speaker": "Steve McConnell"}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS)
    b = _buckets(_run(root, gold))
    assert b == {
        "i:steve": "S7_correct",
        "i:first": "S7_wrong",
        "i:wrong": "S7_wrong",
        "i:unnamed": "S7_unnamed",
        "i:speakerless": "S7_unnamed",
        "i:none": "no_grounded_quote",
    }


def test_one_name_on_two_voices_counts_each_insight_once(tmp_path: Path) -> None:
    """The v0.1 defect: joined by stored label, both voices' insights counted under the name."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    turns = [
        TURNS[0],
        ("SPEAKER_01", "Steve McConnell", "One."),
        ("SPEAKER_02", "Steve McConnell", "Two."),
    ]
    meta = _episode(
        root,
        turns=turns,
        insights=[
            ("i:a", [1], {"speaker": "Steve McConnell"}),
            ("i:b", [2], {"speaker": "Steve McConnell"}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS + [_lab("SPEAKER_02", "Someone Else")])
    report = _run(root, gold)
    assert _buckets(report) == {"i:a": "S7_correct", "i:b": "S7_wrong"}
    t = report["sets"]["g"]["total"]
    assert t["insights"] == 2 == sum(t[b] for b in gi.BUCKETS)


def test_hidden_and_shown_are_split_by_surfaceable_false_only(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root,
        insights=[
            ("i:h", [1], {"speaker": "Steve McConnell", "surfaceable": False}),
            ("i:s", [1], {"speaker": "Steve McConnell"}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS)
    t = _run(root, gold)["sets"]["g"]["total"]
    assert (t["S7_correct_hidden"], t["S7_correct_shown"]) == (1, 1)


@pytest.mark.parametrize(
    "label, bucket",
    [
        (_lab("SPEAKER_01", "Steve McConnell", confidence="low"), "S8_undecidable"),
        (_lab("SPEAKER_01", "Steve McConnell", role="unknown"), "S8_undecidable"),
        (_lab("SPEAKER_01", None, role="clip"), "scored_no_name"),
    ],
)
def test_unscored_and_nameless_labels(tmp_path: Path, label, bucket) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, [LABELS[0], label])
    assert _buckets(_run(root, gold)) == {"i": bucket}


def test_a_voice_with_no_label_and_a_quote_outside_every_segment(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root, insights=[("i:unlab", [1], {"speaker": "X"}), ("i:gap", [0], {"speaker": "X"})]
    )
    gfile = root / meta.replace(".metadata.json", ".gi.json")
    doc = json.loads(gfile.read_text())
    for n in doc["nodes"]:
        if n["id"] == "quote:0":
            n["properties"]["char_start"] = 0  # inside the "Gergely Orosz: " label, no segment
    gfile.write_text(json.dumps(doc))
    _gold(gold, "g", "c1", meta, [LABELS[0]])
    report = _run(root, gold)
    assert _buckets(report) == {"i:unlab": "voice_not_labelled", "i:gap": "quote_not_placeable"}
    assert report["sets"]["g"]["total"]["quote_not_placeable_by_reason"] == {"outside_segments": 1}


def test_refs_resolve_by_path() -> None:
    root, run = Path("/c"), Path("/c/feeds/f/run_B")
    assert gi.resolve_ref("transcripts/x.adfree.txt", root, run) == Path(
        "/c/feeds/f/run_B/transcripts/x.adfree.txt"
    )
    assert gi.resolve_ref("../run_A/transcripts/x.txt", root, run) == Path(
        "/c/feeds/f/run_A/transcripts/x.txt"
    )
    assert gi.resolve_ref(
        "/app/output/feeds/f/run_A/../run_A/transcripts/x.txt", root, run
    ) == Path("/c/feeds/f/run_A/transcripts/x.txt")


def test_a_cross_run_ref_is_placed_not_lost(tmp_path: Path) -> None:
    """A quote citing another run's transcript (val_v2 w038): a join by file name inside the served
    run's own transcripts/ finds nothing and loses the insight."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    _episode(root, run="run_A", eid="other")
    meta = _episode(
        root,
        run="run_B",
        insights=[("i", [1], {"speaker": "Steve McConnell"})],
        ref="../run_A/transcripts/ep.txt",
    )
    (root / "feeds/f/run_B/transcripts/ep.txt").unlink()
    _gold(gold, "g", "c1", meta, LABELS)
    assert _buckets(_run(root, gold)) == {"i": "S7_correct"}


def test_a_case_no_longer_served_is_skipped_and_named(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    old = _episode(root, run="run_1", insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _episode(root, run="run_2", insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", old, LABELS)
    s = _run(root, gold)["sets"]["g"]
    assert s["skipped_not_served"] == ["c1"] and s["included"] == 0


def test_the_newest_label_set_is_chosen_by_number(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, [LABELS[0], _lab("SPEAKER_01", "Wrong Name")], labels="labels_v2")
    _gold(gold, "g", "c1", meta, LABELS, labels="labels_v10")
    s = _run(root, gold)["sets"]["g"]
    assert s["labels"] == "labels_v10" and _buckets(_run(root, gold)) == {"i": "S7_correct"}


def test_a_case_without_a_label_file_is_counted_not_scored(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, LABELS)
    _gold(gold, "g", "c2", meta.replace("run_1", "run_9"), None)
    s = _run(root, gold)["sets"]["g"]
    assert s["included"] == 1


def test_two_languages_are_split(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    a = _episode(
        root, run="run_1", eid="e1", insights=[("i1", [1], {"speaker": "Steve McConnell"})]
    )
    b = _episode(
        root,
        run="run_2",
        eid="e2",
        language="de",
        insights=[("i2", [1], {"speaker": "SPEAKER_01"})],
    )
    _gold(gold, "g", "c1", a, LABELS)
    _gold(gold, "g", "c2", b, LABELS)
    per = _run(root, gold)["sets"]["g"]["per_language"]
    assert per["en"]["S7_correct"] == 1 and per["de"]["S7_unnamed"] == 1


def test_matcher_tiers() -> None:
    assert gi.tier("Steve McConnell", "steve mcconnell") == "fold_exact"
    assert gi.tier("Josh Elman", "Josh") == "token_subset"
    assert gi.tier("Josh Dawsey", "Josh Dawsy") == "fuzzy_or_title"


def test_the_scoring_rules_equal_the_naming_gates() -> None:
    spec = importlib.util.spec_from_file_location(
        "naming_gate", REPO / "scripts/measure/naming_gate.py"
    )
    assert spec and spec.loader
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    labels = [
        _lab("v", "A", role=r, confidence=c)
        for r in ("guest", "host", "unknown", "clip")
        for c in ("high", "medium", "low", "")
    ]
    assert [gate.is_scored(x) for x in labels] == [gi.is_scored(x) for x in labels]
    pairs = [
        ("Josh Elman", "Josh"),
        ("Dr. Adam Rodman", "Adam Rodman"),
        ("A", "B"),
        ("Steve", "steve"),
    ]
    assert [gate._same(a, b) for a, b in pairs] == [gi.same_name(a, b) for a, b in pairs]


def test_it_writes_nothing_and_is_deterministic(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, LABELS)
    for p in list(root.rglob("*")) + list(gold.rglob("*")):
        p.chmod(0o555 if p.is_dir() else 0o444)
    try:
        a, b = _run(root, gold), _run(root, gold)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    finally:
        for p in list(root.rglob("*")) + list(gold.rglob("*")):
            p.chmod(0o755 if p.is_dir() else 0o644)


def test_conservation_is_asserted_when_an_insight_would_be_lost(tmp_path: Path) -> None:
    """Two Insight nodes with one id: one would vanish from the buckets, so the run must refuse."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    gfile = root / meta.replace(".metadata.json", ".gi.json")
    doc = json.loads(gfile.read_text())
    doc["nodes"].append({"id": "i", "type": "Insight", "properties": {}})
    gfile.write_text(json.dumps(doc))
    _gold(gold, "g", "c1", meta, LABELS)
    with pytest.raises(AssertionError, match="rows for"):
        _run(root, gold)


def test_a_quote_without_offsets_is_not_placeable_not_ungrounded(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    gfile = root / meta.replace(".metadata.json", ".gi.json")
    doc = json.loads(gfile.read_text())
    for n in doc["nodes"]:
        if n["id"] == "quote:1":
            n["properties"].pop("char_start")
    gfile.write_text(json.dumps(doc))
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert _buckets(report) == {"i": "quote_not_placeable"}
    assert report["sets"]["g"]["total"]["quote_not_placeable_by_reason"] == {"no_offsets": 1}


def _gi_file(root: Path, meta: str) -> Path:
    return root / meta.replace(".metadata.json", ".gi.json")


def _edit_quote(root: Path, meta: str, qid: str, **props: Any) -> None:
    f = _gi_file(root, meta)
    doc = json.loads(f.read_text())
    for n in doc["nodes"]:
        if n["id"] == qid:
            n["properties"].update(props)
    f.write_text(json.dumps(doc))


@pytest.mark.parametrize(
    "label", [_lab("SPEAKER_01", None, role="unknown"), _lab("SPEAKER_01", None, confidence="low")]
)
def test_a_nameless_label_that_is_not_scored_is_s8_not_scored_no_name(
    tmp_path: Path, label
) -> None:
    """Reviewer R3: the common real case, a low-confidence or unknown label with no name."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, [LABELS[0], label])
    assert _buckets(_run(root, gold)) == {"i": "S8_undecidable"}


def test_a_served_case_with_no_label_file_is_counted_and_not_scored(tmp_path: Path) -> None:
    """Reviewer R12: the case is served, so only the missing label keeps it out."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, None)
    (gold / "g" / "labels_v1").mkdir()
    s = _run(root, gold)["sets"]["g"]
    assert (s["no_label"], s["included"], s["skipped_not_served"]) == (1, 0, [])
    assert s["total"]["insights"] == 0


def _plain_segments(root: Path, meta: str, pad: int = 0) -> None:
    """Segments without char offsets: the pipeline falls back to cumulative text lengths and its
    alignment guard. The transcript is the segment texts joined, plus ``pad`` characters."""
    t = root / meta.replace("metadata/ep.metadata.json", "transcripts/ep.txt")
    segf = t.with_name("ep.segments.json")
    segs = json.loads(segf.read_text())
    pos = 0
    for s in segs:
        s.pop("char_start"), s.pop("char_end")
        _edit_quote(root, meta, f"quote:{segs.index(s)}", char_start=pos)
        pos += len(s["text"])
    segf.write_text(json.dumps(segs))
    t.write_text("".join(s["text"] for s in segs) + "x" * pad)


def test_segments_without_offsets_are_placed_by_text_length(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _plain_segments(root, meta)
    _gold(gold, "g", "c1", meta, LABELS)
    assert _buckets(_run(root, gold)) == {"i": "S7_correct"}


def test_spans_refused_by_the_alignment_guard_are_not_outside_segments(tmp_path: Path) -> None:
    """Reviewer R5: the guard's refusal is its own reason."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _plain_segments(root, meta, pad=200)
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert _buckets(report) == {"i": "quote_not_placeable"}
    assert report["sets"]["g"]["total"]["quote_not_placeable_by_reason"] == {"spans_none": 1}


@pytest.mark.parametrize("segments", [None, []])
def test_no_segments_is_spans_none_not_outside_segments(tmp_path: Path, segments) -> None:
    """Reviewer R19."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    segf = root / "feeds/f/run_1/transcripts/ep.segments.json"
    if segments is None:
        segf.unlink()
    else:
        segf.write_text(json.dumps(segments))
    _gold(gold, "g", "c1", meta, LABELS)
    reasons = _run(root, gold)["sets"]["g"]["total"]["quote_not_placeable_by_reason"]
    assert reasons == {"spans_none": 1}


def test_a_ref_to_a_missing_transcript_is_unresolved_not_ungrounded(tmp_path: Path) -> None:
    """Reviewer R15."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _edit_quote(root, meta, "quote:1", transcript_ref="transcripts/gone.txt")
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert _buckets(report) == {"i": "quote_not_placeable"}
    assert report["sets"]["g"]["total"]["quote_not_placeable_by_reason"] == {"unresolved_ref": 1}


def test_a_quote_through_an_adfree_ref_uses_the_adfree_segments(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root, insights=[("i", [1], {"speaker": "Steve McConnell"})], ref="transcripts/ep.adfree.txt"
    )
    tdir = root / "feeds/f/run_1/transcripts"
    (tdir / "ep.adfree.txt").write_text((tdir / "ep.txt").read_text())
    segs = json.loads((tdir / "ep.segments.json").read_text())
    segs[1]["speaker"] = "SPEAKER_00"  # the ad-free base's voice differs: only it can decide
    (tdir / "ep.adfree.segments.json").write_text(json.dumps(segs))
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert _buckets(report) == {"i": "S7_wrong"}
    assert [r["voice"] for r in report["rows"]] == ["SPEAKER_00"]


def test_the_later_quote_sensitivity_counts_a_wrong_insight_whose_later_quote_is_the_stored_person(
    tmp_path: Path,
) -> None:
    """Reviewer R8."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root,
        insights=[
            (
                "i:later",
                [0, 1],
                {"speaker": "Steve McConnell"},
            ),  # first quote Gergely's, then Steve's
            ("i:none", [0], {"speaker": "Steve McConnell"}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert _buckets(report) == {"i:later": "S7_wrong", "i:none": "S7_wrong"}
    assert report["sets"]["g"]["total"]["sensitivity_wrong_with_later_quote_on_stored"] == 1


def test_the_flag_absent_count_is_insights_from_gi_excluding_a_flagless_insight(
    tmp_path: Path,
) -> None:
    """Reviewer R9: val_v1's 74 — not hidden by the flag, yet the app's projection drops them."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root,
        insights=[
            ("i:text", [1], {"speaker": "Steve McConnell", "text": "Books matter."}),
            ("i:drop", [1], {"speaker": "Steve McConnell", "text": "Um.", "routing_tag": "drop"}),
            ("i:notext", [1], {"speaker": "Steve McConnell"}),
            ("i:hidden", [1], {"speaker": "Steve McConnell", "text": "X.", "surfaceable": False}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    flagged = {r["insight"] for r in report["rows"] if r["excluded_unflagged"]}
    assert flagged == {"i:drop", "i:notext"}
    assert report["sets"]["g"]["total"]["flag_absent_excluded_by_insights_from_gi"] == 2


def test_a_labelled_voice_missing_from_the_segments_is_counted(tmp_path: Path) -> None:
    """Reviewer R11: SPEAKER_00 has no insight but is in the segments; SPEAKER_09 is in neither."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, LABELS + [_lab("SPEAKER_09", "Nobody Here")])
    assert _run(root, gold)["sets"]["g"]["labelled_voices_missing_from_segments"] == 1


def test_an_explicit_label_directory_overrides_the_newest(tmp_path: Path) -> None:
    """Reviewer R17."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, LABELS, labels="labels_v1")
    _gold(gold, "g", "c1", meta, [LABELS[0], _lab("SPEAKER_01", "Wrong Name")], labels="labels_v2")
    pinned = gi.run(root, gold, ["g"], "snap-test", {"g": "labels_v1"})
    assert pinned["sets"]["g"]["labels"] == "labels_v1"
    assert _buckets(pinned) == {"i": "S7_correct"}
    assert _buckets(_run(root, gold)) == {"i": "S7_wrong"}


@pytest.mark.parametrize("stored", ["SPEAKER", "SPEAKER_01", ""])
def test_a_raw_or_empty_stored_name_is_unnamed(tmp_path: Path, stored: str) -> None:
    """Reviewer R20: the bare ``SPEAKER`` id too."""
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": stored})])
    _gold(gold, "g", "c1", meta, LABELS)
    assert _buckets(_run(root, gold)) == {"i": "S7_unnamed"}


def test_entry_voices_counts_a_lists_voices_by_bucket_and_split(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(
        root,
        insights=[
            ("i:a", [1], {"speaker": "Steve McConnell"}),
            ("i:b", [1], {"speaker": "SPEAKER_01", "surfaceable": False}),
            ("i:c", [0], {"speaker": "Gergely Orosz"}),
        ],
    )
    _gold(gold, "g", "c1", meta, LABELS)
    out = gi.entry_voices(_run(root, gold), [("ep", "SPEAKER_01"), ("other", "SPEAKER_00")])
    assert out == {
        "g/c1 SPEAKER_01": {
            "S7_correct": 1,
            "S7_correct_shown": 1,
            "S7_unnamed": 1,
            "S7_unnamed_hidden": 1,
        }
    }


def test_a_served_case_whose_gi_file_is_missing_is_counted(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gi_file(root, meta).unlink()
    _gold(gold, "g", "c1", meta, LABELS)
    s = _run(root, gold)["sets"]["g"]
    assert (s["included"], s["gi_missing"], s["total"]["insights"]) == (1, 1, 0)


def test_the_report_names_what_the_ledger_records(tmp_path: Path) -> None:
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _w(root / "upgrade_ledger.json", {"applied": [{"id": "0029_x"}, {"id": "0001_a"}]})
    _gold(gold, "g", "c1", meta, LABELS)
    report = _run(root, gold)
    assert report["ledger_applied"] == ["0001_a", "0029_x"] and "ledger_0030" not in report


def test_the_cli_passes_labels_through(tmp_path: Path, monkeypatch, capsys) -> None:
    """Reviewer N6: ``--labels SET=DIR`` reaches ``run``."""
    spec = importlib.util.spec_from_file_location(
        "gold_cli", REPO / "scripts/measure/gold_insights.py"
    )
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    root, gold = tmp_path / "c", tmp_path / "gold"
    meta = _episode(root, insights=[("i", [1], {"speaker": "Steve McConnell"})])
    _gold(gold, "g", "c1", meta, LABELS, labels="labels_v1")
    _gold(gold, "g", "c1", meta, [LABELS[0], _lab("SPEAKER_01", "Wrong Name")], labels="labels_v2")
    out = tmp_path / "r.json"
    argv = ["x", "--corpus", str(root), "--gold", str(gold), "--snapshot", "s", "--out", str(out)]
    monkeypatch.setattr("sys.argv", argv + ["--labels", "g=labels_v1"])
    assert cli.main() == 0
    assert json.loads(out.read_text())["sets"]["g"]["labels"] == "labels_v1"
    monkeypatch.setattr("sys.argv", argv)
    assert cli.main() == 0
    assert json.loads(out.read_text())["sets"]["g"]["labels"] == "labels_v2"
    monkeypatch.setattr("sys.argv", argv + ["--labels", "nope"])
    with pytest.raises(SystemExit):
        cli.main()
