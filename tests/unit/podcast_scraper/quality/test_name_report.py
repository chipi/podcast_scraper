"""The name report on a synthetic corpus built in the shapes the arc measures.

Every shape the report must see is here once, with synthetic names:

* a raw diarization label as an insight speaker (scoped placeholder Person named ``SPEAKER_07``);
* the detector's default name on a global id (``person:unknown-guest-1`` / ``unknown_guest_1``),
  which the shared placeholder predicate does not match;
* a substantive unnamed voice in the transcript (``voice_type`` unknown, so a raw label is shown);
* two spellings of one name in one feed (rule b), and two names equal after normalisation (rule a);
* a roster entry that is listed but never placed on a voice (``placed: false``);
* an older record without ``placed`` / ``voices``, placed through its segment label;
* a superseded run of the same episode, which must not count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.quality import name_report


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _gi(insights: List[Dict[str, Any]], persons: Dict[str, str]) -> Dict[str, Any]:
    """A GI artifact: one Insight per entry, each supported by one quote spoken by ``person``."""
    nodes: List[Dict[str, Any]] = [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons.items()
    ]
    edges: List[Dict[str, Any]] = []
    for i, ins in enumerate(insights):
        iid, qid = f"insight:{i}", f"quote:{i}"
        nodes.append(
            {
                "id": iid,
                "type": "Insight",
                "properties": {"text": f"claim {i}", **ins.get("props", {})},
            }
        )
        nodes.append({"id": qid, "type": "Quote", "properties": {"text": f"said {i}"}})
        edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
        edges.append({"type": "SPOKEN_BY", "from": qid, "to": ins["person"]})
    return {"nodes": nodes, "edges": edges}


def _episode(
    root: Path,
    feed: str,
    run: str,
    stem: str,
    episode_id: str,
    *,
    speakers: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
    gi: Optional[Dict[str, Any]] = None,
    kg_persons: Optional[Dict[str, str]] = None,
) -> None:
    run_dir = root / "feeds" / feed / run
    _write(
        run_dir / "metadata" / f"{stem}.metadata.json",
        {
            "episode": {"episode_id": episode_id, "title": stem},
            "content": {"speakers": speakers, "transcript_file_path": f"transcripts/{stem}.txt"},
        },
    )
    _write(run_dir / "transcripts" / f"{stem}.segments.json", segments)
    if gi is not None:
        _write(run_dir / "metadata" / f"{stem}.gi.json", gi)
    if kg_persons is not None:
        _write(
            run_dir / "metadata" / f"{stem}.kg.json",
            {
                "episode_id": episode_id,
                "nodes": [
                    {"id": pid, "type": "Person", "properties": {"name": name}}
                    for pid, name in kg_persons.items()
                ],
            },
        )


def _seg(
    start: float, voice: str, label: Optional[str], voice_type: Optional[str] = None
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"start": start, "end": start + 5, "text": "words", "speaker": voice}
    if label is not None:
        row["speaker_label"] = label
    if voice_type is not None:
        row["voice_type"] = voice_type
    return row


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    # Feed A, episode 1 (new record): a raw-label insight speaker; an unnamed substantive voice in
    # the transcript; "Jorna Velt" placed, "Jorna Vellt" listed but never placed.
    _episode(
        root,
        "feed_a",
        "run_x_20261001-000000",
        "ep1",
        "a1",
        speakers=[
            {"name": "Jorna Velt", "placed": True, "voices": ["SPEAKER_00"]},
            {"name": "Jorna Vellt", "placed": False, "voices": []},
        ],
        segments=[
            _seg(0, "SPEAKER_00", "Jorna Velt"),
            _seg(5, "SPEAKER_07", "SPEAKER_07", "unknown"),
        ],
        gi=_gi(
            [{"person": "person:jorna-velt"}, {"person": "person:speaker-a1-07"}],
            {"person:jorna-velt": "Jorna Velt", "person:speaker-a1-07": "SPEAKER_07"},
        ),
        kg_persons={"person:jorna-velt": "Jorna Velt"},
    )
    # Feed A, episode 2 (new record): the second spelling, placed; and the detector's default
    # name on a global id, which the placeholder predicate misses.
    _episode(
        root,
        "feed_a",
        "run_y_20261002-000000",
        "ep2",
        "a2",
        speakers=[{"name": "Jorna Vellt", "placed": True, "voices": ["SPEAKER_01"]}],
        # The segment label differs from the roster spelling: only the voices field places it.
        segments=[_seg(0, "SPEAKER_01", "J. Vellt")],
        gi=_gi(
            [{"person": "person:jorna-vellt"}, {"person": "person:unknown-guest-1"}],
            {"person:jorna-vellt": "Jorna Vellt", "person:unknown-guest-1": "unknown_guest_1"},
        ),
        kg_persons={
            "person:jorna-vellt": "Jorna Vellt",
            "person:unknown-guest-1": "unknown_guest_1",
        },
    )
    # Feed A, episode 2 again in an OLDER run: superseded, must not count.
    _episode(
        root,
        "feed_a",
        "run_old_20260901-000000",
        "ep2",
        "a2",
        speakers=[{"name": "Zed Superseded", "placed": True, "voices": ["SPEAKER_00"]}],
        segments=[_seg(0, "SPEAKER_00", "SPEAKER_00", "unknown")],
    )
    # Feed B (old record, no placed/voices): rule a pair, placed through segment labels.
    _episode(
        root,
        "feed_b",
        "run_z_20261003-000000",
        "ep3",
        "b1",
        speakers=[{"name": "Kai Moreno-Lund"}, {"name": "Kai Moreno Lund"}],
        segments=[_seg(0, "SPEAKER_00", "Kai Moreno-Lund")],
    )
    # Feed C: an entry marked unplaced (no ``voices`` field), whose name IS a segment label, must
    # not count as heard; and the episode's only insight is unattributed, spoken by a raw-label
    # voice, so it reaches the panel only on the fallback path.
    _episode(
        root,
        "feed_c",
        "run_w_20261004-000000",
        "ep4",
        "c1",
        speakers=[{"name": "Odile Brand", "placed": False}],
        segments=[_seg(0, "SPEAKER_04", "Odile Brand")],
        gi=_gi(
            [{"person": "person:speaker-c1-04", "props": {"surfaceable": False}}],
            {"person:speaker-c1-04": "SPEAKER_04"},
        ),
    )
    return root


def test_served_episodes_keep_only_the_newest_run(corpus: Path) -> None:
    eps = name_report.served_episodes(corpus)
    assert [(e.feed, e.episode_id, e.run_ts) for e in eps] == [
        ("feed_a", "a1", "20261001-000000"),
        ("feed_a", "a2", "20261002-000000"),
        ("feed_b", "b1", "20261003-000000"),
        ("feed_c", "c1", "20261004-000000"),
    ]


def test_r1_counts_both_raw_shapes_on_insights_and_the_transcript(corpus: Path) -> None:
    report = name_report.run(corpus)
    r1 = report["r1_raw_labels"]
    assert r1["insights_default:raw"] == {"episodes": 2, "items": 2}
    # Feed C's unattributed raw-label insight appears on the fallback path only.
    assert r1["insights_fallback:raw"] == {"episodes": 3, "items": 3}
    assert r1["transcript_segments:raw"] == {"episodes": 1, "items": 1}
    # The people chips drop scoped placeholders and SPEAKER_NN, but not the detector default.
    assert r1["episode_people_chips:raw"] == {"episodes": 1, "items": 1}
    names = {row["name"] for row in report["r1_rows"]}
    assert names == {"SPEAKER_07", "SPEAKER_04", "unknown_guest_1"}
    assert "Zed Superseded" not in json.dumps(report)


def test_roster_mode_pairs_listings_and_voice_mode_pairs_voices(corpus: Path) -> None:
    report = name_report.run(corpus)
    roster, voice = report["r2_modes"]["roster"], report["r2_modes"]["voice"]
    assert (roster["pairs_rule_a"], roster["pairs_rule_b"]) == (1, 1)
    # Voice mode: "Jorna Vellt" is placed in ep2, so the rule-b pair stays; in feed_b only
    # "Kai Moreno-Lund" is a segment label, so the old record's rule-a twin is listing-only.
    assert (voice["pairs_rule_a"], voice["pairs_rule_b"]) == (0, 1)
    # An entry carrying ``placed`` is the newer record format even without ``voices`` (feed C).
    assert report["records"] == {"new": 3, "old": 1}
    # Listed but not heard: "Jorna Vellt" in ep1 (placed false), "Kai Moreno Lund" (no segment
    # label), "Odile Brand" (marked unplaced although her name is a segment label).
    assert report["r5"]["listed_not_heard_entries"] == 3


def test_label_kind_separates_raw_role_and_deliberate() -> None:
    assert name_report.label_kind("SPEAKER_03") == "raw"
    assert name_report.label_kind("unknown_guest_1") == "raw"
    assert name_report.label_kind("Speaker John Knight") is None
    assert name_report.label_kind("Unidentified speaker") == "deliberate"
    assert name_report.label_kind("Host") == "deliberate"
    assert name_report.label_kind("Guest") == "role"
    assert name_report.label_kind("Jorna Velt") is None


@pytest.mark.parametrize(
    "a,b,rule",
    [
        ("Kai Moreno-Lund", "Kai Moreno Lund", "a"),
        ("N. T. Wright", "N.T. Wright", "a"),
        ("Jorna Velt", "Jorna Vellt", "b"),
        ("Yu", "Yo", None),  # under the rule-b length floor
        ("Jorna Velt", "Marta Velt", None),
    ],
)
def test_pair_rules_match_the_scan(a: str, b: str, rule: Optional[str]) -> None:
    assert name_report.pair_rule(a, b) == rule
