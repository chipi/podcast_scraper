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
    # Feed D: the same person under the same spelling on another show (an identical cross-show
    # group the map joins, since both mint person:jorna-velt), which is also a cross-show variant
    # of feed A's "Jorna Vellt".
    _episode(
        root,
        "feed_d",
        "run_v_20261005-000000",
        "ep5",
        "d1",
        speakers=[
            {"name": "Jorna Velt", "placed": True, "voices": ["SPEAKER_00"]},
            {"name": "Brinna Sol", "placed": True, "voices": ["SPEAKER_01"]},
        ],
        segments=[_seg(0, "SPEAKER_00", "Jorna Velt"), _seg(5, "SPEAKER_01", "Brinna Sol")],
        kg_persons={"person:jorna-velt": "Jorna Velt"},
    )
    # Feed D again: a within-show variant only ("Brinna Soll"), which R2 pairs and R4 must not.
    _episode(
        root,
        "feed_d",
        "run_u_20261006-000000",
        "ep6",
        "d2",
        speakers=[{"name": "Brinna Soll", "placed": True, "voices": ["SPEAKER_00"]}],
        segments=[_seg(0, "SPEAKER_00", "Brinna Soll")],
    )
    return root


def test_served_episodes_keep_only_the_newest_run(corpus: Path) -> None:
    eps = name_report.served_episodes(corpus)
    assert [(e.feed, e.episode_id, e.run_dir) for e in eps] == [
        ("feed_a", "a1", "run_x_20261001-000000"),
        ("feed_a", "a2", "run_y_20261002-000000"),
        ("feed_b", "b1", "run_z_20261003-000000"),
        ("feed_c", "c1", "run_w_20261004-000000"),
        ("feed_d", "d1", "run_v_20261005-000000"),
        ("feed_d", "d2", "run_u_20261006-000000"),
    ]


def test_r1_finds_no_raw_label_on_any_surface_after_s3(corpus: Path) -> None:
    # RFC-131 S3: the app no longer shows raw labels, so R1 measures zero on this corpus. The
    # report's own detection is tested below against builders that still show them.
    report = name_report.run(corpus)
    assert {k: v for k, v in report["r1_raw_labels"].items() if k.endswith(":raw")} == {}
    assert "Zed Superseded" not in json.dumps(report)


def test_r1_detects_both_raw_shapes_when_a_surface_shows_them(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report must catch what the app shows; here the pre-S3 behaviour is restored by fakes."""
    from podcast_scraper.server import app_gi_view, segments_view
    from podcast_scraper.server.schemas import AppInsight, AppQuote, TranscriptSegment

    def fake_insights(gi, *, limit=None, unattributed_fallback=False):
        names = sorted(
            n["properties"]["name"]
            for n in gi.get("nodes", [])
            if n.get("type") == "Person"
            and name_report.label_kind(n["properties"]["name"]) == "raw"
            and (unattributed_fallback or n["id"] != "person:speaker-c1-04")
        )
        return [
            AppInsight(id=f"i{k}", text="t", grounded=True, quotes=[AppQuote(text="q", speaker=n)])
            for k, n in enumerate(names)
        ]

    def fake_segments(raw):
        return [
            TranscriptSegment(id="s", start=0, end=1, text="x", speaker=r.get("speaker_label"))
            for r in raw
        ]

    monkeypatch.setattr(app_gi_view, "insights_from_gi", fake_insights)
    monkeypatch.setattr(segments_view, "to_contract_segments", fake_segments)
    report = name_report.run(corpus)
    r1 = report["r1_raw_labels"]
    assert r1["insights_default:raw"] == {"episodes": 2, "items": 2}
    # Feed C's unattributed raw-label insight appears on the fallback path only.
    assert r1["insights_fallback:raw"] == {"episodes": 3, "items": 3}
    assert r1["transcript_segments:raw"] == {"episodes": 1, "items": 1}
    names = {row["name"] for row in report["r1_rows"]}
    assert names == {"SPEAKER_07", "SPEAKER_04", "unknown_guest_1"}


def test_roster_mode_pairs_listings_and_voice_mode_pairs_voices(corpus: Path) -> None:
    report = name_report.run(corpus)
    roster, voice = report["r2_modes"]["roster"], report["r2_modes"]["voice"]
    assert (roster["pairs_rule_a"], roster["pairs_rule_b"]) == (1, 2)
    # Voice mode: "Jorna Vellt" is placed in ep2, so the rule-b pair stays; in feed_b only
    # "Kai Moreno-Lund" is a segment label, so the old record's rule-a twin is listing-only.
    assert (voice["pairs_rule_a"], voice["pairs_rule_b"]) == (0, 2)
    # An entry carrying ``placed`` is the newer record format even without ``voices`` (feed C).
    assert report["records"] == {"new": 5, "old": 1}
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


def test_r4_finds_identical_names_and_variants_across_shows(corpus: Path) -> None:
    r4 = name_report.run(corpus)["r4_voice"]
    assert r4["identical_groups"] == 1
    (group,) = r4["identical"]
    assert {(m["feed"], m["name"]) for m in group["members"]} == {
        ("feed_a", "Jorna Velt"),
        ("feed_d", "Jorna Velt"),
    }
    assert group["map_joined"] is True
    # "Jorna Vellt" (feed A only) against "Jorna Velt" (feeds A and D): a cross-show variant.
    assert r4["variant_pairs"] == 1
    assert r4["variants"][0]["norms"] == ["jornavelt", "jornavellt"]


def test_report_records_the_corpus_state_and_the_served_runs(corpus: Path) -> None:
    report = name_report.run(corpus)
    assert report["corpus_token"]["newest_kg_count"] == 3
    assert ["feed_a", "a2", "run_y_20261002-000000"] in report["served_runs"]


def test_a_suffixed_newer_run_wins_as_in_the_catalog(tmp_path: Path) -> None:
    """Run dirs often carry a suffix after the timestamp (``…_20260819-060035_285e51f2``).

    The 2026-10-30 scans and the first report read the timestamp with ``_(\\d{8}-\\d{6})$``,
    so a suffixed run read as "" and an older run could win (prod episode f58130f3). The report
    must serve the run the catalog serves.
    """
    root = tmp_path / "corpus"
    for run, name in (
        ("run_0194f6d1-aa_20260812-182337_285e51f2", "Old Spelling"),
        ("run_20260819-060035_285e51f2", "New Spelling"),
    ):
        _episode(
            root,
            "feed_s",
            run,
            "ep",
            "s1",
            speakers=[{"name": name, "placed": True, "voices": ["SPEAKER_00"]}],
            segments=[_seg(0, "SPEAKER_00", name)],
        )
    (ep,) = name_report.served_episodes(root)
    assert ep.metadata_path.parent.parent.name == "run_20260819-060035_285e51f2"
    assert ep.roster()[0]["name"] == "New Spelling"


def test_a_relative_corpus_root_works(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(corpus.parent)
    assert len(name_report.served_episodes(Path(corpus.name))) == 6


def test_languages_and_lossy_person_ids_are_reported(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _episode(
        root,
        "feed_zh",
        "run_t_20261007-000000",
        "ep",
        "z1",
        speakers=[{"name": "张川红", "placed": True, "voices": ["SPEAKER_00"]}],
        segments=[_seg(0, "SPEAKER_00", "张川红")],
        kg_persons={"person:topic": "张川红", "person:jorna-velt": "Jorna Velt"},
    )
    meta = root / "feeds/feed_zh/run_t_20261007-000000/metadata/ep.metadata.json"
    doc = json.loads(meta.read_text())
    doc["episode"]["language"] = "zh"
    meta.write_text(json.dumps(doc))
    report = name_report.run(root)
    assert report["languages"]["zh"]["episodes"] == 1
    assert report["l6_person_ids"]["lossy_ids"] == 1
    assert report["l6_person_ids"]["non_latin_names"] == 1
    (row,) = report["l6_person_ids"]["rows"]
    assert row["person_id"] == "person:topic" and row["language"] == "zh"


def test_r1_detects_raw_labels_on_search_and_the_mcp_roster(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read-model, lifted-speaker and roster surfaces, with the pre-S3 behaviour restored."""
    from podcast_scraper.gi import load
    from podcast_scraper.mcp.tools import enrichment
    from podcast_scraper.search import transcript_chunk_lift as lift

    def old_meta(artifact, quote_node_id):
        for e in artifact.get("edges", []):
            if e.get("type") == "SPOKEN_BY" and e.get("from") == quote_node_id:
                node = next(n for n in artifact["nodes"] if n.get("id") == e.get("to"))
                return e.get("to"), node["properties"]["name"]
        return None, None

    monkeypatch.setattr(load, "_speaker_graph_meta", old_meta)
    monkeypatch.setattr(
        lift,
        "_lifted_speaker",
        lambda pid, name: {"id": pid, "display_name": name} if pid else None,
    )
    monkeypatch.setattr(
        enrichment,
        "_roster_from_record",
        lambda content, diag: [{"name": "SPEAKER_09"}] + list(content.get("speakers") or []),
    )
    r1 = name_report.run(corpus)["r1_raw_labels"]
    assert r1["search_insight_quotes:raw"]["items"] >= 2
    assert r1["search_lifted_speaker:raw"]["items"] >= 1
    assert r1["mcp_speaker_roster:raw"]["episodes"] == 6
