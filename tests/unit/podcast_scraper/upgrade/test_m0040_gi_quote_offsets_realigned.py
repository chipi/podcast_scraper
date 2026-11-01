"""m0040: GI quotes that do not slice to their text, realigned by their own time window."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0040_gi_quote_offsets_realigned as m38b
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

# The same phrase twice: only the time window can tell which a quote means.
LINES = [
    ("Mara Tull", 0.0, 4.0, "the franchise model works here."),
    ("Odile Brand", 4.0, 8.0, "I disagree with that."),
    ("Mara Tull", 8.0, 12.0, "Then the franchise model fails there."),
]


def _episode(root: Path, quotes: list) -> dict:
    run = root / "feeds" / "f" / "run_1"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    meta = run / "metadata" / "ep.metadata.json"
    meta.write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep"},
                "content": {"transcript_file_path": "transcripts/ep.txt"},
            }
        )
    )
    text, rows, pos = "", [], 0
    for who, t0, t1, words in LINES:
        line = f"{who}: {words}\n"
        start = pos + len(who) + 2
        rows.append(
            {
                "start": t0,
                "end": t1,
                "text": words,
                "char_start": start,
                "char_end": start + len(words),
            }
        )
        text += line
        pos += len(line)
    (run / "transcripts" / "ep.adfree.txt").write_text(text)
    (run / "transcripts" / "ep.adfree.segments.json").write_text(json.dumps(rows))
    nodes = [
        {
            "id": f"quote:{i}",
            "type": "Quote",
            "properties": {"transcript_ref": "transcripts/ep.adfree.txt", **q},
        }
        for i, q in enumerate(quotes)
    ]
    gi = run / "metadata" / "ep.gi.json"
    gi.write_text(json.dumps({"episode_id": "ep", "nodes": nodes, "edges": []}))
    return {"gi": gi, "text": text, "seg": run / "transcripts" / "ep.adfree.segments.json"}


def _q(words: str, cs: int, t0: float, t1: float) -> dict:
    return {
        "text": words,
        "char_start": cs,
        "char_end": cs + len(words),
        "timestamp_start_ms": int(t0 * 1000),
        "timestamp_end_ms": int(t1 * 1000),
    }


def _apply(root: Path, dry_run: bool = False):
    return m38b.GiQuoteOffsetsRealignedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _verify(root: Path):
    return m38b.GiQuoteOffsetsRealignedMigration().verify(MigrationContext(corpus_root=root))


def _quotes(p: dict) -> list:
    return [n["properties"] for n in json.loads(p["gi"].read_text())["nodes"]]


def test_registered_between_0039_and_0041() -> None:
    ids = [m.id for m in get_migrations()]
    i = ids.index(m38b.MIGRATION_ID)
    assert ids[i - 1] == "0039_adfree_segment_offsets_realigned"
    assert ids[i + 1] == "0041_joined_names_respelled"


def test_a_quote_takes_the_occurrence_inside_its_own_time(tmp_path: Path) -> None:
    # Its stored offsets point nowhere useful; its time says the third line.
    p = _episode(tmp_path, [_q("the franchise model", 3, 8.5, 10.0), _q("I disagree", 0, 4.0, 5.0)])
    p2 = json.loads(p["gi"].read_text())
    p2["nodes"][1]["properties"]["char_start"] = p["text"].index("I disagree")
    p2["nodes"][1]["properties"]["char_end"] = p2["nodes"][1]["properties"]["char_start"] + len(
        "I disagree"
    )
    p["gi"].write_text(json.dumps(p2))
    before = _quotes(p)
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert _quotes(p) == before
    res = _apply(tmp_path)
    assert res.details["episodes"][0]["quotes"] == 1
    q, other = _quotes(p)
    third = p["text"].index("Then the franchise model") + len("Then ")
    assert (q["char_start"], q["char_end"]) == (third, third + len("the franchise model"))
    assert {k: v for k, v in q.items() if k not in ("char_start", "char_end")} == {
        k: v for k, v in before[0].items() if k not in ("char_start", "char_end")
    }
    assert other == before[1]  # a quote that sliced is unchanged
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["episodes"] == []
    restored, refused = m38b.undo(tmp_path)
    assert restored and not refused and _quotes(p) == before


def test_a_quote_time_cannot_place_is_left_and_listed(tmp_path: Path) -> None:
    # Its window covers both lines with the phrase: two candidates.
    p = _episode(tmp_path, [_q("the franchise model", 3, 0.0, 12.0), _q("never said", 0, 0.0, 4.0)])
    before = _quotes(p)
    res = _apply(tmp_path)
    assert res.details["episodes"] == [] and _quotes(p) == before
    found = sorted(q["found"] for row in res.details["left"] for q in row["quotes"])
    assert found == [0, 2]
    assert _verify(tmp_path)[0]  # nothing time can place is left: verify passes, the rest listed


def test_segments_that_do_not_slice_refuse_the_episode(tmp_path: Path) -> None:
    p = _episode(tmp_path, [_q("the franchise model", 3, 8.5, 10.0)])
    rows = json.loads(p["seg"].read_text())
    rows[2]["char_start"] += 2
    rows[2]["char_end"] += 2
    p["seg"].write_text(json.dumps(rows))
    before = p["gi"].read_bytes()
    assert not _verify(tmp_path)[0]
    _apply(tmp_path)
    assert p["gi"].read_bytes() == before
