"""Unit tests for GIL Quote vs FAISS chunk offset verification (#528)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.search import gil_chunk_offset_verify as v


def test_half_open_ranges_overlap() -> None:
    assert v.half_open_ranges_overlap(0, 10, 5, 15) is True
    assert v.half_open_ranges_overlap(0, 5, 5, 10) is False
    assert v.half_open_ranges_overlap(5, 10, 0, 6) is True


def test_overlap_width() -> None:
    assert v.overlap_width(0, 10, 5, 15) == 5
    assert v.overlap_width(0, 5, 5, 10) == 0


def test_transcript_chunk_spans_by_episode() -> None:
    meta: Dict[str, Dict[str, Any]] = {
        "a": {
            "doc_type": "transcript",
            "episode_id": "episode:1",
            "char_start": 0,
            "char_end": 100,
        },
        "b": {"doc_type": "insight", "episode_id": "episode:1"},
    }
    out = v.transcript_chunk_spans_by_episode(meta)
    assert out == {"episode:1": [(0, 100)]}


def test_quote_spans_from_gi() -> None:
    gi = {
        "nodes": [
            {
                "id": "q1",
                "type": "Quote",
                "properties": {"char_start": 10, "char_end": 40},
            },
            {"id": "x", "type": "Insight", "properties": {}},
        ]
    }
    rows = v.quote_spans_from_gi(gi)
    assert rows == [("q1", 10, 40)]


def test_build_offset_alignment_report_aligned(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {
                "id": "q1",
                "type": "Quote",
                "properties": {"char_start": 5, "char_end": 25},
            }
        ]
    }
    gpath = tmp_path / "a.gi.json"
    gpath.write_text(__import__("json").dumps(gi), encoding="utf-8")
    meta = {
        "c0": {
            "doc_type": "transcript",
            "episode_id": "episode:x",
            "char_start": 0,
            "char_end": 50,
        }
    }
    gi_map = {"episode:x": gpath}
    rep = v.build_offset_alignment_report(gi_by_episode=gi_map, metadata_by_doc=meta)
    assert rep["quotes_total"] == 1
    assert rep["quotes_with_chunk_overlap"] == 1
    assert rep["overlap_rate"] == 1.0
    assert rep["verdict"] == "aligned"


def test_build_offset_alignment_report_no_overlap(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {
                "id": "q1",
                "type": "Quote",
                "properties": {"char_start": 100, "char_end": 120},
            }
        ]
    }
    gpath = tmp_path / "b.gi.json"
    gpath.write_text(__import__("json").dumps(gi), encoding="utf-8")
    meta = {
        "c0": {
            "doc_type": "transcript",
            "episode_id": "episode:y",
            "char_start": 0,
            "char_end": 50,
        }
    }
    rep = v.build_offset_alignment_report(
        gi_by_episode={"episode:y": gpath},
        metadata_by_doc=meta,
    )
    assert rep["quotes_total"] == 1
    assert rep["quotes_with_chunk_overlap"] == 0
    assert rep["overlap_rate"] == 0.0
    assert rep["verdict"] == "divergent"


def test_load_index_metadata_map_roundtrip(tmp_path: Path) -> None:
    blob = {"d1": {"doc_type": "transcript", "char_start": 1, "char_end": 2}}
    (tmp_path / "metadata.json").write_text(__import__("json").dumps(blob), encoding="utf-8")
    loaded = v.load_index_metadata_map(tmp_path)
    assert loaded["d1"]["doc_type"] == "transcript"


def test_load_index_metadata_map_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        v.load_index_metadata_map(tmp_path)


def test_transcript_chunk_spans_skips_bad_char_range() -> None:
    meta = {
        "a": {
            "doc_type": "transcript",
            "episode_id": "e1",
            "char_start": 10,
            "char_end": 5,
        },
        "b": {
            "doc_type": "transcript",
            "episode_id": "e2",
            "char_start": "x",
            "char_end": 10,
        },
    }
    assert v.transcript_chunk_spans_by_episode(meta) == {}


def test_quote_spans_skips_invalid_quote_nodes() -> None:
    gi = {
        "nodes": [
            {"id": "q1", "type": "Quote", "properties": {"char_start": 2, "char_end": 1}},
            {"id": "q2", "type": "Quote", "properties": "bad"},
            {"id": "q3", "type": "Insight", "properties": {"char_start": 0, "char_end": 1}},
        ]
    }
    assert v.quote_spans_from_gi(gi) == []


def test_build_offset_alignment_report_gi_load_failed(tmp_path: Path) -> None:
    bad = tmp_path / "missing.gi.json"
    rep = v.build_offset_alignment_report(
        gi_by_episode={"ep": bad},
        metadata_by_doc={},
    )
    assert rep["episodes"][0]["error"] == "gi_load_failed"


def test_build_offset_alignment_report_gi_not_object(tmp_path: Path) -> None:
    p = tmp_path / "arr.gi.json"
    p.write_text("[1]", encoding="utf-8")
    rep = v.build_offset_alignment_report(
        gi_by_episode={"ep": p},
        metadata_by_doc={},
    )
    assert rep["episodes"][0]["error"] == "gi_not_object"


def test_build_offset_verdict_mostly_aligned(tmp_path: Path) -> None:
    """17/20 overlap => 0.85 (mostly_aligned)."""
    nodes = [
        {
            "id": f"q{i}",
            "type": "Quote",
            "properties": {"char_start": i * 4, "char_end": i * 4 + 3},
        }
        for i in range(17)
    ]
    nodes.extend(
        [
            {
                "id": f"far{i}",
                "type": "Quote",
                "properties": {"char_start": 9000 + i, "char_end": 9005 + i},
            }
            for i in range(3)
        ]
    )
    gi = {"nodes": nodes}
    gpath = tmp_path / "m.gi.json"
    gpath.write_text(__import__("json").dumps(gi), encoding="utf-8")
    meta = {
        "c": {
            "doc_type": "transcript",
            "episode_id": "episode:x",
            "char_start": 0,
            "char_end": 120,
        }
    }
    rep = v.build_offset_alignment_report(
        gi_by_episode={"episode:x": gpath},
        metadata_by_doc=meta,
    )
    assert rep["verdict"] == "mostly_aligned"
    assert rep["overlap_rate"] == pytest.approx(17 / 20)


def test_merge_report_dict() -> None:
    target: dict = {"a": 1}
    v.merge_report_dict(target, {"b": 2})
    assert target == {"a": 1, "b": 2}
