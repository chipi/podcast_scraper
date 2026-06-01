"""Unit tests for segment-document build + insight linking (RFC-090 §3.8, #857)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import SegmentDocument
from podcast_scraper.search.segments import build_segment_documents, link_insights_to_segments

pytestmark = pytest.mark.unit


def test_build_segment_documents_ids_and_text():
    text = "Sam Altman discussed AI. Tim Cook talked earnings. The market reacted."
    segs = build_segment_documents(text, episode_id="ep1", show_id="showA", target_tokens=5)
    assert len(segs) >= 1
    assert all(isinstance(s, SegmentDocument) for s in segs)
    assert segs[0].id == "ep1_chunk_0"
    assert segs[0].show_id == "showA" and segs[0].episode_id == "ep1"
    assert segs[0].source_tier == "segment" and segs[0].embedding == []


def test_build_segment_documents_timestamps_to_seconds():
    text = "Hello world this is a sentence. And another one here please."
    ts = [
        {"char_start": 0, "char_end": 31, "start_ms": 1000, "end_ms": 4000},
        {"char_start": 31, "char_end": 60, "start_ms": 4000, "end_ms": 7000},
    ]
    segs = build_segment_documents(
        text, episode_id="ep1", show_id="s", timestamps=ts, target_tokens=50
    )
    # ms -> seconds.
    assert segs[0].start_time == 1.0 and segs[0].end_time == 7.0


def test_build_segment_documents_no_timestamps_defaults_zero():
    segs = build_segment_documents("one two three.", episode_id="ep1", show_id="s")
    assert segs[0].start_time == 0.0 and segs[0].end_time == 0.0


def _seg(sid, start, end):
    return SegmentDocument(
        id=sid, text="t", show_id="s", episode_id="ep1", start_time=start, end_time=end
    )


def test_link_insight_within_segment_time():
    segs = [_seg("ep1_chunk_0", 0.0, 10.0), _seg("ep1_chunk_1", 10.0, 20.0)]
    mapping = link_insights_to_segments(segs, [("insight:1", 12.0, 15.0)])
    assert mapping == {"insight:1": "ep1_chunk_1"}
    assert segs[1].linked_insight_ids == ["insight:1"]
    assert segs[0].linked_insight_ids == []


def test_link_respects_tolerance():
    segs = [_seg("ep1_chunk_0", 0.0, 10.0)]
    # quote at 10.5s is just past the segment but within the 2s tolerance.
    mapping = link_insights_to_segments(segs, [("insight:1", 9.5, 10.5)], tolerance_seconds=2.0)
    assert mapping == {"insight:1": "ep1_chunk_0"}


def test_link_no_match_when_far_outside():
    segs = [_seg("ep1_chunk_0", 0.0, 10.0)]
    mapping = link_insights_to_segments(segs, [("insight:1", 50.0, 55.0)])
    assert mapping == {}
    assert segs[0].linked_insight_ids == []


def test_link_skips_insight_without_timestamp():
    segs = [_seg("ep1_chunk_0", 0.0, 10.0)]
    assert link_insights_to_segments(segs, [("insight:1", None, None)]) == {}
