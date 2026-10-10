"""Transcript chunks are indexed with the time they were said (operator 2026-10-10).

Chunks took their time only from ``summary.timestamps``, which the pipeline does not write, so on
prod every transcript passage in the search index sat at 0 ms while the timed transcript
(``*.segments.json``) was right beside it. The app worked around it at read time. The indexer now
times each chunk from the segments of the body it chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.indexer import _collect_docs_for_episode

pytestmark = pytest.mark.unit

SENTENCES = [f"Sentence number {i} says something about risk and time." for i in range(60)]


def _episode(root: Path, *, with_segments: bool = True, adfree_offsets: bool = False) -> tuple:
    (root / "metadata").mkdir(parents=True)
    (root / "transcripts").mkdir()
    # A raw transcript puts separators between segments, so cumulative lengths do not line up.
    body = "\n\n".join(SENTENCES)
    (root / "transcripts" / "ep.txt").write_text(body, encoding="utf-8")
    if with_segments:
        segs = []
        pos = 0
        for i, s in enumerate(SENTENCES):
            seg = {"start": 10.0 * i, "end": 10.0 * i + 8.0, "text": s}
            if adfree_offsets:
                at = body.index(s, pos)
                seg.update(char_start=at, char_end=at + len(s))
                pos = at + len(s)
            segs.append(seg)
        (root / "transcripts" / "ep.segments.json").write_text(json.dumps(segs), encoding="utf-8")
    doc = {
        "feed": {"feed_id": "showx", "title": "Show X", "language": "en"},
        "episode": {"episode_id": "ep1", "title": "T", "published_date": "2026-06-15T00:00:00Z"},
        "content": {"transcript_file_path": "transcripts/ep.txt"},
        # No summary.timestamps: what the pipeline actually writes.
        "summary": {"bullets": ["A point."]},
    }
    meta = root / "metadata" / "ep.metadata.json"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    return meta, doc


def _chunks(root: Path, meta: Path, doc: dict) -> list[dict]:
    rows = _collect_docs_for_episode(
        root,
        meta,
        doc,
        target_tokens=60,
        overlap_tokens=0,
        metadata_relative_path="metadata/ep.metadata.json",
    )
    return [m for _, _, m in rows if m.get("doc_type") == "transcript"]


@pytest.mark.parametrize("adfree_offsets", [False, True], ids=["raw", "segment-offsets"])
def test_every_chunk_is_timed_from_the_segments_it_spans(tmp_path: Path, adfree_offsets) -> None:
    meta, doc = _episode(tmp_path, adfree_offsets=adfree_offsets)
    chunks = _chunks(tmp_path, meta, doc)
    assert len(chunks) > 3
    body = (tmp_path / "transcripts" / "ep.txt").read_text(encoding="utf-8")
    for ch in chunks:
        text = body[ch["char_start"] : ch["char_end"]]
        said = [i for i, s in enumerate(SENTENCES) if s in text]
        assert said, text
        # Starts with the first sentence the chunk holds, ends with the last.
        assert ch["timestamp_start_ms"] <= said[0] * 10_000
        assert ch["timestamp_end_ms"] >= said[-1] * 10_000 + 8_000
    assert chunks[-1]["timestamp_start_ms"] > 0


def test_no_timed_transcript_no_invented_time(tmp_path: Path) -> None:
    meta, doc = _episode(tmp_path, with_segments=False)
    assert all(ch["timestamp_start_ms"] is None for ch in _chunks(tmp_path, meta, doc))
