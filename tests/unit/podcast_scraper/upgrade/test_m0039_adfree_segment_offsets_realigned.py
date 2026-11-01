"""m0039: ad-free segment offsets that no longer slice to their text, realigned from the text."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0039_adfree_segment_offsets_realigned as m38a
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

LINES = [
    "Mara Tull: Welcome to the show.",
    "Odile Brand: Thanks for having me.",
    "Mara Tull: Tell me more.",
]


def _episode(root: Path, text: str, rows: list) -> dict:
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
        ),
        encoding="utf-8",
    )
    (run / "transcripts" / "ep.txt").write_text(text, encoding="utf-8")
    (run / "transcripts" / "ep.adfree.txt").write_text(text, encoding="utf-8")
    seg = run / "transcripts" / "ep.adfree.segments.json"
    seg.write_text(json.dumps(rows), encoding="utf-8")
    return {"seg": seg, "txt": run / "transcripts" / "ep.adfree.txt"}


def _rows_for(text: str, shift_after: int = 0, shift: int = 0) -> list:
    """Rows for each line's spoken text; rows from line ``shift_after`` on stored ``shift`` late."""
    out, pos = [], 0
    for i, line in enumerate(text.split("\n")):
        words = line.split(": ", 1)[1]
        start = text.index(words, pos)
        off = shift if i >= shift_after else 0
        out.append(
            {
                "start": float(i),
                "end": i + 0.5,
                "speaker": f"S{i}",
                "text": words,
                "char_start": start + off,
                "char_end": start + off + len(words),
            }
        )
        pos = start + len(words)
    return out


def _apply(root: Path, dry_run: bool = False):
    return m38a.AdfreeSegmentOffsetsRealignedMigration().apply(
        MigrationContext(corpus_root=root, dry_run=dry_run)
    )


def _verify(root: Path):
    return m38a.AdfreeSegmentOffsetsRealignedMigration().verify(MigrationContext(corpus_root=root))


def test_registered_between_0038_and_0041() -> None:
    ids = [m.id for m in get_migrations()]
    i = ids.index(m38a.MIGRATION_ID)
    assert ids[i - 1] == "0038_raw_label_speakers_removed"
    assert ids[i + 1] == "0040_gi_quote_offsets_realigned"


def test_rows_left_early_by_a_shortened_prefix_take_their_text_again(tmp_path: Path) -> None:
    text = "\n".join(LINES)
    rows = _rows_for(text, shift_after=1, shift=3)  # stored 3 late, as if a prefix lost 3 chars
    p = _episode(tmp_path, text, rows)
    text_before = p["txt"].read_bytes()
    assert not _verify(tmp_path)[0]
    _apply(tmp_path, dry_run=True)
    assert json.loads(p["seg"].read_text()) == rows  # a dry run writes nothing
    res = _apply(tmp_path)
    assert res.details["refused"] == [] and res.details["episodes"][0]["rows"] == 2
    after = json.loads(p["seg"].read_text())
    assert all(text[r["char_start"] : r["char_end"]] == r["text"] for r in after)

    def strip(rs):
        return [{k: v for k, v in r.items() if k not in ("char_start", "char_end")} for r in rs]

    assert strip(after) == strip(rows)  # only the offsets move
    assert after[0] == rows[0]  # a row that sliced is unchanged
    assert p["txt"].read_bytes() == text_before
    assert _verify(tmp_path)[0]
    assert _apply(tmp_path).details["episodes"] == []  # a second run writes nothing
    restored, refused = m38a.undo(tmp_path)
    assert restored and not refused
    assert json.loads(p["seg"].read_text()) == rows


def _row(text: str, start: int) -> dict:
    return {"text": text, "char_start": start, "char_end": start + len(text)}


def test_a_repeated_short_line_near_the_row_refuses_the_episode() -> None:
    text = "A: Yeah. Yeah.\nB: Right."  # "Yeah." twice within the window of a drifted row
    with pytest.raises(m38a.Refused, match="found 2 times"):
        m38a.realigned(text, [_row("Yeah.", 4)])


def test_a_drift_that_shrinks_refuses_the_episode() -> None:
    text = "\n".join(LINES)
    rows = _rows_for(text)
    for r, k in ((rows[1], 3), (rows[2], 1)):  # stored 3 late, then 1 late: drift -3 then -1
        r["char_start"] += k
        r["char_end"] += k
    with pytest.raises(m38a.Refused, match="shrinks"):
        m38a.realigned(text, rows)


def test_a_span_that_would_overlap_the_row_before_refuses_the_episode() -> None:
    text = "A: hello world\nB: something else entirely, then world"
    rows = [_row("hello world", 3), _row("world", 4)]  # the only "world" near 4 is inside row 0
    with pytest.raises(m38a.Refused, match="overlaps"):
        m38a.realigned(text, rows)


def test_a_drift_that_reverses_refuses_the_episode() -> None:
    text = "\n".join(LINES)
    rows = _rows_for(text)
    rows[1]["char_start"] += 2
    rows[1]["char_end"] += 2
    rows[2]["char_start"] -= 2
    rows[2]["char_end"] -= 2
    with pytest.raises(m38a.Refused):
        m38a.realigned(text, rows)


def test_an_episode_whose_rows_slice_is_untouched(tmp_path: Path) -> None:
    text = "\n".join(LINES)
    p = _episode(tmp_path, text, _rows_for(text))
    before = p["seg"].read_bytes()
    assert _apply(tmp_path).details["episodes"] == []
    assert p["seg"].read_bytes() == before and _verify(tmp_path)[0]
