"""The overlap scan counts #2370 drift and ignores rounding between adjacent segments."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.asr_segment_overlaps import main, overlaps, scan


def test_back_to_back_segments_and_rounding_are_not_overlaps():
    segs = [{"start": 0, "end": 5.0}, {"start": 4.98, "end": 9}, {"start": 9, "end": 12}]
    assert overlaps(segs) == {"segs": 3, "overlaps": 0, "worst_s": 0.0}


def test_a_drifted_passage_counts_against_the_latest_end_seen():
    # A segment that jumps back counts, and so does every later one still inside the old span.
    segs = [{"start": 0, "end": 60}, {"start": 30, "end": 35}, {"start": 40, "end": 45}]
    assert overlaps(segs) == {"segs": 3, "overlaps": 2, "worst_s": 30.0}


def _episode(run: Path, guid: str, segments) -> None:
    stem = f"0001 - {guid}"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    (run / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps({"feed": {"url": "https://x/rss"}, "episode": {"guid": guid}}), encoding="utf-8"
    )
    if segments is not None:
        (run / "transcripts" / f"{stem}.segments.json").write_text(
            json.dumps({"segments": segments}), encoding="utf-8"
        )


def test_scan_reports_each_served_episode_and_one_without_segments(tmp_path: Path, capsys):
    feed = tmp_path / "feeds" / "rss_x"
    _episode(feed / "run_a", "drifted", [{"start": 0, "end": 60}, {"start": 1, "end": 2}])
    _episode(feed / "run_b", "clean", [{"start": 0, "end": 1}, {"start": 1, "end": 2}])
    _episode(feed / "run_c", "nosegs", None)

    rows = {r["guid"]: r for r in scan(tmp_path)}
    assert rows["drifted"]["overlaps"] == 1
    assert rows["clean"]["overlaps"] == 0
    assert rows["nosegs"] == {
        "guid": "nosegs",
        "md": rows["nosegs"]["md"],
        "segs": 0,
        "overlaps": None,
    }

    assert main(["--corpus-dir", str(tmp_path), "--min-overlaps", "1"]) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["guid"] for r in printed] == ["drifted"]
