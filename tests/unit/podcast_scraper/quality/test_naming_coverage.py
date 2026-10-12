"""quality.naming_coverage: the served corpus's absolute naming coverage (#2304 S8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.quality import naming_coverage as nc

pytestmark = [pytest.mark.unit]


def _episode(root: Path, name: str, labels: list, quotes: list, language: str = "en") -> None:
    run = root / "feeds" / "f1" / f"run_{name}"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    (run / "metadata" / f"0001 - {name}.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f1", "title": "Show", "language": language},
                "episode": {"episode_id": name, "title": name},
                "content": {"transcript_file_path": f"transcripts/0001 - {name}.txt"},
            }
        ),
        encoding="utf-8",
    )
    nodes = [
        {"id": f"quote:{i}", "type": "Quote", "properties": {"speaker_id": sid}}
        for i, sid in enumerate(quotes)
    ]
    (run / "metadata" / f"0001 - {name}.gi.json").write_text(
        json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8"
    )
    (run / "transcripts" / f"0001 - {name}.segments.json").write_text(
        json.dumps(
            [{"speaker": f"SPEAKER_0{i}", "speaker_label": x} for i, x in enumerate(labels)]
        ),
        encoding="utf-8",
    )


def test_the_corpus_is_measured_and_the_snapshot_written(tmp_path: Path) -> None:
    _episode(tmp_path, "a", ["Tom Holland", "SPEAKER_01"], ["person:tom-holland", None])
    _episode(tmp_path, "b", ["SPEAKER_00", "SPEAKER_01"], [None, None])
    _episode(tmp_path, "c", ["SPEAKER_00"], [None], language="de")  # another language: out
    summary = nc.measure_and_record(tmp_path)
    assert summary is not None
    assert (summary["episodes"], summary["episodes_no_named_voice"]) == (2, 1)
    assert (summary["quotes"], summary["named_quotes"]) == (4, 1)
    snap = nc.read_snapshot(tmp_path)
    assert snap is not None and snap["episodes_no_named_voice"] == 1
    assert snap["language"] == "en" and snap["measured_at"] > 0
    assert not list((tmp_path / ".podcast_scraper").glob("*.tmp"))  # the write was atomic


def test_a_failed_measurement_never_raises_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    def boom(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(nc, "scan_corpus", boom)
    assert nc.measure_and_record(tmp_path) is None
    assert nc.read_snapshot(tmp_path) is None


def test_an_unreadable_snapshot_reads_as_none(tmp_path: Path) -> None:
    path = tmp_path / nc.SNAPSHOT_RELPATH
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert nc.read_snapshot(tmp_path) is None
