"""#2359 option A: per-episode maps from transcript time to the time in the stream served now."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.app_audio_drift import MAPS_FILE, time_maps, to_stream

pytestmark = pytest.mark.unit


def test_to_stream_applies_the_offset_in_force_at_each_time() -> None:
    anchors = ((0.0, 8.0), (1200.0, -75.0), (2500.0, 34.0))
    assert to_stream(anchors, 10.0) == 18.0
    assert to_stream(anchors, 1199.9) == pytest.approx(1207.9)
    assert to_stream(anchors, 1200.0) == 1125.0
    assert to_stream(anchors, 3000.0) == 3034.0


def test_time_maps_reads_the_file_and_ignores_broken_entries(tmp_path: Path) -> None:
    assert time_maps(tmp_path) == {}
    (tmp_path / MAPS_FILE).write_text(
        json.dumps(
            {
                "a.metadata.json": {"anchors": [[900.0, -3.0], [0.0, 12.5]]},
                "b.metadata.json": {"anchors": [["x", 1]]},
                "c.metadata.json": {"anchors": []},
            }
        ),
        encoding="utf-8",
    )
    assert time_maps(tmp_path) == {"a.metadata.json": ((0.0, 12.5), (900.0, -3.0))}


def test_an_unreadable_maps_file_means_no_maps(tmp_path: Path) -> None:
    (tmp_path / MAPS_FILE).write_text("{nope", encoding="utf-8")
    assert time_maps(tmp_path) == {}
