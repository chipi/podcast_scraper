"""#2359 option A: per-episode maps from transcript time to the time in the stream served now."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import List, Tuple

import pytest

from podcast_scraper.server.app_audio_drift import MAPS_FILE, time_maps, to_stream

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "ops" / "audio_retime.py"
_spec = importlib.util.spec_from_file_location("audio_retime", _SCRIPT)
assert _spec and _spec.loader
retime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retime)

WORDS = (
    "harbour lantern copper meadow violin granite orchard falcon ember tundra quartz saddle "
    "pepper glacier marble cobalt thimble walnut beacon prairie canyon velvet juniper anchor "
    "compass lagoon pebble sorrel mantle cinder"
).split()


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


def _transcript(n: int, spacing: float = 1.0) -> List[Tuple[str, float]]:
    return [(WORDS[i % len(WORDS)] + str(i // len(WORDS)), i * spacing) for i in range(n)]


def test_find_votes_on_the_offset_most_word_runs_agree_on() -> None:
    tw = _transcript(400)
    heard = [(w, t + 40.0) for w, t in tw[100:118]]
    heard[5] = ("static", heard[5][1])  # a misheard word breaks three runs, not the answer
    hit = retime.find(tw, heard)
    assert hit is not None
    transcript_t, stream_t = hit
    assert stream_t - transcript_t == pytest.approx(40.0)


def test_find_needs_three_votes() -> None:
    tw = _transcript(400)
    assert retime.find(tw, [(w, t) for w, t in tw[10:13]]) is None
    assert retime.find(tw, [("ad", 1.0), ("words", 2.0), ("only", 3.0), ("here", 4.0)]) is None


class FakeStream:
    """A stream that plays transcript words [0, cut) with offset ``a``, then an ad, then the rest
    with offset ``b``; listening returns the words a 20 s window would hear."""

    def __init__(self, tw: List[Tuple[str, float]], cut: float, a: float, b: float) -> None:
        self.words = [(w, t + (a if t < cut else b)) for w, t in tw]
        self.calls = 0

    def heard(self, at: float, length: float = 20.0) -> List[Tuple[str, float]]:
        self.calls += 1
        return [(w, t) for w, t in self.words if at <= t < at + length]


def test_build_map_places_the_change_where_the_talk_resumes() -> None:
    tw = _transcript(3000)
    stream = FakeStream(tw, cut=1500.0, a=10.0, b=70.0)  # a 60 s ad inserted at 1510 s
    anchors = retime.build_map(tw, stream, served_len=3070.0)  # type: ignore[arg-type]
    assert anchors[0] == [0.0, 10.0]
    assert len(anchors) == 2
    at, off = anchors[1]
    assert off == 70.0
    assert 1500.0 <= at <= 1504.0
    for t in (1000.0, 1499.0, 1504.0, 2500.0):
        expected = t + (10.0 if t < 1500.0 else 70.0)
        assert to_stream(tuple(map(tuple, anchors)), t) == pytest.approx(expected)


def test_build_map_with_no_change_is_one_anchor() -> None:
    tw = _transcript(1000)
    stream = FakeStream(tw, 9999.0, 5.0, 5.0)
    anchors = retime.build_map(tw, stream, 1005.0)  # type: ignore[arg-type]
    assert anchors == [[0.0, 5.0]]
