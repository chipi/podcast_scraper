"""RFC-129: line up a stream with ads inserted per request against the original we transcribed.

The original is a fixture episode. A "listener's" stream is built from it with ads (speech from
another fixture episode) inserted at known points, then re-encoded at another bitrate, as a host
stitching ads would. The alignment must find each stretch of talk, its offset to one frame, and its
edges within a fraction of a second.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from podcast_scraper.audio_sync.fingerprint import align_files, decode, SR, time_map

pytestmark = pytest.mark.integration

AUDIO = Path(__file__).resolve().parents[2] / "fixtures" / "audio" / "v3"
ORIGINAL = AUDIO / "p01_e01.mp3"  # 488 s
ADS = AUDIO / "p01_e02.mp3"  # other speech, used as the inserted ads


def _encode(samples: np.ndarray, path: Path, bitrate: str = "128k") -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-"]
        + ["-b:a", bitrate, str(path)],
        input=samples.astype(np.float32).tobytes(),
        check=True,
    )
    return path


def _s(x: np.ndarray, a: float, b: float | None = None) -> np.ndarray:
    return x[int(a * SR) : None if b is None else int(b * SR)]


@pytest.fixture(scope="module")
def original() -> np.ndarray:
    return decode(str(ORIGINAL))


@pytest.fixture(scope="module")
def ads() -> np.ndarray:
    return decode(str(ADS))


def test_finds_each_stretch_and_how_far_it_moved(
    tmp_path: Path, original: np.ndarray, ads: np.ndarray
) -> None:
    # A 30 s ad after 120 s of talk, a 45 s ad after 300 s.
    stream = np.concatenate(
        [
            _s(original, 0, 120),
            _s(ads, 10, 40),
            _s(original, 120, 300),
            _s(ads, 100, 145),
            _s(original, 300),
        ]
    )
    found = align_files(str(ORIGINAL), str(_encode(stream, tmp_path / "listener.mp3")))
    assert [s.offset for s in found] == pytest.approx([0.0, 30.0, 75.0], abs=0.04)
    assert found[0].start == pytest.approx(0.0, abs=0.3)
    # The fixture pauses at 119.2-119.9 s: anywhere in the pause is the edge.
    assert found[0].end == pytest.approx(120.0, abs=0.75)
    assert found[1].start == pytest.approx(120.0, abs=0.75)
    assert found[1].end == pytest.approx(300.0, abs=0.5)
    assert found[2].start == pytest.approx(300.0, abs=0.5)
    assert time_map(found)[0] == [0.0, 0.0]


def test_a_recording_heard_twice_is_not_taken_for_talk(
    tmp_path: Path, original: np.ndarray
) -> None:
    """An ad used in two breaks matches in two places (seen on prod 2026-10-16). Here 40 s of the
    original itself is inserted again at 200 s: the talk is the chain in order, without the copy."""
    stream = np.concatenate([_s(original, 0, 200), _s(original, 400, 440), _s(original, 200)])
    found = align_files(str(ORIGINAL), str(_encode(stream, tmp_path / "listener.mp3")))
    assert [s.offset for s in found] == pytest.approx([0.0, 40.0], abs=0.04)
    assert found[0].end == pytest.approx(200.0, abs=0.5)
    assert found[1].start == pytest.approx(200.0, abs=0.5)


def test_an_unchanged_stream_is_one_stretch_at_zero(tmp_path: Path, original: np.ndarray) -> None:
    found = align_files(str(ORIGINAL), str(_encode(original, tmp_path / "same.mp3", "96k")))
    assert len(found) == 1
    assert found[0].offset == pytest.approx(0.0, abs=0.04)
    assert found[0].seconds > 480


def test_each_checkpoint_of_the_reference_is_found_in_a_short_chunk(
    tmp_path: Path, original: np.ndarray, ads: np.ndarray
) -> None:
    """What the app will do: a chunk of its own stream around where a checkpoint should be, and
    the checkpoint's window located in it, at the offset the inserted ads make."""
    from podcast_scraper.audio_sync import reference

    # The original "had" an ad at 200-215 s (our transcript marks it); the listener's stream has a
    # 40 s ad there instead, and a 25 s one at the very start.
    stream = np.concatenate(
        [_s(ads, 0, 25), _s(original, 0, 200), _s(ads, 50, 90), _s(original, 215)]
    )
    decoded = decode(str(_encode(stream, tmp_path / "listener.mp3")))
    doc = reference.build(original, [(200_000, 215_000)])
    kinds = [c["kind"] for c in doc["checkpoints"]]
    assert kinds[0] == "talk_start" and "after_ad" in kinds
    for cp in doc["checkpoints"]:
        expected = 25.0 if cp["t"] < 200 else 25.0 + 40.0 - 15.0
        a = cp["t"] + expected - 10.0
        at, match = reference.locate(
            reference.decode_window(cp["frames"]), _s(decoded, a, a + 30.0), a
        )
        assert match >= 0.8, cp
        assert at - cp["t"] == pytest.approx(expected, abs=0.04), cp


def test_every_checkpoint_carries_a_long_window(original: np.ndarray) -> None:
    """RFC-129 v2: a window reaching ``REACH_S`` either side of a guess plus the probe, at intervals
    too (a host can insert a break where the original had none), starting with the short one."""
    from podcast_scraper.audio_sync import reference

    doc = reference.build(original, [(20_000, 25_000)])
    assert doc["version"] == 2 and doc["reach_s"] == reference.REACH_S
    kinds = {c["kind"] for c in doc["checkpoints"]}
    assert {"talk_start", "after_ad", "interval"} <= kinds
    for cp in doc["checkpoints"]:
        long = reference.decode_window(cp["long"])
        short = reference.decode_window(cp["frames"])
        assert np.array_equal(long[: len(short)], short), cp["t"]  # the short window is its start
    after = next(c for c in doc["checkpoints"] if c["kind"] == "after_ad")
    assert len(reference.decode_window(after["long"])) == int(reference.LONG_WINDOW_S * 50)


def test_a_long_window_stops_where_the_next_ad_starts(original: np.ndarray) -> None:
    """Past the next ad the offset is different: a probe matching there must not be read as this
    checkpoint's (RFC-129 v2)."""
    from podcast_scraper.audio_sync import reference

    doc = reference.build(original, [(100_000, 110_000), (150_000, 160_000)])
    first = next(c for c in doc["checkpoints"] if c["kind"] == "after_ad")
    assert first["t"] == pytest.approx(115.0)
    frames = len(reference.decode_window(first["long"]))
    assert frames == pytest.approx((150.0 - 115.0) * 50, abs=2)
