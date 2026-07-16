"""Audio preprocessing must preserve the transcript↔audio timeline (#1173).

Transcript timestamps are stored against the *original* audio (the player seeks it, the KG
cites it), but the transcriber only ever sees the *preprocessed* file. So preprocessing must
not change the duration. It used to: ``silenceremove`` with ``stop_periods=-1`` deleted every
interior pause, which shortened the audio the transcriber saw and pulled every later timestamp
early — accumulating to -20 s on a 25-min prod episode and -162 s on a 1h54m one.

These tests pin the invariant on the *real* ffmpeg pipeline, not a mock: same duration in, same
duration out.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from podcast_scraper.preprocessing.audio.ffmpeg_processor import FFmpegAudioPreprocessor

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


@pytest.fixture()
def speech_with_pauses(tmp_path: Path) -> Path:
    """20 s of audio: tone / 4 s silence / tone / 4 s silence / tone — 8 s of interior pause."""
    src = tmp_path / "src.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-filter_complex",
            "sine=frequency=300:duration=4[a];anullsrc=r=44100:cl=mono,atrim=0:4[s1];"
            "sine=frequency=300:duration=4[b];anullsrc=r=44100:cl=mono,atrim=0:4[s2];"
            "sine=frequency=300:duration=4[c];[a][s1][b][s2][c]concat=n=5:v=0:a=1",
            "-y",
            str(src),
        ],
        check=True,
        capture_output=True,
    )
    return src


def test_preprocessing_preserves_duration(speech_with_pauses: Path, tmp_path: Path) -> None:
    """The default (prod) preprocessor must not change the audio duration (#1173)."""
    out = tmp_path / "out.mp3"
    # Prod settings (speech_optimal_v1): aggressive thresholds, but removal off.
    pre = FFmpegAudioPreprocessor(
        sample_rate=16000,
        silence_threshold="-30dB",
        silence_duration=0.5,
        mp3_bitrate_kbps=32,
    )
    ok, _ = pre.preprocess(str(speech_with_pauses), str(out))
    assert ok
    src_dur, out_dur = _duration(speech_with_pauses), _duration(out)
    # mp3 frame padding moves the edges a hair; anything beyond that is a cut pause.
    assert abs(out_dur - src_dur) < 0.5, (
        f"preprocessing changed the timeline: {src_dur:.1f}s -> {out_dur:.1f}s. Transcript "
        f"timestamps are stored against the original audio, so this becomes seek drift (#1173)."
    )


def test_silence_removal_opt_in_still_cuts_the_timeline(
    speech_with_pauses: Path, tmp_path: Path
) -> None:
    """The destructive behaviour is still reachable — but only when explicitly asked for.

    This is the guard's other half: it proves the fixture really does contain removable silence,
    so the test above passes because removal is *off*, not because there was nothing to cut.
    """
    out = tmp_path / "out.mp3"
    pre = FFmpegAudioPreprocessor(
        sample_rate=16000,
        silence_threshold="-30dB",
        silence_duration=0.5,
        mp3_bitrate_kbps=32,
        silence_removal=True,
    )
    ok, _ = pre.preprocess(str(speech_with_pauses), str(out))
    assert ok
    assert _duration(speech_with_pauses) - _duration(out) > 4.0  # the interior pauses are gone


def test_cache_key_separates_silence_removal(speech_with_pauses: Path) -> None:
    """Flipping silence removal must invalidate the preprocessed-audio cache.

    Without this, a corpus built before the fix keeps serving its silence-stripped (drifted)
    cache entries and the fix silently does nothing.
    """
    src = str(speech_with_pauses)
    off = FFmpegAudioPreprocessor(sample_rate=16000, silence_removal=False).get_cache_key(src)
    on = FFmpegAudioPreprocessor(sample_rate=16000, silence_removal=True).get_cache_key(src)
    assert off != on
