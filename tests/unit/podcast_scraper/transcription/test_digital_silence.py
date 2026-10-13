"""Digital silence at the ends of an episode is cut off before ASR (Camera seduta 717, 2026-10-12).

717 opens with 103 s of zeros; Whisper wrote "Grazie a tutti." over the whole 66-minute sitting
(417 words, every attempt). Cut, the same audio gave 8,957 words against a reference of 8,836.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.transcription import digital_silence as ds

# ffmpeg 6/7 silencedetect output for 717's shape: zeros at both ends.
LOG_717 = """
  Duration: 01:06:12.13, start: 0.000000, bitrate: 32 kb/s
[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 102.912 | silence_duration: 102.912
[silencedetect @ 0x1] silence_start: 3906.1
[silencedetect @ 0x1] silence_end: 3972.13 | silence_duration: 66.03
"""


def test_both_silent_ends_are_read() -> None:
    edges = ds.silent_edges_from_log(LOG_717)
    assert edges is not None
    assert edges.lead_s == pytest.approx(102.912)
    assert edges.tail_s == pytest.approx(66.03, abs=0.01)
    assert edges.duration_s == pytest.approx(3972.13)


def test_a_silence_left_open_at_the_end_of_the_stream_reaches_the_end() -> None:
    log = LOG_717.replace(
        "[silencedetect @ 0x1] silence_end: 3972.13 | silence_duration: 66.03\n", ""
    )
    edges = ds.silent_edges_from_log(log)
    assert edges is not None and edges.tail_s == pytest.approx(66.03, abs=0.01)


def test_a_silence_in_the_middle_is_not_an_edge() -> None:
    log = """
  Duration: 00:30:00.00, start: 0.000000
[silencedetect @ 0x1] silence_start: 600
[silencedetect @ 0x1] silence_end: 640 | silence_duration: 40
"""
    edges = ds.silent_edges_from_log(log)
    assert edges is not None and edges.lead_s == 0.0 and edges.tail_s == 0.0
    assert ds.kept_span(edges) is None


def test_the_cut_keeps_a_margin_and_only_long_silence_counts() -> None:
    assert ds.kept_span(ds.SilentEdges(102.9, 66.0, 3972.1)) == pytest.approx((101.9, 3907.1))
    # Throughline 10-08 opens with 2.2 s of zeros: under the minimum, nothing is cut.
    assert ds.kept_span(ds.SilentEdges(2.2, 0.0, 2000.0)) is None
    assert ds.kept_span(ds.SilentEdges(12.0, 0.0, 2000.0)) == pytest.approx((11.0, 2000.0))


def test_an_all_silent_file_is_left_alone() -> None:
    log = """
  Duration: 00:01:00.00, start: 0.000000
[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 60 | silence_duration: 60
"""
    edges = ds.silent_edges_from_log(log)
    assert edges is not None and ds.kept_span(edges) is None


def test_an_unreadable_log_is_none() -> None:
    assert ds.silent_edges_from_log("no duration here") is None


def test_the_result_is_shifted_back_to_episode_time() -> None:
    result: Dict[str, Any] = {
        "text": "La seduta è ripresa.",
        "segments": [
            {
                "start": 1.4,
                "end": 7.5,
                "text": "La seduta è ripresa.",
                "words": [{"start": 1.4, "end": 1.6}],
            }
        ],
        "words": [{"start": 1.4, "end": 1.6, "word": "La"}],
    }
    out = ds.shift_result(result, 101.9)
    seg = out["segments"][0]
    assert seg["start"] == pytest.approx(103.3) and seg["end"] == pytest.approx(109.4)
    assert seg["words"][0]["start"] == pytest.approx(103.3)
    assert out["words"][0]["end"] == pytest.approx(103.5)
    assert result["segments"][0]["start"] == 1.4  # the input is not mutated


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_a_real_file_is_measured_and_cut(tmp_path: Path) -> None:
    src = tmp_path / "ep.mp3"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=15",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=20",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=12",
            "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1",
            "-ac", "1", str(src),
        ],
        check=True,
    )  # fmt: skip
    trim = ds.trim_for_asr(str(src), str(tmp_path))
    assert trim is not None
    assert trim.edges.lead_s == pytest.approx(15.0, abs=0.2)
    assert trim.edges.tail_s == pytest.approx(12.0, abs=0.3)
    assert trim.kept_start_s == pytest.approx(14.0, abs=0.2)
    cut = ds.silent_edges(trim.path)
    assert cut is not None and cut.duration_s == pytest.approx(22.0, abs=0.5)


def test_gap_recovery_does_not_re_transcribe_the_cut_silence(tmp_path: Path) -> None:
    """Camera seduta 722: silent-gap recovery added invented words in the 70 s of zeros at the
    end."""
    from podcast_scraper.transcription import gap_recovery

    clips: List[str] = []

    def transcribe_clip(path: str) -> Dict[str, Any]:
        clips.append(path)
        return {"segments": []}

    result = {
        "segments": [{"start": 102.0, "end": 110.0, "text": "La seduta è ripresa."}],
        "asr_digital_silence_trimmed": {"lead_s": 102.9, "tail_s": 66.0, "kept": [101.9, 3907.1]},
    }
    gaps = [
        {"start": 16.98, "end": 44.02, "speaker": None},  # inside the cut lead
        {"start": 4187.24, "end": 4217.2, "speaker": None},  # inside the cut tail
    ]
    out = gap_recovery.recover_silent_gaps(result, gaps, str(tmp_path / "x.mp3"), transcribe_clip)
    assert clips == []
    assert [r["reason"] for r in out["asr_silent_gap_recovery"]] == [
        "digital_silence",
        "digital_silence",
    ]


def test_the_episode_is_transcribed_trimmed_and_read_in_episode_time(monkeypatch, tmp_path) -> None:
    from unittest.mock import MagicMock

    from podcast_scraper import config as config_mod
    from podcast_scraper.workflow import episode_processor as ep

    trimmed = tmp_path / "asr_trimmed.mp3"
    edges = ds.SilentEdges(lead_s=102.9, tail_s=66.0, duration_s=3972.1)
    monkeypatch.setattr(
        ep.digital_silence,
        "trim_for_asr",
        lambda path, out_dir: ds.Trim(str(trimmed), 101.9, 3907.1, edges),
    )
    sent: List[str] = []

    class Provider:
        def transcribe_with_segments(self, path, **kw):
            sent.append(path)
            seg = {"start": 1.4, "end": 7.5, "text": "La seduta è ripresa."}
            return {"text": seg["text"], "segments": [seg]}, 1.0

    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"feed_declared_language": "it", "transcription_repair_unpunctuated_windows": False}
    )
    job = MagicMock()
    job.idx = 2
    result, _ = ep._transcribe_with_segments_maybe_chunked(
        str(tmp_path / "episode.mp3"),
        cfg=cfg,
        job=job,
        transcription_provider=Provider(),
        pipeline_metrics=None,
        episode_duration_seconds=3972.1,
        call_metrics=None,
    )
    assert sent == [str(trimmed)]
    assert result["segments"][0]["start"] == pytest.approx(103.3)
    assert result["asr_digital_silence_trimmed"] == {
        "lead_s": 102.9,
        "tail_s": 66.0,
        "kept": [101.9, 3907.1],
    }


def test_the_end_is_where_decoding_ended_not_the_header_estimate() -> None:
    """ffmpeg 7.1 (and CI's 6.1) on an MP3: the header estimates 47.09 s, decoding ends at 47.00 and
    the final silence ends there. Read against the header, the 12 s tail was 0 (CI, 2026-10-13)."""
    log = """
  Duration: 00:00:47.09, start: 0.000000, bitrate: 32 kb/s
[Parsed_silencedetect_0 @ 0x1] silence_start: 0
[Parsed_silencedetect_0 @ 0x1] silence_end: 14.983937 | silence_duration: 14.983937
[Parsed_silencedetect_0 @ 0x1] silence_start: 35.01775
size=N/A time=00:00:30.00 bitrate=N/A speed= 600x
[Parsed_silencedetect_0 @ 0x1] silence_end: 47 | silence_duration: 11.98225
[out#0/null @ 0x2] video:0KiB audio:0KiB
size=N/A time=00:00:47.00 bitrate=N/A speed= 610x
"""
    edges = ds.silent_edges_from_log(log)
    assert edges is not None
    assert edges.duration_s == pytest.approx(47.0)
    assert edges.tail_s == pytest.approx(11.98, abs=0.01)
