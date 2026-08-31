"""Audio duration falls back to ffprobe when soundfile can't answer (#1886).

The DGX diarization breaker tripped during the 2026-08-30 pilot because
``probe_audio_duration_sec`` returned None for every episode: ``soundfile`` is not in the
pipeline image, and even where it is installed it commonly cannot decode MP3. A None
duration collapses ``effective_timeout_sec`` to the flat base, so a 2-hour episode was
being given a 45-minute episode's budget — the duration-scaling code had effectively never
run in production.

ffmpeg (hence ffprobe) is installed unconditionally in the pipeline runtime image, so it is
available exactly where soundfile is not.
"""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from podcast_scraper.providers.resilience import sockets


def test_falls_through_to_ffprobe_when_soundfile_returns_none():
    with mock.patch.object(sockets, "_duration_via_soundfile", return_value=None):
        with mock.patch.object(sockets, "_duration_via_ffprobe", return_value=1234.5) as ff:
            assert sockets.probe_audio_duration_sec("/tmp/a.mp3") == 1234.5
    ff.assert_called_once()


def test_soundfile_wins_when_it_answers():
    """The in-process read is preferred; ffprobe must not be spawned needlessly."""
    with mock.patch.object(sockets, "_duration_via_soundfile", return_value=99.0):
        with mock.patch.object(sockets, "_duration_via_ffprobe") as ff:
            assert sockets.probe_audio_duration_sec("/tmp/a.mp3") == 99.0
    ff.assert_not_called()


def test_returns_none_when_neither_backend_can_answer():
    with mock.patch.object(sockets, "_duration_via_soundfile", return_value=None):
        with mock.patch.object(sockets, "_duration_via_ffprobe", return_value=None):
            assert sockets.probe_audio_duration_sec("/tmp/a.mp3") is None


def test_missing_ffprobe_binary_is_not_an_error():
    with mock.patch("shutil.which", return_value=None):
        assert sockets._duration_via_ffprobe("/tmp/a.mp3") is None


@pytest.mark.parametrize("stdout", ["", "N/A", "0", "-1", "not-a-number"])
def test_unusable_ffprobe_output_returns_none(stdout):
    """A duration of 0/NaN/garbage cannot size a budget — must not become a timeout."""
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    with mock.patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with mock.patch("subprocess.run", return_value=completed):
            assert sockets._duration_via_ffprobe("/tmp/a.mp3") is None


def test_nonzero_exit_returns_none():
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with mock.patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with mock.patch("subprocess.run", return_value=completed):
            assert sockets._duration_via_ffprobe("/tmp/a.mp3") is None


def test_a_wedged_ffprobe_cannot_outlive_the_call_it_sizes():
    with mock.patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with mock.patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=20)
        ):
            assert sockets._duration_via_ffprobe("/tmp/a.mp3") is None


def test_invoked_without_a_shell_because_the_path_comes_from_a_feed():
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="10.0", stderr="")
    with mock.patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with mock.patch("subprocess.run", return_value=completed) as run:
            sockets._duration_via_ffprobe("/tmp/a; rm -rf /.mp3")
    args, kwargs = run.call_args
    assert isinstance(args[0], list), "must use argv form, never a shell string"
    assert args[0][-1] == "/tmp/a; rm -rf /.mp3", "path passed as one argv element"
    assert kwargs.get("timeout") == sockets._FFPROBE_TIMEOUT_SEC
    assert "shell" not in kwargs or kwargs["shell"] is False


def test_duration_actually_scales_the_budget():
    """The point of the fix: a real duration must change the timeout."""
    flat = sockets.effective_timeout_sec(900.0, 3.0, None)
    scaled = sockets.effective_timeout_sec(900.0, 3.0, 2645.75)
    assert flat == 900.0
    assert scaled > flat
