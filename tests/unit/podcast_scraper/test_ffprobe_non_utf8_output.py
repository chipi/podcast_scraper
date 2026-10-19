"""ffprobe durations survive non-UTF-8 bytes on ffprobe's stderr.

ffmpeg/ffprobe print a file's tags and a URL's headers as stored; on prod's 2026-10-18 nightly
those bytes were not UTF-8 and a strict ``subprocess.run(text=True)`` raised UnicodeDecodeError
in the digital-silence trim. UnicodeDecodeError is a ValueError, so these three callers did not
crash: each swallowed it and returned no duration (None / 0.0), silently losing the timeout
sizing or drift check the duration is for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.audio_sync import drift
from podcast_scraper.providers.mistral.mistral_provider import MistralProvider
from podcast_scraper.providers.resilience import sockets

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_ffprobe(monkeypatch, tmp_path: Path) -> Path:
    (tmp_path / "err.bin").write_bytes(b"[mp3 @ 0x1] title : Caf\xe9 \xe2\x28 Cr\xe8me\n")
    fake = tmp_path / "ffprobe"
    fake.write_text(f'#!/bin/sh\ncat "{tmp_path / "err.bin"}" >&2\necho 1834.5\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    return tmp_path


def test_drift_served_seconds(fake_ffprobe: Path) -> None:
    assert drift._served_seconds(str(fake_ffprobe / "ep.mp3"), timeout=10) == pytest.approx(1834.5)


def test_mistral_probe_audio_duration(fake_ffprobe: Path) -> None:
    got = MistralProvider._probe_audio_duration_s(str(fake_ffprobe / "ep.mp3"))
    assert got == pytest.approx(1834.5)


def test_resilience_duration_via_ffprobe(fake_ffprobe: Path) -> None:
    got = sockets._duration_via_ffprobe(str(fake_ffprobe / "ep.mp3"))
    assert got == pytest.approx(1834.5)
