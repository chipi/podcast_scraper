"""Regression guard for #1646: speaker detection must never be gated on media size.

The defect ran from 2026-06-05 to 2026-08-14 and cost, measured on the live corpus:

    488 / 678 episodes (72 %)   speaker detection skipped
    2,112 / 8,952 insights (23.6 %)  became unsurfaceable
    82 episodes                 lost every insight they had

The cause was a 25 MB OpenAI-Whisper *upload* limit, applied while transcribing with
Deepgram, to a stage that reads the episode TITLE and DESCRIPTION and never opens the media
file. It survived because the guard's premise (#327: "skip speaker detection when
transcription will be skipped") stopped being true without the guard being removed.

These tests are deliberately narrow and behavioural: they assert the *decoupling*, not the
implementation. Any future re-coupling — a new provider added to a size tuple, a helpful
"optimisation" that skips detection for big files — fails here.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]

MB = 1024 * 1024


class _Episode:
    def __init__(self, media_url: str = "https://example.com/a.mp3", transcript_urls=None) -> None:
        self.idx = 1
        self.title = "Notion's Token Town — Simon Last & Sarah Sachs of Notion"
        self.media_url = media_url
        self.transcript_urls: List[str] = transcript_urls or []
        self.item = MagicMock()


class _Cfg:
    def __init__(self, **kwargs: Any) -> None:
        self.dry_run = False
        self.transcribe_missing = True
        self.transcription_provider = "deepgram"
        self.user_agent = "test"
        self.timeout = 5
        for key, value in kwargs.items():
            setattr(self, key, value)


def _head(size_bytes: int):
    response = MagicMock()
    response.headers = {"Content-Length": str(size_bytes)}
    return lambda *a, **k: response


def _size_skip(cfg: Any, episode: Any) -> Any:
    """Call the gate with duck-typed stubs.

    ``_check_episode_size_skip`` is annotated for the real ``Config``/``Episode``; these tests
    deliberately use minimal stubs so they stay in the unit tier (a real Config pulls provider
    validation and credentials). One typed seam beats an ``arg-type`` ignore on every call.
    """
    return processing._check_episode_size_skip(cfg, episode)


@pytest.mark.parametrize(
    "size_mb",
    [
        26,  # just over the limit
        41,  # The a16z Show, 44.6 min
        48,  # The a16z Show, 52.0 min
        76,  # Ideas of India, 91.3 min
        86,  # The Pragmatic Engineer, 85.5 min
    ],
)
def test_oversize_media_never_disables_speaker_detection(monkeypatch, size_mb: int) -> None:
    """Real file sizes measured on the damaged corpus, plus the boundary."""
    monkeypatch.setattr(processing, "http_head", _head(size_mb * MB))
    result = _size_skip(_Cfg(), _Episode())
    assert result.skip_speaker_detection is False, f"{size_mb} MB re-gated speaker detection"
    assert result.media_oversize is True


@pytest.mark.parametrize("provider", ["openai", "gemini", "mistral", "deepgram"])
def test_no_provider_re_enables_the_gate(monkeypatch, provider: str) -> None:
    """Deepgram inherited this cap by being added to a tuple. No provider may re-acquire it."""
    monkeypatch.setattr(processing, "http_head", _head(80 * MB))
    result = _size_skip(_Cfg(transcription_provider=provider), _Episode())
    assert result.skip_speaker_detection is False


def test_the_episode_itself_is_never_skipped_for_size(monkeypatch) -> None:
    """#327's premise: oversize meant the episode would not be transcribed. It is chunked now."""
    monkeypatch.setattr(processing, "http_head", _head(80 * MB))
    assert _size_skip(_Cfg(), _Episode()).skip_episode is False


def test_transcript_urls_do_not_change_the_decision(monkeypatch) -> None:
    """Both branches used to disable detection; neither may now."""
    monkeypatch.setattr(processing, "http_head", _head(80 * MB))
    episode = _Episode(transcript_urls=["https://example.com/t.vtt"])
    assert _size_skip(_Cfg(), episode).skip_speaker_detection is False


def test_under_the_limit_is_unremarkable(monkeypatch) -> None:
    monkeypatch.setattr(processing, "http_head", _head(20 * MB))
    result = _size_skip(_Cfg(), _Episode())
    assert result.skip_speaker_detection is False
    assert result.media_oversize is False


def test_the_size_probe_still_happens_so_operators_learn_chunking_is_coming(monkeypatch) -> None:
    """#557's real purpose survives: report the upload-limit breach, gate nothing."""
    calls: List[Any] = []

    def _spy(*args: Any, **kwargs: Any):
        calls.append(args)
        response = MagicMock()
        response.headers = {"Content-Length": str(80 * MB)}
        return response

    monkeypatch.setattr(processing, "http_head", _spy)
    result = _size_skip(_Cfg(), _Episode())
    assert len(calls) == 1
    assert result.detail is not None and result.detail["media_bytes"] == 80 * MB
