"""Platform fix: native-screenplay providers (deepgram/moss, no roster pass) must still get
host/guest roles from the single role authority (the roster), not the provider's positional guess.

These exercise the real roster (deterministic, no infra) through the ``precomputed_diarization``
seam that ``_apply_native_speaker_roster`` uses.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper import config
from podcast_scraper.workflow.episode_processor import (
    _segments_carry_native_speakers,
)

pytestmark = pytest.mark.unit


def _cfg() -> config.Config:
    # The role pass is provider-agnostic; use a keyless transcription provider — what matters is
    # the segments carry native ``speaker`` ids and the roster runs against them.
    return config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=False,  # native mode: provider self-diarized, no local pass
        screenplay=True,
        known_hosts=["Kevin Roose"],
    )


def _job() -> SimpleNamespace:
    return SimpleNamespace(idx=1, detected_speaker_names=[], metadata_named=[], feed_hosts=[])


class TestNativeSpeakerDetection:
    def test_detects_int_speaker_zero(self) -> None:
        # int 0 is a valid native speaker id — truthiness would drop it.
        assert _segments_carry_native_speakers(
            {"segments": [{"start": 0, "end": 1, "text": "a", "speaker": 0}]}
        )

    def test_ignores_plain_transcript(self) -> None:
        # whisper/openai segments never carry ``speaker`` -> the pass is inert.
        assert not _segments_carry_native_speakers(
            {"segments": [{"start": 0, "end": 1, "text": "a"}]}
        )
        assert not _segments_carry_native_speakers({"segments": []})
        assert not _segments_carry_native_speakers("nope")
