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
    _apply_native_speaker_roster,
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


class TestNativeSpeakerRoster:
    def test_native_segments_get_roles_from_the_roster(self) -> None:
        # Deepgram-style: integer speakers (incl. 0), no names, no roles. The host self-introduces.
        result = {
            "text": "Welcome to the show. I'm Kevin Roose. Thanks for having me, I am the guest.",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "speaker": 0,
                    "text": "Welcome to the show. I'm Kevin Roose.",
                },
                {
                    "start": 3.0,
                    "end": 6.0,
                    "speaker": 1,
                    "text": "Thanks for having me, I am the guest.",
                },
            ],
        }
        out = _apply_native_speaker_roster(result, _cfg(), _job())
        segs = out["segments"]
        # The roster ran: each segment now carries a resolved label + a role.
        by_label = {s.get("speaker_label"): s.get("speaker_role") for s in segs}
        assert "Kevin Roose" in by_label
        assert by_label["Kevin Roose"] == "host"
        # The non-host voice is NOT a host (it is guest or unknown, never defaulted to host).
        non_host_roles = [r for lbl, r in by_label.items() if lbl != "Kevin Roose"]
        assert "host" not in non_host_roles

    def test_noop_on_plain_transcript(self) -> None:
        result = {"text": "hi", "segments": [{"start": 0, "end": 1, "text": "hi"}]}
        out = _apply_native_speaker_roster(result, _cfg(), _job())
        assert out is result  # unchanged — no native speaker ids
