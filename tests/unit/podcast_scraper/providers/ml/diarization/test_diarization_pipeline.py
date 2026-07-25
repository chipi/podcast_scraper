"""Unit tests for diarization pipeline integration helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

pytestmark = pytest.mark.unit


# A real pre-roll ad read: enough distinct _AD_PATTERNS hits (>= PREROLL_THRESHOLD) that the
# ad-region detector actually fires on it, rather than a string that only "looks like" an ad.
_PREROLL_AD = (
    "This episode is sponsored by Acme. This show is brought to you by our sponsors at Acme. "
    "Go to acme dot com slash deal to support the show. "
)


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_preroll_ad_narrator_is_not_crowned_host(mock_create_provider) -> None:
    """An episode that opens with a sponsor read must not make the ad narrator the host (#1169).

    The host rule is "the voice that opens the episode". On real, ad-laden feeds the opening voice
    is the *ad narrator*, so without ad intervals the sponsor voice becomes the host and the real
    host's name gets pinned onto it. The roster has always accepted ``ad_intervals`` — the pipeline
    simply never passed any, leaving the guard dead in production.
    """
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=20.0, speaker="SPEAKER_09"),  # the ad read
            DiarizationSegment(start=20.0, end=120.0, speaker="SPEAKER_00"),  # the real host
            DiarizationSegment(start=120.0, end=400.0, speaker="SPEAKER_01"),  # the guest
        ],
        num_speakers=3,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    # Ad detection only runs on a real-length transcript (MIN_TRANSCRIPT_CHARS), so the body has
    # to be episode-sized — a toy transcript would skip detection and pass for the wrong reason.
    host_turn = "Welcome back to the show. I'm Katie Martin. " + ("Let's dig into markets. " * 60)
    guest_turn = "Thanks for having me, glad to be here. " + ("Here is my long answer. " * 60)
    segments = [
        {"start": 0.0, "end": 20.0, "text": _PREROLL_AD},
        {"start": 20.0, "end": 120.0, "text": host_turn},
        {"start": 120.0, "end": 400.0, "text": guest_turn},
    ]
    enriched = apply_diarization_to_result(
        {"text": "".join(str(s["text"]) for s in segments), "segments": segments},
        "/tmp/audio.wav",
        cfg,
        ["Guest"],
    )

    labels = {seg["speaker"]: seg["speaker_label"] for seg in enriched["segments"]}
    # The host name from the self-intro lands on the voice that speaks *after* the ad...
    assert labels["SPEAKER_00"] == "Katie Martin"
    # ...and never on the sponsor voice.
    assert labels["SPEAKER_09"] != "Katie Martin"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_enriches_segments(mock_create_provider) -> None:
    mock_provider = MagicMock()
    # SPEAKER_00 owns the intro -> host (kept raw); SPEAKER_01 talks most -> guest (#876).
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=400.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    result = {
        "text": "hello world",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "hello"},
            {"start": 60.0, "end": 400.0, "text": "world"},
        ],
    }

    # detected_speaker_names is guest-only; the guest name must land on the guest, not the host.
    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Guest"])

    assert enriched["segments"][0]["speaker"] == "SPEAKER_00"
    assert enriched["segments"][0]["speaker_label"] == "SPEAKER_00", "host kept raw"
    assert enriched["segments"][1]["speaker_label"] == "Guest", "guest named"
    # enrichment sidecar + role hint (harden #1170): the diagnostics dict is attached,
    # and an unnamed host segment carries speaker_role="host" (renders as "Host", not SPEAKER).
    assert "speaker_diagnostics" in enriched, "speaker_diagnostics sidecar attached"
    assert enriched["segments"][0]["speaker_role"] == "host", "unnamed host tagged for display"
    assert "speaker_role" not in enriched["segments"][1], "named guest needs no role hint"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_names_host_from_transcript_self_intro(mock_create_provider) -> None:
    """End-to-end (#876): host self-intro in the transcript names the diarized host voice."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),  # intro -> host
            DiarizationSegment(start=60.0, end=400.0, speaker="SPEAKER_01"),  # guest
        ],
        num_speakers=2,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    result = {
        "text": "Hello and welcome. I'm Patrick O'Shaughnessy. My guest is Brian Chesky.",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": "Hello and welcome. I'm Patrick O'Shaughnessy."},
            {"start": 60.0, "end": 400.0, "text": "Thanks for having me."},
        ],
    }

    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Brian Chesky"])

    assert enriched["segments"][0]["speaker_label"] == "Patrick O'Shaughnessy", "host named"
    assert enriched["segments"][1]["speaker_label"] == "Brian Chesky", "guest named"


def _two_voice_result(host_intro: str):
    return {
        "text": f"Hello and welcome. {host_intro} My guest is Brian Chesky.",
        "segments": [
            {"start": 0.0, "end": 60.0, "text": f"Hello and welcome. {host_intro}"},
            {"start": 60.0, "end": 400.0, "text": "Thanks for having me."},
        ],
    }


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_feed_hosts_canonicalize_garbled_surname(mock_create_provider) -> None:
    """feed_hosts anchors the roster so an ASR-garbled self-intro surname canonicalizes to the
    feed's spelling: "I'm Kevin Russo" -> "Kevin Roose" (the reprocess canonicalization path)."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=400.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )

    enriched = apply_diarization_to_result(
        _two_voice_result("I'm Kevin Russo."),
        "/tmp/audio.wav",
        cfg,
        ["Brian Chesky"],
        feed_hosts=["Kevin Roose"],
    )
    assert enriched["segments"][0]["speaker_label"] == "Kevin Roose", "canonicalized"

    # Without the anchor the garbled surname survives — proves feed_hosts is load-bearing.
    no_anchor = apply_diarization_to_result(
        _two_voice_result("I'm Kevin Russo."),
        "/tmp/audio.wav",
        cfg,
        ["Brian Chesky"],
    )
    assert no_anchor["segments"][0]["speaker_label"] == "Kevin Russo", "no anchor"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_feed_hosts_merges_with_cfg_known_hosts(mock_create_provider) -> None:
    """feed_hosts is merged with cfg.known_hosts (both anchor the roster, neither shadows)."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00")],
        num_speakers=1,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
        known_hosts=["Someone Else"],
    )
    enriched = apply_diarization_to_result(
        {
            "text": "Hello and welcome. I'm Casey Noon.",
            "segments": [{"start": 0.0, "end": 60.0, "text": "Hello and welcome. I'm Casey Noon."}],
        },
        "/tmp/audio.wav",
        cfg,
        None,
        feed_hosts=["Casey Newton"],
    )
    # cfg.known_hosts did not shadow feed_hosts: the feed-stated host still anchored the match.
    assert enriched["segments"][0]["speaker_label"] == "Casey Newton"


@patch("podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider")
def test_apply_diarization_degrades_when_no_turns(mock_create_provider) -> None:
    """Zero diarization turns → segments returned unlabelled so the caller degrades (A3)."""
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[],
        num_speakers=0,
        model_name="test",
    )
    mock_create_provider.return_value = mock_provider

    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
    )
    result = {
        "text": "hello",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}],
    }

    enriched = apply_diarization_to_result(result, "/tmp/audio.wav", cfg, ["Host"])

    # No phantom SPEAKER_00: segments carry no speaker_label, so the screenplay
    # gate falls back to gap-based formatting.
    assert "speaker_label" not in enriched["segments"][0]
    assert "speaker" not in enriched["segments"][0]


def test_recurring_text_index_rebuilds_as_more_transcripts_land(tmp_path) -> None:
    # D3: the mid-roll-ad recurring-script index must rebuild once a batch's later episodes have
    # written more transcripts. Keying on output_dir alone froze it at the first episode's
    # near-empty state, so the rule ran blind for the whole first pass of a fresh feed.
    from podcast_scraper import config as _config
    from podcast_scraper.providers.ml.diarization import pipeline as _pipeline

    _pipeline._recurring_cache.clear()
    tdir = tmp_path / "feeds" / "f" / "run_1" / "transcripts"
    tdir.mkdir(parents=True)
    cfg = _config.Config(
        rss="https://example.com/f.xml",
        transcription_provider="whisper",
        output_dir=str(tmp_path),
    )
    passage = "and now a quick word from our longtime friends at the acme house of widgets"

    def write(n: int) -> None:
        (tdir / f"{n}.txt").write_text(passage + f". Unique content number {n}. " * 20)

    write(1)
    write(2)
    assert _pipeline._feed_recurring_text(cfg) == set()  # <3 transcripts -> abstain
    write(3)
    write(4)
    assert _pipeline._feed_recurring_text(cfg)  # >=3 sharing a passage -> rebuilt, non-empty
