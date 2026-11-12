"""Tests for pipeline_stage=relabel_only (re-resolve speaker NAMES on a finished corpus's
frozen diarization — no audio, no re-ASR, no re-diarize).

Guards the two bugs found before the v2.1 reprocess scaled:
  1. Relabel must read the identity from ``speaker_label`` (a finished corpus stores ``speaker``
     as None), not the raw ``speaker`` field — else it silently no-ops on every real episode.
  2. The frozen labels are v2's *resolved names* ("Ada Lowell"), so relabel anonymizes the
     clustering before re-resolving, and canonicalizes ASR-garbled host surnames against the
     feed-stated hosts ("Kevin Russo" -> "Kevin Roose").
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode
from podcast_scraper.workflow import episode_processor as ep_mod
from podcast_scraper.workflow.episode_processor import download_media_for_transcription

pytestmark = pytest.mark.unit

_FEED_DESC = "Each week, journalists Kevin Roose and Casey Newton explore the world of tech."


def _cfg(
    pipeline_stage: Literal[
        "full",
        "audio_only",
        "enrich_only",
        "download_only",
        "relabel_only",
        "rediarize_only",
        "retranscript_only",
        "rederive_only",
        "repunctuate_only",
    ] = "full",
) -> config.Config:
    # The clip-calling stage needs the provider that has a clip call; the rest need none.
    provider = (
        {
            "transcription_provider": "tailnet_dgx_whisper",
            "transcription_fallback_provider": "openai",
            "openai_api_key": "sk-test-not-a-key",
            "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
        }
        if pipeline_stage == "repunctuate_only"
        else {"transcription_provider": "whisper"}
    )
    return config.Config(
        rss="https://example.com/feed.xml",
        **provider,  # type: ignore[arg-type]
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
        speaker_resolution_llm=False,  # deterministic + airgapped (no LLM in tests/CI)
        pipeline_stage=pipeline_stage,
        feed_declared_language="en",  # #2283: an English feed
    )


# --- _feed_hosts_from_sibling_metadata (unit) ---


# --- _relabel_existing_transcript (integration: files -> relabel -> corrected files) ---


# --- dispatch plumbing (coercion -> no-download job -> transcribe dispatch -> relabel) ---


def test_download_media_returns_no_download_job_for_relabel_only(tmp_path: Path) -> None:
    """relabel_only must NOT download audio: the job is created with an empty temp_media so the
    transcribe stage reaches the relabel branch instead of transcribing."""
    episode = Episode(
        idx=1,
        title="Ep",
        title_safe="Ep",
        item=ET.Element("item"),
        transcript_urls=[],
        media_url="https://example.com/ep1.mp3",
        media_type="audio/mpeg",
    )
    job = download_media_for_transcription(
        episode,
        _cfg("relabel_only"),
        str(tmp_path / "tmp"),
        str(tmp_path / "out"),
        "20260101-000000_t",
        detected_speaker_names=["Guest"],
        metadata_named=["Someone"],
    )
    assert job is not None
    assert job.temp_media == ""  # no download happened
    assert job.idx == 1
    assert job.detected_speaker_names == ["Guest"]


# --- the speaker record follows the relabel (#2075) ---


# --- rediarize_only (v2.2): fresh diarization aligned to the existing ASR transcript ---


# --- the cleaned transcript follows the relabel (#2075) ---


@pytest.mark.parametrize(
    "stage",
    ["relabel_only", "retranscript_only", "rediarize_only", "rederive_only", "repunctuate_only"],
)
def test_a_reprocess_stage_never_redownloads_a_publisher_transcript(
    tmp_path: Path, monkeypatch, stage
) -> None:
    """#2075, found verifying retranscript_only on an Odd Lots episode: the publisher-transcript
    branch ran before any stage check, so a feed that ships its own transcript was re-ingested into
    a NEW run and the repair never ran — exit 0, stored transcript untouched."""
    import queue

    def _must_not_download(*_a, **_k):
        raise AssertionError("a reprocess stage re-downloaded the publisher transcript")

    monkeypatch.setattr(ep_mod, "process_transcript_download", _must_not_download)
    # This test is about the PUBLISHER-TRANSCRIPT branch only. `rediarize_only` and
    # `retranscript_only` legitimately fetch the AUDIO afterwards (they re-diarize / re-ASR it),
    # which under the suite's `--disable-socket` reaches a real connect to example.com. Stub the
    # media fetch so the failure surface stays the branch under test.
    monkeypatch.setattr(ep_mod, "download_media_for_transcription", lambda *_a, **_k: None)
    episode = Episode(
        idx=1,
        title="Ep",
        title_safe="Ep",
        item=ET.Element("item"),
        transcript_urls=[("https://example.com/ep1.srt", "application/x-subrip")],
        media_url="https://example.com/ep1.mp3",
        media_type="audio/mpeg",
    )
    jobs: "queue.Queue" = queue.Queue()
    ep_mod.process_episode_download(
        episode,
        _cfg(stage),  # type: ignore[arg-type]
        str(tmp_path / "tmp"),
        str(tmp_path / "out"),
        "20260101-000000_t",
        jobs,
        None,
    )


# --- rederive_only routing (#2075): re-derive from the transcript already on disk ---


def _rederive_episode() -> Episode:
    return Episode(
        idx=1,
        title="Ep",
        title_safe="Ep",
        item=ET.Element("item"),
        transcript_urls=[],
        media_url="https://example.com/ep1.mp3",
        media_type="audio/mpeg",
    )


def test_rederive_only_reuses_the_on_disk_transcript(tmp_path: Path, monkeypatch) -> None:
    """rederive_only was a silent no-op: it coerces ``transcribe_missing=false``, and the only
    other exit from ``process_episode_download`` was the ``transcribe_missing and temp_dir`` gate,
    so the function returned (False, None, None, 0), nothing was ever queued, and the run exited 0
    having re-derived nothing. The stage must resolve the stored transcript HERE."""
    import queue

    from podcast_scraper.utils import filesystem

    out_dir = tmp_path / "out"
    run_suffix = "20260101-000000_t"
    stored = Path(filesystem.build_whisper_output_path(1, "Ep", run_suffix, str(out_dir)))
    stored.parent.mkdir(parents=True)
    stored.write_text("SPEAKER_00: Welcome to the show.\n", encoding="utf-8")

    def _must_not_download_media(*_a, **_k):
        raise AssertionError("rederive_only downloaded media for transcription")

    monkeypatch.setattr(ep_mod, "download_media_for_transcription", _must_not_download_media)

    jobs: "queue.Queue" = queue.Queue()
    ok, path, source, downloaded = ep_mod.process_episode_download(
        _rederive_episode(),
        _cfg("rederive_only"),
        str(tmp_path / "tmp"),
        str(out_dir),
        run_suffix,
        jobs,
        None,
    )

    assert ok is True  # the stage now reports the episode as processed
    # ...from the transcript already on disk, named RELATIVE to the run's output dir like every
    # transcript path the pipeline hands on (an absolute one was written into metadata and GI/KG).
    assert path is not None and not Path(path).is_absolute()
    assert (out_dir / path).resolve() == stored.resolve()
    # Provenance is READ from the episode's own metadata, never assumed, so a direct-download feed
    # is not relabelled `whisper_transcription`. This fixture stores no metadata, and the
    # documented fallback for unreadable metadata is exactly one value — accepting either would
    # let a wrong provenance label through on a reused transcript.
    assert source == "whisper_transcription"
    assert downloaded == 0  # nothing was fetched
    assert jobs.empty()  # and no ASR job was queued


def test_rederive_only_reports_failure_when_nothing_is_on_disk(tmp_path: Path) -> None:
    """The other half of the silent-success class: with no stored transcript there is nothing to
    re-derive, and the episode must NOT be counted as processed."""
    import queue

    jobs: "queue.Queue" = queue.Queue()
    ok, path, source, downloaded = ep_mod.process_episode_download(
        _rederive_episode(),
        _cfg("rederive_only"),
        str(tmp_path / "tmp"),
        str(tmp_path / "out"),
        "20260101-000000_t",
        jobs,
        None,
    )

    assert (ok, path, source, downloaded) == (False, None, None, 0)
    assert jobs.empty()


# --- the rediarize audio guard: the downloaded file must be the one the transcript came from ---


@pytest.mark.parametrize(
    "downloaded, ingest, audio_s, end, refused",
    [
        (
            9159716,
            6909843,
            None,
            416.0,
            True,
        ),  # the publisher's file changed (DGX block, 2026-10-23)
        (62138364, 62138364, 3865.0, 3865.0, False),  # same file
        (None, None, 3000.0, 2995.0, False),  # no recorded size; a few seconds of outro
        (None, None, 365.0, 416.0, True),  # the transcript runs past the audio's end
        (None, None, 4000.0, 3500.0, True),  # 500 s of audio after the last speech (12.5%)
        (None, None, 600.0, 470.0, True),  # 130 s and 22% after the last speech
        (None, None, 7200.0, 7050.0, False),  # 150 s of a 2 h file (2%): an outro
        (None, None, None, 416.0, False),  # nothing to compare: the run proceeds as before
    ],
)
def test_rediarize_audio_mismatch(downloaded, ingest, audio_s, end, refused) -> None:
    from podcast_scraper.workflow.episode_processor import rediarize_audio_mismatch

    assert (rediarize_audio_mismatch(downloaded, ingest, audio_s, end) is not None) is refused
