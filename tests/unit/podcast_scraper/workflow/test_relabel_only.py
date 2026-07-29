"""Tests for pipeline_stage=relabel_only (re-resolve speaker NAMES on a finished corpus's
frozen diarization — no audio, no re-ASR, no re-diarize).

Guards the two bugs found before the v2.1 reprocess scaled:
  1. Relabel must read the identity from ``speaker_label`` (a finished corpus stores ``speaker``
     as None), not the raw ``speaker`` field — else it silently no-ops on every real episode.
  2. The frozen labels are v2's *resolved names* ("Amy Lawrence"), so relabel anonymizes the
     clustering before re-resolving, and canonicalizes ASR-garbled host surnames against the
     feed-stated hosts ("Kevin Russo" -> "Kevin Roose").
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode, TranscriptionJob
from podcast_scraper.workflow import episode_processor as ep_mod
from podcast_scraper.workflow.episode_processor import (
    _feed_hosts_from_sibling_metadata,
    _relabel_existing_transcript,
    download_media_for_transcription,
    transcribe_media_to_text,
)

pytestmark = pytest.mark.unit

_FEED_DESC = "Each week, journalists Kevin Roose and Casey Newton explore the world of tech."


def _write_corpus(
    base: Path,
    run_tag: str,
    *,
    seg_labels,
    texts,
    feed_desc=_FEED_DESC,
    with_metadata: bool = True,
):
    """Lay down a finished-corpus run dir: transcript + real-schema segments + feed metadata."""
    old_run = base / f"run_{run_tag}"
    tdir = old_run / "transcripts"
    mdir = old_run / "metadata"
    tdir.mkdir(parents=True)
    mdir.mkdir(parents=True)
    stem = f"0001 - Ep_{run_tag}"
    # Real finished-corpus schema: ``speaker`` is None; the identity lives in ``speaker_label``.
    t = 0.0
    segs = []
    for lbl, txt in zip(seg_labels, texts):
        segs.append(
            {"start": t, "end": t + 60.0, "speaker": None, "speaker_label": lbl, "text": txt}
        )
        t += 60.0
    (tdir / f"{stem}.txt").write_text(
        "\n".join(f"{lbl}: {txt}" for lbl, txt in zip(seg_labels, texts)), encoding="utf-8"
    )
    (tdir / f"{stem}.segments.json").write_text(json.dumps(segs), encoding="utf-8")
    if with_metadata:
        (mdir / f"{stem}.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {
                        "title": "Hard Fork",
                        "description": feed_desc,
                        "authors": ["The New York Times"],
                    }
                }
            ),
            encoding="utf-8",
        )
    return old_run, stem


def _cfg(
    pipeline_stage: Literal[
        "full", "audio_only", "enrich_only", "download_only", "relabel_only", "rediarize_only"
    ] = "full",
) -> config.Config:
    return config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        screenplay=True,
        hf_token="hf-test",
        speaker_resolution_llm=False,  # deterministic + airgapped (no LLM in tests/CI)
        pipeline_stage=pipeline_stage,
    )


def _job() -> TranscriptionJob:
    return TranscriptionJob(
        idx=1,
        ep_title="Ep",
        ep_title_safe="Ep",
        temp_media="",
        detected_speaker_names=None,
        metadata_named=None,
        episode=None,
    )


# --- _feed_hosts_from_sibling_metadata (unit) ---


def test_feed_hosts_from_sibling_metadata_reads_feed_block(tmp_path: Path) -> None:
    old_run, stem = _write_corpus(
        tmp_path / "feed",
        "20260101-000000_t",
        seg_labels=["SPEAKER_00"],
        texts=["hello"],
    )
    txt_path = old_run / "transcripts" / f"{stem}.txt"
    assert _feed_hosts_from_sibling_metadata(txt_path) == ["Casey Newton", "Kevin Roose"]


def test_feed_hosts_from_sibling_metadata_missing_file_returns_empty(tmp_path: Path) -> None:
    old_run, stem = _write_corpus(
        tmp_path / "feed",
        "20260101-000000_t",
        seg_labels=["SPEAKER_00"],
        texts=["hello"],
        with_metadata=False,
    )
    txt_path = old_run / "transcripts" / f"{stem}.txt"
    assert _feed_hosts_from_sibling_metadata(txt_path) == []


def test_feed_hosts_from_sibling_metadata_malformed_json_returns_empty(tmp_path: Path) -> None:
    old_run, stem = _write_corpus(
        tmp_path / "feed",
        "20260101-000000_t",
        seg_labels=["SPEAKER_00"],
        texts=["hello"],
    )
    md = old_run / "metadata" / f"{stem}.metadata.json"
    md.write_text("{ not valid json", encoding="utf-8")
    txt_path = old_run / "transcripts" / f"{stem}.txt"
    assert _feed_hosts_from_sibling_metadata(txt_path) == []


# --- _relabel_existing_transcript (integration: files -> relabel -> corrected files) ---


def test_relabel_reads_speaker_label_anonymizes_and_canonicalizes(tmp_path: Path) -> None:
    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    # v2 crowned the ad-reader "Amy Lawrence" on the host voice and left the co-host a raw
    # SPEAKER_NN; both self-intros carry ASR-garbled surnames.
    host_text = "Welcome to Hard Fork. I'm Kevin Russo, tech columnist. " + ("Host turn. " * 60)
    co_text = "I'm Casey Noon from Platformer. " + ("Co-host turn. " * 60)
    old_run, stem = _write_corpus(
        base,
        run_tag,
        seg_labels=["Amy Lawrence", "SPEAKER_07"],
        texts=[host_text, co_text],
    )
    new_run = base / "run_20260102-000000_t"  # the empty new run dir = effective_output_dir
    new_run.mkdir(parents=True)

    ok, rel_path, _ = _relabel_existing_transcript(
        _job(), _cfg(), run_tag, str(new_run), None, None
    )

    # Regression #1: real schema has speaker=None — relabel must read speaker_label, not no-op.
    assert ok is True
    out = (old_run / "transcripts" / f"{stem}.txt").read_text(encoding="utf-8")
    # Regression #2a: v2's wrong name is not inherited (clustering was anonymized before resolving).
    assert "Amy Lawrence" not in out
    # Regression #2b: the garbled host surname canonicalized against the feed-stated host. Assert on
    # the speaker LABEL (colon-suffixed) — relabel fixes attribution, NOT the in-dialogue ASR text,
    # so "Kevin Russo" legitimately survives in the body ("I'm Kevin Russo, tech columnist").
    assert "Kevin Roose:" in out
    assert "Kevin Russo:" not in out


def test_relabel_feeds_episode_title_and_description_to_resolution_like_full(
    tmp_path: Path, monkeypatch
) -> None:
    """relabel_only must hand the roster the SAME episode title/description a FULL run does.

    The structural half of the relabel!=full confound (root-cause #2): FULL passes
    ``episode_title`` / ``episode_description`` into ``apply_diarization_to_result`` (ADR-135 — both
    feed the LLM's host/guest role determination and gate role-only resolution), but relabel_only
    passed neither, so every reprocess resolved on a strictly weaker prompt showing
    "(not provided)". That is a deterministic divergence, independent of LLM sampling. Pin that
    relabel forwards both, mirroring the FULL call site.
    """
    from podcast_scraper.providers.ml.diarization import pipeline as _pipe

    captured: dict = {}

    def _spy(result, media, cfg, detected, **kwargs):
        captured.update(kwargs)
        return result  # a valid result dict (text + segments) for the downstream render

    monkeypatch.setattr(_pipe, "apply_diarization_to_result", _spy)

    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    _write_corpus(
        base,
        run_tag,
        seg_labels=["SPEAKER_00", "SPEAKER_01"],
        texts=["Welcome to the show. " * 40, "Glad to be here. " * 40],
    )
    new_run = base / "run_20260102-000000_t"
    new_run.mkdir(parents=True)

    import xml.etree.ElementTree as ET

    from podcast_scraper.models import Episode

    episode = Episode(
        idx=1,
        title="The Real Episode Title",
        title_safe="Ep",
        item=ET.Element("item"),
        transcript_urls=[],
        description="Kevin and Casey dig into AI agents.",
    )
    job = TranscriptionJob(
        idx=1,
        ep_title="The Real Episode Title",
        ep_title_safe="Ep",
        temp_media="",
        detected_speaker_names=None,
        metadata_named=None,
        episode=episode,
    )

    ok, _, _ = _relabel_existing_transcript(job, _cfg(), run_tag, str(new_run), None, None)

    assert ok is True
    # THE FIX: the episode title now reaches the roster (it was absent, so the LLM saw
    # "(not provided)" where FULL showed the real title).
    assert captured.get("episode_title") == "The Real Episode Title"
    # And the REAL description is forwarded (Episode now carries it; review found it was dead code
    # returning None everywhere), so the two call sites carry identical resolution context.
    assert captured.get("episode_description") == "Kevin and Casey dig into AI agents."


def test_relabel_feed_hosts_falls_back_to_live_when_sibling_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Q3 (advisor): a frozen sibling-metadata host anchor is preferred for a reproducible relabel,
    but when it is MISSING (returns []), fall back to the live job.feed_hosts already computed — an
    empty anchor is the worst state (no ASR-garble canonicalization, no host pool)."""
    from podcast_scraper.providers.ml.diarization import pipeline as _pipe

    captured: dict = {}

    def _spy(result, media, cfg, detected, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(_pipe, "apply_diarization_to_result", _spy)

    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    _write_corpus(  # no sibling metadata.json -> _feed_hosts_from_sibling_metadata returns []
        base,
        run_tag,
        seg_labels=["SPEAKER_00"],
        texts=["welcome to the show " * 30],
        with_metadata=False,
    )
    new_run = base / "run_20260102-000000_t"
    new_run.mkdir(parents=True)

    job = TranscriptionJob(
        idx=1,
        ep_title="Ep",
        ep_title_safe="Ep",
        temp_media="",
        detected_speaker_names=None,
        metadata_named=None,
        episode=None,
    )
    job.feed_hosts = ["Kevin Roose", "Casey Newton"]  # live detection, already computed

    ok, _, _ = _relabel_existing_transcript(job, _cfg(), run_tag, str(new_run), None, None)

    assert ok is True
    assert captured.get("feed_hosts") == ["Kevin Roose", "Casey Newton"]


def test_relabel_feed_hosts_freeze_wins_over_live_when_sibling_present(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Q3: when the sibling metadata NAMES hosts, they win — a relabel of a stored corpus stays
    reproducible and does not track live feed drift, even if live job.feed_hosts differs — AND the
    sibling-vs-live divergence is LOGGED (the visibility half of the Q3 fix)."""
    import logging

    from podcast_scraper.providers.ml.diarization import pipeline as _pipe

    captured: dict = {}

    def _spy(result, media, cfg, detected, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(_pipe, "apply_diarization_to_result", _spy)

    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    _write_corpus(  # default with_metadata=True -> sibling names Casey Newton + Kevin Roose
        base,
        run_tag,
        seg_labels=["SPEAKER_00"],
        texts=["welcome to the show " * 30],
    )
    new_run = base / "run_20260102-000000_t"
    new_run.mkdir(parents=True)

    job = TranscriptionJob(
        idx=1,
        ep_title="Ep",
        ep_title_safe="Ep",
        temp_media="",
        detected_speaker_names=None,
        metadata_named=None,
        episode=None,
    )
    job.feed_hosts = ["Someone Else"]  # live detection drifted; the frozen sibling must win

    with caplog.at_level(logging.INFO):
        ok, _, _ = _relabel_existing_transcript(job, _cfg(), run_tag, str(new_run), None, None)

    assert ok is True
    assert captured.get("feed_hosts") == ["Casey Newton", "Kevin Roose"]  # sibling, not live
    # the divergence is surfaced so live-vs-freeze can later be decided from data
    assert any("feed_hosts divergence" in r.message for r in caplog.records)


def test_relabel_skips_when_no_segments_identity(tmp_path: Path) -> None:
    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    old_run, stem = _write_corpus(base, run_tag, seg_labels=["SPEAKER_00"], texts=["hi"])
    # Blank out every speaker identity → nothing to relabel.
    seg_path = old_run / "transcripts" / f"{stem}.segments.json"
    segs = json.loads(seg_path.read_text(encoding="utf-8"))
    for s in segs:
        s["speaker"] = None
        s["speaker_label"] = None
    seg_path.write_text(json.dumps(segs), encoding="utf-8")
    new_run = base / "run_20260102-000000_t"
    new_run.mkdir(parents=True)

    ok, _, _ = _relabel_existing_transcript(_job(), _cfg(), run_tag, str(new_run), None, None)
    assert ok is False


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


def test_transcribe_media_to_text_dispatches_to_relabel(tmp_path: Path, monkeypatch) -> None:
    """With pipeline_stage=relabel_only, transcribe_media_to_text hands off to the relabel path
    (no transcription/diarization of new audio)."""
    sentinel = (True, "run_x/transcripts/0001 - Ep.txt", 0)
    seen = {}

    def _fake_relabel(job, cfg, run_suffix, out_dir, provider, metrics):
        seen["dispatched"] = True
        return sentinel

    monkeypatch.setattr(ep_mod, "_relabel_existing_transcript", _fake_relabel)

    result = transcribe_media_to_text(
        _job(),
        _cfg("relabel_only"),
        None,
        "20260101-000000_t",
        str(tmp_path),
        None,
        None,
    )
    assert seen.get("dispatched") is True
    assert result == sentinel


# --- rediarize_only (v2.2): fresh diarization aligned to the existing ASR transcript ---


def test_rediarize_reruns_diarization_and_aligns(tmp_path: Path) -> None:
    from unittest.mock import MagicMock, patch

    from podcast_scraper.providers.ml.diarization.base import (
        DiarizationResult,
        DiarizationSegment,
    )
    from podcast_scraper.workflow.episode_processor import _rediarize_existing_transcript

    base = tmp_path / "feed"
    run_tag = "20260101-000000_t"
    host_text = "Welcome to Hard Fork. I'm Kevin Russo. " + ("Host turn. " * 60)
    co_text = "I'm Casey Noon from Platformer. " + ("Co-host turn. " * 60)
    old_run, stem = _write_corpus(
        base, run_tag, seg_labels=["Amy Lawrence", "SPEAKER_07"], texts=[host_text, co_text]
    )
    new_run = base / "run_20260102-000000_t"
    new_run.mkdir(parents=True)
    audio = tmp_path / "ep.mp3"
    audio.write_bytes(b"AUDIO")  # the stage checks the downloaded audio exists
    job = TranscriptionJob(idx=1, ep_title="Ep", ep_title_safe="Ep", temp_media=str(audio))

    # fresh diarization aligned to the fixture's 0-60 / 60-120 ASR segments
    mock_provider = MagicMock()
    mock_provider.diarize.return_value = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=60.0, speaker="SPEAKER_00"),
            DiarizationSegment(start=60.0, end=120.0, speaker="SPEAKER_01"),
        ],
        num_speakers=2,
    )
    with patch(
        "podcast_scraper.providers.ml.diarization.pipeline.create_diarization_provider",
        return_value=mock_provider,
    ):
        ok, _rel, _ = _rediarize_existing_transcript(job, _cfg(), run_tag, str(new_run), None, None)

    assert ok is True
    assert mock_provider.diarize.called  # re-diarized fresh (bypassed the audio-hash cache)
    out = (old_run / "transcripts" / f"{stem}.txt").read_text(encoding="utf-8")
    assert "Amy Lawrence" not in out  # fresh diarization + resolution, v2's name not inherited
    assert "Kevin Roose:" in out  # feed_hosts canonicalized the self-intro on the fresh voices


def test_rediarize_skips_when_no_audio(tmp_path: Path) -> None:
    from podcast_scraper.workflow.episode_processor import _rediarize_existing_transcript

    job = TranscriptionJob(idx=1, ep_title="Ep", ep_title_safe="Ep", temp_media="")  # no audio
    ok, _, _ = _rediarize_existing_transcript(job, _cfg(), "tag", str(tmp_path), None, None)
    assert ok is False


def test_transcribe_dispatches_to_rediarize(tmp_path: Path, monkeypatch) -> None:
    sentinel = (True, "run/transcripts/0001 - Ep.txt", 0)
    seen = {}

    def _fake(job, cfg, run_suffix, out_dir, provider, metrics):
        seen["hit"] = True
        return sentinel

    monkeypatch.setattr(ep_mod, "_rediarize_existing_transcript", _fake)
    result = transcribe_media_to_text(
        _job(), _cfg("rediarize_only"), None, "tag", str(tmp_path), None, None
    )
    assert seen.get("hit") is True
    assert result == sentinel
