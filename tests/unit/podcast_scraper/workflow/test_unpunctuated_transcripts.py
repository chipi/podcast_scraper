"""Unpunctuated transcripts are never passed on silently (#2284).

A publisher transcript with no punctuation is refused so the episode is transcribed by us instead
(the same fall-through as a transcript that separates no speaker turns), and an ASR transcript
that stays unpunctuated after the prompted retry is KEPT but recorded: ``punctuation`` in the
``.asr.json`` provenance and an ``asr_unpunctuated`` flag on the processing manifest.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper import config as config_module
from podcast_scraper.models.entities import Episode
from podcast_scraper.utils import correlation
from podcast_scraper.workflow import episode_processor as epx, processing_manifest as pm

pytestmark = pytest.mark.unit

LOWER = "so the ports moved north over a decade and the trading families followed the water"
UNPUNCTUATED_VTT = "WEBVTT\n\n" + "\n".join(
    f"00:00:{i:02d}.000 --> 00:00:{i + 1:02d}.000\n<v Maya>{LOWER}</v>\n" for i in range(25)
)
PUNCTUATED_TEXT = (
    "So the ports moved north over a decade. Why? The delta silted up, and trade moved. " * 25
)


def _episode() -> Episode:
    return Episode(
        idx=1,
        title="Ports",
        title_safe="ports",
        item=ET.Element("item"),
        transcript_urls=[("http://feed.example/t.vtt", "text/vtt")],
    )


def _download(tmp_path, monkeypatch, body: str, ctype: str, url: str):
    monkeypatch.setattr(epx, "_fetch_transcript_content", lambda u, cfg: (body.encode(), ctype))
    cfg = config_module.Config(output_dir=str(tmp_path))
    return epx.process_transcript_download(
        _episode(), url, ctype, cfg, str(tmp_path), None, detected_speaker_names=["Maya"]
    )


def test_an_unpunctuated_publisher_vtt_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
) -> None:
    ok, rel_path, source, nbytes = _download(
        tmp_path, monkeypatch, UNPUNCTUATED_VTT, "text/vtt", "http://feed.example/t.vtt"
    )
    assert (ok, rel_path, source) == (False, None, epx.TRANSCRIPT_UNPUNCTUATED)
    assert nbytes > 0, "the fetched bytes are still reported"
    assert not list(Path(tmp_path).rglob("*.txt"))


def test_an_unpunctuated_plain_text_transcript_is_refused(tmp_path, monkeypatch) -> None:
    ok, _, source, _ = _download(
        tmp_path, monkeypatch, (LOWER + " ") * 25, "text/plain", "http://feed.example/t.txt"
    )
    assert not ok and source == epx.TRANSCRIPT_UNPUNCTUATED


def test_a_punctuated_plain_text_transcript_is_accepted(tmp_path, monkeypatch) -> None:
    ok, rel_path, source, _ = _download(
        tmp_path, monkeypatch, PUNCTUATED_TEXT, "text/plain", "http://feed.example/t.txt"
    )
    assert ok and rel_path and source == "direct_download"


def test_the_refusal_falls_through_to_transcription_like_the_speaker_refusal() -> None:
    assert epx._REFUSED_TRANSCRIPT_REASONS[epx.TRANSCRIPT_UNPUNCTUATED] == "unpunctuated"
    assert epx.TRANSCRIPT_LACKS_SPEAKERS in epx._REFUSED_TRANSCRIPT_REASONS


@pytest.fixture
def _no_run_id():
    saved = correlation.get_run_id()
    correlation.set_run_id(None)
    yield
    correlation.set_run_id(saved)


def _manifest_inputs(tmp_path):
    os.makedirs(tmp_path / "transcripts")
    rel = "transcripts/0001 - Ports.txt"
    (tmp_path / rel).write_text("so the ports moved north")
    cfg = SimpleNamespace(
        rss_url="https://example.com/feed.xml",
        run_id="run-1",
        transcription_provider="tailnet_dgx_whisper",
        dgx_whisper_model="turbo",
        diarization_provider="pyannote",
        transcription_speech_coverage_min=0.85,
    )
    job = epx.TranscriptionJob(
        idx=1, ep_title="Ports", ep_title_safe="Ports", temp_media="", episode=None
    )
    return rel, cfg, job


def test_a_kept_unpunctuated_transcript_is_flagged_in_the_manifest_and_provenance(
    tmp_path, _no_run_id
) -> None:
    rel, cfg, job = _manifest_inputs(tmp_path)
    outcome = {"mode": "on_retry", "retried": True, "unpunctuated": True}
    result = {"speech_audio_ratio": 0.9, "model_used": "turbo", "punctuation": outcome}

    epx._write_processing_manifest(result, cfg, job, rel, str(tmp_path))
    epx._save_asr_provenance_file(result, cfg, rel, str(tmp_path))

    manifest = json.load(open(pm.manifest_path(str(tmp_path), rel)))
    assert "asr_unpunctuated" in manifest["quality_flags"]
    asr = json.load(open(tmp_path / "transcripts" / "0001 - Ports.asr.json"))
    assert asr["punctuation"] == outcome


def test_a_restored_transcript_is_not_flagged(tmp_path, _no_run_id) -> None:
    rel, cfg, job = _manifest_inputs(tmp_path)
    result = {
        "speech_audio_ratio": 0.9,
        "punctuation": {"mode": "on_retry", "retried": True, "unpunctuated": False},
    }
    epx._write_processing_manifest(result, cfg, job, rel, str(tmp_path))
    manifest = json.load(open(pm.manifest_path(str(tmp_path), rel)))
    assert "asr_unpunctuated" not in manifest.get("quality_flags", [])
