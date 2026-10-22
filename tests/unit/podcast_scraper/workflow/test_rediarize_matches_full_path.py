"""rediarize_only diarizes what a new episode's full run diarizes, and names on the same inputs.

The full path (``transcribe_media_to_text``) diarizes the PREPROCESSED audio
(``_preprocess_audio_if_needed``: the profile's audio preset, e.g. speech_optimal_v1 at 16 kHz,
32 kbps, loudness -16) and gives the naming step the episode title and description (ADR-137).
rediarize_only diarized the raw download and passed neither, so a re-diarized episode was not
comparable with one the pipeline had processed: different audio into pyannote, a weaker naming
prompt. These tests pin the two inputs to the full path's.
"""

from __future__ import annotations

import json
import types

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as diar_pipeline
from podcast_scraper.workflow import episode_processor as ep


class _Captured(Exception):
    pass


def _setup(tmp_path, monkeypatch):
    raw = tmp_path / "raw.mp3"
    raw.write_bytes(b"raw audio")
    pre = tmp_path / "preprocessed.mp3"
    pre.write_bytes(b"preprocessed audio")
    txt = tmp_path / "transcripts" / "0001 - ep.txt"
    txt.parent.mkdir()
    txt.write_text("hello there", encoding="utf-8")
    txt.with_name("0001 - ep.segments.json").write_text(
        json.dumps([{"start": 0.0, "end": 2.0, "text": "hello there"}]), encoding="utf-8"
    )
    job = types.SimpleNamespace(
        idx=1,
        temp_media=str(raw),
        ep_title="Episode title",
        episode=types.SimpleNamespace(description="Episode description"),
        detected_speaker_names=[],
        metadata_named=[],
    )
    preprocess_calls = []

    def fake_preprocess(job_, cfg_, media, metrics):
        preprocess_calls.append(media)
        return str(pre)

    captured = {}

    def fake_apply(result, audio_path, cfg_, names, **kwargs):
        captured["audio_path"] = audio_path
        captured.update(kwargs)
        raise _Captured()

    monkeypatch.setattr(ep, "_preprocess_audio_if_needed", fake_preprocess)
    monkeypatch.setattr(ep, "_existing_transcript_for", lambda *a, **k: txt)
    monkeypatch.setattr(ep, "_feed_hosts_from_sibling_metadata", lambda *a, **k: [])
    monkeypatch.setattr(ep, "_detection_report_of", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_operator_renames", lambda *a, **k: None)
    monkeypatch.setattr(diar_pipeline, "apply_diarization_to_result", fake_apply)
    return job, raw, pre, preprocess_calls, captured


def _run(job, tmp_path):
    with pytest.raises(_Captured):
        ep._rediarize_existing_transcript(
            job, types.SimpleNamespace(), None, str(tmp_path), None, None
        )


def test_rediarize_diarizes_the_preprocessed_audio_like_a_full_run(tmp_path, monkeypatch):
    job, raw, pre, preprocess_calls, captured = _setup(tmp_path, monkeypatch)
    _run(job, tmp_path)
    assert preprocess_calls == [str(raw)]
    assert captured["audio_path"] == str(pre)


def test_rediarize_names_with_the_episode_title_and_description(tmp_path, monkeypatch):
    job, _, _, _, captured = _setup(tmp_path, monkeypatch)
    _run(job, tmp_path)
    assert captured["episode_title"] == "Episode title"
    assert captured["episode_description"] == "Episode description"
