"""Unit tests for the #925 scoped reprocess (``--reprocess-source``) skip override.

Verifies the filter logic in isolation (no pipeline run): an episode whose
existing ``content.transcript_source`` matches ``cfg.reprocess_source`` is forced
through transcription again (so diarization re-runs), overriding ``--skip-existing``;
non-matching episodes are unaffected, and with the flag off behaviour is unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor as ep
from podcast_scraper.workflow import metadata_generation

pytestmark = pytest.mark.unit


def _cfg(**kw):
    base = dict(skip_existing=True, reprocess_source=None, dry_run=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _episode(idx=1, title_safe="Ep One"):
    return SimpleNamespace(idx=idx, title_safe=title_safe)


# --- _episode_existing_transcript_source -----------------------------------


def test_existing_source_reads_content_field(tmp_path, monkeypatch):
    meta = tmp_path / "ep.metadata.json"
    meta.write_text(json.dumps({"content": {"transcript_source": "whisper_transcription"}}))
    monkeypatch.setattr(metadata_generation, "_determine_metadata_path", lambda *a, **k: str(meta))
    assert (
        ep._episode_existing_transcript_source(_episode(), str(tmp_path), None, _cfg())
        == "whisper_transcription"
    )


def test_existing_source_none_when_metadata_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        metadata_generation, "_determine_metadata_path", lambda *a, **k: str(tmp_path / "nope.json")
    )
    assert ep._episode_existing_transcript_source(_episode(), str(tmp_path), None, _cfg()) is None


def test_existing_source_none_when_field_absent(tmp_path, monkeypatch):
    meta = tmp_path / "ep.metadata.json"
    meta.write_text(json.dumps({"content": {}}))
    monkeypatch.setattr(metadata_generation, "_determine_metadata_path", lambda *a, **k: str(meta))
    assert ep._episode_existing_transcript_source(_episode(), str(tmp_path), None, _cfg()) is None


# --- _force_reprocess_for_source -------------------------------------------


def test_force_off_when_flag_unset(monkeypatch):
    monkeypatch.setattr(
        ep, "_episode_existing_transcript_source", lambda *a, **k: "whisper_transcription"
    )
    assert (
        ep._force_reprocess_for_source(_episode(), "/out", None, _cfg(reprocess_source=None))
        is False
    )


def test_force_on_when_source_matches(monkeypatch):
    monkeypatch.setattr(
        ep, "_episode_existing_transcript_source", lambda *a, **k: "whisper_transcription"
    )
    cfg = _cfg(reprocess_source="whisper_transcription")
    assert ep._force_reprocess_for_source(_episode(), "/out", None, cfg) is True


def test_force_off_when_source_differs(monkeypatch):
    # A direct_download episode is left untouched when re-diarizing the whisper set.
    monkeypatch.setattr(
        ep, "_episode_existing_transcript_source", lambda *a, **k: "direct_download"
    )
    cfg = _cfg(reprocess_source="whisper_transcription")
    assert ep._force_reprocess_for_source(_episode(), "/out", None, cfg) is False


# --- _check_existing_transcript override -----------------------------------


def _make_existing_transcript(tmp_path, episode):
    tdir = tmp_path / filesystem.TRANSCRIPTS_SUBDIR
    tdir.mkdir(parents=True, exist_ok=True)
    base = f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - {episode.title_safe}"
    (tdir / f"{base}.txt").write_text("existing transcript")


def test_check_existing_skips_when_not_forced(tmp_path, monkeypatch):
    episode = _episode()
    _make_existing_transcript(tmp_path, episode)
    monkeypatch.setattr(ep, "_force_reprocess_for_source", lambda *a, **k: False)
    # transcript present + skip_existing + not forced -> skip (True)
    assert ep._check_existing_transcript(episode, str(tmp_path), None, _cfg()) is True


def test_check_existing_forces_reprocess_when_source_matches(tmp_path, monkeypatch):
    episode = _episode()
    _make_existing_transcript(tmp_path, episode)
    monkeypatch.setattr(ep, "_force_reprocess_for_source", lambda *a, **k: True)
    # transcript present but forced -> do NOT skip (False) so it re-transcribes/diarizes
    cfg = _cfg(reprocess_source="whisper_transcription")
    assert ep._check_existing_transcript(episode, str(tmp_path), None, cfg) is False
