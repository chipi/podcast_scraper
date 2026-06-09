"""Unit tests for the #925 scoped reprocess (``--reprocess-source``) skip override.

Verifies the filter logic in isolation (no pipeline run): an episode whose
existing ``content.transcript_source`` matches ``cfg.reprocess_source`` is forced
through transcription again (so diarization re-runs), overriding ``--skip-existing``;
non-matching episodes are unaffected, and with the flag off behaviour is unchanged.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor as ep, metadata_generation
from tests.conftest import create_test_config, create_test_episode

pytestmark = pytest.mark.unit

_MEDIA_URL = "https://example.com/audio.mp3"
_MEDIA_TYPE = "audio/mpeg"


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


def test_existing_source_reads_yaml_metadata(tmp_path, monkeypatch):
    # #925 MED2: YAML corpora must not silently no-op (json.load would raise).
    meta = tmp_path / "ep.metadata.yaml"
    meta.write_text("content:\n  transcript_source: whisper_transcription\n")
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


# --- download_media_for_transcription override (the real Whisper path, #925) ----
# A whisper_transcription episode has no transcript URL, so it routes through
# download_media_for_transcription (NOT _check_existing_transcript). These drive
# that function directly -- the path the original #925 override missed.


def _whisper_episode_with_existing_transcript(tmp_path, *, transcript_source, reprocess_source):
    temp_dir = os.path.join(str(tmp_path), ".tmp_media")
    os.makedirs(temp_dir, exist_ok=True)
    ep_title = "Episode 1"
    ep_title_safe = filesystem.sanitize_filename(ep_title)
    os.makedirs(os.path.join(str(tmp_path), filesystem.TRANSCRIPTS_SUBDIR), exist_ok=True)
    final_path = filesystem.build_whisper_output_path(1, ep_title_safe, None, str(tmp_path))
    with open(final_path, "wb") as fh:
        fh.write(b"existing transcript")  # so skip_existing would normally skip

    cfg = create_test_config(
        output_dir=str(tmp_path),
        skip_existing=True,
        transcribe_missing=True,
        dry_run=True,  # short-circuit the real download; returns a job if not skipped
        reprocess_source=reprocess_source,
    )
    item = ET.Element("item")
    ET.SubElement(item, "enclosure", attrib={"url": _MEDIA_URL, "type": _MEDIA_TYPE})
    episode = create_test_episode(
        idx=1,
        title=ep_title,
        title_safe=ep_title_safe,
        item=item,
        transcript_urls=[],  # no transcript URL -> Whisper path
        media_url=_MEDIA_URL,
        media_type=_MEDIA_TYPE,
    )
    # Existing metadata carrying the source the filter keys off.
    meta_path = metadata_generation._determine_metadata_path(episode, str(tmp_path), None, cfg)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"content": {"transcript_source": transcript_source}}, fh)
    return cfg, episode, temp_dir


def test_whisper_episode_skipped_without_reprocess_source(tmp_path):
    cfg, episode, temp_dir = _whisper_episode_with_existing_transcript(
        tmp_path, transcript_source="whisper_transcription", reprocess_source=None
    )
    job = ep.download_media_for_transcription(episode, cfg, temp_dir, str(tmp_path), None)
    assert job is None, "no --reprocess-source: existing transcript should be skipped"


def test_whisper_episode_forced_when_source_matches(tmp_path):
    # THE regression guard: a matching whisper episode must NOT be skipped, so
    # download/transcribe (and thus re-diarization) is scheduled.
    cfg, episode, temp_dir = _whisper_episode_with_existing_transcript(
        tmp_path,
        transcript_source="whisper_transcription",
        reprocess_source="whisper_transcription",
    )
    job = ep.download_media_for_transcription(episode, cfg, temp_dir, str(tmp_path), None)
    assert job is not None, "matching --reprocess-source must force re-transcription (not skip)"


def test_direct_download_episode_not_forced_by_whisper_filter(tmp_path):
    # A direct_download episode is left untouched when re-diarizing the whisper set.
    cfg, episode, temp_dir = _whisper_episode_with_existing_transcript(
        tmp_path,
        transcript_source="direct_download",
        reprocess_source="whisper_transcription",
    )
    job = ep.download_media_for_transcription(episode, cfg, temp_dir, str(tmp_path), None)
    assert job is None, "source mismatch must not force reprocess"
