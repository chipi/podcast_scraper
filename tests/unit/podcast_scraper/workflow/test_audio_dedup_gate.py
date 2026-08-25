# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — the field declares alias="rss"; populate-by-name accepts
# either at runtime (same pragma as test_uninitialized_provider_never_fakes_a_result.py).
"""The content-duplicate gate: a republished episode must not bill ASR twice (#1656).

``skip_existing`` is GUID-keyed, so a feed republishing the same content under a new GUID used
to schedule a second full transcription. The gate fingerprints the downloaded bytes and refuses
to schedule ASR for content the corpus already transcribed — across runs (persistent index) and
within one run (in-process pending claim). Acceptance from the issue: detected BEFORE
transcription is billed, by a fingerprint, not a title heuristic.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode
from podcast_scraper.utils import audio_fingerprint
from podcast_scraper.workflow import episode_processor

pytestmark = [pytest.mark.unit]


AUDIO_BYTES = b"identical enclosure bytes, byte for byte" * 64


def _episode(idx: int, guid: str, title: str, media_url: str) -> Episode:
    item = ET.Element("item")
    guid_el = ET.SubElement(item, "guid")
    guid_el.text = guid
    return Episode(
        idx=idx,
        title=title,
        title_safe=f"ep-{idx}",
        item=item,
        transcript_urls=[],
        media_url=media_url,
        media_type="audio/mpeg",
    )


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch) -> Path:
    audio_fingerprint.reset_pending_for_tests()

    def _fake_download(episode, cfg, temp_media, pipeline_metrics, effective_output_dir):
        Path(temp_media).write_bytes(AUDIO_BYTES)
        return True, len(AUDIO_BYTES), 0.01

    monkeypatch.setattr(episode_processor, "_download_or_reuse_media", _fake_download)
    (tmp_path / "tmp").mkdir()
    return tmp_path


def _cfg(tmp_path: Path, **overrides) -> config.Config:
    return config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(tmp_path),
        **overrides,
    )


def _schedule(cfg: config.Config, tmp_path: Path, episode: Episode):
    return episode_processor.download_media_for_transcription(
        episode,
        cfg,
        temp_dir=str(tmp_path / "tmp"),
        effective_output_dir=str(tmp_path),
        run_suffix=None,
    )


def test_same_run_republish_with_new_guid_never_reaches_asr(corpus: Path):
    """Two identical enclosures in one work queue: the second must not get a job."""
    cfg = _cfg(corpus)
    first = _schedule(cfg, corpus, _episode(1, "guid-original", "The Episode", "https://cdn/a.mp3"))
    assert first is not None, "the first copy is new content and must transcribe"

    second = _schedule(
        cfg, corpus, _episode(2, "guid-republish", "The Episode (rerun)", "https://cdn/b.mp3")
    )
    assert second is None, (
        "identical bytes under a new GUID were scheduled for transcription — "
        "the republish would be billed to ASR a second time (#1656)"
    )


def test_cross_run_republish_is_caught_by_the_persistent_index(corpus: Path):
    """A republish months later (fresh process): the recorded fingerprint must catch it."""
    cfg = _cfg(corpus)
    first = _schedule(cfg, corpus, _episode(1, "guid-original", "The Episode", "https://cdn/a.mp3"))
    assert first is not None
    # The first copy's transcript was saved; the pipeline records its fingerprint.
    episode_processor._register_audio_fingerprint(first, cfg, str(corpus), "transcripts/0001.txt")
    # Fresh process: the in-memory pending set is gone, only the on-disk index remains.
    audio_fingerprint.reset_pending_for_tests()

    second = _schedule(
        cfg, corpus, _episode(7, "guid-bestof", "Best of: The Episode", "https://cdn/z.mp3")
    )
    assert second is None, "the persistent fingerprint index missed a cross-run republish"


def test_a_retry_of_the_same_episode_is_not_a_duplicate(corpus: Path):
    """Same GUID re-encountering its own bytes is a retry and must proceed to ASR."""
    cfg = _cfg(corpus)
    assert _schedule(cfg, corpus, _episode(1, "guid-x", "Ep", "https://cdn/a.mp3")) is not None
    retry = _schedule(cfg, corpus, _episode(1, "guid-x", "Ep", "https://cdn/a.mp3"))
    assert retry is not None, "an episode was treated as a duplicate of itself"


def test_the_gate_has_a_kill_switch(corpus: Path):
    """audio_dedup_enabled=False restores the old behaviour entirely."""
    cfg = _cfg(corpus, audio_dedup_enabled=False)
    assert _schedule(cfg, corpus, _episode(1, "guid-a", "Ep", "https://cdn/a.mp3")) is not None
    assert _schedule(cfg, corpus, _episode(2, "guid-b", "Ep 2", "https://cdn/b.mp3")) is not None


def test_registration_survives_and_is_atomic_on_disk(corpus: Path):
    """The index is real JSON at the corpus root, keyed by digest, carrying the identity."""
    cfg = _cfg(corpus)
    job = _schedule(cfg, corpus, _episode(1, "guid-a", "Ep", "https://cdn/a.mp3"))
    assert job is not None and job.audio_sha256, "the job must carry the digest it was hashed with"
    episode_processor._register_audio_fingerprint(job, cfg, str(corpus), "transcripts/0001.txt")

    index_file = corpus / audio_fingerprint.INDEX_RELPATH
    assert index_file.exists()
    import json

    entry = json.loads(index_file.read_text())[job.audio_sha256]
    assert entry["identity"] == "guid-a"
    assert entry["transcript_path"] == "transcripts/0001.txt"
