"""Skip-existing must key on the STABLE guid, not the run-local feed idx (incremental-add #1).

The blocker for cautious incremental prod adds: ``--skip-existing`` / ``--append`` resolved
"already processed?" from ``episode.idx`` — the feed's enumerate position — which shifts the moment
a feed publishes a new item between runs. Every downstream check (transcript path, metadata lookup)
then misses the already-present episode and silently re-processes it (wasted cloud spend + a
duplicate in the corpus). These tests pin the drift scenario: an episode whose on-disk idx is 1 but
whose run-local idx is now 2 (a newer item was prepended) is still recognised as present.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import Config, models
from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor, run_index
from podcast_scraper.workflow.helpers import get_episode_id_from_episode

pytestmark = pytest.mark.unit

FEED_URL = "https://example.com/podcast.xml"


@pytest.fixture(autouse=True)
def _reset_index_cache():
    run_index.reset_corpus_metadata_index_cache_for_tests()
    yield
    run_index.reset_corpus_metadata_index_cache_for_tests()


def _episode(guid: str, idx: int, title: str = "Ep A", title_safe: str = "Ep A") -> models.Episode:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = title
    ET.SubElement(item, "guid").text = guid
    return models.Episode(
        idx=idx, title=title, title_safe=title_safe, item=item, transcript_urls=[]
    )


def _seed_processed_episode(corpus: Path, guid: str, on_disk_idx: int = 1) -> str:
    """Write a processed episode at ``on_disk_idx`` (transcript + metadata). Returns episode_id."""
    name = f"{on_disk_idx:04d} - Ep A"
    (corpus / "transcripts").mkdir(parents=True, exist_ok=True)
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    (corpus / "transcripts" / f"{name}.txt").write_text("hello", encoding="utf-8")
    eid, _ = get_episode_id_from_episode(_episode(guid, on_disk_idx), FEED_URL)
    (corpus / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": guid, "episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{name}.txt"},
            }
        ),
        encoding="utf-8",
    )
    return str(eid)


def test_resolve_ondisk_idx_uses_guid_not_run_local_idx(tmp_path: Path):
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    # The feed grew: this episode is now at run-local idx 2, but on disk it is 0001.
    shifted = _episode("gA", idx=2)
    assert run_index.resolve_ondisk_idx_for_episode(shifted, str(tmp_path)) == 1


def test_resolve_ondisk_idx_falls_back_for_new_episode(tmp_path: Path):
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    new_ep = _episode("gBRAND_NEW", idx=3)  # guid not on disk → keep its run-local idx
    assert run_index.resolve_ondisk_idx_for_episode(new_ep, str(tmp_path)) == 3


def test_check_existing_transcript_skips_across_idx_shift(tmp_path: Path):
    """Direct-download skip path: the already-processed episode is skipped even after idx shift."""
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    cfg = Config(rss=FEED_URL, output_dir=str(tmp_path), skip_existing=True)
    shifted = _episode("gA", idx=2)

    # The run-local idx path (0002) does not exist — the old idx-keyed logic would MISS it.
    assert not (tmp_path / "transcripts" / "0002 - Ep A.txt").exists()
    assert episode_processor._check_existing_transcript(shifted, str(tmp_path), None, cfg) is True


def test_check_existing_transcript_new_episode_is_not_skipped(tmp_path: Path):
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    cfg = Config(rss=FEED_URL, output_dir=str(tmp_path), skip_existing=True)
    new_ep = _episode("gNEW", idx=2, title="Ep B", title_safe="Ep B")
    assert episode_processor._check_existing_transcript(new_ep, str(tmp_path), None, cfg) is False


def test_whisper_skip_path_resolves_existing_transcript_across_shift(tmp_path: Path):
    """Whisper path: build_whisper_output_path(resolved idx) hits the on-disk transcript (the
    exact existence check download_media_for_transcription runs)."""
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    shifted = _episode("gA", idx=2)
    resolved = run_index.resolve_ondisk_idx_for_episode(shifted, str(tmp_path))
    path = filesystem.build_whisper_output_path(resolved, shifted.title_safe, None, str(tmp_path))
    assert Path(path).exists()  # found via guid-resolved idx
    # Run-local idx would have looked for 0002 and re-transcribed.
    run_local = filesystem.build_whisper_output_path(2, shifted.title_safe, None, str(tmp_path))
    assert not Path(run_local).exists()


def test_find_metadata_relative_path_locates_by_guid_across_shift(tmp_path: Path):
    _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    shifted = _episode("gA", idx=2)
    rel = run_index.find_episode_metadata_relative_path(shifted, str(tmp_path), None)
    assert rel is not None
    assert rel.endswith("0001 - Ep A.metadata.json")


def test_corpus_metadata_index_maps_guid_and_episode_id(tmp_path: Path):
    eid = _seed_processed_episode(tmp_path, "gA", on_disk_idx=1)
    idx = run_index.corpus_metadata_index(str(tmp_path))
    assert idx["by_guid"]["gA"].idx == 1
    assert idx["by_id"][eid].idx == 1
    assert idx["by_guid"]["gA"].metadata_rel.endswith("0001 - Ep A.metadata.json")
