"""Guardrail: skip-existing must be CORPUS-WIDE, not scoped to the current run dir (D7).

The prod defect (Step 1 NO-GO, 2026-08-11): under ``--single-feed-uses-corpus-layout`` each run
writes a FRESH ``feeds/<slug>/run_<id>/`` dir. ``--skip-existing`` resolved the episode's existence
against ``effective_output_dir`` (that fresh, empty run dir), so an episode already present in a
PRIOR run dir was NOT found → the pipeline re-transcribed it (wasted Deepgram + LLM spend), though
episode_id dedup kept the catalog count stable.

The guid fix (#1) is present but was run-dir-scoped. These tests pin the cross-run-dir scenario:
an episode processed under ``run_A`` must be recognised as present when the next run is ``run_B``.

``Config(single_feed_uses_corpus_layout=True)`` rewrites ``output_dir`` to ``<root>/feeds/<slug>``,
so ``cfg.output_dir`` is the FEED dir that contains every ``run_*`` for the feed — the correct
corpus-wide scan root for skip-existing.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import Config, models
from podcast_scraper.workflow import episode_processor, run_index
from podcast_scraper.workflow.helpers import get_episode_id_from_episode

pytestmark = pytest.mark.unit

FEED_URL = "https://example.com/podcast.xml"


@pytest.fixture(autouse=True)
def _reset_index_cache():
    run_index.reset_corpus_metadata_index_cache_for_tests()
    yield
    run_index.reset_corpus_metadata_index_cache_for_tests()


def _episode(guid: str, idx: int) -> models.Episode:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = "Ep A"
    ET.SubElement(item, "guid").text = guid
    return models.Episode(idx=idx, title="Ep A", title_safe="Ep A", item=item, transcript_urls=[])


def _corpus_layout_cfg(tmp_path: Path) -> Config:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    return Config(
        rss=FEED_URL,
        output_dir=str(corpus),
        skip_existing=True,
        single_feed_uses_corpus_layout=True,
    )


def _seed_prior_run(feed_dir: Path, guid: str, on_disk_idx: int = 1) -> None:
    """Write a processed episode under <feed_dir>/run_A/{metadata,transcripts}/ (a PRIOR run)."""
    run_dir = feed_dir / "run_20260101-000000_priorAA"
    name = f"{on_disk_idx:04d} - Ep A"
    (run_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "transcripts" / f"{name}.txt").write_text("hello", encoding="utf-8")
    eid, _ = get_episode_id_from_episode(_episode(guid, on_disk_idx), FEED_URL)
    (run_dir / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"guid": guid, "episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{name}.txt"},
            }
        ),
        encoding="utf-8",
    )


def _fresh_run(feed_dir: Path) -> Path:
    fresh = feed_dir / "run_20260102-000000_freshBB"
    fresh.mkdir(parents=True, exist_ok=True)
    return fresh


def test_skip_existing_finds_episode_in_prior_run_dir(tmp_path: Path):
    """Direct-download skip path: episode from run_A is recognised when the fresh run is run_B."""
    cfg = _corpus_layout_cfg(tmp_path)
    feed_dir = Path(str(cfg.output_dir))
    _seed_prior_run(feed_dir, "gA", on_disk_idx=1)
    fresh_run = _fresh_run(feed_dir)

    # The fresh run dir has NO transcript — the run-dir-scoped check would MISS the episode.
    assert not (fresh_run / "transcripts" / "0001 - Ep A.txt").exists()
    result = episode_processor._check_existing_transcript(
        _episode("gA", idx=1), str(fresh_run), None, cfg
    )
    assert result is True


def test_asr_path_skips_episode_in_prior_run_dir(tmp_path: Path):
    """ASR/Deepgram skip path (the actual prod bug): download_media_for_transcription returns None
    (skipped) for an already-present episode instead of re-downloading + re-transcribing."""
    cfg = _corpus_layout_cfg(tmp_path)
    feed_dir = Path(str(cfg.output_dir))
    _seed_prior_run(feed_dir, "gA", on_disk_idx=1)
    fresh_run = _fresh_run(feed_dir)
    ep = _episode("gA", idx=1)
    ep.media_url = "https://example.com/media/gA.mp3"  # would be downloaded if NOT skipped

    job = episode_processor.download_media_for_transcription(
        ep, cfg, str(tmp_path / "tmp"), str(fresh_run), None
    )
    assert job is None, "an already-present episode must be skipped, not re-transcribed (D7)"


def test_new_episode_not_skipped_in_corpus_layout(tmp_path: Path):
    cfg = _corpus_layout_cfg(tmp_path)
    feed_dir = Path(str(cfg.output_dir))
    _seed_prior_run(feed_dir, "gA", on_disk_idx=1)
    fresh_run = _fresh_run(feed_dir)

    assert (
        episode_processor._check_existing_transcript(
            _episode("gBRAND_NEW", idx=2), str(fresh_run), None, cfg
        )
        is False
    )


def test_non_corpus_layout_unchanged_flat_dir(tmp_path: Path):
    """Non-corpus-layout (single flat dir) keeps the original run-dir-scoped behaviour."""
    corpus = tmp_path / "flat"
    name = "0001 - Ep A"
    (corpus / "transcripts").mkdir(parents=True, exist_ok=True)
    (corpus / "metadata").mkdir(parents=True, exist_ok=True)
    (corpus / "transcripts" / f"{name}.txt").write_text("hi", encoding="utf-8")
    eid, _ = get_episode_id_from_episode(_episode("gA", 1), FEED_URL)
    (corpus / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps({"episode": {"guid": "gA", "episode_id": eid}}), encoding="utf-8"
    )
    cfg = Config(rss=FEED_URL, output_dir=str(corpus), skip_existing=True)
    assert (
        episode_processor._check_existing_transcript(_episode("gA", 1), str(corpus), None, cfg)
        is True
    )


def _episode_no_guid(idx: int = 1) -> models.Episode:
    item = ET.Element("item")
    ET.SubElement(item, "title").text = "Ep A"
    return models.Episode(idx=idx, title="Ep A", title_safe="Ep A", item=item, transcript_urls=[])


def _seed_metadata_only(feed_dir: Path, run_name: str, guid: str) -> Path:
    run_dir = feed_dir / run_name
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    eid, _ = get_episode_id_from_episode(_episode(guid, 1), FEED_URL)
    (run_dir / "metadata" / "0001 - Ep A.metadata.json").write_text(
        json.dumps({"episode": {"guid": guid, "episode_id": eid}}), encoding="utf-8"
    )
    return run_dir


def test_metadata_rel_none_without_guid(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    _seed_prior_run(feed_dir, "gA")
    assert run_index.episode_metadata_rel_in_corpus(_episode_no_guid(), str(feed_dir)) is None


def test_metadata_rel_none_when_not_present(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    _seed_prior_run(feed_dir, "gA")
    assert run_index.episode_metadata_rel_in_corpus(_episode("gMISS", 9), str(feed_dir)) is None


def test_existing_transcript_prefers_txt(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    _seed_prior_run(feed_dir, "gA")  # writes a .txt transcript
    p = run_index.existing_transcript_path_in_corpus(_episode("gA", 1), str(feed_dir))
    assert p is not None and p.endswith("0001 - Ep A.txt")


def test_existing_transcript_globs_non_txt(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    run = _seed_metadata_only(feed_dir, "run_20260101-000000_vtt00000", "gA")
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    (run / "transcripts" / "0001 - Ep A.vtt").write_text("v", encoding="utf-8")
    p = run_index.existing_transcript_path_in_corpus(_episode("gA", 1), str(feed_dir))
    assert p is not None and p.endswith("0001 - Ep A.vtt")


def test_existing_transcript_metadata_only_fallback(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    _seed_metadata_only(feed_dir, "run_20260101-000000_meta0000", "gA")  # no transcripts/ dir
    p = run_index.existing_transcript_path_in_corpus(_episode("gA", 1), str(feed_dir))
    assert p is not None and p.endswith("0001 - Ep A.metadata.json")


def test_existing_transcript_none_when_absent(tmp_path: Path):
    feed_dir = Path(str(_corpus_layout_cfg(tmp_path).output_dir))
    assert run_index.existing_transcript_path_in_corpus(_episode("gNONE", 1), str(feed_dir)) is None
