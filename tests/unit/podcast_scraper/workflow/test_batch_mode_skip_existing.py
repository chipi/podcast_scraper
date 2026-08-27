# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
"""Batch-mode skip_existing must see PRIOR run dirs, exactly like corpus-layout mode (D7).

2026-08-27, caught by the supervised cap-10 sweep: the D7 corpus-wide presence lookup was
gated on ``single_feed_uses_corpus_layout`` — a flag only per-feed jobs set. The nightly's
batch mode writes the IDENTICAL fresh-run-dir shape (feeds/<slug>/run_*) with the flag off,
so its presence check hit the brand-new empty run dir and NEVER skipped: every scheduled run
re-ingested its whole window. Masked for one night by max_episodes=1; exposed at window 10
(all 14 feeds re-selecting episodes already in the corpus; run stopped at operator's order).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.models import Episode
from podcast_scraper.workflow import episode_processor

pytestmark = [pytest.mark.unit]


def _episode(guid: str, title: str) -> Episode:
    item = ET.Element("item")
    g = ET.SubElement(item, "guid")
    g.text = guid
    return Episode(
        idx=1,
        title=title,
        title_safe=title.replace(" ", "_"),
        item=item,
        transcript_urls=[],
        media_url="https://cdn.example/ep.mp3",
        media_type="audio/mpeg",
    )


def _prior_run_with_episode(corpus: Path, guid: str, title: str) -> None:
    """A completed prior run for this feed holding the episode's metadata + transcript."""
    run = corpus / "feeds" / "rss_feed_a" / "run_20260826-000000_old"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir(parents=True)
    stem = f"0001 - {title}_old"
    (run / "metadata" / f"{stem}.metadata.json").write_text(
        json.dumps({"episode": {"guid": guid, "episode_id": guid, "title": title}}),
        encoding="utf-8",
    )
    (run / "transcripts" / f"{stem}.txt").write_text("a transcript", encoding="utf-8")


def test_batch_mode_skips_episode_already_in_a_prior_run(tmp_path: Path, monkeypatch) -> None:
    """The nightly shape: corpus output_dir + per-feed fresh run dir, NO corpus-layout flag."""
    guid = "guid-already-ingested"
    _prior_run_with_episode(tmp_path, guid, "Bombing the bond market")
    fresh_run = tmp_path / "feeds" / "rss_feed_a" / "run_20260827-055555_new"
    (fresh_run / "metadata").mkdir(parents=True)

    def _download_must_not_run(*_a, **_k):
        raise AssertionError(
            "download was reached — skip_existing must decide BEFORE any download; "
            "returning None via a failed download would fake a pass"
        )

    monkeypatch.setattr(episode_processor, "_download_or_reuse_media", _download_must_not_run)
    cfg = config.Config(
        rss_url="https://feed-a.example/rss",
        output_dir=str(tmp_path),
        skip_existing=True,
        # deliberately NOT single_feed_uses_corpus_layout — batch mode leaves it False
    )
    job = episode_processor.download_media_for_transcription(
        _episode(guid, "Bombing the bond market"),
        cfg,
        temp_dir=str(tmp_path / "tmp"),
        effective_output_dir=str(fresh_run),
        run_suffix=None,
    )
    assert job is None, (
        "an episode present in a PRIOR run dir was scheduled for re-transcription — "
        "batch-mode skip_existing is blind to the corpus (the 2026-08-27 sweep burn)"
    )


def test_batch_mode_still_ingests_a_genuinely_new_episode(tmp_path: Path, monkeypatch) -> None:
    _prior_run_with_episode(tmp_path, "guid-old", "Old episode")
    fresh_run = tmp_path / "feeds" / "rss_feed_a" / "run_20260827-055555_new"
    (fresh_run / "metadata").mkdir(parents=True)

    def _fake_download(episode, cfg, temp_media, pipeline_metrics, effective_output_dir):
        Path(temp_media).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_media).write_bytes(b"x" * (300 * 1024))
        return True, 300 * 1024, 0.01

    monkeypatch.setattr(episode_processor, "_download_or_reuse_media", _fake_download)
    cfg = config.Config(
        rss_url="https://feed-a.example/rss",
        output_dir=str(tmp_path),
        skip_existing=True,
    )
    job = episode_processor.download_media_for_transcription(
        _episode("guid-brand-new", "New episode"),
        cfg,
        temp_dir=str(tmp_path / "tmp"),
        effective_output_dir=str(fresh_run),
        run_suffix=None,
    )
    assert job is not None, "a genuinely new episode must still be ingested"


def test_non_corpus_single_feed_layout_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Legacy single-feed runs (output NOT under <corpus>/feeds/) keep prior behaviour."""
    out = tmp_path / "output" / "rss_feed_a"
    (out / "metadata").mkdir(parents=True)

    def _fake_download(episode, cfg, temp_media, pipeline_metrics, effective_output_dir):
        Path(temp_media).parent.mkdir(parents=True, exist_ok=True)
        Path(temp_media).write_bytes(b"x" * (300 * 1024))
        return True, 300 * 1024, 0.01

    monkeypatch.setattr(episode_processor, "_download_or_reuse_media", _fake_download)
    cfg = config.Config(
        rss_url="https://feed-a.example/rss",
        output_dir=str(out),
        skip_existing=True,
    )
    job = episode_processor.download_media_for_transcription(
        _episode("guid-x", "Ep"),
        cfg,
        temp_dir=str(tmp_path / "tmp"),
        effective_output_dir=str(out),
        run_suffix=None,
    )
    assert job is not None
