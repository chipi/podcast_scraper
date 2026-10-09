"""A language-guard refusal is remembered, so the DGX is not asked the same question every run.

A refusal (#2187, ADR-158) writes no transcript, and skip-existing keys on the transcript — so
before this every scheduled run re-downloaded and re-transcribed a mis-tagged episode, only to
refuse it again (deep review, 2026-10-08). The record is corpus-wide (each run has a fresh run
dir) and keyed on the language declared at refusal time: a changed declaration is the operator's
answer, and the episode is tried again.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from podcast_scraper import config
from podcast_scraper.models import Episode
from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import episode_processor as EP

FEED_URL = "https://feeds.example/es.xml"


def _episode(guid: str = "guid-mistagged") -> Episode:
    item = ET.Element("item")
    ET.SubElement(item, "guid").text = guid
    return Episode(
        idx=1,
        title="Un episodio",
        title_safe="Un_episodio",
        item=item,
        transcript_urls=[],
        media_url="https://cdn.example/ep.mp3",
        media_type="audio/mpeg",
    )


def _cfg(corpus: Path, language: str = "es", **overrides) -> config.Config:
    fields = {
        "rss_url": FEED_URL,
        "output_dir": filesystem.corpus_feed_output_dir(str(corpus), FEED_URL),
        "skip_existing": True,
        "feed_declared_language": language,
        **overrides,
    }
    return config.Config(**fields)


def _decide(ep: Episode, cfg: config.Config) -> EP.SkipExisting:
    return EP.media_route_skip_existing(ep, cfg, str(cfg.output_dir), None)


def test_a_refused_episode_is_skipped_while_its_language_is_unchanged(tmp_path: Path) -> None:
    ep, cfg = _episode(), _cfg(tmp_path)
    assert _decide(ep, cfg).action == EP.NEW
    EP._record_language_refusal(ep, cfg, "reads as English, declared es")
    decision = _decide(ep, cfg)
    assert decision.action == EP.REFUSED
    assert decision.path is not None and Path(decision.path).exists()


def test_the_record_is_corpus_wide_not_per_run(tmp_path: Path) -> None:
    # Each run writes a fresh run dir; the record must be found from the next one.
    ep = _episode()
    EP._record_language_refusal(ep, _cfg(tmp_path), "detail")
    assert ".language_refusals" in str(_decide(ep, _cfg(tmp_path)).path)
    assert str(tmp_path) in str(_decide(ep, _cfg(tmp_path)).path)


def test_a_changed_declaration_tries_again(tmp_path: Path) -> None:
    ep = _episode()
    EP._record_language_refusal(ep, _cfg(tmp_path, "es"), "detail")
    assert _decide(ep, _cfg(tmp_path, "en")).action == EP.NEW


def test_another_episode_is_not_affected(tmp_path: Path) -> None:
    EP._record_language_refusal(_episode("guid-a"), _cfg(tmp_path), "detail")
    assert _decide(_episode("guid-b"), _cfg(tmp_path)).action == EP.NEW


def test_without_skip_existing_nothing_is_skipped(tmp_path: Path) -> None:
    ep = _episode()
    EP._record_language_refusal(ep, _cfg(tmp_path), "detail")
    assert _decide(ep, _cfg(tmp_path, skip_existing=False)).action == EP.NEW


def test_the_early_presence_skip_agrees(tmp_path: Path) -> None:
    ep, cfg = _episode(), _cfg(tmp_path, transcribe_missing=True)
    EP._record_language_refusal(ep, cfg, "detail")
    evidence = EP.presence_skip_evidence(ep, cfg, str(cfg.output_dir), None, str(tmp_path))
    assert evidence is not None and ".language_refusals" in evidence


def test_an_unreadable_record_is_ignored(tmp_path: Path) -> None:
    ep, cfg = _episode(), _cfg(tmp_path)
    raw = EP._language_refusal_path(ep, cfg)
    assert raw is not None
    path = Path(raw)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    assert _decide(ep, cfg).action == EP.NEW


def test_the_refusal_site_writes_the_record() -> None:
    """Structural: the record only helps if the guard's refusal branch writes it. Driving the whole
    transcription path for one branch would test the fixture; this pins the call in place."""
    source = Path(EP.__file__).read_text(encoding="utf-8")
    start = source.index('"[%s] REFUSING episode: %s"')
    branch = source[start : source.index("return False, None, bytes_downloaded", start)]
    assert "_record_language_refusal(job.episode, cfg, wrong_language)" in branch


def test_with_no_corpus_and_no_output_dir_nothing_is_written_anywhere(
    tmp_path: Path, monkeypatch
) -> None:
    # The first version fell back to "." and a test run scattered a record into the repo root.
    monkeypatch.chdir(tmp_path)
    cfg = _cfg(tmp_path, output_dir=None)
    assert EP._language_refusal_path(_episode(), cfg) is None
    EP._record_language_refusal(_episode(), cfg, "detail")
    assert list(tmp_path.iterdir()) == []
