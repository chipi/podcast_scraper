"""CLI helpers for single-feed corpus manifest stamp (#807)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.cli import single_feed_corpus_parent_for_manifest_stamp


@pytest.mark.unit
def test_single_feed_corpus_parent_with_feeds_spec_uses_output_dir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(corpus),
            "openai_api_key": "sk-test",
        }
    )
    args = argparse.Namespace(feeds_spec="feeds.yaml", output_dir=str(corpus))
    assert single_feed_corpus_parent_for_manifest_stamp(cfg, args) == str(corpus.resolve())


@pytest.mark.unit
def test_single_feed_corpus_parent_delegates_to_cfg_helper(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "feeds").mkdir(parents=True)
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(corpus),
            "single_feed_uses_corpus_layout": True,
            "openai_api_key": "sk-test",
        }
    )
    args = argparse.Namespace(feeds_spec=None, output_dir=None)
    assert single_feed_corpus_parent_for_manifest_stamp(cfg, args) == str(corpus.resolve())
