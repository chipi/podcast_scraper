"""Regression: single-feed ``--feeds-spec`` must use ``<corpus>/feeds/`` layout (#440)."""

from __future__ import annotations

from argparse import Namespace

import pytest

from podcast_scraper.cli import resolve_cli_feed_targets, single_feeds_spec_output_dir
from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME

pytestmark = pytest.mark.unit


def test_single_feeds_spec_with_corpus_output_dir_uses_feeds_subdir(tmp_path) -> None:
    corpus = tmp_path / "my-corpus"
    corpus.mkdir()
    spec = corpus / FEEDS_SPEC_DEFAULT_BASENAME
    url = "https://example.com/show/feed.xml"
    spec.write_text(f"feeds:\n  - url: {url}\n", encoding="utf-8")
    args = Namespace(output_dir=str(corpus), feeds_spec=str(spec))
    only = resolve_cli_feed_targets(args)[0]
    out = single_feeds_spec_output_dir(args, only)
    norm = out.replace("\\", "/")
    assert "/feeds/rss_" in norm
    assert str(corpus).replace("\\", "/") in norm


def test_single_feeds_spec_without_output_dir_uses_legacy_derive(tmp_path) -> None:
    spec = tmp_path / "feeds.spec.yaml"
    url = "https://example.com/show/feed.xml"
    spec.write_text(f"feeds:\n  - url: {url}\n", encoding="utf-8")
    args = Namespace(output_dir=None, feeds_spec=str(spec))
    only = resolve_cli_feed_targets(args)[0]
    out = single_feeds_spec_output_dir(args, only)
    assert out.replace("\\", "/").startswith("output/rss_")
