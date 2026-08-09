"""Per-feed job scoping resolution — feed URL / slug → RSS URL (incremental-add P1.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException

from podcast_scraper.rss.feeds_spec import FEEDS_SPEC_DEFAULT_BASENAME
from podcast_scraper.server.routes.jobs import _resolve_feed_url
from podcast_scraper.utils.filesystem import feed_workspace_dirname

pytestmark = [pytest.mark.integration]

_URL = "https://a.example/podcast.xml"


def _corpus_with_feed(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / FEEDS_SPEC_DEFAULT_BASENAME).write_text(f"feeds:\n  - {_URL}\n", encoding="utf-8")
    return corpus


def test_url_passes_through_verbatim(tmp_path: Path) -> None:
    corpus = _corpus_with_feed(tmp_path)
    assert _resolve_feed_url(corpus, _URL) == _URL


def test_slug_resolves_to_url(tmp_path: Path) -> None:
    corpus = _corpus_with_feed(tmp_path)
    slug = feed_workspace_dirname(_URL)
    assert _resolve_feed_url(corpus, slug) == _URL


def test_unknown_slug_is_404(tmp_path: Path) -> None:
    corpus = _corpus_with_feed(tmp_path)
    with pytest.raises(HTTPException) as ei:
        _resolve_feed_url(corpus, "rss_nope_0000")
    assert ei.value.status_code == 404
