"""Shared reads take podcast episodes only unless asked for another type (RFC-134, ADR-177 D8).

A corpus can hold another content type before any surface reads it. The catalog builders and the
search indexer's walk must then keep returning exactly the podcast rows they return today, so the
Player and the MCP tools see a new type only once they ask for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.content_types import (
    is_wanted,
    item_content_type,
    PODCAST_EPISODE,
    PODCAST_ONLY,
)
from podcast_scraper.search.two_tier_indexer import _servable_docs
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows,
    build_catalog_rows_cumulative,
    catalog_row_for_metadata_path,
)

pytestmark = pytest.mark.unit

OTHER = "test_text_item"


def _write(root: Path, feed: str, item: str, content_type: str | None = None) -> Path:
    doc: dict = {
        "feed": {"feed_id": feed, "title": f"Source {feed}"},
        "episode": {"episode_id": item, "title": f"Item {item}", "published_date": "2026-11-01"},
    }
    if content_type is not None:
        doc["content_type"] = content_type
    path = root / "feeds" / feed / "run_20261101-000000" / "metadata" / f"{item}.metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.fixture
def mixed_corpus(tmp_path: Path) -> Path:
    _write(tmp_path, "podcast-a", "e1")
    _write(tmp_path, "podcast-a", "e2", content_type=PODCAST_EPISODE)
    _write(tmp_path, "texts", "t1", content_type=OTHER)
    return tmp_path


def test_a_document_without_a_type_is_a_podcast_episode() -> None:
    assert item_content_type({"episode": {}}) == PODCAST_EPISODE
    assert item_content_type({"content_type": "  "}) == PODCAST_EPISODE
    assert item_content_type(None) == PODCAST_EPISODE
    assert item_content_type({"content_type": OTHER}) == OTHER
    assert is_wanted({}, PODCAST_ONLY)
    assert not is_wanted({"content_type": OTHER}, PODCAST_ONLY)


@pytest.mark.parametrize("builder", [build_catalog_rows, build_catalog_rows_cumulative])
def test_catalog_builders_return_podcast_rows_by_default(mixed_corpus: Path, builder) -> None:
    assert sorted(r.episode_id for r in builder(mixed_corpus)) == ["e1", "e2"]


@pytest.mark.parametrize("builder", [build_catalog_rows, build_catalog_rows_cumulative])
def test_catalog_builders_return_another_type_when_asked(mixed_corpus: Path, builder) -> None:
    rows = builder(mixed_corpus, content_types={PODCAST_EPISODE, OTHER})
    assert sorted(r.episode_id for r in rows) == ["e1", "e2", "t1"]
    assert [r.episode_id for r in builder(mixed_corpus, content_types={OTHER})] == ["t1"]


def test_the_single_row_builder_refuses_another_type_by_default(mixed_corpus: Path) -> None:
    rel = "feeds/texts/run_20261101-000000/metadata/t1.metadata.json"
    assert catalog_row_for_metadata_path(mixed_corpus, rel) is None
    row = catalog_row_for_metadata_path(mixed_corpus, rel, content_types={OTHER})
    assert row is not None and row.episode_id == "t1"


def test_the_indexer_walks_podcast_episodes_only(mixed_corpus: Path) -> None:
    paths = sorted(mixed_corpus.rglob("*.metadata.json"))
    served: set[str] = set()
    withheld: set[str] = set()
    got = sorted(
        doc["episode"]["episode_id"] for _, doc in _servable_docs(paths, {}, served, withheld)
    )
    assert got == ["e1", "e2"]
    # Another type's item is neither served nor withheld: its rows are not this index's to prune.
    assert served == {"e1", "e2"} and withheld == set()
