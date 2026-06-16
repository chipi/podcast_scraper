"""Integration tests for the LanceDB index-stats reader (ADR-099 / #995).

``read_lance_index_stats`` aggregates row counts, doc-type breakdown, indexed feeds and
on-disk size from a real two-tier LanceDB index. Uses a real embedded LanceDB in a temp
dir with tiny 4-dim vectors (fast, no ML model) — mirrors tests/integration/search/
test_lancedb_backend.py.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import InsightDocument, SegmentDocument

pytestmark = pytest.mark.integration

lancedb = pytest.importorskip("lancedb")

from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from podcast_scraper.search.lance_index_stats import (  # noqa: E402
    read_lance_index_stats,
)


def _seg(i, text, show, emb):
    return SegmentDocument(
        id=i, text=text, show_id=show, episode_id="ep1", start_time=0.0, end_time=1.0, embedding=emb
    )


@pytest.fixture
def lance_dir(tmp_path):
    """Build a small two-tier index (2 segments across 2 feeds + 1 insight)."""
    d = tmp_path / "lance"
    b = LanceDBBackend(str(d), embed_dim=4)
    b.upsert_segment(_seg("s1", "Sam Altman talks about OpenAI", "feed-A", [0.1, 0.2, 0.3, 0.4]))
    b.upsert_segment(_seg("s2", "Tim Cook and Apple earnings", "feed-B", [0.9, 0.1, 0.0, 0.1]))
    b.upsert_insight(
        InsightDocument(
            id="insight:1",
            text="Altman argues AI scaling continues",
            show_id="feed-A",
            episode_id="ep1",
            entity_type="claim",
            confidence=0.9,
            derived=False,
            embedding=[0.1, 0.2, 0.3, 0.45],
        )
    )
    b.create_indices()
    return d


def test_aggregates_rows_doc_types_and_feeds(lance_dir):
    st = read_lance_index_stats(lance_dir)
    assert st is not None
    # 2 segments + 1 insight across the two tiers.
    assert st.total_vectors == 3
    # segment rows default to "transcript" (the tier's implicit doc_type); insight to "insight".
    assert st.doc_type_counts == {"transcript": 2, "insight": 1}
    # show_id values become the indexed-feeds roster (sorted, de-duped).
    assert st.feeds_indexed == ["feed-A", "feed-B"]


def test_reports_on_disk_size_and_last_updated(lance_dir):
    st = read_lance_index_stats(lance_dir)
    assert st is not None
    # _dir_size walks the index dir — a populated index is non-empty on disk.
    assert st.index_size_bytes > 0
    # last_updated is an ISO-8601 UTC stamp ending in Z.
    assert st.last_updated is not None and st.last_updated.endswith("Z")


def test_returns_none_for_absent_directory(tmp_path):
    assert read_lance_index_stats(tmp_path / "does-not-exist") is None


def test_accepts_str_path(lance_dir):
    # The signature is Path | str; a plain string must work too.
    st = read_lance_index_stats(str(lance_dir))
    assert st is not None and st.total_vectors == 3


def test_returns_none_when_backend_construction_fails(tmp_path, monkeypatch):
    # A present-but-corrupt index dir: the backend raises on open -> stats reader returns None
    # rather than propagating (the dir exists, so it's past the is_dir() guard).
    (tmp_path / "lance").mkdir()

    def _boom(*_a, **_k):
        raise RuntimeError("corrupt lance index")

    monkeypatch.setattr("podcast_scraper.search.lance_index_stats.LanceDBBackend", _boom)
    assert read_lance_index_stats(tmp_path / "lance") is None
