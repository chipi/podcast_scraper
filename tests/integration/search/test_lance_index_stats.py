"""Integration tests for the LanceDB index-stats reader (ADR-099 / #995).

``read_lance_index_stats`` aggregates row counts, doc-type breakdown, indexed feeds and
on-disk size from a real two-tier LanceDB index. Uses a real embedded LanceDB in a temp
dir with tiny 4-dim vectors (fast, no ML model) — mirrors tests/integration/search/
test_lancedb_backend.py.
"""

from __future__ import annotations

import pytest

from podcast_scraper.search.backend import InsightDocument, SegmentDocument

# critical_path so these run in ``test-integration-fast`` on PRs (which installs
# the ``search`` extra → lancedb), keeping the LanceDB stats readers — including
# ``read_lance_doc_type_by_month`` — covered in PR codecov, not just nightly.
pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

lancedb = pytest.importorskip("lancedb")

from podcast_scraper.search import lance_index_stats as lis  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402
from podcast_scraper.search.lance_index_stats import (  # noqa: E402
    _dir_size,
    read_lance_doc_type_by_month,
    read_lance_index_stats,
)


def _seg(i, text, show, emb, publish_date=None):
    return SegmentDocument(
        id=i,
        text=text,
        show_id=show,
        episode_id="ep1",
        start_time=0.0,
        end_time=1.0,
        embedding=emb,
        publish_date=publish_date,
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


def test_dir_size_skips_unreadable_files(tmp_path, monkeypatch):
    # _dir_size swallows per-file OSError (e.g. a vanished/permission-denied file) and
    # keeps summing the rest (line 42-43). Force getsize to raise for every file.
    (tmp_path / "a.bin").write_bytes(b"x" * 16)

    def _boom(_p):
        raise OSError("permission denied")

    monkeypatch.setattr("podcast_scraper.search.lance_index_stats.os.path.getsize", _boom)
    # Every file errored -> total falls back to 0, no exception propagates.
    assert _dir_size(tmp_path) == 0


def test_last_updated_stays_none_on_stat_oserror(lance_dir, monkeypatch):
    # When the last_updated stat/timestamp lookup raises OSError, the stamp is left unset
    # (line 86-87) while the rest of the stats (counts, size) still populate. We make the
    # datetime.fromtimestamp inside that try-block raise (the stat result feeds it), which
    # is exactly the failure mode the except OSError guards.
    real_fromtimestamp = lis.datetime.fromtimestamp

    class _RaisingDatetime:
        @staticmethod
        def fromtimestamp(*_a, **_k):
            raise OSError("clock/stat failure")

    monkeypatch.setattr(lis, "datetime", _RaisingDatetime)
    st = read_lance_index_stats(lance_dir)
    assert st is not None
    assert st.last_updated is None  # the OSError in the try-block left the stamp unset
    assert st.total_vectors == 3  # counts unaffected
    # sanity: the real fromtimestamp is restored after the test (monkeypatch undoes it).
    assert real_fromtimestamp is not None


class _StubTable:
    """A tier table with NO doc_type/show_id columns -> exercises the _TIER_DEFAULT path."""

    def __init__(self, n: int) -> None:
        self._n = n

        class _Schema:
            names = ["id", "text", "embedding"]  # neither doc_type nor show_id

        self.schema = _Schema()

    def count_rows(self) -> int:
        return self._n


def test_schema_light_tier_uses_default_doc_type(tmp_path, monkeypatch):
    # A tier whose schema lacks both doc_type and show_id can't be searched per-row, so the
    # whole count folds into the tier's implicit doc_type (line 70-72). The real schemas
    # always carry show_id, so this is exercised with a stub table.
    (tmp_path / "lance").mkdir()
    be = LanceDBBackend.__new__(LanceDBBackend)  # bypass lancedb.connect

    def _open(tier):
        # Only the segment tier exists; it has the schema-light stub.
        return _StubTable(5) if tier == "segment" else None

    monkeypatch.setattr(be, "_open_if_exists", _open, raising=False)
    monkeypatch.setattr(be, "read_index_meta", lambda: {}, raising=False)
    monkeypatch.setattr(lis, "LanceDBBackend", lambda _p: be)

    st = read_lance_index_stats(tmp_path / "lance")
    assert st is not None
    assert st.total_vectors == 5
    # No show_id/doc_type columns -> the 5 rows fold into "transcript" (segment's default),
    # and no feeds can be derived.
    assert st.doc_type_counts == {"transcript": 5}
    assert st.feeds_indexed == []


class _SearchableStubTable:
    """A tier table WITH a show_id column whose rows carry an empty show_id, exercising the
    per-row path where the `if sid:` guard is false (no feed recorded)."""

    def __init__(self, rows):
        self._rows = rows

        class _Schema:
            names = ["id", "doc_type", "show_id"]

        self.schema = _Schema()

    def count_rows(self):
        return len(self._rows)

    def search(self):
        rows = self._rows

        class _Q:
            def limit(self, _n):
                return self

            def select(self, _cols):
                return self

            def to_list(self):
                return rows

        return _Q()


def test_row_with_empty_show_id_records_no_feed(tmp_path, monkeypatch):
    # A populated tier whose rows have a falsy show_id counts the doc_type but adds no feed
    # (covers the per-row `if sid:` false branch).
    (tmp_path / "lance").mkdir()
    be = LanceDBBackend.__new__(LanceDBBackend)
    table = _SearchableStubTable([{"doc_type": "quote", "show_id": ""}])

    monkeypatch.setattr(
        be, "_open_if_exists", lambda tier: table if tier == "aux" else None, raising=False
    )
    monkeypatch.setattr(be, "read_index_meta", lambda: {}, raising=False)
    monkeypatch.setattr(lis, "LanceDBBackend", lambda _p: be)

    st = read_lance_index_stats(tmp_path / "lance")
    assert st is not None
    assert st.doc_type_counts == {"quote": 1}  # the row's own doc_type was used
    assert st.feeds_indexed == []  # empty show_id -> no feed recorded


def test_schema_light_unknown_tier_falls_back_to_tier_name(tmp_path, monkeypatch):
    # A schema-light tier with no _TIER_DEFAULT_DOC_TYPE entry (aux) folds into the tier name.
    (tmp_path / "lance").mkdir()
    be = LanceDBBackend.__new__(LanceDBBackend)

    def _open(tier):
        return _StubTable(3) if tier == "aux" else None

    monkeypatch.setattr(be, "_open_if_exists", _open, raising=False)
    monkeypatch.setattr(be, "read_index_meta", lambda: {}, raising=False)
    monkeypatch.setattr(lis, "LanceDBBackend", lambda _p: be)

    st = read_lance_index_stats(tmp_path / "lance")
    assert st is not None
    # aux is not in _TIER_DEFAULT_DOC_TYPE -> falls back to the literal tier name "aux".
    assert st.doc_type_counts == {"aux": 3}


@pytest.fixture
def lance_dir_dated(tmp_path):
    """Two-tier index whose rows carry publish_date across two months."""
    d = tmp_path / "lance-dated"
    b = LanceDBBackend(str(d), embed_dim=4)
    b.upsert_segment(_seg("s1", "Jan one", "feed-A", [0.1, 0.2, 0.3, 0.4], "2026-01-15"))
    b.upsert_segment(_seg("s2", "Jan two", "feed-B", [0.9, 0.1, 0.0, 0.1], "2026-01-20"))
    b.upsert_segment(_seg("s3", "Feb one", "feed-A", [0.2, 0.2, 0.2, 0.2], "2026-02-03"))
    b.upsert_insight(
        InsightDocument(
            id="insight:1",
            text="Jan insight",
            show_id="feed-A",
            episode_id="ep1",
            entity_type="claim",
            confidence=0.9,
            derived=False,
            embedding=[0.1, 0.2, 0.3, 0.45],
            publish_date="2026-01-15",
        )
    )
    b.create_indices()
    return d


def test_doc_type_by_month_buckets_by_publish_month(lance_dir_dated):
    out = read_lance_doc_type_by_month(lance_dir_dated)
    # Jan: 2 transcripts + 1 insight; Feb: 1 transcript.
    assert out == {
        "2026-01": {"transcript": 2, "insight": 1},
        "2026-02": {"transcript": 1},
    }


def test_doc_type_by_month_skips_rows_without_publish_date(tmp_path):
    d = tmp_path / "lance-mixed"
    b = LanceDBBackend(str(d), embed_dim=4)
    b.upsert_segment(_seg("s1", "dated", "feed-A", [0.1, 0.2, 0.3, 0.4], "2026-03-01"))
    b.upsert_segment(_seg("s2", "undated", "feed-A", [0.2, 0.2, 0.2, 0.2], None))
    b.create_indices()
    out = read_lance_doc_type_by_month(d)
    # The undated row is dropped (no month to bucket into); only the dated one survives.
    assert out == {"2026-03": {"transcript": 1}}


def test_doc_type_by_month_returns_empty_for_absent_dir(tmp_path):
    assert read_lance_doc_type_by_month(tmp_path / "nope") == {}


def test_doc_type_by_month_accepts_str_path(lance_dir_dated):
    out = read_lance_doc_type_by_month(str(lance_dir_dated))
    assert out["2026-02"] == {"transcript": 1}
