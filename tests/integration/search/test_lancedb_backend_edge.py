"""Edge-branch coverage for LanceDBBackend (RFC-090 #855 + hardening)."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search.backend import (  # noqa: E402
    SearchQuery,
    SegmentDocument,
)
from podcast_scraper.search.backends import lancedb_backend as lb  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import (  # noqa: E402
    LANCE_SCHEMA_VERSION,
    LanceDBBackend,
    stored_schema_version,
)


def test_search_on_empty_backend_returns_nothing(tmp_path):
    # No tables created → read path skips missing tiers (no autocreate), returns [].
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    assert b.search_bm25(SearchQuery(text="x", embedding=[], tier="all")) == []
    assert b.search_vector(SearchQuery(text="x", embedding=[0.0] * 4, tier="all")) == []


def test_health_on_empty_backend(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    h = b.health()
    assert h["status"] == "ok" and h["segments"] == 0 and h["insights"] == 0


def test_meta_rejects_unsafe_path(tmp_path):
    # safe_resolve_directory rejects a '..' path → read returns None, write is a no-op.
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.path = str(tmp_path / "a" / ".." / "b")  # contains '..' → sanitizer rejects
    assert b.read_index_meta() is None
    b.write_index_meta("some-model")  # must not raise


def test_meta_round_trip(tmp_path):
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.write_index_meta("sentence-transformers/all-MiniLM-L6-v2")
    meta = b.read_index_meta()
    assert meta is not None and meta["embedding_model"].endswith("MiniLM-L6-v2")
    assert meta["embed_dim"] == 4


def test_read_index_meta_returns_none_on_invalid_json(tmp_path):
    # Covers the read_index_meta try/except (ValueError on malformed JSON) -> None.
    d = tmp_path / "lance"
    d.mkdir()
    (d / LanceDBBackend.INDEX_META_FILE).write_text("{not: valid json", encoding="utf-8")
    b = LanceDBBackend(str(d), embed_dim=4)
    assert b.read_index_meta() is None


def test_read_index_meta_returns_none_when_meta_is_not_a_dict(tmp_path):
    # A JSON list (valid JSON, wrong shape) -> isinstance(dict) guard returns None.
    d = tmp_path / "lance"
    d.mkdir()
    (d / LanceDBBackend.INDEX_META_FILE).write_text("[1, 2, 3]", encoding="utf-8")
    b = LanceDBBackend(str(d), embed_dim=4)
    assert b.read_index_meta() is None


def test_write_index_meta_noop_when_relpath_verifier_rejects(tmp_path, monkeypatch):
    # safe_relpath_under_corpus_root returning falsy short-circuits write (line ~146) and
    # read (line ~170) before any file IO. Force it falsy to exercise both guards.
    d = tmp_path / "lance"
    d.mkdir()
    b = LanceDBBackend(str(d), embed_dim=4)
    monkeypatch.setattr(lb, "safe_relpath_under_corpus_root", lambda *a, **k: None)
    b.write_index_meta("some-model")  # must not raise, writes nothing
    assert not (d / LanceDBBackend.INDEX_META_FILE).exists()
    assert b.read_index_meta() is None


def test_meta_noop_when_normpath_guard_rejects(tmp_path, monkeypatch):
    # normpath_if_under_root returning falsy short-circuits write (line ~149) and
    # read (line ~173) at the final sanitizer hop.
    d = tmp_path / "lance"
    d.mkdir()
    b = LanceDBBackend(str(d), embed_dim=4)
    monkeypatch.setattr(lb, "normpath_if_under_root", lambda *a, **k: None)
    b.write_index_meta("some-model")
    assert not (d / LanceDBBackend.INDEX_META_FILE).exists()
    assert b.read_index_meta() is None


def test_compact_swallows_optimize_failure(tmp_path):
    # A table whose optimize() raises must not fail compact() (best-effort, line 238-239).
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.upsert_segment(_seg("s1"))

    table = b._open_if_exists("segment")

    def _boom(*_a, **_k):
        raise RuntimeError("optimize unavailable")

    table.optimize = _boom  # type: ignore[assignment]
    b.compact()  # must not raise despite the failing optimize
    # The row is still there — compact failure left the data intact.
    assert b.health()["segments"] == 1


def test_health_reports_error_when_open_raises(tmp_path, monkeypatch):
    # An exception inside health() is caught -> {"status": "error", ...} (line 406-407).
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)

    def _boom(_tier):
        raise RuntimeError("table cache corrupt")

    monkeypatch.setattr(b, "_open_if_exists", _boom)
    h = b.health()
    assert h["status"] == "error" and "table cache corrupt" in h["error"]


def test_upsert_empty_batch_is_noop(tmp_path):
    # _upsert_many([]) returns early without creating a table (line 350).
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.upsert_segments([])
    b.upsert_insights([])
    b.upsert_auxes([])
    # No table was ever created -> health reports all zero, no on-disk tables.
    assert b._open_if_exists("segment") is None
    assert b.health() == {"status": "ok", "segments": 0, "insights": 0, "aux": 0}


def test_search_applies_where_filter(tmp_path):
    # search_bm25 with filters exercises the _run where-clause branch. (search_hybrid was removed
    # for #1205; the where clause now only lives on the single-modality _run path.)
    b = LanceDBBackend(str(tmp_path / "lance"), embed_dim=4)
    b.upsert_segment(_seg("s1", "central bank policy", show="A", emb=[0.1, 0.2, 0.3, 0.4]))
    b.upsert_segment(_seg("s2", "central bank policy", show="B", emb=[0.1, 0.2, 0.3, 0.4]))
    b.create_indices()
    res = b.search_bm25(
        SearchQuery(
            text="central bank policy",
            embedding=[0.1, 0.2, 0.3, 0.4],
            tier="segment",
            filters={"show_id": "A"},
            k=10,
        )
    )
    ids = {r.doc_id for r in res}
    assert ids == {"s1"}  # show B filtered out by the where clause
    assert all(r.signal == "bm25" for r in res)


# --- stored_schema_version None-guards ---------------------------------------


def test_stored_schema_version_none_for_traversal_path(tmp_path):
    # safe_resolve_directory rejects '..' -> root_res None -> None (line 425).
    assert stored_schema_version(tmp_path / "a" / ".." / "b") is None


def test_stored_schema_version_none_when_meta_missing(tmp_path):
    # Existing dir, but no index_meta.json file -> isfile() guard -> None.
    d = tmp_path / "lance"
    d.mkdir()
    assert stored_schema_version(d) is None


def test_stored_schema_version_none_on_invalid_json(tmp_path):
    # Malformed JSON in meta -> try/except (ValueError) -> None (line 440-441).
    d = tmp_path / "lance"
    d.mkdir()
    (d / "index_meta.json").write_text("{broken", encoding="utf-8")
    assert stored_schema_version(d) is None


def test_stored_schema_version_none_when_meta_not_dict(tmp_path):
    # Valid JSON list (not a dict) -> isinstance guard -> None (line 443).
    d = tmp_path / "lance"
    d.mkdir()
    (d / "index_meta.json").write_text("[1, 2]", encoding="utf-8")
    assert stored_schema_version(d) is None


def test_stored_schema_version_defaults_to_one_without_key(tmp_path):
    # Meta present but no schema_version key -> pre-versioning index reports version 1.
    d = tmp_path / "lance"
    d.mkdir()
    (d / "index_meta.json").write_text(json.dumps({"embedding_model": "m"}), encoding="utf-8")
    assert stored_schema_version(d) == 1


def test_stored_schema_version_reads_recorded_version(tmp_path):
    # A real backend writes the current schema_version, which is read back verbatim.
    d = tmp_path / "lance"
    b = LanceDBBackend(str(d), embed_dim=4)
    b.write_index_meta("m")
    assert stored_schema_version(d) == LANCE_SCHEMA_VERSION


def test_stored_schema_version_none_when_relpath_verifier_rejects(tmp_path, monkeypatch):
    # safe_relpath_under_corpus_root falsy -> None (line 429).
    d = tmp_path / "lance"
    d.mkdir()
    (d / "index_meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lb, "safe_relpath_under_corpus_root", lambda *a, **k: None)
    assert stored_schema_version(d) is None


def test_stored_schema_version_none_when_normpath_guard_rejects(tmp_path, monkeypatch):
    # normpath_if_under_root falsy -> None (line 432).
    d = tmp_path / "lance"
    d.mkdir()
    (d / "index_meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lb, "normpath_if_under_root", lambda *a, **k: None)
    assert stored_schema_version(d) is None


def _seg(i, text="some text", show="A", emb=None):
    return SegmentDocument(
        id=i,
        text=text,
        show_id=show,
        episode_id="ep1",
        start_time=0.0,
        end_time=1.0,
        embedding=emb if emb is not None else [0.1, 0.2, 0.3, 0.4],
    )
