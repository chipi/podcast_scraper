"""Unit tests for migration 0001 verify branches (#858/#862)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.faiss_store import VECTORS_FILE
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0001_faiss_to_lance import FaissToLanceMigration

pytestmark = pytest.mark.unit


def _ctx(tmp_path):
    return MigrationContext(corpus_root=tmp_path)


def test_verify_noop_when_no_faiss(tmp_path):
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))
    assert ok and "no source" in msg  # nothing to migrate → vacuously ok


def test_verify_fails_when_faiss_present_but_lance_missing(tmp_path):
    (tmp_path / "search").mkdir()
    (tmp_path / "search" / VECTORS_FILE).write_text("", encoding="utf-8")  # FAISS exists
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))  # but no lance_index
    assert not ok and "missing" in msg


def test_plan_noop_string_when_no_faiss(tmp_path):
    assert "nothing to migrate" in FaissToLanceMigration().plan(_ctx(tmp_path))
