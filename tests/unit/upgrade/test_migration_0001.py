"""Unit tests for migration 0001 — a recorded no-op since FAISS was retired (#995)."""

from __future__ import annotations

import pytest

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0001_faiss_to_lance import FaissToLanceMigration

pytestmark = pytest.mark.unit


def _ctx(tmp_path):
    return MigrationContext(corpus_root=tmp_path)


def test_verify_is_vacuous_noop(tmp_path):
    ok, msg = FaissToLanceMigration().verify(_ctx(tmp_path))
    assert ok and "no-op" in msg


def test_plan_is_noop(tmp_path):
    assert "nothing to migrate" in FaissToLanceMigration().plan(_ctx(tmp_path))


def test_apply_records_noop(tmp_path):
    res = FaissToLanceMigration().apply(_ctx(tmp_path))
    assert res.applied
