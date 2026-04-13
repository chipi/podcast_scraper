"""Integration tests for in-process index rebuild mutex (GitHub #507 follow-up)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server.app import create_app
from podcast_scraper.server.index_rebuild import CorpusRebuildGate, gate_for_corpus

pytestmark = [pytest.mark.integration]


def test_corpus_rebuild_gate_try_begin_excludes_second() -> None:
    g = CorpusRebuildGate()
    assert g.try_begin() is True
    assert g.try_begin() is False
    assert g.snapshot() == (True, None)
    g.end("oops")
    assert g.snapshot() == (False, "oops")
    assert g.try_begin() is True
    g.end(None)
    assert g.snapshot() == (False, None)


def test_gate_for_corpus_same_resolved_path(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    a = tmp_path / "c"
    a.mkdir()
    g1 = gate_for_corpus(app, a)
    g2 = gate_for_corpus(app, a)
    assert g1 is g2
