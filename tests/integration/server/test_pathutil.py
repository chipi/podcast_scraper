"""Tests for server corpus path anchoring (CodeQL path-injection mitigation).

Must not import FastAPI: unit jobs may run with only ``.[dev]`` for import checks,
and pathutil is HTTP-framework agnostic (errors map to responses in route code).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.pathutil import CorpusPathRequestError, resolve_corpus_path_param

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def test_rejects_override_when_no_anchor(tmp_path: Path) -> None:
    with pytest.raises(CorpusPathRequestError) as exc_info:
        resolve_corpus_path_param(str(tmp_path), None)
    assert exc_info.value.status_code == 400


def test_accepts_exact_anchor(tmp_path: Path) -> None:
    out = resolve_corpus_path_param(str(tmp_path), tmp_path)
    assert out == tmp_path.resolve()


def test_accepts_subdirectory_of_anchor(tmp_path: Path) -> None:
    sub = tmp_path / "metadata"
    sub.mkdir()
    out = resolve_corpus_path_param(str(sub), tmp_path)
    assert out == sub.resolve()


def test_rejects_empty_path_param(tmp_path: Path) -> None:
    with pytest.raises(CorpusPathRequestError) as exc_info:
        resolve_corpus_path_param("   ", tmp_path)
    assert exc_info.value.status_code == 400
    assert "non-empty" in (exc_info.value.detail or "")


def test_rejects_path_outside_anchor(tmp_path: Path) -> None:
    anchor = tmp_path / "allowed"
    anchor.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(CorpusPathRequestError) as exc_info:
        resolve_corpus_path_param(str(outside), anchor)
    assert exc_info.value.status_code == 400
