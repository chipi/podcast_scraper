"""Unit tests for corpus manifest reads under anchor (health preflight)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.pathutil import read_manifest_produced_by_under_anchor

pytestmark = [pytest.mark.unit]


def test_read_manifest_under_anchor_returns_produced_by(tmp_path: Path) -> None:
    anchor = tmp_path / "corpus"
    anchor.mkdir()
    manifest = {
        "produced_by": {
            "code_version": "2.6.0",
            "git_sha": "abc1234",
            "produced_at": "2026-05-23T12:00:00Z",
        }
    }
    (anchor / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    parsed = read_manifest_produced_by_under_anchor(anchor, anchor)
    assert parsed == manifest["produced_by"]


def test_read_manifest_rejects_path_outside_anchor(tmp_path: Path) -> None:
    anchor = tmp_path / "corpus"
    anchor.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "corpus_manifest.json").write_text("{}", encoding="utf-8")
    assert read_manifest_produced_by_under_anchor(outside, anchor) is None


def test_read_manifest_returns_none_when_file_missing(tmp_path: Path) -> None:
    anchor = tmp_path / "corpus"
    anchor.mkdir()
    assert read_manifest_produced_by_under_anchor(anchor, anchor) is None
