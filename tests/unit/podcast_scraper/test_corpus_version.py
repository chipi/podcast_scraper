"""Unit tests for corpus/code version contract (#796)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from podcast_scraper.corpus_version import (
    assess_corpus_version_compat,
    build_produced_by,
    corpus_code_version,
    MIN_SUPPORTED_CORPUS_CODE_VERSION,
    parse_produced_by_from_manifest_doc,
    read_produced_by,
    resolve_git_commit_sha,
)


def test_build_produced_by_has_required_keys() -> None:
    stamp = build_produced_by(produced_at="2026-05-23T12:00:00Z")
    assert stamp["code_version"]
    assert stamp["git_sha"]
    assert stamp["produced_at"] == "2026-05-23T12:00:00Z"


def test_assess_warns_when_produced_by_missing() -> None:
    ver, warning = assess_corpus_version_compat(None)
    assert ver is None
    assert warning is not None
    assert MIN_SUPPORTED_CORPUS_CODE_VERSION in warning


def test_assess_ok_for_current_version() -> None:
    produced_by = {
        "code_version": MIN_SUPPORTED_CORPUS_CODE_VERSION,
        "git_sha": "abc1234",
        "produced_at": "2026-05-23T12:00:00Z",
    }
    ver, warning = assess_corpus_version_compat(produced_by)
    assert ver == MIN_SUPPORTED_CORPUS_CODE_VERSION
    assert warning is None


def test_assess_warns_when_corpus_below_minimum() -> None:
    produced_by = {
        "code_version": "2.5.0",
        "git_sha": "abc1234",
        "produced_at": "2026-05-23T12:00:00Z",
    }
    ver, warning = assess_corpus_version_compat(produced_by)
    assert ver == "2.5.0"
    assert warning is not None
    assert "2.5.0" in warning


def test_assess_warns_on_invalid_semver() -> None:
    produced_by = {
        "code_version": "not-a-version",
        "git_sha": "abc1234",
        "produced_at": "2026-05-23T12:00:00Z",
    }
    ver, warning = assess_corpus_version_compat(produced_by)
    assert ver == "not-a-version"
    assert warning is not None
    assert "not a valid semver" in warning


def test_parse_produced_by_from_manifest_doc_legacy_tool_version() -> None:
    doc = {"tool_version": "2.5.0", "updated_at": "2026-01-01T00:00:00Z"}
    parsed = parse_produced_by_from_manifest_doc(doc)
    assert parsed == {
        "code_version": "2.5.0",
        "git_sha": "unknown",
        "produced_at": "2026-01-01T00:00:00Z",
    }


def test_parse_produced_by_from_manifest_doc_rejects_non_object() -> None:
    assert parse_produced_by_from_manifest_doc([]) is None
    assert parse_produced_by_from_manifest_doc({"tool_version": "  "}) is None


def test_read_produced_by_returns_none_when_manifest_missing(tmp_path: Path) -> None:
    assert read_produced_by(tmp_path) is None


def test_read_produced_by_reads_produced_by_stamp(tmp_path: Path) -> None:
    manifest = {
        "produced_by": {
            "code_version": "2.6.0",
            "git_sha": "deadbeef",
            "produced_at": "2026-05-23T12:00:00Z",
        }
    }
    (tmp_path / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    parsed = read_produced_by(tmp_path)
    assert parsed == manifest["produced_by"]


def test_corpus_code_version_returns_none_for_empty_string() -> None:
    assert corpus_code_version({"code_version": "   "}) is None


@patch("podcast_scraper.corpus_version.subprocess.run")
def test_resolve_git_commit_sha_success(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=0, stdout="abc123def456\n")
    assert resolve_git_commit_sha() == "abc123def456"


@patch("podcast_scraper.corpus_version.subprocess.run")
def test_resolve_git_commit_sha_failure(mock_run: MagicMock) -> None:
    mock_run.return_value = MagicMock(returncode=1, stdout="")
    assert resolve_git_commit_sha() == "unknown"
