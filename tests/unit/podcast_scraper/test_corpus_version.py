"""Unit tests for corpus/code version contract (#796)."""

from __future__ import annotations

from podcast_scraper.corpus_version import (
    assess_corpus_version_compat,
    build_produced_by,
    MIN_SUPPORTED_CORPUS_CODE_VERSION,
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
