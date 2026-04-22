"""viewer_operator_extras_source for Docker job mode (#660)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import operator_paths as op


class _App:
    def __init__(self, fixed: Path | None) -> None:
        self.state = type("S", (), {})()
        self.state.operator_config_fixed_path = fixed


def test_extras_source_prefers_corpus_file_in_docker_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    per = corpus / op.VIEWER_OPERATOR_BASENAME
    per.write_text("pipeline_install_extras: ml\n", encoding="utf-8")
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text("rss: x\n", encoding="utf-8")
    app = _App(fixed)
    assert op.viewer_operator_extras_source(app, corpus) == per


def test_extras_source_falls_back_to_fixed_when_no_corpus_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PODCAST_PIPELINE_EXEC_MODE", "docker")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text("pipeline_install_extras: llm\n", encoding="utf-8")
    app = _App(fixed)
    assert op.viewer_operator_extras_source(app, corpus) == fixed


def test_extras_source_native_ignores_missing_corpus_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("PODCAST_PIPELINE_EXEC_MODE", raising=False)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    fixed = tmp_path / "fixed.yaml"
    fixed.write_text("k: v\n", encoding="utf-8")
    app = _App(fixed)
    assert op.viewer_operator_extras_source(app, corpus) == fixed
