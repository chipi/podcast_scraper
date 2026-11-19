"""The SPA served by ``serve --output-dir`` starts with that corpus path filled in.

nginx injects ``window.__PODCAST_DEFAULT_CORPUS_PATH__`` (and seeds the saved path) for the
deployed Workbench (docker/workbench/default.conf.template). A server that serves the built app
itself injected nothing, so the Workbench opened on "Set Corpus path in the status bar below" even
though the server was started with ``--output-dir`` (public-only review, 2026-11-17).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.unit]

_INDEX_HTML = "<!doctype html><html><head><title>Workbench</title></head><body></body></html>"


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_dir: Path | None) -> TestClient:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "appdata"))
    return TestClient(create_app(output_dir=output_dir, static_dir=static))


@pytest.mark.parametrize("path", ["/", "/index.html", "/library"])
def test_the_served_document_names_the_corpus(
    path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus <x>"
    corpus.mkdir()
    body = _client(tmp_path, monkeypatch, corpus).get(path).text
    expected = json.dumps(str(corpus.resolve())).replace("<", "\\u003c")
    assert f"window.__PODCAST_DEFAULT_CORPUS_PATH__={expected}" in body
    assert "corpus <x>" not in body  # "<" is escaped, so a path cannot close the script early
    assert body.index("__PODCAST_DEFAULT_CORPUS_PATH__") < body.index("</head>")


def test_no_output_dir_injects_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = _client(tmp_path, monkeypatch, None).get("/").text
    assert "__PODCAST_DEFAULT_CORPUS_PATH__" not in body


@pytest.mark.parametrize("mode", ["PODCAST_SERVE_APP_ONLY", "PODCAST_SERVE_OPERATOR_PUBLIC"])
def test_a_public_mode_does_not_publish_the_corpus_path(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    monkeypatch.setenv(mode, "1")
    monkeypatch.setenv("APP_SESSION_SECRET", "test-secret")
    body = _client(tmp_path, monkeypatch, corpus).get("/").text
    assert "__PODCAST_DEFAULT_CORPUS_PATH__" not in body
    assert str(corpus) not in body
