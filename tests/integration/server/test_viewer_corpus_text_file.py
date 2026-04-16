"""GET /api/corpus/text-file — transcript-oriented files under corpus root."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_corpus_text_file_returns_plain_text(tmp_path: Path) -> None:
    sub = tmp_path / "feeds" / "show"
    sub.mkdir(parents=True)
    target = sub / "notes.txt"
    target.write_text("hello transcript", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "feeds/show/notes.txt"},
    )
    assert r.status_code == 200
    assert r.text == "hello transcript"
    assert "text/plain" in r.headers.get("content-type", "")


def test_corpus_text_file_rejects_bad_suffix(tmp_path: Path) -> None:
    p = tmp_path / "secret.exe"
    p.write_bytes(b"MZ")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "secret.exe"},
    )
    assert r.status_code == 400


def test_corpus_text_file_returns_json(tmp_path: Path) -> None:
    p = tmp_path / "transcript.json"
    p.write_text('{"segments":[]}', encoding="utf-8")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "transcript.json"},
    )
    assert r.status_code == 200
    assert r.text == '{"segments":[]}'
    assert "application/json" in r.headers.get("content-type", "")


def test_corpus_text_file_blocks_traversal(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "../outside.txt"},
    )
    assert r.status_code == 400


def test_corpus_text_file_missing_404(tmp_path: Path) -> None:
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "missing.txt"},
    )
    assert r.status_code == 404


def test_corpus_text_file_serves_cleaned_when_raw_txt_missing(tmp_path: Path) -> None:
    """GI/metadata often reference raw Whisper ``.txt`` while only ``.cleaned.txt`` exists on disk."""
    sub = tmp_path / "transcripts"
    sub.mkdir(parents=True)
    (sub / "ep.cleaned.txt").write_text("cleaned only", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "transcripts/ep.txt"},
    )
    assert r.status_code == 200
    assert r.text == "cleaned only"


def test_corpus_text_file_prefers_raw_txt_when_both_exist(tmp_path: Path) -> None:
    sub = tmp_path / "transcripts"
    sub.mkdir(parents=True)
    (sub / "ep.txt").write_text("raw", encoding="utf-8")
    (sub / "ep.cleaned.txt").write_text("cleaned", encoding="utf-8")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/text-file",
        params={"path": str(tmp_path), "relpath": "transcripts/ep.txt"},
    )
    assert r.status_code == 200
    assert r.text == "raw"
