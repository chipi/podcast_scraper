"""Unit tests for :mod:`podcast_scraper.server.app_corpus_access`.

The two filesystem helpers the consumer ``/api/app/*`` routes share: resolve the corpus
root (or 503) and path-safely load a JSON artifact (or ``None`` on missing/unreadable/escape).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503, load_json_artifact

pytestmark = [pytest.mark.integration]


def _request(output_dir: object) -> Request:
    """A minimal stand-in for a Starlette Request exposing ``app.state.output_dir``."""
    return cast(
        Request, SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(output_dir=output_dir)))
    )


def test_corpus_root_resolves_configured_dir(tmp_path: Path) -> None:
    root = corpus_root_or_503(_request(tmp_path))
    assert root == Path(tmp_path)


def test_corpus_root_503_when_unconfigured() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        corpus_root_or_503(_request(None))
    assert exc.value.status_code == 503


def test_load_json_artifact_reads_dict(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "x.kg.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_json_artifact(tmp_path, "metadata/x.kg.json") == {"a": 1}


def test_load_json_artifact_blank_relpath_is_none(tmp_path: Path) -> None:
    assert load_json_artifact(tmp_path, "") is None


def test_load_json_artifact_escaping_relpath_is_none(tmp_path: Path) -> None:
    # A path-traversal attempt fails the safety check → None.
    assert load_json_artifact(tmp_path, "../../etc/passwd") is None


def test_load_json_artifact_missing_file_is_none(tmp_path: Path) -> None:
    assert load_json_artifact(tmp_path, "metadata/absent.json") is None


def test_load_json_artifact_unreadable_json_is_none(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "bad.json").write_text("{not json", encoding="utf-8")
    assert load_json_artifact(tmp_path, "metadata/bad.json") is None


def test_load_json_artifact_non_dict_top_level_is_none(tmp_path: Path) -> None:
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_json_artifact(tmp_path, "metadata/list.json") is None
