"""The hosted-photo map is built once per ``person_web.json`` and still sees it rewritten.

An enrichment run rewrites the file and moves none of the corpus stamps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import relational_view

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean():
    perf_cache.clear()
    yield
    perf_cache.clear()


def _write_person_web(root: Path, persons: list[dict]) -> Path:
    path = root / "enrichments" / "person_web.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": {"persons": persons}}), encoding="utf-8")
    return path


def test_the_photo_map_follows_person_web_json_and_is_a_copy(tmp_path: Path) -> None:
    path = _write_person_web(tmp_path, [{"person_id": "person:ann", "image_hosted": True}])
    first = relational_view.hosted_photo_urls(tmp_path)
    assert set(first) == {"person:ann"}
    first["person:mallory"] = "/x"  # a caller's edit must not reach the next reader
    assert set(relational_view.hosted_photo_urls(tmp_path)) == {"person:ann"}

    # An enrichment run rewrites the file; the run summary and manifest do not move.
    _write_person_web(
        tmp_path,
        [
            {"person_id": "person:ann", "image_hosted": True},
            {"person_id": "person:bo", "image_hosted": True},
        ],
    )
    os.utime(path, (path.stat().st_atime + 5, path.stat().st_mtime + 5))
    assert set(relational_view.hosted_photo_urls(tmp_path)) == {"person:ann", "person:bo"}

    path.unlink()
    assert relational_view.hosted_photo_urls(tmp_path) == {}


def test_the_photo_map_is_not_rebuilt_while_the_file_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_person_web(tmp_path, [{"person_id": "person:ann", "image_hosted": True}])
    built: list[int] = []
    real = relational_view._hosted_photos

    def _counting(root: Any) -> Any:
        built.append(1)
        return real(root)

    monkeypatch.setattr(relational_view, "_hosted_photos", _counting)
    for _ in range(3):
        relational_view.hosted_photo_urls(tmp_path)
    assert built == [1]
