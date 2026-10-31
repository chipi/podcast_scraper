"""Reading one large JSON object entry by entry, in bounded memory (D2 item 6)."""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest

from podcast_scraper.utils.json_stream import iter_json_object_items

pytestmark = pytest.mark.unit


def _write(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


@pytest.mark.parametrize("chunk", [1, 7, 64, 1 << 20])
def test_every_entry_comes_back_whatever_the_chunk_size(tmp_path: Path, chunk: int) -> None:
    obj = {
        "a": {"x": 1, "t": 'é, ünïcode, "quoted", {braces}'},
        "b": [1, 2.5, None, True, False],
        "c": 1234567890,
        "d": "plain",
        "e": {},
        "fé": -0.25e3,
    }
    got = dict(iter_json_object_items(_write(tmp_path / "m.json", obj), chunk_size=chunk))
    assert got == obj


def test_only_keeps_the_keys_asked_for(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.json", {f"k{i}": {"i": i} for i in range(100)})
    assert dict(iter_json_object_items(path, keys={"k3", "k99", "missing"})) == {
        "k3": {"i": 3},
        "k99": {"i": 99},
    }


@pytest.mark.parametrize("text", ['{"a": 1', '["not", "an", "object"]', '{"a" 1}', ""])
def test_a_broken_or_non_object_file_is_an_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "m.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_json_object_items(path))


def _peak(path: Path) -> int:
    tracemalloc.start()
    try:
        for _ in iter_json_object_items(path, keys={"row-0"}, chunk_size=1 << 16):
            pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_memory_does_not_grow_with_the_file(tmp_path: Path) -> None:
    """The operator's requirement: verify's peak must not grow with corpus size (N against 4N)."""
    entry = {"char_start": 12, "char_end": 340, "source_metadata_relative_path": "feeds/x/m.json"}
    small = _write(tmp_path / "n.json", {f"row-{i}": dict(entry, i=i) for i in range(5_000)})
    large = _write(tmp_path / "4n.json", {f"row-{i}": dict(entry, i=i) for i in range(20_000)})
    p_small, p_large = _peak(small), _peak(large)
    assert large.stat().st_size > 3.5 * small.stat().st_size
    assert p_large < 1.5 * p_small + 64 * 1024, (p_small, p_large)
