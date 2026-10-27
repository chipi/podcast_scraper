"""The interest index is read from its own small sidecar, never from ``search/metadata.json``.

Production, 2026-10-26/27: ``search/metadata.json`` is 500 MB (every summary / transcript / KG
row of the two-tier index). ``interest_episode_index`` parsed all of it with one ``json.loads``
on the first home load of any user with interests after a restart or an index rebuild. That
single call holds the GIL while it allocates several GB, so the player api's memory watchdog
never ran, and the kernel killed the process at its 6 GB cap three times (two of them an
onboarding user saving interests). The index needs only the KG rows' token -> episode pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search import interest_index
from podcast_scraper.server import app_discover_view


def _write_sidecars(root: Path, metadata: dict) -> None:
    search = root / "search"
    search.mkdir(parents=True, exist_ok=True)
    (search / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    interest_index.write_interest_index(search, interest_index.build_interest_index(metadata))


METADATA = {
    "kg:a": {
        "doc_type": "kg_topic",
        "source_id": "topic:ai",
        "source_metadata_relative_path": "feeds/f/run_1/metadata/e1.metadata.json",
    },
    "kg:b": {
        "doc_type": "kg_entity",
        "source_id": "person:ada",
        "source_metadata_relative_path": "feeds/f/run_1/metadata/e2.metadata.json",
    },
    "kg:c": {
        "doc_type": "kg_topic",
        "source_id": "topic:ai",
        "source_metadata_relative_path": "feeds/f/run_1/metadata/e2.metadata.json",
    },
    "bullet:x": {
        "doc_type": "summary",
        "source_id": "0",
        "source_metadata_relative_path": "feeds/f/run_1/metadata/e3.metadata.json",
    },
}


@pytest.fixture(autouse=True)
def _clear_cache():
    app_discover_view._INTEREST_INDEX_CACHE.clear()
    yield
    app_discover_view._INTEREST_INDEX_CACHE.clear()


def test_build_keeps_only_kg_tokens():
    built = interest_index.build_interest_index(METADATA)
    assert built == {
        "person:ada": ["feeds/f/run_1/metadata/e2.metadata.json"],
        "topic:ai": [
            "feeds/f/run_1/metadata/e1.metadata.json",
            "feeds/f/run_1/metadata/e2.metadata.json",
        ],
    }


def test_reader_answers_from_the_sidecar(tmp_path: Path):
    _write_sidecars(tmp_path, METADATA)
    index = app_discover_view.interest_episode_index(tmp_path)
    assert index == {
        "person:ada": {"feeds/f/run_1/metadata/e2.metadata.json"},
        "topic:ai": {
            "feeds/f/run_1/metadata/e1.metadata.json",
            "feeds/f/run_1/metadata/e2.metadata.json",
        },
    }


def test_reader_never_parses_search_metadata_json(tmp_path: Path, monkeypatch):
    """The regression guard: reading the big sidecar is what killed the process on prod."""
    _write_sidecars(tmp_path, METADATA)
    big = (tmp_path / "search" / "metadata.json").resolve()
    real_read_text = Path.read_text
    real_open = open

    def guarded_read_text(self: Path, *a, **k):
        if self.resolve() == big:
            raise AssertionError("interest_episode_index parsed search/metadata.json")
        return real_read_text(self, *a, **k)

    def guarded_open(file, *a, **k):
        if Path(str(file)).resolve() == big:
            raise AssertionError("interest_episode_index opened search/metadata.json")
        return real_open(file, *a, **k)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr("builtins.open", guarded_open)
    assert app_discover_view.interest_episode_index(tmp_path)["topic:ai"]


def test_missing_sidecar_is_empty_not_a_fallback_to_the_big_file(tmp_path: Path):
    """Without the small sidecar the answer is ``{}`` (recency fallback), as for a corpus with no
    index; it must not fall back to parsing ``metadata.json``."""
    search = tmp_path / "search"
    search.mkdir()
    (search / "metadata.json").write_text(json.dumps(METADATA), encoding="utf-8")
    assert app_discover_view.interest_episode_index(tmp_path) == {}


def test_cache_follows_the_sidecar(tmp_path: Path):
    _write_sidecars(tmp_path, METADATA)
    assert set(app_discover_view.interest_episode_index(tmp_path)) == {"topic:ai", "person:ada"}
    smaller = {k: v for k, v in METADATA.items() if k != "kg:b"}
    interest_index.write_interest_index(
        tmp_path / "search", interest_index.build_interest_index(smaller)
    )
    # A rewrite must be seen even within the same mtime second (size changes too).
    assert set(app_discover_view.interest_episode_index(tmp_path)) == {"topic:ai"}


def test_write_is_atomic_and_rebuild_from_metadata(tmp_path: Path):
    search = tmp_path / "search"
    search.mkdir()
    (search / "metadata.json").write_text(json.dumps(METADATA), encoding="utf-8")
    written = interest_index.rebuild_from_metadata(search)
    assert written == 2
    on_disk = json.loads((search / interest_index.INTEREST_INDEX_FILE).read_text("utf-8"))
    assert on_disk["topic:ai"] == [
        "feeds/f/run_1/metadata/e1.metadata.json",
        "feeds/f/run_1/metadata/e2.metadata.json",
    ]
    assert not list(search.glob("*.tmp"))
