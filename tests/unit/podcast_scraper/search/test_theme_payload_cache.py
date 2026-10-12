"""The theme artifact is parsed once per version of the file (review N1, 2026-10-11).

``theme_map_by_topic`` re-read and re-parsed ``search/topic_clusters.json`` on every call, while the
storyline artifact was cached on its own mtime. A short listen on a theme and every Home load with a
saved theme reel paid for the parse. It is cached the same way now, and a rewrite still shows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.search import topic_clusters


def _write(root: Path, label: str, mtime: float) -> None:
    path = root / "search" / "topic_clusters.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "clusters": [
                    {
                        "graph_compound_parent_id": "tc:safety-practices",
                        "canonical_label": label,
                        "members": [
                            {"topic_id": "topic:safety-practices", "label": "safety practices"},
                            {"topic_id": "topic:dive-planning", "label": "dive planning"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    os.utime(path, (mtime, mtime))


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_the_theme_artifact_is_parsed_once_per_version(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path, "Safety practices", 1_000_000.0)
    reads: list[str] = []
    real = topic_clusters._read_theme_payload

    def counting(joined: str):
        reads.append(joined)
        return real(joined)

    monkeypatch.setattr(topic_clusters, "_read_theme_payload", counting)

    first = topic_clusters.theme_map_by_topic(tmp_path)
    second = topic_clusters.theme_map_by_topic(tmp_path)
    assert first == second
    assert first["topic:safety-practices"]["cluster_id"] == "tc:safety-practices"
    assert len(reads) == 1, "parsed again although the file had not changed"

    # A rewrite (new mtime) is read, not served from the cache.
    _write(tmp_path, "Safety habits", 1_000_100.0)
    third = topic_clusters.theme_map_by_topic(tmp_path)
    assert third["topic:safety-practices"]["cluster_label"] == "Safety habits"
    assert len(reads) == 2
