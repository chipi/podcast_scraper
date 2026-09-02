"""Guards for the theme-cluster sweep harness.

Two things here are worth protecting. The harness must rebuild the enricher's lift-edge set the
same way the enricher does — otherwise it sweeps a graph production never had. And it must refuse
to print a table when ``n`` exceeds ``_MAX_LINKAGE_TOPICS``, because past that point the enricher
silently returns all-singletons and every row of the sweep would read "zero themes" as if the
threshold caused it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[5]
    / "scripts"
    / "eval"
    / "score"
    / "theme_cluster_threshold_sweep.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("theme_cluster_threshold_sweep", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load()


# --- feed attribution: x_feed is what the sweep is scored on ---------------------------------


def test_feed_of_extracts_the_feed_directory() -> None:
    rel = "feeds/rss_example_abc123/run_dead/metadata/0001 - Ep.metadata.json"
    assert sweep.feed_of(rel) == "rss_example_abc123"


@pytest.mark.parametrize("rel", ["", "nope", "feeds", "feeds/two"])
def test_feed_of_degrades_to_input(rel: str) -> None:
    assert sweep.feed_of(rel) == rel


# --- topic -> episode mapping ----------------------------------------------------------------


def _jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "topics.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_topic_episodes_collects_distinct_episodes(tmp_path: Path) -> None:
    src = _jsonl(
        tmp_path,
        [
            {"ep": "feeds/f1/run_a/metadata/1.json", "topics": [{"id": "topic:a"}]},
            {"ep": "feeds/f2/run_b/metadata/2.json", "topics": [{"id": "topic:a"}]},
            {"ep": "feeds/f2/run_b/metadata/2.json", "topics": [{"id": "topic:b"}]},
        ],
    )
    got = sweep.load_topic_episodes(src)
    assert len(got["topic:a"]) == 2
    assert len(got["topic:b"]) == 1


def test_load_topic_episodes_ignores_rows_without_an_episode(tmp_path: Path) -> None:
    """A topic with no episode contributes nothing to reach and must not fake a span."""
    src = _jsonl(tmp_path, [{"ep": "", "topics": [{"id": "topic:a"}]}])
    assert sweep.load_topic_episodes(src).get("topic:a", set()) == set()


def test_load_topic_episodes_survives_a_bad_line(tmp_path: Path) -> None:
    p = tmp_path / "topics.jsonl"
    p.write_text(
        '{"ep": "feeds/f1/run_a/metadata/1.json", "topics": [{"id": "topic:a"}]}\n'
        "not json at all\n"
        '{"ep": "feeds/f1/run_a/metadata/2.json", "topics": [{"id": "topic:b"}]}\n',
        encoding="utf-8",
    )
    got = sweep.load_topic_episodes(p)
    assert set(got) == {"topic:a", "topic:b"}


# --- the cliff -------------------------------------------------------------------------------
# ``min_pair_episode_count`` reads like a recall knob and is not one: it also decides how many
# topics enter the linkage, and above _MAX_LINKAGE_TOPICS the enricher returns all-singletons
# without erroring. Measured on the 1,066-episode corpus: min_pair=1 puts 9,344 topics in the
# subgraph against a cap of 400, so "loosening" the filter empties the surface entirely.


def test_max_linkage_topics_cap_is_what_the_enricher_uses() -> None:
    """The harness must guard on production's cap, not a copy that can drift."""
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import _MAX_LINKAGE_TOPICS

    assert isinstance(_MAX_LINKAGE_TOPICS, int) and _MAX_LINKAGE_TOPICS > 0


def test_average_linkage_returns_all_singletons_past_the_cap() -> None:
    """Pin the silent-degradation behaviour the harness exists to warn about.

    If this ever starts raising or clustering instead, the warning text in the harness (and the
    tuning advice built on it) is stale and must be revisited.
    """
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        _MAX_LINKAGE_TOPICS,
        _average_linkage,
    )

    n = _MAX_LINKAGE_TOPICS + 1

    def weight(i: int, j: int) -> float:
        return 100.0  # every pair wildly above any threshold

    clusters = _average_linkage(n, weight, 2.0)
    assert len(clusters) == n, "past the cap every topic must come back as its own singleton"
    assert all(len(c) == 1 for c in clusters)


def test_average_linkage_below_the_cap_still_merges() -> None:
    """The contrast case — so the test above is about the cap, not about linkage being broken."""
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import _average_linkage

    def weight(i: int, j: int) -> float:
        return 100.0

    clusters = _average_linkage(4, weight, 2.0)
    assert any(len(c) > 1 for c in clusters)
