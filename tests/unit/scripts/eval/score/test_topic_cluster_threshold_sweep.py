"""Guards for the topic-cluster threshold sweep harness.

The sweep rebuilds topic vectors locally instead of reading the production LanceDB index, so its
whole value rests on one thing: the text it embeds must be the text production embeds. If
``search/indexer.py`` ever changes how a Topic node becomes embeddable text and this harness does
not, the sweep keeps producing confident numbers that describe a corpus nobody has.

That is the failure this file exists to make loud — the rest is ordinary parsing cover.
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
    / "topic_cluster_threshold_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("topic_cluster_threshold_sweep", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep = _load_module()


# --- the load-bearing one -------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,desc",
    [
        ("ai safety", ""),
        ("ai safety", "alignment of frontier models"),
        ("", "description only"),
        ("  padded  ", "  also padded  "),
        ("", ""),
    ],
)
def test_embed_text_matches_production_indexer(label: str, desc: str) -> None:
    """The harness must build the SAME string ``search/indexer.py`` embeds.

    Compared against the real function rather than a copied expectation, so a change to
    production's rule fails here instead of silently skewing a threshold recommendation.
    """
    from podcast_scraper.search.indexer import _kg_embed_text_topic

    props = {}
    if label:
        props["label"] = label
    if desc:
        props["description"] = desc

    expected = _kg_embed_text_topic(props) or ""
    assert sweep._embed_text(label, desc) == expected


def test_embed_text_joins_label_and_description_in_that_order() -> None:
    assert sweep._embed_text("topic", "detail") == "topic detail"


# --- feed attribution -----------------------------------------------------------------------
# x-feed reach is the metric the sweep is scored on, so mis-deriving the feed would silently
# rewrite the recommendation.


def test_feed_of_extracts_the_feed_directory() -> None:
    rel = "feeds/rss_example_abc123/run_deadbeef_20260902/metadata/0001 - Ep.metadata.json"
    assert sweep.feed_of(rel) == "rss_example_abc123"


@pytest.mark.parametrize("rel", ["", "not-a-corpus-path", "feeds", "feeds/only-two"])
def test_feed_of_degrades_to_the_input_when_not_a_corpus_path(rel: str) -> None:
    """Never raise, and never silently collapse unrelated episodes into one feed."""
    assert sweep.feed_of(rel) == rel


def test_two_episodes_of_the_same_feed_share_a_feed_key() -> None:
    a = "feeds/rss_x_1/run_a/metadata/1.json"
    b = "feeds/rss_x_1/run_b/metadata/2.json"
    assert sweep.feed_of(a) == sweep.feed_of(b)


# --- parsing / aggregation ------------------------------------------------------------------


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "topics.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_rows_aggregates_occurrences_per_topic(tmp_path: Path) -> None:
    """One topic seen in two episodes yields two occurrences and two episode ids.

    Production averages the per-occurrence vectors, so dropping duplicates here would change
    the vector and therefore the clustering.
    """
    src = _write(
        tmp_path,
        [
            {"ep": "feeds/f1/run_a/metadata/1.json", "topics": [{"id": "topic:ai", "label": "ai"}]},
            {"ep": "feeds/f2/run_b/metadata/2.json", "topics": [{"id": "topic:ai", "label": "ai"}]},
        ],
    )
    ids, texts, episodes = sweep.load_rows(src)
    assert ids == ["topic:ai"]
    assert len(texts["topic:ai"]) == 2
    assert len(set(episodes["topic:ai"])) == 2


def test_load_rows_skips_unusable_rows(tmp_path: Path) -> None:
    src = _write(
        tmp_path,
        [
            {"ep": "feeds/f1/run_a/metadata/1.json", "topics": [{"id": "topic:ok", "label": "ok"}]},
            {"ep": "feeds/f1/run_a/metadata/2.json", "topics": [{"id": "", "label": "no id"}]},
            {
                "ep": "feeds/f1/run_a/metadata/3.json",
                "topics": [{"id": "topic:blank", "label": ""}],
            },
        ],
    )
    ids, _texts, _eps = sweep.load_rows(src)
    assert ids == ["topic:ok"], "an empty id or an empty embed text must not become a topic"


def test_load_rows_survives_a_malformed_line(tmp_path: Path) -> None:
    p = tmp_path / "topics.jsonl"
    p.write_text(
        '{"ep": "feeds/f1/run_a/metadata/1.json", "topics": [{"id": "topic:a", "label": "a"}]}\n'
        "{ this is not json\n"
        "\n"
        '{"ep": "feeds/f1/run_a/metadata/2.json", "topics": [{"id": "topic:b", "label": "b"}]}\n',
        encoding="utf-8",
    )
    ids, _texts, _eps = sweep.load_rows(p)
    assert ids == ["topic:a", "topic:b"], "one bad line must not discard the rest of the corpus"


def test_load_rows_returns_ids_sorted(tmp_path: Path) -> None:
    """Vector row order must be deterministic — the similarity matrix is indexed by position."""
    src = _write(
        tmp_path,
        [
            {
                "ep": "feeds/f1/run_a/metadata/1.json",
                "topics": [
                    {"id": "topic:z", "label": "z"},
                    {"id": "topic:a", "label": "a"},
                    {"id": "topic:m", "label": "m"},
                ],
            }
        ],
    )
    ids, _texts, _eps = sweep.load_rows(src)
    assert ids == sorted(ids)
