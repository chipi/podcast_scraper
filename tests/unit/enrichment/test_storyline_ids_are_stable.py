"""#2345: a storyline keeps its id when it is re-clustered under a new label."""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.unit


def _kg(labels: dict[str, str]) -> dict:
    return {
        "nodes": [
            {"type": "Topic", "id": t, "properties": {"label": lb}} for t, lb in labels.items()
        ],
        "edges": [],
    }


def _storylines(tmp_path, labels: dict[str, str]) -> dict:
    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        TopicThemeClustersEnricher,
    )
    from tests.unit.enrichment.test_deterministic_enrichers import _bundle, _ctx, _run

    # Two topics on half the episodes and two others on the rest: each pair co-occurs above
    # chance (lift 2), so each forms a storyline.
    pair, other = dict(list(labels.items())[:2]), {"topic:c": "cedar", "topic:d": "dune"}
    bundles = [
        _bundle(tmp_path / "metadata", f"ep-{i}", kg=_kg(pair if i < 2 else other))
        for i in range(4)
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    out = tmp_path / "enrichments" / "topic_theme_clusters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"data": data}), encoding="utf-8")
    return data


def test_a_relabelled_storyline_keeps_its_id(tmp_path) -> None:
    first = _storylines(tmp_path, {"topic:a": "alpha", "topic:b": "beta"})
    ids = {
        frozenset(m["topic_id"] for m in c["members"]): c["graph_compound_parent_id"]
        for c in first["clusters"]
    }
    sid = ids[frozenset({"topic:a", "topic:b"})]
    assert first["id_history"] == {"successors": {}, "gone": []}

    # Same members, new labels: the canonical label (and so the slug) changes.
    second = _storylines(tmp_path, {"topic:a": "gamma", "topic:b": "delta"})
    ab = next(
        c
        for c in second["clusters"]
        if {m["topic_id"] for m in c["members"]} == {"topic:a", "topic:b"}
    )
    assert ab["graph_compound_parent_id"] == sid
    assert ab["canonical_label"] in {"gamma", "delta"}
    assert second["id_history"] == {"successors": {}, "gone": []}


def test_a_first_run_has_an_empty_history(tmp_path) -> None:
    data = _storylines(tmp_path, {"topic:a": "alpha", "topic:b": "beta"})
    assert data["id_history"] == {"successors": {}, "gone": []}
