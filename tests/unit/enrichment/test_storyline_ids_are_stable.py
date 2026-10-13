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


def _storylines(
    tmp_path, labels: dict[str, str], now: float = 1_000_000.0, monkeypatch=None
) -> dict:
    from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc
    from tests.unit.enrichment.test_deterministic_enrichers import _bundle, _ctx, _run

    if monkeypatch is not None:
        monkeypatch.setattr(ttc, "_now", lambda: now)
    # Two topics on half the episodes and two others on the rest: each pair co-occurs above
    # chance (lift 2), so each forms a storyline.
    pair, other = dict(list(labels.items())[:2]), {"topic:c": "cedar", "topic:d": "dune"}
    bundles = [
        _bundle(tmp_path / "metadata", f"ep-{i}", kg=_kg(pair if i < 2 else other))
        for i in range(4)
    ]
    data = _run(
        ttc.TopicThemeClustersEnricher(),
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


def _ab(data: dict) -> dict:
    return next(
        c
        for c in data["clusters"]
        if {m["topic_id"] for m in c["members"]} == {"topic:a", "topic:b"}
    )


DAY = 86400.0


def test_a_relabelled_storyline_keeps_its_id_and_its_name_until_the_new_label_holds(
    tmp_path, monkeypatch
) -> None:
    """RFC-128: the id never moves; the name changes only after another label has led the
    storyline's proposals for 60 days, and the old name stays as an alias."""
    first = _ab(_storylines(tmp_path, {"topic:a": "alpha", "topic:b": "beta"}, 0.0, monkeypatch))
    sid, name = first["graph_compound_parent_id"], first["canonical_label"]

    # Same members under new labels: the next day, nothing a listener sees changes.
    nxt = _ab(_storylines(tmp_path, {"topic:a": "gamma", "topic:b": "delta"}, DAY, monkeypatch))
    assert (nxt["graph_compound_parent_id"], nxt["canonical_label"]) == (sid, name)

    # 61 days of the new label leading: renamed, the old name kept.
    later = _ab(
        _storylines(tmp_path, {"topic:a": "gamma", "topic:b": "delta"}, 62 * DAY, monkeypatch)
    )
    assert later["graph_compound_parent_id"] == sid
    assert later["canonical_label"] in {"gamma", "delta"} and later["aliases"] == [name]


def test_a_first_run_seeds_the_registry_from_its_own_clusters(tmp_path, monkeypatch) -> None:
    data = _storylines(tmp_path, {"topic:a": "alpha", "topic:b": "beta"}, 0.0, monkeypatch)
    assert data["cluster_count"] == 2 and data["proposal_count"] == 2
    assert set(data["registry"]["entries"]) == {
        c["graph_compound_parent_id"] for c in data["clusters"]
    }
    assert data["id_history"] == {"successors": {}, "gone": []}
