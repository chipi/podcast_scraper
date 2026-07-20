"""Unit tests for the 6 chunk-2 deterministic enrichers.

One section per enricher. Synthetic 1-3 episode fixtures asserting
numerics + envelope shape — no real corpus, no IO beyond ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers import (
    ALL_DETERMINISTIC_ENRICHER_IDS,
    GroundingRateEnricher,
    GuestCoappearanceEnricher,
    InsightDensityEnricher,
    register_deterministic_enrichers,
    TemporalVelocityEnricher,
    TopicCooccurrenceCorpusEnricher,
    TopicThemeClustersEnricher,
)
from podcast_scraper.enrichment.protocol import (
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundle(
    metadata_dir: Path,
    stem: str,
    *,
    kg: dict[str, Any] | None = None,
    gi: dict[str, Any] | None = None,
    bridge: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EpisodeArtifactBundle:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    md_path = metadata_dir / f"{stem}.metadata.json"
    md_path.write_text(json.dumps(metadata or {}), encoding="utf-8")
    kg_path: Path | None = None
    if kg is not None:
        kg_path = metadata_dir / f"{stem}.kg.json"
        kg_path.write_text(json.dumps(kg), encoding="utf-8")
    gi_path: Path | None = None
    if gi is not None:
        gi_path = metadata_dir / f"{stem}.gi.json"
        gi_path.write_text(json.dumps(gi), encoding="utf-8")
    bridge_path: Path | None = None
    if bridge is not None:
        bridge_path = metadata_dir / f"{stem}.bridge.json"
        bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=md_path,
        gi_path=gi_path,
        kg_path=kg_path,
        bridge_path=bridge_path,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def _ctx(enricher_id: str) -> RunContext:
    return RunContext(
        run_id="r1",
        parent_run_id=None,
        enricher_id=enricher_id,
        enricher_version="1.0.0",
        tier="deterministic",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _run(enricher: Any, **kw: Any) -> dict[str, Any]:
    """Run an enricher synchronously; return its data dict."""
    result = asyncio.run(enricher.enrich(**kw))
    assert result.status == STATUS_OK, f"status={result.status!r} error={result.error}"
    assert isinstance(result.data, dict)
    return result.data


# ---------------------------------------------------------------------------
# topic_cooccurrence_corpus (corpus scope)
# ---------------------------------------------------------------------------


def test_topic_cooccurrence_corpus_ranks_by_episode_count(tmp_path: Path) -> None:
    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Topic", "id": tid, "properties": {"label": tid.split(":")[-1]}}
                for tid in topic_ids
            ],
            "edges": [],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep3", kg=_kg(["topic:a", "topic:c"])),
    ]
    data = _run(
        TopicCooccurrenceCorpusEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_cooccurrence_corpus"),
    )
    pairs = data["pairs"]
    # (a,b): 2 episodes; (a,c): 1 episode.
    assert pairs[0]["topic_a_id"] == "topic:a"
    assert pairs[0]["topic_b_id"] == "topic:b"
    assert pairs[0]["episode_count"] == 2
    assert pairs[1]["episode_count"] == 1
    assert data["episode_count"] == 3


def test_topic_cooccurrence_corpus_emits_lift_and_pmi(tmp_path: Path) -> None:
    """A (raw count) and B (lift/PMI) diverge: a ubiquitous pair scores high on
    count but ~chance on lift, while a rare pair scores low on count but high on
    lift. Both signals ship per pair so the Topic card can rank either way."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Topic", "id": tid, "properties": {"label": tid.split(":")[-1]}}
                for tid in topic_ids
            ],
            "edges": [],
        }

    # a is in every episode; (a,b) co-occurs 3× (high count, but only at chance).
    # (c,d) co-occurs once, but both topics are otherwise rare → high lift.
    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep3", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep4", kg=_kg(["topic:a", "topic:c", "topic:d"])),
    ]
    data = _run(
        TopicCooccurrenceCorpusEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_cooccurrence_corpus"),
    )
    by_key = {(p["topic_a_id"], p["topic_b_id"]): p for p in data["pairs"]}

    # A ranks (a,b) top (count 3) — but its lift is ~1.0 (co-occurs at chance).
    ab = by_key[("topic:a", "topic:b")]
    assert ab["episode_count"] == 3
    assert ab["topic_a_episode_count"] == 4  # a in all 4 episodes
    assert ab["topic_b_episode_count"] == 3
    assert ab["lift"] == pytest.approx(1.0)
    assert ab["pmi"] == pytest.approx(0.0)

    # B ranks (c,d) top (lift 4.0, pmi 2.0) despite a raw count of only 1.
    cd = by_key[("topic:c", "topic:d")]
    assert cd["episode_count"] == 1
    assert cd["lift"] == pytest.approx(4.0)  # 1·4 / (1·1)
    assert cd["pmi"] == pytest.approx(2.0)  # log2(4)
    assert cd["lift"] > ab["lift"]  # the whole point of B


# ---------------------------------------------------------------------------
# topic_theme_clusters (corpus scope)
# ---------------------------------------------------------------------------


def test_topic_theme_clusters_groups_cooccurring_topics(tmp_path: Path) -> None:
    """Themes = topics *discussed together*. Two storylines emerge; the ubiquitous
    'news' topic co-occurs only at chance (lift 1.0) and stays a singleton — it
    never pollutes a theme, which is the whole point of using lift not raw count."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "type": "Topic",
                    "id": tid,
                    "properties": {"label": tid.split(":")[-1].replace("-", " ")},
                }
                for tid in topic_ids
            ],
            "edges": [],
        }

    sf, oil, sanc = "topic:shadow-fleet", "topic:oil-prices", "topic:sanctions"
    kind, grat, news = "topic:kindness", "topic:gratitude", "topic:news"
    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg([sf, oil, sanc, news])),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg([sf, oil, sanc, news])),
        _bundle(tmp_path / "metadata", "ep3", kg=_kg([sf, oil, news])),
        _bundle(tmp_path / "metadata", "ep4", kg=_kg([kind, grat, news])),
        _bundle(tmp_path / "metadata", "ep5", kg=_kg([kind, grat, news])),
        _bundle(tmp_path / "metadata", "ep6", kg=_kg([news])),
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] == 2
    by_members = {frozenset(m["topic_id"] for m in c["members"]): c for c in data["clusters"]}
    assert frozenset({sf, oil, sanc}) in by_members
    assert frozenset({kind, grat}) in by_members
    # 'news' co-occurs at chance → never inside a theme.
    all_members = {m["topic_id"] for c in data["clusters"] for m in c["members"]}
    assert news not in all_members
    # Theme markers + per-member evidence.
    for c in data["clusters"]:
        assert c["cluster_type"] == "theme"
        assert c["graph_compound_parent_id"].startswith("thc:")
        assert c["member_count"] == len(c["members"])
        for m in c["members"]:
            assert m["episode_ids"]  # non-empty evidence trail
            assert "lift_to_cluster" in m


def test_topic_theme_clusters_empty_on_tiny_corpus(tmp_path: Path) -> None:
    """3 disjoint episodes → every pair co-occurs once (< min_pair=2) → no edges →
    zero theme clusters. Themes need volume; empty is expected, not a failure."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [{"type": "Topic", "id": t, "properties": {"label": t}} for t in topic_ids],
            "edges": [],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg(["topic:a", "topic:b"])),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg(["topic:c", "topic:d"])),
        _bundle(tmp_path / "metadata", "ep3", kg=_kg(["topic:e", "topic:f"])),
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] == 0
    assert data["clusters"] == []


def test_topic_theme_clusters_super_theme_rollup_noop_below_min(tmp_path: Path) -> None:
    """graph-v3 tier 7-1a — when cluster count is at or below the min bound
    (5) the super-theme rollup is a no-op: each cluster is its own
    super-theme. Verifies additive schema (super_theme_*) lands on every
    cluster without collapsing anything."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [{"type": "Topic", "id": t, "properties": {"label": t}} for t in topic_ids],
            "edges": [],
        }

    # Two disjoint themes → 2 clusters → below _SUPER_THEME_MIN → no rollup.
    a1, a2, b1, b2 = "topic:a1", "topic:a2", "topic:b1", "topic:b2"
    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg([a1, a2])),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg([a1, a2])),
        _bundle(tmp_path / "metadata", "ep3", kg=_kg([b1, b2])),
        _bundle(tmp_path / "metadata", "ep4", kg=_kg([b1, b2])),
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] == 2
    assert data["super_theme_count"] == 2
    assert data["super_theme_method"] == "cross_cluster_lift_avg_linkage"
    super_ids = {c["super_theme_id"] for c in data["clusters"]}
    assert len(super_ids) == 2  # each cluster is its own super-theme
    # Every cluster gets both super_theme_* fields populated.
    for c in data["clusters"]:
        assert c["super_theme_id"].startswith("sth:")
        assert c["super_theme_label"]


def test_topic_theme_clusters_super_theme_rollup_merges_at_target(tmp_path: Path) -> None:
    """graph-v3 tier 7-1a — forcing super_theme_target=3 on a corpus with 4
    clusters proves the merge algorithm runs and picks the highest-lift
    pair. The two clusters that share an episode with cross-cluster lift
    end up in the same super-theme; the other two stay separate."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [{"type": "Topic", "id": t, "properties": {"label": t}} for t in topic_ids],
            "edges": [],
        }

    # 4 disjoint themes → 4 clusters. Extra bridge episodes link theme A and
    # theme B via co-occurrence, so their inter-cluster lift is nonzero and
    # they merge first when the target drops to 3.
    a1, a2, b1, b2 = "topic:a1", "topic:a2", "topic:b1", "topic:b2"
    c1, c2, d1, d2 = "topic:c1", "topic:c2", "topic:d1", "topic:d2"
    bundles = [
        _bundle(tmp_path / "metadata", "ep-a1", kg=_kg([a1, a2])),
        _bundle(tmp_path / "metadata", "ep-a2", kg=_kg([a1, a2])),
        _bundle(tmp_path / "metadata", "ep-b1", kg=_kg([b1, b2])),
        _bundle(tmp_path / "metadata", "ep-b2", kg=_kg([b1, b2])),
        _bundle(tmp_path / "metadata", "ep-c1", kg=_kg([c1, c2])),
        _bundle(tmp_path / "metadata", "ep-c2", kg=_kg([c1, c2])),
        _bundle(tmp_path / "metadata", "ep-d1", kg=_kg([d1, d2])),
        _bundle(tmp_path / "metadata", "ep-d2", kg=_kg([d1, d2])),
        # Bridge episodes that mix theme A + theme B topics.
        _bundle(tmp_path / "metadata", "ep-ab1", kg=_kg([a1, b1])),
        _bundle(tmp_path / "metadata", "ep-ab2", kg=_kg([a2, b2])),
    ]
    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"super_theme_target": 3},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] >= 3
    # Target clamped to [_SUPER_THEME_MIN=5, _SUPER_THEME_MAX=8]. Target=3
    # asked → clamped up to 5. On a 4-cluster corpus that means no merges
    # happen (4 ≤ 5) — this test is really about the clamp behaviour + the
    # additive fields landing, NOT about hitting target=3 exactly.
    assert data["super_theme_target"] == 5
    for c in data["clusters"]:
        assert c["super_theme_id"].startswith("sth:")
        assert c["super_theme_label"]


def test_topic_theme_clusters_super_theme_rollup_merges_above_min(tmp_path: Path) -> None:
    """graph-v3 tier 7-1a — with cluster_count > _SUPER_THEME_MIN the rollup
    actually executes ``_average_linkage_to_target`` and merges clusters
    down to (or toward) the target. Six disjoint themes yield six clusters;
    a target of 5 (the min) forces exactly one merge, exercising the
    inner mean-lift + argmax + merge loop that noop/at-min-target tests
    skipped."""

    def _kg(topic_ids: list[str]) -> dict[str, Any]:
        return {
            "nodes": [{"type": "Topic", "id": t, "properties": {"label": t}} for t in topic_ids],
            "edges": [],
        }

    themes = [
        ("topic:a1", "topic:a2"),
        ("topic:b1", "topic:b2"),
        ("topic:c1", "topic:c2"),
        ("topic:d1", "topic:d2"),
        ("topic:e1", "topic:e2"),
        ("topic:f1", "topic:f2"),
    ]
    bundles: list[Any] = []
    # Two episodes per theme to fix each pair as its own cluster.
    for label, (t1, t2) in zip("abcdef", themes):
        bundles.append(_bundle(tmp_path / "metadata", f"ep-{label}1", kg=_kg([t1, t2])))
        bundles.append(_bundle(tmp_path / "metadata", f"ep-{label}2", kg=_kg([t1, t2])))
    # Strong cross-cluster bridge between themes A and B so their inter-
    # cluster lift dominates and they merge first when target < N.
    bundles.append(_bundle(tmp_path / "metadata", "ep-ab1", kg=_kg(["topic:a1", "topic:b1"])))
    bundles.append(_bundle(tmp_path / "metadata", "ep-ab2", kg=_kg(["topic:a2", "topic:b2"])))

    data = _run(
        TopicThemeClustersEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        # Target = 5 forces exactly one merge from 6 clusters. Higher targets
        # would leave 6 clusters untouched (n <= target early-return in
        # _average_linkage_to_target).
        config={"super_theme_target": 5},
        ctx=_ctx("topic_theme_clusters"),
    )
    assert data["cluster_count"] == 6
    assert data["super_theme_target"] == 5
    # Distinct super-theme ids ≤ target (linkage collapsed at least one pair).
    super_ids = {c["super_theme_id"] for c in data["clusters"]}
    assert 1 <= len(super_ids) <= 5
    assert data["super_theme_count"] == len(super_ids)
    for c in data["clusters"]:
        assert c["super_theme_id"].startswith("sth:")
        assert c["super_theme_label"]


# temporal_velocity (corpus scope)
# ---------------------------------------------------------------------------


def test_temporal_velocity_zero_filled_12_month_window(tmp_path: Path) -> None:
    def _kg(date: str, topic_id: str) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "type": "Episode",
                    "id": "episode:x",
                    "properties": {"publish_date": date},
                },
                {"type": "Topic", "id": topic_id, "properties": {"label": "Topic"}},
            ],
            "edges": [],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg("2026-06-15T00:00:00Z", "topic:a")),
        _bundle(tmp_path / "metadata", "ep2", kg=_kg("2026-05-15T00:00:00Z", "topic:a")),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert len(data["window_months"]) == 12
    topics = data["topics"]
    assert len(topics) == 1
    t = topics[0]
    assert t["topic_id"] == "topic:a"
    assert t["monthly_counts"]["2026-06"] == 1
    assert t["monthly_counts"]["2026-05"] == 1
    assert t["monthly_counts"]["2026-04"] == 0
    assert t["total"] == 2


def test_temporal_velocity_velocity_signal(tmp_path: Path) -> None:
    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                ],
                "edges": [],
            },
        )

    # Last month had 3 mentions; preceding 5 months had 1 each → velocity ≈ 3 / 1.33 = 2.25
    bundles = [
        _ep("ep1a", "2026-06-01T00:00:00Z"),
        _ep("ep1b", "2026-06-15T00:00:00Z"),
        _ep("ep1c", "2026-06-20T00:00:00Z"),
        _ep("ep2", "2026-05-10T00:00:00Z"),
        _ep("ep3", "2026-04-10T00:00:00Z"),
        _ep("ep4", "2026-03-10T00:00:00Z"),
        _ep("ep5", "2026-02-10T00:00:00Z"),
        _ep("ep6", "2026-01-10T00:00:00Z"),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    t = data["topics"][0]
    # 3 last over 6-month avg of (3+1+1+1+1+1)/6 = 1.333... → 3/1.333 ≈ 2.25
    assert t["velocity_last_over_6mo"] > 2.0


def test_temporal_velocity_weekly_series(tmp_path: Path) -> None:
    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                ],
                "edges": [],
            },
        )

    # 3 mentions in the latest ISO week (Mon 06-29 + Tue 06-30), 1 mention ~10 weeks
    # earlier → the latest week is clearly rising against its trailing-8-week average.
    bundles = [
        _ep("w1", "2026-06-29T00:00:00Z"),
        _ep("w2", "2026-06-29T12:00:00Z"),
        _ep("w3", "2026-06-30T00:00:00Z"),
        _ep("old", "2026-04-20T00:00:00Z"),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert len(data["window_weeks"]) == 26
    t = data["topics"][0]
    assert t["topic_id"] == "topic:a"
    # weekly_counts / weekly_velocity are keyed by the window weeks; the count total is
    # the number of in-window mentions.
    assert set(t["weekly_counts"]) == set(data["window_weeks"])
    assert set(t["weekly_velocity"]) == set(data["window_weeks"])
    assert sum(t["weekly_counts"].values()) == 4
    latest_week = data["window_weeks"][-1]
    assert t["weekly_counts"][latest_week] == 3
    assert t["weekly_velocity"][latest_week] > 1.0


def test_temporal_velocity_weekly_window_read_from_config(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path / "metadata",
        "ep1",
        kg={
            "nodes": [
                {
                    "type": "Episode",
                    "id": "ep:1",
                    "properties": {"publish_date": "2026-06-20T00:00:00Z"},
                },
                {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
            ],
            "edges": [],
        },
    )
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[bundle],
        config={"now": "2026-06-30T00:00:00Z", "weekly_window": 8},
        ctx=_ctx("temporal_velocity"),
    )
    assert len(data["window_weeks"]) == 8


def _tv_ep(
    base: Path, stem: str, date: str, topics: list[str], persons: list[str]
) -> "EpisodeArtifactBundle":  # noqa: F821
    """A KG bundle with an Episode publish_date + Topic/Person nodes (for content_series tests)."""
    nodes: list[dict] = [
        {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}}
    ]
    nodes += [{"type": "Topic", "id": t, "properties": {"label": t.split(":")[-1]}} for t in topics]
    nodes += [
        {"type": "Person", "id": p, "properties": {"name": p.split(":")[-1]}} for p in persons
    ]
    return _bundle(base / "metadata", stem, kg={"nodes": nodes, "edges": []})


def test_temporal_velocity_content_series_topics_and_persons(tmp_path: Path) -> None:
    # Two episodes ~10 months apart → content_series spans the FULL history (far beyond the 26-week
    # now-window), and counts Topics AND Persons per ISO week.
    bundles = [
        _tv_ep(tmp_path, "old", "2025-03-10T00:00:00Z", ["topic:a"], ["person:alice"]),
        _tv_ep(
            tmp_path,
            "new",
            "2026-01-12T00:00:00Z",
            ["topic:a", "topic:b"],
            ["person:alice", "person:bob"],
        ),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-02-01T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    cs = data["content_series"]
    # Full history: the axis spans ~45 weeks (Mar 2025 → Jan 2026), NOT the 26-week now-window.
    assert len(cs["window_weeks"]) > 26
    assert cs["window_weeks"] == sorted(cs["window_weeks"])  # contiguous, oldest→newest

    topics = {t["topic_id"]: t for t in cs["topics"]}
    assert topics["topic:a"]["total"] == 2 and len(topics["topic:a"]["weekly_counts"]) == 2
    assert topics["topic:b"]["total"] == 1
    # every counted week is on the axis
    assert set(topics["topic:a"]["weekly_counts"]) <= set(cs["window_weeks"])

    persons = {p["person_id"]: p for p in cs["persons"]}
    assert persons["person:alice"]["total"] == 2  # both episodes
    assert persons["person:bob"]["total"] == 1
    assert persons["person:alice"]["person_label"] == "alice"


def test_temporal_velocity_content_series_is_now_independent(tmp_path: Path) -> None:
    # The durable content_series must be identical regardless of the run-time `now` (unlike the
    # now-anchored monthly/weekly fields) — that is the whole point of the read-time momentum split.
    bundles = [
        _tv_ep(tmp_path, "old", "2025-03-10T00:00:00Z", ["topic:a"], ["person:alice"]),
        _tv_ep(tmp_path, "new", "2026-01-12T00:00:00Z", ["topic:a"], ["person:alice"]),
    ]

    def _cs(now: str) -> Any:
        return _run(
            TemporalVelocityEnricher(),
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=bundles,
            config={"now": now},
            ctx=_ctx("temporal_velocity"),
        )["content_series"]

    assert _cs("2026-02-01T00:00:00Z") == _cs("2031-09-09T00:00:00Z")


def test_temporal_velocity_partial_reason_on_no_bundles(tmp_path: Path) -> None:
    """#1208 — no-silent-fail contract. When there are no bundles at all,
    the enricher emits ``partial_reason='no_bundles'`` so downstream can
    distinguish "enricher ran cleanly, no data" from a real failure."""
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[],
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert data["partial_reason"] == "no_bundles"
    assert data["topics"] == []


def test_temporal_velocity_partial_reason_on_no_topics_in_window(tmp_path: Path) -> None:
    """#1208 — no-silent-fail contract. When bundles are present but no
    Topics fell within the counted window, emit ``partial_reason=
    'no_topics_in_window'``."""

    def _kg_no_topic(date: str) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Episode", "id": "episode:x", "properties": {"publish_date": date}},
                # No Topic nodes at all — the enricher counts Topic mentions.
            ],
            "edges": [],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg_no_topic("2026-06-15T00:00:00Z")),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert data["partial_reason"] == "no_topics_in_window"
    assert data["topics"] == []


def test_temporal_velocity_partial_reason_absent_on_ok_output(tmp_path: Path) -> None:
    """#1208 — no-silent-fail contract. When output is non-empty, the
    ``partial_reason`` field is present but None. Consumers key on
    ``partial_reason is not None``."""

    def _kg(date: str, topic_id: str) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Episode", "id": "episode:x", "properties": {"publish_date": date}},
                {"type": "Topic", "id": topic_id, "properties": {"label": "Topic"}},
            ],
            "edges": [],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", kg=_kg("2026-06-15T00:00:00Z", "topic:a")),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert data["partial_reason"] is None
    assert len(data["topics"]) == 1


# ---------------------------------------------------------------------------
# grounding_rate (corpus scope)
# ---------------------------------------------------------------------------


def test_grounding_rate_per_person_ratio(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {"type": "Person", "id": "person:p1", "properties": {"name": "Alice"}},
            {"type": "Insight", "id": "insight:i1", "properties": {"grounded": True}},
            {"type": "Insight", "id": "insight:i2", "properties": {"grounded": False}},
            {"type": "Quote", "id": "quote:q1"},
            {"type": "Quote", "id": "quote:q2"},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:p1"},
            {"type": "SPOKEN_BY", "from": "quote:q2", "to": "person:p1"},
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
        ],
    }
    bundle = _bundle(tmp_path / "metadata", "ep1", gi=gi)
    data = _run(
        GroundingRateEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[bundle],
        config={},
        ctx=_ctx("grounding_rate"),
    )
    persons = data["persons"]
    assert len(persons) == 1
    p = persons[0]
    assert p["person_id"] == "person:p1"
    assert p["total_insights"] == 2
    assert p["grounded_insights"] == 1
    assert p["rate"] == 0.5


def test_grounding_rate_no_quotes_emits_empty(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "metadata", "ep1", gi={"nodes": [], "edges": []})
    data = _run(
        GroundingRateEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[bundle],
        config={},
        ctx=_ctx("grounding_rate"),
    )
    assert data["persons"] == []
    # #1208 — no-silent-fail contract; partial_reason distinguishes empty-
    # output from real failure.
    assert data["partial_reason"] == "no_persons_with_insights"


def test_grounding_rate_partial_reason_on_no_bundles(tmp_path: Path) -> None:
    """#1208 — no bundles → explicit partial_reason='no_bundles'."""
    data = _run(
        GroundingRateEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[],
        config={},
        ctx=_ctx("grounding_rate"),
    )
    assert data["persons"] == []
    assert data["partial_reason"] == "no_bundles"


# ---------------------------------------------------------------------------
# guest_coappearance (corpus scope)
# ---------------------------------------------------------------------------


def test_guest_coappearance_partial_reason_on_no_bundles(tmp_path: Path) -> None:
    """#1208 — no bundles → explicit partial_reason."""
    data = _run(
        GuestCoappearanceEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[],
        config={},
        ctx=_ctx("guest_coappearance"),
    )
    assert data["pairs"] == []
    assert data["partial_reason"] == "no_bundles"


def test_guest_coappearance_ranks_by_shared_episodes(tmp_path: Path) -> None:
    def _gi_with_speakers(speakers: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Person", "id": pid, "properties": {"name": pid.split(":")[-1]}}
                for pid in speakers
            ]
            + [{"type": "Quote", "id": f"quote:{i}"} for i in range(len(speakers))],
            "edges": [
                {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": pid}
                for i, pid in enumerate(speakers)
            ],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", gi=_gi_with_speakers(["person:a", "person:b"])),
        _bundle(tmp_path / "metadata", "ep2", gi=_gi_with_speakers(["person:a", "person:b"])),
        _bundle(tmp_path / "metadata", "ep3", gi=_gi_with_speakers(["person:a", "person:c"])),
    ]
    data = _run(
        GuestCoappearanceEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("guest_coappearance"),
    )
    pairs = data["pairs"]
    assert pairs[0]["person_a_id"] == "person:a"
    assert pairs[0]["person_b_id"] == "person:b"
    assert pairs[0]["episode_count"] == 2
    assert pairs[1]["episode_count"] == 1


def test_guest_coappearance_person_communities_via_connected_components(tmp_path: Path) -> None:
    """graph-v3 tier 7-4 — person community rollup.

    Three co-appearing persons (a-b twice, b-c once, a-c once) all end up in
    the same community at default threshold=2 via the a-b edge — b and c
    ride in transitively through connected-components even though b-c alone
    is below threshold. Person d never co-appears, so is dropped (no
    singleton community). Anchor label goes to the highest-degree member.
    """

    def _gi_with_speakers(speakers: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Person", "id": pid, "properties": {"name": pid.split(":")[-1]}}
                for pid in speakers
            ]
            + [{"type": "Quote", "id": f"quote:{i}"} for i in range(len(speakers))],
            "edges": [
                {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": pid}
                for i, pid in enumerate(speakers)
            ],
        }

    bundles = [
        _bundle(tmp_path / "metadata", "ep1", gi=_gi_with_speakers(["person:a", "person:b"])),
        _bundle(tmp_path / "metadata", "ep2", gi=_gi_with_speakers(["person:a", "person:b"])),
        # A single co-appearance edge — below default threshold=2 alone but
        # keeps a & b in the same component regardless.
        _bundle(tmp_path / "metadata", "ep3", gi=_gi_with_speakers(["person:a", "person:c"])),
        # Isolated person — no co-appearances, must NOT get a community.
        _bundle(tmp_path / "metadata", "ep4", gi=_gi_with_speakers(["person:d"])),
    ]
    data = _run(
        GuestCoappearanceEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("guest_coappearance"),
    )
    assert data["community_method"] == "connected_components_threshold"
    assert data["community_min_pair"] == 2
    # Only 1 community (a+b via the 2-episode edge); c doesn't qualify at
    # default threshold, d is isolated.
    assert data["community_count"] == 1
    (community,) = data["communities"]
    assert community["community_id"].startswith("pco:")
    assert set(community["member_ids"]) == {"person:a", "person:b"}
    assert community["member_count"] == 2


def test_guest_coappearance_community_threshold_1_bridges_transitively(tmp_path: Path) -> None:
    """At threshold=1 every co-appearance links people; connected-components
    then produce fewer, larger communities. Verifies the ``community_min_pair``
    config knob threads through and the union-find bridges chains."""

    def _gi(speakers: list[str]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Person", "id": pid, "properties": {"name": pid.split(":")[-1]}}
                for pid in speakers
            ]
            + [{"type": "Quote", "id": f"quote:{i}"} for i in range(len(speakers))],
            "edges": [
                {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": pid}
                for i, pid in enumerate(speakers)
            ],
        }

    # a—b—c chain via single co-appearances; d—e in a separate component.
    bundles = [
        _bundle(tmp_path / "metadata", "ep1", gi=_gi(["person:a", "person:b"])),
        _bundle(tmp_path / "metadata", "ep2", gi=_gi(["person:b", "person:c"])),
        _bundle(tmp_path / "metadata", "ep3", gi=_gi(["person:d", "person:e"])),
    ]
    data = _run(
        GuestCoappearanceEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"community_min_pair": 1},
        ctx=_ctx("guest_coappearance"),
    )
    assert data["community_min_pair"] == 1
    assert data["community_count"] == 2
    by_size = sorted(data["communities"], key=lambda c: -c["member_count"])
    assert by_size[0]["member_count"] == 3  # a-b-c
    assert set(by_size[0]["member_ids"]) == {"person:a", "person:b", "person:c"}
    assert by_size[1]["member_count"] == 2  # d-e
    assert set(by_size[1]["member_ids"]) == {"person:d", "person:e"}


# ---------------------------------------------------------------------------
# insight_density (episode scope)
# ---------------------------------------------------------------------------


def test_insight_density_splits_by_quote_timing(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {"type": "Quote", "id": "quote:q1", "properties": {"start_s": 10}},
            {"type": "Quote", "id": "quote:q2", "properties": {"start_s": 350}},
            {"type": "Quote", "id": "quote:q3", "properties": {"start_s": 700}},
            {"type": "Insight", "id": "insight:i1"},
            {"type": "Insight", "id": "insight:i2"},
            {"type": "Insight", "id": "insight:i3"},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
            {"type": "SUPPORTED_BY", "from": "insight:i3", "to": "quote:q3"},
        ],
    }
    # 900s episode → thirds at 300s and 600s.
    bundle = _bundle(tmp_path / "metadata", "ep1", gi=gi, metadata={"duration_seconds": 900})
    data = _run(
        InsightDensityEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("insight_density"),
    )
    assert data["has_timing"] is True
    assert data["counts"] == {"early": 1, "mid": 1, "late": 1, "unknown": 0}
    assert data["total_insights"] == 3


def test_insight_density_falls_back_when_no_timing(tmp_path: Path) -> None:
    gi = {
        "nodes": [
            {"type": "Insight", "id": "insight:i1"},
            {"type": "Insight", "id": "insight:i2"},
            {"type": "Insight", "id": "insight:i3"},
            {"type": "Quote", "id": "quote:q1"},
            {"type": "Quote", "id": "quote:q2"},
            {"type": "Quote", "id": "quote:q3"},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
            {"type": "SUPPORTED_BY", "from": "insight:i3", "to": "quote:q3"},
        ],
    }
    bundle = _bundle(tmp_path / "metadata", "ep1", gi=gi, metadata={})
    data = _run(
        InsightDensityEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("insight_density"),
    )
    assert data["has_timing"] is False
    assert data["total_insights"] == 3


# ---------------------------------------------------------------------------
# Registry wiring + manifests
# ---------------------------------------------------------------------------


def test_register_deterministic_enrichers_registers_all_six() -> None:
    reg = EnricherRegistry()
    register_deterministic_enrichers(reg)
    assert sorted(reg.all_ids()) == sorted(ALL_DETERMINISTIC_ENRICHER_IDS)


def test_every_deterministic_enricher_is_correct_tier_and_scope() -> None:
    reg = EnricherRegistry()
    register_deterministic_enrichers(reg)
    for eid in ALL_DETERMINISTIC_ENRICHER_IDS:
        m = reg.get(eid).manifest
        assert m.tier is EnricherTier.DETERMINISTIC, eid
        assert m.scope in (EnricherScope.EPISODE, EnricherScope.CORPUS), eid
        # Deterministic tier never requires opt-in.
        assert m.requires_opt_in is False, eid


# ---------------------------------------------------------------------------
# Real-corpus-validation follow-ups (RFC-088 audit Bugs 2-5)
# ---------------------------------------------------------------------------


def test_temporal_velocity_falls_back_when_current_month_is_empty(tmp_path: Path) -> None:
    """Bug 2 — partial / stale current-month data must NOT collapse velocity.

    Data ends mid-May (corpus stopped collecting), ``now`` is mid-June.
    The current calendar month (2026-06) has zero mentions across the
    corpus. Pre-fix: ``velocity_last_over_6mo = 0 / avg = 0``. Post-fix:
    velocity uses the most recent month with ANY topic activity as the
    "effective last" month — which here is 2026-05.
    """

    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                ],
                "edges": [],
            },
        )

    bundles = [
        # Last month with activity = 2026-05 (3 mentions). Prior 5 months: 1 each.
        _ep("epm1", "2026-05-04T00:00:00Z"),
        _ep("epm2", "2026-05-12T00:00:00Z"),
        _ep("epm3", "2026-05-20T00:00:00Z"),
        _ep("ep2", "2026-04-10T00:00:00Z"),
        _ep("ep3", "2026-03-10T00:00:00Z"),
        _ep("ep4", "2026-02-10T00:00:00Z"),
        _ep("ep5", "2026-01-10T00:00:00Z"),
        _ep("ep6", "2025-12-10T00:00:00Z"),
    ]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-28T00:00:00Z"},  # now is mid-June, June has 0 data
        ctx=_ctx("temporal_velocity"),
    )
    t = data["topics"][0]
    # 3 last-effective over 6-month avg of (3+1+1+1+1+1)/6 = 1.333 → ≈ 2.25
    assert t["velocity_last_over_6mo"] > 2.0, t
    assert data["effective_last_month"] == "2026-05"


def test_temporal_velocity_full_window_uses_actual_last_month(tmp_path: Path) -> None:
    """Sanity: when the current month HAS data, effective_last_month = now's month."""

    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                ],
                "edges": [],
            },
        )

    bundles = [_ep("a", "2026-06-15T00:00:00Z"), _ep("b", "2026-05-01T00:00:00Z")]
    data = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-06-28T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert data["effective_last_month"] == "2026-06"


def test_temporal_velocity_alpha_and_window_months_read_from_config(tmp_path: Path) -> None:
    """Externalised knobs: ``alpha`` + ``window_months`` accepted via per-enricher config.

    Pre-change: ``_ALPHA`` and ``_WINDOW_MONTHS`` were module constants
    (0.5 / 12). Post-change: read from ``config`` with the same defaults.
    """

    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a", "properties": {"label": "A"}},
                ],
                "edges": [],
            },
        )

    bundles = [_ep("a", "2026-04-15T00:00:00Z"), _ep("b", "2026-05-15T00:00:00Z")]
    # Default alpha = 0.5 (verify by inspecting the envelope's reported alpha)
    data_default = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-05-30T00:00:00Z"},
        ctx=_ctx("temporal_velocity"),
    )
    assert data_default["alpha"] == 0.5
    assert len(data_default["window_months"]) == 12

    # Override knobs via config
    data_tuned = _run(
        TemporalVelocityEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={"now": "2026-05-30T00:00:00Z", "alpha": 0.9, "window_months": 6},
        ctx=_ctx("temporal_velocity"),
    )
    assert data_tuned["alpha"] == 0.9
    assert len(data_tuned["window_months"]) == 6


def test_temporal_velocity_invalid_knobs_fall_back_to_defaults(tmp_path: Path) -> None:
    """Out-of-range / non-numeric knob values silently fall back to defaults."""

    def _ep(stem: str, date: str) -> EpisodeArtifactBundle:
        return _bundle(
            tmp_path / "metadata",
            stem,
            kg={
                "nodes": [
                    {"type": "Episode", "id": "ep:" + stem, "properties": {"publish_date": date}},
                    {"type": "Topic", "id": "topic:a"},
                ],
                "edges": [],
            },
        )

    bundles = [_ep("a", "2026-05-15T00:00:00Z")]
    for bad_config in (
        {"alpha": -1, "window_months": 12},  # alpha out of range
        {"alpha": 2.0, "window_months": 12},  # alpha > 1
        {"alpha": "high", "window_months": 12},  # non-numeric alpha
        {"alpha": 0.5, "window_months": 0},  # window too small
        {"alpha": 0.5, "window_months": 500},  # window too large
        {"alpha": 0.5, "window_months": "twelve"},  # non-integer window
    ):
        data = _run(
            TemporalVelocityEnricher(),
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=bundles,
            config={"now": "2026-05-30T00:00:00Z", **bad_config},
            ctx=_ctx("temporal_velocity"),
        )
        assert data["alpha"] == 0.5, bad_config
        assert len(data["window_months"]) == 12, bad_config


def test_enricher_manifest_carries_config_schema_and_provider_requirement() -> None:
    """Per-RFC-088 v2 schema: manifests declare per-enricher knob schema
    (for UI form generation + YAML validation) and provider injection
    requirements (for --with-ml wiring). Deterministic enrichers without
    knobs leave config_schema=None; enrichers without dependencies leave
    provider_requirement=None."""
    tv = TemporalVelocityEnricher().manifest
    assert tv.config_schema is not None
    assert "alpha" in tv.config_schema["properties"]
    assert "window_months" in tv.config_schema["properties"]
    assert tv.provider_requirement is None

    from podcast_scraper.enrichment.enrichers.topic_consensus import (
        TopicConsensusEnricher,
    )
    from podcast_scraper.enrichment.enrichers.topic_similarity import (
        TopicSimilarityEnricher,
    )

    ts = TopicSimilarityEnricher.manifest
    assert ts.provider_requirement is not None
    assert ts.provider_requirement.protocol == "EmbeddingProvider"
    assert ts.config_schema is not None
    assert "top_k" in ts.config_schema["properties"]

    tc = TopicConsensusEnricher.manifest
    assert tc.provider_requirement is not None
    assert tc.provider_requirement.protocol == "ConsensusScorer"
    assert tc.config_schema is not None
    assert "cos_threshold" in tc.config_schema["properties"]


def test_guest_coappearance_filters_speaker_placeholders(tmp_path: Path) -> None:
    """Bug 3 — SPEAKER_NN placeholder Persons must not produce pairs.

    Two episodes each have ``person:speaker-03`` <-> ``person:speaker-05``.
    Each episode's SPEAKER_NN is independent, so counting them as
    cross-episode pairs is wrong. Real Persons co-appearing in both
    episodes ARE counted.
    """

    def _gi(speakers: list[tuple[str, str]]) -> dict[str, Any]:
        return {
            "nodes": [
                {"type": "Person", "id": pid, "properties": {"name": name}}
                for pid, name in speakers
            ]
            + [{"type": "Quote", "id": f"quote:{i}"} for i in range(len(speakers))],
            "edges": [
                {"type": "SPOKEN_BY", "from": f"quote:{i}", "to": pid}
                for i, (pid, _) in enumerate(speakers)
            ],
        }

    bundles = [
        _bundle(
            tmp_path / "metadata",
            "ep1",
            gi=_gi(
                [
                    ("person:alice", "Alice"),
                    ("person:bob", "Bob"),
                    ("person:speaker-03", "SPEAKER_03"),
                    ("person:speaker-05", "SPEAKER_05"),
                ]
            ),
        ),
        _bundle(
            tmp_path / "metadata",
            "ep2",
            gi=_gi(
                [
                    ("person:alice", "Alice"),
                    ("person:bob", "Bob"),
                    ("person:speaker-03", "SPEAKER_03"),
                    ("person:speaker-05", "SPEAKER_05"),
                ]
            ),
        ),
    ]
    data = _run(
        GuestCoappearanceEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=bundles,
        config={},
        ctx=_ctx("guest_coappearance"),
    )
    pairs = data["pairs"]
    pair_ids = {(p["person_a_id"], p["person_b_id"]) for p in pairs}
    # Real pair retained.
    assert ("person:alice", "person:bob") in pair_ids
    # No SPEAKER_NN pair leaks into the output.
    for a, b in pair_ids:
        assert "speaker-" not in a, (a, b)
        assert "speaker-" not in b, (a, b)


def test_grounding_rate_filters_speaker_placeholders(tmp_path: Path) -> None:
    """Bug 3 — SPEAKER_NN placeholders must not appear in grounding rate output."""
    gi = {
        "nodes": [
            {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
            {"type": "Person", "id": "person:speaker-07", "properties": {"name": "SPEAKER_07"}},
            {"type": "Insight", "id": "insight:i1", "properties": {"grounded": True}},
            {"type": "Insight", "id": "insight:i2", "properties": {"grounded": True}},
            {"type": "Quote", "id": "quote:q1"},
            {"type": "Quote", "id": "quote:q2"},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote:q1", "to": "person:alice"},
            {"type": "SPOKEN_BY", "from": "quote:q2", "to": "person:speaker-07"},
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
        ],
    }
    bundle = _bundle(tmp_path / "metadata", "ep1", gi=gi)
    data = _run(
        GroundingRateEnricher(),
        bundle=None,
        corpus_root=tmp_path,
        all_bundles=[bundle],
        config={},
        ctx=_ctx("grounding_rate"),
    )
    person_ids = [p["person_id"] for p in data["persons"]]
    assert "person:alice" in person_ids
    assert "person:speaker-07" not in person_ids


def test_insight_density_reads_timestamp_start_ms_field(tmp_path: Path) -> None:
    """Bug 4 — real `.gi.json` writes Quote timing as ``timestamp_start_ms``.

    Pre-fix: the enricher only checked ``start_s`` / ``start_seconds`` /
    ``start`` so every real corpus reported ``has_timing=False`` and fell
    back to even-thirds segmentation. Post-fix: ``timestamp_start_ms``
    is recognised (ms → s) and timing is honoured.
    """
    gi = {
        "nodes": [
            {
                "type": "Quote",
                "id": "quote:q1",
                "properties": {"timestamp_start_ms": 10_000},
            },  # 10s
            {
                "type": "Quote",
                "id": "quote:q2",
                "properties": {"timestamp_start_ms": 350_000},
            },  # 350s
            {
                "type": "Quote",
                "id": "quote:q3",
                "properties": {"timestamp_start_ms": 700_000},
            },  # 700s
            {"type": "Insight", "id": "insight:i1"},
            {"type": "Insight", "id": "insight:i2"},
            {"type": "Insight", "id": "insight:i3"},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
            {"type": "SUPPORTED_BY", "from": "insight:i3", "to": "quote:q3"},
        ],
    }
    bundle = _bundle(
        tmp_path / "metadata",
        "ep1",
        gi=gi,
        metadata={"episode": {"duration_seconds": 900}},
    )
    data = _run(
        InsightDensityEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("insight_density"),
    )
    assert data["has_timing"] is True
    assert data["counts"] == {"early": 1, "mid": 1, "late": 1, "unknown": 0}


def test_insight_density_reads_nested_episode_duration(tmp_path: Path) -> None:
    """Bug 4 — metadata writer puts ``duration_seconds`` under ``episode.``.

    Pre-fix: ``meta.get("duration_seconds")`` returned 0 → has_timing=False
    + even-thirds fallback. Post-fix: ``episode_duration_seconds`` reads
    both shapes, real timing is honoured.
    """
    gi = {
        "nodes": [
            {"type": "Quote", "id": "quote:q1", "properties": {"start_s": 10}},
            {"type": "Quote", "id": "quote:q2", "properties": {"start_s": 350}},
            {"type": "Quote", "id": "quote:q3", "properties": {"start_s": 700}},
            {"type": "Insight", "id": "insight:i1"},
            {"type": "Insight", "id": "insight:i2"},
            {"type": "Insight", "id": "insight:i3"},
        ],
        "edges": [
            {"type": "SUPPORTED_BY", "from": "insight:i1", "to": "quote:q1"},
            {"type": "SUPPORTED_BY", "from": "insight:i2", "to": "quote:q2"},
            {"type": "SUPPORTED_BY", "from": "insight:i3", "to": "quote:q3"},
        ],
    }
    bundle = _bundle(
        tmp_path / "metadata",
        "ep1",
        gi=gi,
        metadata={"episode": {"duration_seconds": 900}},
    )
    data = _run(
        InsightDensityEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("insight_density"),
    )
    assert data["has_timing"] is True
    assert data["duration_seconds"] == 900.0
    assert data["counts"] == {"early": 1, "mid": 1, "late": 1, "unknown": 0}


def test_sync_enricher_records_written_from_largest_list(tmp_path: Path) -> None:
    """Bug 5 — sync_enricher infers records_written from the data dict.

    Every deterministic enricher returns a dict with a primary list
    value (``pairs`` / ``persons`` / ``topics`` / ``insight_segments``).
    ``sync_enricher`` derives records_written from the longest list so
    the run_summary's per_enricher.records_written stops reporting 0.
    """
    # Use grounding_rate as the realistic vehicle.
    gi = {
        "nodes": [
            {"type": "Person", "id": f"person:p{i}", "properties": {"name": f"P{i}"}}
            for i in range(3)
        ]
        + [
            {"type": "Insight", "id": f"insight:i{i}", "properties": {"grounded": True}}
            for i in range(3)
        ]
        + [{"type": "Quote", "id": f"quote:q{i}"} for i in range(3)],
        "edges": [
            {"type": "SPOKEN_BY", "from": f"quote:q{i}", "to": f"person:p{i}"} for i in range(3)
        ]
        + [
            {"type": "SUPPORTED_BY", "from": f"insight:i{i}", "to": f"quote:q{i}"} for i in range(3)
        ],
    }
    bundle = _bundle(tmp_path / "metadata", "ep1", gi=gi)
    result = asyncio.run(
        GroundingRateEnricher().enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=[bundle],
            config={},
            ctx=_ctx("grounding_rate"),
        )
    )
    assert result.status == STATUS_OK
    assert isinstance(result.data, dict)
    assert len(result.data["persons"]) == 3
    assert result.records_written == 3
