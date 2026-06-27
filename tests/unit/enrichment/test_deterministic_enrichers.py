"""Unit tests for the 6 chunk-2 deterministic enrichers.

One section per enricher. Synthetic 1-3 episode fixtures asserting
numerics + envelope shape — no real corpus, no IO beyond ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers import (
    ALL_DETERMINISTIC_ENRICHER_IDS,
    GroundingRateEnricher,
    GuestCoappearanceEnricher,
    InsightDensityEnricher,
    register_deterministic_enrichers,
    TemporalVelocityEnricher,
    TopicCooccurrenceCorpusEnricher,
    TopicCooccurrenceEnricher,
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
# topic_cooccurrence (episode scope)
# ---------------------------------------------------------------------------


def test_topic_cooccurrence_lists_unordered_pairs(tmp_path: Path) -> None:
    kg = {
        "nodes": [
            {"type": "Topic", "id": "topic:a", "properties": {"label": "Alpha"}},
            {"type": "Topic", "id": "topic:b", "properties": {"label": "Beta"}},
            {"type": "Topic", "id": "topic:c", "properties": {"label": "Gamma"}},
            {"type": "Person", "id": "person:x"},  # ignored
        ],
        "edges": [],
    }
    bundle = _bundle(tmp_path / "metadata", "ep1", kg=kg)
    data = _run(
        TopicCooccurrenceEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("topic_cooccurrence"),
    )
    pairs = data["pairs"]
    # 3 topics → C(3,2) = 3 pairs.
    assert len(pairs) == 3
    pair_ids = {(p["topic_a_id"], p["topic_b_id"]) for p in pairs}
    assert pair_ids == {("topic:a", "topic:b"), ("topic:a", "topic:c"), ("topic:b", "topic:c")}
    assert all(p["episode_count"] == 1 for p in pairs)


def test_topic_cooccurrence_no_topics_emits_empty(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "metadata", "ep1", kg={"nodes": [], "edges": []})
    data = _run(
        TopicCooccurrenceEnricher(),
        bundle=bundle,
        corpus_root=tmp_path,
        all_bundles=None,
        config={},
        ctx=_ctx("topic_cooccurrence"),
    )
    assert data["pairs"] == []


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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# guest_coappearance (corpus scope)
# ---------------------------------------------------------------------------


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
