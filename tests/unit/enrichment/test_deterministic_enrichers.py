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
