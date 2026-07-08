"""#1075 chunk 1 — v3 GI emission contract sanity test.

The shipped Person Profile / Position Tracker viewer surfaces
(``PersonLandingView.vue``, ``PositionTrackerPanel.vue``) read four
v3-shape fields/edges from every Insight artifact they render:

- ``Insight ─MENTIONS_PERSON→ Person`` edges (powers
  ``rankedPersonTopicMentions`` + ``personInsightsByTopic`` +
  ``personTopicPositionArc``)
- ``Insight ─ABOUT→ Topic`` edges (same surfaces)
- ``Insight.properties.insight_type`` (strict-required per RFC-097
  chunk 9 / ADR-101; filter chips on Position Tracker depend on it)
- ``Insight.properties.position_hint`` (strict-required per ADR-101;
  Position Tracker timeline sort depends on it)

The active validation corpus
(``tests/fixtures/viewer-validation-corpus/v3/``) now carries all four
v3 fields/edges. Without this contract test the GI pipeline could
silently regress to emitting artifacts that the viewer panels then
render as blank rails, and the viewer's own unit tests (which craft
synthetic artifacts inline) would not catch it.

This test is **the contract** between the pipeline's emit side and the
viewer's consume side. If it fails:

1. If the pipeline regressed, fix the pipeline.
2. If the contract changed deliberately, update this test AND every
   viewer helper that reads the affected field/edge.

Two layers now exist: ``test_v3_emission_contract_real_fields_present``
exercises the same emit functions the real pipeline calls
(``_artifact_from_multi_insight`` for topic/about edges +
``add_insight_entity_edges`` for typed mentions) on an inline artifact;
``test_v3_emission_contract_real_validation_corpus_artifacts`` asserts the
same contract against the real on-disk ``viewer-validation-corpus/v3/``
fixtures (the chunk-2 sibling — landed with the v3 corpus).
"""

from __future__ import annotations

import numpy as np
import pytest

from podcast_scraper.gi.pipeline import _artifact_from_multi_insight
from podcast_scraper.gi.relational_edges import add_insight_entity_edges
from podcast_scraper.gi.schema import validate_artifact

pytestmark = [pytest.mark.integration]


class _StubEncoder:
    """Returns unit vectors whose dot products produce deterministic cosines.

    Mirrors the helper used in tests/unit/podcast_scraper/gi/test_pipeline.py
    so the topic-ABOUT edge emission can be exercised without a real
    sentence-transformers load.
    """

    def __init__(self) -> None:
        # 2 insights × 2 topics: cosines designed so both topics survive
        # the 0.25 floor for at least one insight. Insight texts include
        # the person names Alice and Bob so add_insight_entity_edges
        # matches them as whole-word phrases.
        self._by_text = {
            "Alice argues climate policy needs urgent action": np.array([1.0, 0.0, 0.0]),
            "Bob says energy transition is the bigger challenge": np.array([0.0, 1.0, 0.0]),
            "Climate Policy": np.array([0.9, np.sqrt(1 - 0.81), 0.0]),
            "Energy": np.array([np.sqrt(1 - 0.81), 0.9, 0.0]),
        }

    def encode(self, texts, normalize_embeddings=True):  # noqa: D401
        return np.stack([self._by_text[t] for t in texts])


def test_v3_emission_contract_real_fields_present() -> None:
    """All four v3 fields/edges land in one round-trip through the emit
    functions the real pipeline calls.

    Asserted invariants (the Person Profile / Position Tracker viewer
    surfaces depend on every single one of these):

    - ≥1 ``ABOUT`` edge from Insight → Topic
    - ≥1 ``MENTIONS_PERSON`` edge from Insight → Person
    - every Insight has ``insight_type`` (strict v3.0 schema)
    - every Insight has ``position_hint`` (strict v3.0 schema)
    """
    # Stage 1 — build a 2-insight GI artifact with topic edges via the
    # real ``_artifact_from_multi_insight`` emit function.
    artifact = _artifact_from_multi_insight(
        "ep:demo",
        [
            ("Alice argues climate policy needs urgent action", "claim"),
            ("Bob says energy transition is the bigger challenge", "observation"),
        ],
        [[]],
        model_version="m",
        prompt_version="v1",
        podcast_id="podcast:test",
        episode_title="Test Episode",
        date_str="2026-06-23T00:00:00Z",
        transcript_ref="transcript.txt",
        topic_labels=["Climate Policy", "Energy"],
        about_edge_encoder=_StubEncoder(),
    )

    # Stage 2 — materialize typed MENTIONS_PERSON edges via the same
    # emit function the workflow's metadata-generation step calls. It
    # synthesizes the missing Person node when the edge target isn't
    # already in ``nodes`` (RFC-097 chunk-4 retroactive sweep). Map
    # shape is ``entity_id → (name, kind)``; the function matches the
    # name as a whole-word phrase in each Insight's text.
    added = add_insight_entity_edges(
        artifact,
        entity_index={
            "person:alice": ("Alice", "person"),
            "person:bob": ("Bob", "person"),
        },
    )
    assert added > 0, "add_insight_entity_edges added no edges"

    # === The contract assertions ===

    # 1. ≥1 ABOUT edge — Topic Tab + Position Tracker timeline both
    #    pivot on this. Without it the ranked-topics list is empty.
    about_edges = [e for e in artifact["edges"] if e.get("type") == "ABOUT"]
    assert about_edges, "no ABOUT edges in v3 artifact"

    # 2. ≥1 MENTIONS_PERSON edge — the Person Profile grouping key.
    mp_edges = [e for e in artifact["edges"] if e.get("type") == "MENTIONS_PERSON"]
    assert mp_edges, "no MENTIONS_PERSON edges in v3 artifact"

    # 3. Every Insight carries insight_type (strict v3 schema requires
    #    it; Position Tracker filter chips depend on it).
    insights = [n for n in artifact["nodes"] if n.get("type") == "Insight"]
    assert insights, "no Insight nodes — emission test ill-formed"
    for ins in insights:
        props = ins.get("properties", {})
        assert "insight_type" in props, f"Insight {ins.get('id')} missing insight_type"

    # 4. Every Insight carries position_hint (strict v3 schema requires
    #    it; Position Tracker timeline sort depends on it).
    for ins in insights:
        props = ins.get("properties", {})
        assert "position_hint" in props, f"Insight {ins.get('id')} missing position_hint"

    # 5. The whole thing passes strict schema validation — catches
    #    a forgotten field or a malformed edge that the contract above
    #    might miss.
    validate_artifact(artifact, strict=True)


def test_v3_emission_contract_real_validation_corpus_artifacts() -> None:
    """The same contract holds when loaded from the real on-disk
    ``tests/fixtures/viewer-validation-corpus/v3/`` artifacts (#1075
    chunk 2 upgrade).

    Catches a regression where the on-disk fixtures fall out of sync
    with the in-memory emission contract — e.g. a future migration that
    drops one of the four required fields/edges from disk but leaves
    the emit functions unchanged.
    """
    import glob
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    corpus_glob = str(
        repo_root
        / "tests"
        / "fixtures"
        / "viewer-validation-corpus"
        / "v3"
        / "feeds"
        / "*"
        / "metadata"
        / "*.gi.json"
    )
    files = sorted(glob.glob(corpus_glob))
    assert files, "no viewer-validation-corpus/v3 GI files found"

    insights_seen = 0
    mp_total = 0
    about_total = 0
    for path in files:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = raw.get("data", {}).get("nodes") or raw.get("nodes") or []
        edges = raw.get("data", {}).get("edges") or raw.get("edges") or []

        # Per-file required-field invariant.
        for n in nodes:
            if not isinstance(n, dict) or n.get("type") != "Insight":
                continue
            insights_seen += 1
            props = n.get("properties") or {}
            assert "insight_type" in props, f"{path}: Insight {n.get('id')} missing insight_type"
            assert "position_hint" in props, f"{path}: Insight {n.get('id')} missing position_hint"

        for e in edges:
            if not isinstance(e, dict):
                continue
            if e.get("type") == "MENTIONS_PERSON":
                mp_total += 1
            elif e.get("type") == "ABOUT":
                about_total += 1

    # Corpus-wide invariants — at least some edges of each type exist
    # so the chunk-2 upgrade isn't a silent no-op.
    assert insights_seen > 0, "no Insight nodes across the entire validation corpus"
    assert mp_total > 0, "no MENTIONS_PERSON edges across the entire validation corpus"
    assert about_total > 0, "no ABOUT edges across the entire validation corpus"
