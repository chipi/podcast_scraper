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

pytestmark = [pytest.mark.integration]


class _StubEncoder:
    """Returns unit vectors whose dot products produce deterministic cosines.

    Mirrors the GI pipeline unit tests' helper
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
