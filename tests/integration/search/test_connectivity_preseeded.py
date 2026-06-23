"""Connectivity over the pre-seeded production-shaped fixture (#1055).

The exploration/relational connectivity (#1054/#1055) is exercised here against the
**checked-in** production-shaped corpus (`web/gi-kg-viewer/e2e/fixtures/production-shaped`,
the Tier-2 deterministic fixture extracted from a real snapshot) — no LLM, no live
pipeline. This guards the *real* relational traversal (not a hand-built graph) against
realistic data: `related_topics` should surface adjacent themes that share insights.

Person-side connectivity (`topics_of` / `co_speakers`) needs grounded
`person→STATES→insight→ABOUT→topic` chains, which this snapshot slice doesn't carry;
that path is covered on a grounded on-disk corpus in ``test_host_reconciliation_repro``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search import relational_queries as rq
from podcast_scraper.search.corpus_graph import CorpusGraph

pytestmark = pytest.mark.integration


def _repo_fixture() -> Path:
    # Resolve from the repo root regardless of where pytest is invoked.
    root = Path(__file__).resolve()
    for parent in root.parents:
        candidate = parent / "web/gi-kg-viewer/e2e/fixtures/production-shaped/artifacts"
        if candidate.is_dir():
            return candidate
    pytest.skip("production-shaped fixture not found")


def test_related_topics_surfaces_adjacent_themes_on_real_fixture() -> None:
    graph = CorpusGraph.build(_repo_fixture(), derive_speaker_links=True, reconcile_hosts=True)
    topics = graph.nodes_by_type("topic")
    assert topics, "fixture should carry topic nodes"
    # At least one topic has adjacent themes (topics sharing an insight) — the
    # TopicEntityView "Related topics" surface is non-empty on real pre-seeded data.
    with_related = [t for t in topics if rq.related_topics(graph, t, k=5)]
    assert with_related, "expected related_topics to be non-empty on the production-shaped fixture"
    # Adjacency is symmetric-ish and never returns the subject itself.
    sample = with_related[0]
    related = rq.related_topics(graph, sample, k=5)
    assert sample not in {n.id for n in related}


def test_reconciliation_is_safe_on_real_fixture() -> None:
    # Reconciliation must never crash or empty a real corpus; node count is stable
    # (this fixture has no network-feed unnamed-host pattern to merge).
    plain = CorpusGraph.build(_repo_fixture(), derive_speaker_links=True)
    reconciled = CorpusGraph.build(_repo_fixture(), derive_speaker_links=True, reconcile_hosts=True)
    assert len(reconciled) == len(plain)  # nothing to merge here → identical
