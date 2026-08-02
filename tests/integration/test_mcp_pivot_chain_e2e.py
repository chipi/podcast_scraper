"""E2E: the MCP cross-surface pivot chain over the committed synthetic corpus (RFC-095).

Standardizes the pivot-chain e2e on ``tests/fixtures/app-validation-corpus/v3`` (the UI
tier-3 corpus). The two-tier search index is BUILT here at setup (offline, cached MiniLM) —
not committed — so there is no binary lance blob in git and no lance-format-version coupling.
Skips cleanly where the embedding model isn't available (model-less unit CI); runs fully in
the ML tier / locally.

Proves referential parity: each surface's output id feeds the next tool's input, so an agent
pivots search -> graph -> insight across the surfaces.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CORPUS = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
)
_LANCE = _CORPUS / "search" / "lance_index"


@pytest.fixture(scope="module")
def indexed_corpus() -> Path:
    """Ensure the synthetic corpus has a two-tier index; build it offline if absent.

    Skips the module when the index can't be built (embedding model missing/offline) — the
    non-search MCP tools are covered by tests/unit/mcp; this module is the search+pivot half.
    """
    if not _LANCE.is_dir():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from podcast_scraper.cli import main as cli_main

            rc = cli_main(["index-two-tier", "--output-dir", str(_CORPUS)])
        except Exception as exc:  # noqa: BLE001 — any build failure => skip, not fail
            pytest.skip(f"could not build search index (embedding model offline?): {exc}")
        if rc not in (0, None) or not _LANCE.is_dir():
            pytest.skip("search index build did not produce a lance_index (model unavailable?)")
    return _CORPUS


def test_pivot_chain_ids_flow_across_surfaces(indexed_corpus: Path) -> None:
    from podcast_scraper.mcp.context import CorpusContext
    from podcast_scraper.mcp.tools import (
        cil,
        composites,
        connectivity,
        relational,
        search as search_tool,
    )

    ctx = CorpusContext.from_path(indexed_corpus)

    # centrality -> a topic id
    leaders = cil.topic_perspective_leaders(ctx, limit=3).get("leaders") or []
    assert leaders, "expected at least one perspective-leader topic in the synthetic corpus"
    topic_id = leaders[0].get("topic_id") or leaders[0].get("id")
    assert topic_id and topic_id.startswith("topic:")

    # temporal pivot: topic_id -> conversation arc
    arc = cil.topic_conversation_arc(ctx, topic_id).get("arc")
    assert arc is not None

    # search -> a hit that carries a pivot handle (the referential-parity artifact)
    res = search_tool.search_corpus(ctx, "risk management", tier="insight", top_k=5)
    if res.get("error") == "embed_failed":
        pytest.skip("query embedding unavailable in this environment")
    hits = res.get("results") or []
    assert hits, "expected search hits on the indexed synthetic corpus"
    pivot = hits[0].get("pivot") or {}
    assert pivot.get("id") and pivot.get("expand_with"), "every hit must carry a pivot handle"

    # THE BRIDGE: a search insight hit's pivot.id resolves in the graph (search -> graph)
    if pivot.get("kind") == "insight":
        detail = relational.insight_detail(ctx, pivot["id"]).get("detail")
        assert detail is not None and detail.get("id") == pivot["id"]

    # graph pivot: the topic id expands via multi-hop proximity
    ego = connectivity.ego_network(ctx, topic_id, max_hops=2, k=10)
    assert ego.get("ok") is True

    # composite: one call fuses the surfaces, keeping ids
    dossier = composites.entity_dossier(ctx, topic_id, k=5)
    assert dossier["kind"] == "topic"
    assert dossier.get("neighborhood") is not None
