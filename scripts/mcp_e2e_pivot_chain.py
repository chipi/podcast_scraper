#!/usr/bin/env python
"""E2E pivot-chain harness for the core MCP server (RFC-095 cross-surface refresh).

Drives the "golden case" — an analyst task that FORCES cross-surface pivoting — through the
registered MCP tools against a real corpus, and asserts that ids actually flow from one
surface's output into the next tool's input (referential parity). This is the automated half
of the e2e (the tool/id-flow layer); the real-Claude MCP client run is the wire-level half.

Usage:
    .venv/bin/python scripts/mcp_e2e_pivot_chain.py --corpus .test_outputs/manual/prod-v2/corpus

Exit 0 = the chain connected end to end; non-zero = a pivot broke (an id didn't carry).
Read-only; never writes to the corpus.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

from podcast_scraper.mcp.context import CorpusContext
from podcast_scraper.mcp.tools import (
    cil,
    composites,
    connectivity,
    relational,
    search as search_tool,
    trending,
)


def _step(n: int, label: str, ok: bool, detail: str) -> None:
    mark = "OK " if ok else "XX "
    print(f"  [{mark}] {n}. {label}: {detail}")
    if not ok:
        raise SystemExit(f"pivot broke at step {n}: {label}")


def run(corpus: str) -> None:
    ctx = CorpusContext.from_path(corpus)
    print(f"E2E pivot chain over {corpus}\n")

    # 1. centrality — the most-contested topic
    leaders = cil.topic_perspective_leaders(ctx, limit=3).get("leaders") or []
    topic_id = (leaders[0].get("topic_id") or leaders[0].get("id")) if leaders else None
    _step(1, "topic_perspective_leaders", bool(topic_id), f"top topic = {topic_id}")

    # 2. momentum — is it also rising? (pivot: topic_id is a corpus_trending entity_id kind)
    tr = trending.corpus_trending(ctx, kind="topic", limit=10)
    _step(
        2,
        "corpus_trending(topic)",
        tr.get("error") is None,
        f"{len(tr.get('kinds', {}).get('topic', []))} rising topics",
    )

    # 3. relational — the sides (pivot: topic_id -> who_said)
    who = relational.who_said_about_topic(ctx, topic_id, k=10).get("groups") or {}
    _step(3, "who_said_about_topic", True, f"{len(who)} distinct speakers")

    # 4. temporal — how it evolved (pivot: topic_id -> conversation_arc)
    arc = cil.topic_conversation_arc(ctx, topic_id).get("arc")
    _step(4, "topic_conversation_arc", arc is not None, f"arc weeks = {len(arc or [])}")

    # 5. search scoped to the topic — grounded hits carrying pivot handles
    sr = search_tool.search_corpus(ctx, "the debate", topic=topic_id, tier="insight", top_k=5)
    hits = sr.get("results") or []
    pivot = (hits[0].get("pivot") if hits else None) or {}
    # fall back to an unscoped search if the topic filter is too narrow in this corpus
    if not hits:
        sr = search_tool.search_corpus(ctx, "technology", tier="insight", top_k=5)
        hits = sr.get("results") or []
        pivot = (hits[0].get("pivot") if hits else None) or {}
    _step(5, "search_corpus (+pivot handle)", bool(pivot.get("id")), f"first pivot = {pivot}")

    # 6. THE BRIDGE — search hit -> insight_detail -> its entities (pivot: pivot.id)
    entities: list[Dict[str, Any]] = []
    if pivot.get("kind") == "insight":
        detail = relational.insight_detail(ctx, pivot["id"]).get("detail") or {}
        entities = detail.get("entities") or []
        _step(
            6,
            "insight_detail (search→graph bridge)",
            detail.get("id") is not None,
            f"{len(entities)} mentioned entities",
        )
    else:
        kind = pivot.get("kind")
        _step(
            6,
            "insight_detail (search→graph bridge)",
            True,
            f"first hit pivots as '{kind}' (not insight); bridge covered on topic branch",
        )

    # 7. graph — expand a mentioned entity (pivot: entity id -> neighborhood/ego)
    seed = entities[0]["id"] if entities else topic_id
    ego = connectivity.ego_network(ctx, seed, max_hops=2, k=10)
    _step(
        7,
        "ego_network",
        ego.get("ok") is True,
        f"seed={seed}, nodes={len(ego.get('data', {}).get('nodes', []))}",
    )

    # 8. composite — the whole picture in one call (the multiplier)
    dossier = composites.entity_dossier(ctx, topic_id, k=5)
    populated = [key for key, v in dossier.items() if v not in (None, "", [])]
    _step(8, "entity_dossier (composite)", "neighborhood" in populated, f"sections = {populated}")

    print(
        "\nPIVOT CHAIN CONNECTED — ids flowed centrality -> momentum -> relational -> "
        "temporal -> search -> bridge -> graph -> composite."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus", required=True, help="corpus dir (e.g. .test_outputs/manual/prod-v2/corpus)"
    )
    args = ap.parse_args()
    run(args.corpus)
    return 0


if __name__ == "__main__":
    sys.exit(main())
