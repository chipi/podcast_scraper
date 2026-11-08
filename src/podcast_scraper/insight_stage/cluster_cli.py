"""The insight-cluster commands: ``insight-clusters`` and the ``gi`` views over its output (moved
verbatim from ``search.cli_handlers``, E04 S4).
"""

from __future__ import annotations

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, cast, Dict, List, Sequence

from podcast_scraper.search.cli_handlers import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS
from podcast_scraper.utils.corpus_walk import corpus_rglob
from podcast_scraper.utils.log_redaction import format_exception_for_log


def parse_insight_clusters_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``insight-clusters`` (#599)."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper insight-clusters",
        description=(
            "Cluster GI insights from gi.json artifacts across episodes; "
            "write insight_clusters.json (#599)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (contains metadata/*.gi.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity to group insights (default: 0.75)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write JSON here (default: <output-dir>/search/insight_clusters.json)",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.command = "insight-clusters"
    return ns


def run_insight_clusters_cli(args: Namespace, logger: logging.Logger) -> int:
    """Build ``insight_clusters.json`` for a corpus (#599)."""
    from podcast_scraper.insight_stage.insight_clusters import build_insight_clusters_for_corpus

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("insight-clusters: --output-dir is required")
        return EXIT_INVALID_ARGS

    out_file = getattr(args, "output_file", None)
    out_path = Path(out_file).resolve() if out_file else None
    threshold = float(getattr(args, "threshold", 0.75) or 0.75)

    try:
        payload = build_insight_clusters_for_corpus(
            output_dir,
            threshold=threshold,
            out_path=out_path,
        )
    except FileNotFoundError as exc:
        logger.error("insight-clusters: %s", exc)
        return EXIT_NO_ARTIFACTS
    except Exception as exc:
        logger.error("insight-clusters: %s", format_exception_for_log(exc))
        return EXIT_INVALID_ARGS

    logger.info(
        "insight-clusters: insights=%s clusters=%s cross_episode=%s singletons=%s",
        payload.get("insight_count"),
        payload.get("cluster_count"),
        payload.get("cross_episode_clusters"),
        payload.get("singletons"),
    )
    return EXIT_SUCCESS


def parse_cluster_browse_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``gi clusters``."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper gi clusters",
        description="Browse insight clusters sorted by member count.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (contains search/insight_clusters.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Show top N clusters by member count (default: 10)",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.gi_subcommand = "clusters"
    return ns


def run_cluster_browse_cli(args: Namespace, logger: logging.Logger) -> int:
    """Browse insight clusters sorted by member count (#601 3b)."""
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("gi clusters: --output-dir is required")
        return EXIT_INVALID_ARGS

    clusters_path = Path(output_dir) / "search" / "insight_clusters.json"
    if not clusters_path.exists():
        logger.error(
            "gi clusters: insight_clusters.json not found at %s " "(run `insight-clusters` first)",
            clusters_path,
        )
        return EXIT_NO_ARTIFACTS

    payload = json.loads(clusters_path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters", [])
    top_n = max(1, int(getattr(args, "top", 10) or 10))
    # Already sorted by member_count (descending) from build step
    shown = clusters[:top_n]

    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        print(json.dumps({"clusters": shown, "total": len(clusters)}, indent=2))
    else:
        print(
            f"Insight clusters: {len(clusters)} total "
            f"({payload.get('cross_episode_clusters', 0)} cross-episode), "
            f"showing top {min(top_n, len(shown))}\n"
        )
        for i, c in enumerate(shown, 1):
            cross = " [cross-episode]" if c.get("cross_episode") else ""
            print(
                f"{i}. [{c['cluster_id']}] "
                f"{c['member_count']} insights, "
                f"{c.get('episode_count', '?')} episodes{cross}"
            )
            canonical = c.get("canonical_insight", "")
            if len(canonical) > 100:
                canonical = canonical[:97] + "..."
            print(f"   Canonical: {canonical}")
            eps = c.get("episode_ids", [])
            if eps:
                print(f"   Episodes: {', '.join(eps[:5])}")
                if len(eps) > 5:
                    print(f"             ... +{len(eps) - 5} more")
            print()

    return EXIT_SUCCESS


def parse_explore_quotes_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``gi explore-quotes``."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper gi explore-quotes",
        description="Search quotes directly via semantic (LanceDB) search.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query for quote text",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (contains search/ index)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="N",
        help="Number of results (default: 10)",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.gi_subcommand = "explore-quotes"
    return ns


def run_explore_quotes_cli(args: Namespace, logger: logging.Logger) -> int:
    """Search quotes directly via semantic (LanceDB) search (#601 3c)."""
    query = (getattr(args, "query", "") or "").strip()
    if not query:
        logger.error("gi explore-quotes: --query is required")
        return EXIT_INVALID_ARGS

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("gi explore-quotes: --output-dir is required")
        return EXIT_INVALID_ARGS

    from podcast_scraper.search.corpus_search import run_corpus_search

    top_k = max(1, int(getattr(args, "top_k", 10) or 10))
    outcome = run_corpus_search(
        Path(output_dir),
        query,
        doc_types=["quote"],
        top_k=top_k,
    )

    if outcome.error == "no_index":
        logger.error("gi explore-quotes: no vector index found (run `index` first)")
        return EXIT_NO_ARTIFACTS
    if outcome.error:
        logger.error("gi explore-quotes: %s — %s", outcome.error, outcome.detail or "")
        return EXIT_INVALID_ARGS

    results = outcome.results
    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        print(json.dumps({"query": query, "results": results}, indent=2))
    else:
        print(f'Quote search: "{query}" — {len(results)} results\n')
        for row in results:
            meta = row.get("metadata", {})
            score = row.get("score", 0.0)
            ep = meta.get("episode_id", "?")
            speaker = meta.get("speaker_id", "")
            speaker_str = f" ({speaker})" if speaker else ""
            text = row.get("text", "")
            if len(text) > 120:
                text = text[:117] + "..."
            print(f"  score={score:.4f}  [{ep}{speaker_str}]")
            print(f'    "{text}"')
            print()

    return EXIT_SUCCESS


def parse_topic_insights_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``gi topic-insights``."""
    parser = argparse.ArgumentParser(
        prog="podcast_scraper gi topic-insights",
        description=(
            "Show insight clusters associated with a topic "
            "(cross-references topic_clusters.json + insight_clusters.json via ABOUT edges)."
        ),
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic label or substring to match",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "json"),
        default="pretty",
        help="Output format (default: pretty)",
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.gi_subcommand = "topic-insights"
    return ns


def run_topic_insights_cli(args: Namespace, logger: logging.Logger) -> int:
    """Show insight clusters for a topic (#601 3d).

    Cross-references topic_clusters.json + insight_clusters.json via
    ABOUT edges in gi.json artifacts.
    """
    topic_query = (getattr(args, "topic", "") or "").strip().lower()
    if not topic_query:
        logger.error("gi topic-insights: --topic is required")
        return EXIT_INVALID_ARGS

    output_dir = Path(getattr(args, "output_dir", "."))
    search_dir = output_dir / "search"

    # Load insight clusters
    ic_path = search_dir / "insight_clusters.json"
    if not ic_path.exists():
        logger.error(
            "gi topic-insights: insight_clusters.json not found (run `insight-clusters` first)"
        )
        return EXIT_NO_ARTIFACTS
    ic_payload = json.loads(ic_path.read_text(encoding="utf-8"))

    # Build insight_id → cluster mapping
    insight_to_cluster: Dict[str, Dict[str, Any]] = {}
    for cluster in ic_payload.get("clusters", []):
        for member in cluster.get("members", []):
            insight_to_cluster[member["insight_id"]] = cluster

    # Scan gi.json artifacts for ABOUT edges: insight → topic
    matched_cluster_ids: set = set()
    matched_clusters: List[Dict[str, Any]] = []

    for gi_path in sorted(corpus_rglob(output_dir, "*.gi.json")):
        try:
            gi = json.loads(gi_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Find topic nodes matching query
        topic_ids: set = set()
        for node in gi.get("nodes", []):
            if node.get("type") == "Topic":
                label = (node.get("properties", {}).get("label", "") or "").lower()
                if topic_query in label:
                    topic_ids.add(node["id"])

        if not topic_ids:
            continue

        # Find insights linked to these topics via ABOUT edges
        for edge in gi.get("edges", []):
            if edge.get("type") == "ABOUT" and edge.get("to") in topic_ids:
                insight_id = edge["from"]
                cluster = insight_to_cluster.get(insight_id)
                if cluster and cluster["cluster_id"] not in matched_cluster_ids:
                    matched_cluster_ids.add(cluster["cluster_id"])
                    matched_clusters.append(cluster)

    fmt = getattr(args, "format", "pretty")
    if fmt == "json":
        print(
            json.dumps(
                {"topic_query": topic_query, "clusters": matched_clusters},
                indent=2,
            )
        )
    else:
        print(f'Topic: "{topic_query}" — ' f"{len(matched_clusters)} insight cluster(s)\n")
        if not matched_clusters:
            print("  No insight clusters found for this topic.")
            print(
                "  (Ensure gi.json has ABOUT edges linking insights to topics, "
                "and insight_clusters.json exists.)"
            )
        for c in sorted(matched_clusters, key=lambda x: -x.get("member_count", 0)):
            cross = " [cross-episode]" if c.get("cross_episode") else ""
            print(
                f"  [{c['cluster_id']}] "
                f"{c['member_count']} insights, "
                f"{c.get('episode_count', '?')} episodes{cross}"
            )
            canonical = c.get("canonical_insight", "")
            if len(canonical) > 100:
                canonical = canonical[:97] + "..."
            print(f"    Canonical: {canonical}")
            eps = c.get("episode_ids", [])
            if eps:
                print(f"    Episodes: {', '.join(eps[:5])}")
            print()

    return EXIT_SUCCESS
