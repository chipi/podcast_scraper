"""The ``cluster-corpus-topics`` command (moved verbatim from ``search.cli_handlers``, E04 S4;
reached through ``cli_hooks``).
"""

from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Sequence

from podcast_scraper.search.cli_handlers import EXIT_INVALID_ARGS, EXIT_NO_ARTIFACTS, EXIT_SUCCESS


def parse_cluster_corpus_topics_argv(argv: Sequence[str]) -> Namespace:
    """Parse ``cluster-corpus-topics`` args (#1058 chunk 3: cross-show
    Topic clustering for airgapped corpora)."""
    parser = argparse.ArgumentParser(prog="podcast_scraper cluster-corpus-topics")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Corpus root (parent of feeds/).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity cutoff for cluster joins (default 0.75).",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=2,
        help=(
            "Minimum distinct episodes per cluster (default 2 — the "
            "cross-show contract). Drop to 1 to also fold "
            "intra-show duplicates."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without mutating any artifacts.",
    )
    args = parser.parse_args(list(argv))
    args.command = "cluster-corpus-topics"
    return args


def run_cluster_corpus_topics_cli(args: Namespace, logger: logging.Logger) -> int:
    """Run the corpus-level Topic-clustering post-pass (#1058 chunk 3).

    Walks every ``*.kg.json`` under ``--output-dir``, clusters Topic
    labels via sentence-transformers cosine, and (unless ``--dry-run``)
    emits ``concept:topic-{slug}`` Topic nodes + ``RELATED_TO`` edges
    back to each source Topic. Idempotent.
    """
    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        return EXIT_INVALID_ARGS
    corpus = Path(output_dir)
    if not corpus.is_dir():
        logger.error("cluster-corpus-topics: corpus dir not found: %s", corpus)
        return EXIT_NO_ARTIFACTS

    from podcast_scraper.kg.topic_clustering import cluster_and_apply_corpus_topics

    summary = cluster_and_apply_corpus_topics(
        corpus,
        threshold=float(getattr(args, "threshold", 0.75)),
        min_episodes=int(getattr(args, "min_episodes", 2)),
        write=not bool(getattr(args, "dry_run", False)),
    )
    msg = (
        f"cluster-corpus-topics: clusters={summary.clusters_found} "
        f"concept_topics_added={summary.concept_topics_added} "
        f"related_to_edges_added={summary.related_to_edges_added} "
        f"artifacts_mutated={summary.artifacts_mutated}"
    )
    logger.info(msg)
    print(msg)
    return EXIT_SUCCESS
