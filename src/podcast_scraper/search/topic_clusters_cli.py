"""``podcast_scraper topic-clusters``: build the themes artifact from the vector index.

Registered by the intelligence extension (ADR-158); the platform CLI dispatches to it by name.
"""

from __future__ import annotations

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import cast, Sequence

from podcast_scraper.search.cli_handlers import (
    _resolve_index_dir,
    EXIT_INVALID_ARGS,
    EXIT_NO_ARTIFACTS,
    EXIT_SUCCESS,
)
from podcast_scraper.utils.log_redaction import format_exception_for_log


def parse_topic_clusters_argv(argv: Sequence[str]) -> Namespace:
    """Parse argv after ``topic-clusters``."""
    from podcast_scraper.search.topic_clusters import DEFAULT_TOPIC_CLUSTER_THRESHOLD

    parser = argparse.ArgumentParser(
        prog="podcast_scraper topic-clusters",
        description=(
            "Cluster KG topics from indexed kg_topic embeddings; write topic_clusters.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pipeline output directory (contains search/ index)",
    )
    parser.add_argument(
        "--index-path",
        dest="vector_index_path",
        default=None,
        help="Vector index directory (default: <output-dir>/search)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_TOPIC_CLUSTER_THRESHOLD,
        help=(
            "Minimum cosine similarity to link topics in the same cluster "
            f"(default: {DEFAULT_TOPIC_CLUSTER_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Write JSON here (default: <index-dir>/topic_clusters.json)",
    )
    parser.add_argument(
        "--validate-config",
        default=None,
        metavar="PATH",
        help=(
            "After clustering, load a validation YAML (path you choose; see topic clustering docs) "
            "and exit non-zero if constraints fail"
        ),
    )
    parser.add_argument(
        "--merge-cil-overrides",
        action="store_true",
        help=(
            "After writing topic_clusters.json, merge derived topic_id_aliases into "
            "cil_lift_overrides.json (existing file entries take precedence over auto)"
        ),
    )
    ns = cast(Namespace, parser.parse_args(list(argv)))
    ns.command = "topic-clusters"
    return ns


def run_topic_clusters_cli(args: Namespace, logger: logging.Logger) -> int:
    """Build ``topic_clusters.json`` for a corpus; optional validation YAML check."""
    from podcast_scraper.search.topic_clusters import (
        build_topic_clusters_for_corpus,
        DEFAULT_TOPIC_CLUSTER_THRESHOLD,
        evaluate_validation_against_topics,
        load_validation_yaml,
    )

    output_dir = getattr(args, "output_dir", None)
    if not output_dir:
        logger.error("topic-clusters: --output-dir is required")
        return EXIT_INVALID_ARGS

    index_dir = _resolve_index_dir(Path(output_dir), getattr(args, "vector_index_path", None))
    out_file = getattr(args, "output_file", None)
    out_path = Path(out_file).resolve() if out_file else None
    threshold = float(
        getattr(args, "threshold", DEFAULT_TOPIC_CLUSTER_THRESHOLD)
        or DEFAULT_TOPIC_CLUSTER_THRESHOLD
    )

    try:
        payload = build_topic_clusters_for_corpus(
            output_dir,
            index_dir=index_dir,
            threshold=threshold,
            out_path=out_path,
        )
    except FileNotFoundError as exc:
        logger.error("topic-clusters: %s", exc)
        return EXIT_NO_ARTIFACTS
    except Exception as exc:
        logger.error("topic-clusters: %s", format_exception_for_log(exc))
        return EXIT_INVALID_ARGS

    vcfg = getattr(args, "validate_config", None)
    if vcfg:
        spec_path = Path(vcfg).resolve()
        if not spec_path.is_file():
            logger.error("topic-clusters: --validate-config not found: %s", spec_path)
            return EXIT_INVALID_ARGS
        spec = load_validation_yaml(spec_path)
        import numpy as np

        from podcast_scraper.search.cluster_math import cluster_labels_by_threshold
        from podcast_scraper.search.topic_clusters import (
            collect_topic_rows_from_lance,
            load_kg_topic_labels_from_corpus,
        )

        rows = collect_topic_rows_from_lance(
            index_dir / "lance_index",
            load_kg_topic_labels_from_corpus(Path(output_dir).resolve()),
        )
        if not rows:
            logger.error("topic-clusters: validate: no kg_topic rows in index")
            return EXIT_INVALID_ARGS
        ids = [r.topic_id for r in rows]
        mat = np.stack([r.vector for r in rows], axis=0)
        labels = cluster_labels_by_threshold(mat, threshold)
        ok, errors = evaluate_validation_against_topics(spec, ids, labels.tolist())
        if not ok:
            for err in errors:
                logger.error("validation: %s", err)
            return EXIT_INVALID_ARGS
        logger.info("topic-clusters: validation OK")

    if getattr(args, "merge_cil_overrides", False):
        if payload.get("skipped_unchanged"):
            # C1 skip-gate: clusters unchanged, so their aliases are unchanged too —
            # deriving aliases from the skip-stub would yield an empty set. Nothing to merge.
            logger.info(
                "topic-clusters: clusters unchanged (skip-gate) — cil_lift_overrides merge skipped"
            )
        else:
            from podcast_scraper.search.cil_lift_overrides import (
                write_cil_lift_overrides_merged_topic_id_aliases,
            )
            from podcast_scraper.search.topic_clusters import (
                topic_id_aliases_from_clusters_payload,
            )

            root = Path(output_dir).resolve()
            auto = topic_id_aliases_from_clusters_payload(payload)
            try:
                merged = write_cil_lift_overrides_merged_topic_id_aliases(root, auto)
            except ValueError as exc:
                logger.error("topic-clusters: %s", exc)
                return EXIT_INVALID_ARGS
            logger.info(
                "topic-clusters: cil_lift_overrides.json topic_id_aliases=%s keys (auto=%s)",
                len(merged),
                len(auto),
            )

    logger.info(
        "topic-clusters: schema_version=%s topics=%s clusters=%s singleton_topic_rows=%s",
        payload.get("schema_version"),
        payload.get("topic_count"),
        payload.get("cluster_count"),
        payload.get("singletons"),
    )
    return EXIT_SUCCESS


# ── insight-clusters (#599) ──────────────────────────────────────────
