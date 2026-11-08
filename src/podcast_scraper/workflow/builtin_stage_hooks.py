"""Today's in-tree stages, on the pipeline's stage hooks (E04 S2).

Each entry imports its stage inside its body, so this module loads without the stage code. An
entry goes when its stage moves to an extension, which then contributes it instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .stage_hooks import PipelineHooks

logger = logging.getLogger(__name__)


def _validate_grounding_dependencies(cfg: Any) -> None:
    from podcast_scraper.gi.deps import validate_gil_grounding_dependencies

    validate_gil_grounding_dependencies(cfg)


def _write_ad_signatures(corpus_root: Path) -> None:
    """Cross-show ad signatures: read segments, not the LanceDB index. Never raises; a missing or
    stale file makes the rule abstain."""
    from podcast_scraper.providers.ml.diarization.ad_signatures import write_for_corpus

    write_for_corpus(Path(corpus_root))


def _measure_naming_coverage(corpus_parent: Path) -> None:
    """The served corpus's naming coverage, once per batch, for the API's /metrics (#2304 S8).
    Absolute, not a diff: an episode no code version names was invisible to every other check."""
    from podcast_scraper.quality.naming_coverage import measure_and_record

    measure_and_record(Path(corpus_parent))


def _build_audio_references(corpus_root: str, run_dir: str) -> None:
    """Each new episode's audio reference, from the audio it was just transcribed from (RFC-129,
    #2359). Before eviction: that is the last moment the exact audio is on local disk. Never
    fatal: a run without numpy or ffmpeg simply has no references."""
    try:
        from podcast_scraper.audio_sync.references import write_for_run

        write_for_run(corpus_root, run_dir)
    except Exception:  # pragma: no cover - a finalize step must never break a finished run
        logger.warning("audio references: end-of-run pass failed (non-fatal)", exc_info=True)


def _cluster_corpus_topics(cfg: Any, output_dir: str) -> None:
    """#1058 chunk 3: corpus-level Topic clustering over every kg.json on disk. Gated on
    cfg.kg_topic_corpus_clustering (default off; airgapped* overlays flip it on). Non-fatal: a
    failure here doesn't bring down a successful run."""
    if not getattr(cfg, "kg_topic_corpus_clustering", False) or cfg.dry_run:
        return
    try:
        from podcast_scraper.kg.topic_clustering import cluster_and_apply_corpus_topics

        summary = cluster_and_apply_corpus_topics(Path(output_dir))
        logger.info(
            "corpus topic clustering: clusters=%d concept_topics_added=%d "
            "related_to_edges_added=%d artifacts_mutated=%d",
            summary.clusters_found,
            summary.concept_topics_added,
            summary.related_to_edges_added,
            summary.artifacts_mutated,
        )
    except Exception as cluster_exc:
        logger.warning(
            "corpus topic clustering failed (non-fatal): %s",
            cluster_exc,
            exc_info=True,
        )


BUILTIN = PipelineHooks(
    name="builtin",
    run_started=(_validate_grounding_dependencies,),
    corpus_derived=(_write_ad_signatures,),
    batch_finalized=(_measure_naming_coverage,),
    run_finalized=(_build_audio_references,),
    run_completed=(_cluster_corpus_topics,),
)
