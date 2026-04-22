"""Per-corpus cost rollup (#650 Finding 20).

Walks ``<corpus_parent>/feeds/<slug>/run_*/metrics.json`` and sums the
per-stage ``llm_*_cost_usd`` fields populated by
:mod:`podcast_scraper.workflow.metrics`. Produces a single dict with
per-stage breakdown plus three authoritative totals — transcription,
non-transcription LLM, and overall — suitable for embedding in
``corpus_manifest.json`` / ``corpus_run_summary.json`` or printing via
the ``corpus-cost`` CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Stage fields populated by Metrics.to_dict() post-#650.
_STAGE_COST_FIELDS: tuple[str, ...] = (
    "llm_transcription_cost_usd",
    "llm_summarization_cost_usd",
    "llm_speaker_detection_cost_usd",
    "llm_cleaning_cost_usd",
    "llm_gi_cost_usd",
    "llm_kg_cost_usd",
    "llm_bundled_clean_summary_cost_usd",
)


def aggregate_corpus_costs(corpus_parent: Path | str) -> Dict[str, Any]:
    """Sum per-stage costs across every ``run_*/metrics.json`` under *corpus_parent*.

    Layout assumed (RFC-corpus):
    ``<corpus_parent>/feeds/<slug>/run_*/metrics.json``

    Returns a dict with:

    - ``total_transcription_cost_usd``: sum of transcription-stage costs.
    - ``total_llm_cost_usd``: sum of non-transcription LLM costs
      (summarization + GI + KG + speaker detection + cleaning).
    - ``total_cost_usd``: overall (transcription + LLM).
    - ``by_stage``: mapping of each ``llm_*_cost_usd`` to its summed value.
    - ``run_count``: number of ``metrics.json`` files aggregated.
    - ``metrics_files_missing_cost_fields``: count of files that lacked
      any of the stage fields (likely pre-#650 artifacts).

    Malformed ``metrics.json`` files are logged and skipped; the aggregate
    reflects everything successfully parsed. Empty corpora return zeros.
    """
    root = Path(corpus_parent)
    feeds_root = root / "feeds"

    by_stage: Dict[str, float] = {f: 0.0 for f in _STAGE_COST_FIELDS}
    run_count = 0
    missing_cost_fields = 0

    # Multi-feed (corpus) layout: <root>/feeds/<slug>/run_*/metrics.json
    # Single-feed layout:         <root>/run_*/metrics.json
    # Support both — single-feed users otherwise see "0 runs" even though
    # the run exists (discovered during #650 Layer 3 validation).
    if feeds_root.exists():
        pattern = feeds_root.glob("*/run_*/metrics.json")
    else:
        pattern = root.glob("run_*/metrics.json")

    for metrics_file in sorted(pattern):
        try:
            doc = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("aggregate_corpus_costs: skipping unreadable %s (%s)", metrics_file, exc)
            continue
        if not isinstance(doc, dict):
            logger.warning("aggregate_corpus_costs: skipping %s (not a JSON object)", metrics_file)
            continue
        run_count += 1
        any_field_present = False
        for field in _STAGE_COST_FIELDS:
            val = doc.get(field)
            if isinstance(val, (int, float)):
                by_stage[field] += float(val)
                any_field_present = True
        if not any_field_present:
            missing_cost_fields += 1

    return _build_result(by_stage, run_count, missing_cost_fields)


def _build_result(
    by_stage: Dict[str, float], run_count: int, missing_cost_fields: int
) -> Dict[str, Any]:
    transcription = by_stage["llm_transcription_cost_usd"]
    llm_non_transcription = sum(
        by_stage[f] for f in _STAGE_COST_FIELDS if f != "llm_transcription_cost_usd"
    )
    total = transcription + llm_non_transcription

    return {
        "total_transcription_cost_usd": round(transcription, 6),
        "total_llm_cost_usd": round(llm_non_transcription, 6),
        "total_cost_usd": round(total, 6),
        "by_stage": {f: round(v, 6) for f, v in by_stage.items()},
        "run_count": run_count,
        "metrics_files_missing_cost_fields": missing_cost_fields,
    }
