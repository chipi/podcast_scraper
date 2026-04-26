"""Corpus aggregates and ``run.json`` discovery for the GI/KG dashboard."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper.search.corpus_scope import normalize_feed_id
from podcast_scraper.server.corpus_catalog import build_catalog_rows
from podcast_scraper.server.corpus_digest import load_digest_topics
from podcast_scraper.server.pathutil import resolved_corpus_root_str
from podcast_scraper.server.routes.corpus_library import _resolve_corpus_root
from podcast_scraper.server.schemas import (
    CorpusRunsSummaryResponse,
    CorpusRunSummaryItem,
    CorpusStatsResponse,
)
from podcast_scraper.utils.path_validation import safe_fixed_file_under_root

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])

_MAX_RUN_JSON_FILES = 150


def _float_metric(m: dict[str, Any], key: str) -> float | None:
    v = m.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _int_metric(m: dict[str, Any], key: str) -> int | None:
    v = m.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return None


# ``run.json`` written before #fix used ``json.dumps(..., default=str)`` on dataclass rows.
_LEGACY_EP_STATUS_RE = re.compile(
    r"""status=(?:'(?P<s1>ok|failed|skipped)'|\"(?P<s2>ok|failed|skipped)\")""",
)


def _episode_outcomes(m: dict[str, Any]) -> dict[str, int]:
    outcomes: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    statuses = m.get("episode_statuses")
    if not isinstance(statuses, list):
        return outcomes
    for row in statuses:
        if isinstance(row, dict):
            st = row.get("status")
            if st in outcomes:
                outcomes[str(st)] += 1
            continue
        if isinstance(row, str):
            match = _LEGACY_EP_STATUS_RE.search(row)
            if match:
                key = match.group("s1") or match.group("s2")
                if key in outcomes:
                    outcomes[key] += 1
    return outcomes


def _parse_run_json(path_str: str, root_safe: str) -> CorpusRunSummaryItem | None:
    try:
        # codeql[py/path-injection] -- path_str from normpath+startswith in os.walk guard.
        with open(path_str, encoding="utf-8") as fh:
            raw_any: Any = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw_any, dict):
        return None
    raw: dict[str, Any] = raw_any
    metrics_any = raw.get("metrics")
    m: dict[str, Any] = metrics_any if isinstance(metrics_any, dict) else {}

    rel = os.path.relpath(path_str, root_safe).replace("\\", "/")
    if rel.startswith(".."):
        rel = os.path.basename(path_str)

    rid = raw.get("run_id")
    cat = raw.get("created_at")
    return CorpusRunSummaryItem(
        relative_path=rel,
        run_id=str(rid) if rid is not None else "",
        created_at=str(cat) if isinstance(cat, str) else None,
        run_duration_seconds=_float_metric(m, "run_duration_seconds"),
        episodes_scraped_total=_int_metric(m, "episodes_scraped_total"),
        errors_total=_int_metric(m, "errors_total"),
        gi_artifacts_generated=_int_metric(m, "gi_artifacts_generated"),
        kg_artifacts_generated=_int_metric(m, "kg_artifacts_generated"),
        time_scraping_seconds=_float_metric(m, "time_scraping"),
        time_parsing_seconds=_float_metric(m, "time_parsing"),
        time_normalizing_seconds=_float_metric(m, "time_normalizing"),
        time_io_and_waiting_seconds=_float_metric(m, "time_io_and_waiting"),
        episode_outcomes=_episode_outcomes(m),
        # #656 Stage B: #652 Part B filter counters (ads, dialogue, topic
        # normalization, entity-kind repair). ``_int_metric`` returns
        # ``None`` when the field is absent — legacy runs pre-#652 render
        # with "—" in the viewer instead of "0".
        ads_filtered_count=_int_metric(m, "ads_filtered_count"),
        dialogue_insights_dropped_count=_int_metric(m, "dialogue_insights_dropped_count"),
        topics_normalized_count=_int_metric(m, "topics_normalized_count"),
        entity_kinds_repaired_count=_int_metric(m, "entity_kinds_repaired_count"),
        # #656 Stage D: #663 pre-extraction ad-region excision counters.
        ad_chars_excised_preroll=_int_metric(m, "ad_chars_excised_preroll"),
        ad_chars_excised_postroll=_int_metric(m, "ad_chars_excised_postroll"),
        ad_episodes_with_excision_count=_int_metric(m, "ad_episodes_with_excision_count"),
    )


@router.get("/corpus/stats", response_model=CorpusStatsResponse)
async def corpus_stats(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit to use server default output_dir.",
    ),
) -> CorpusStatsResponse:
    """Publish-month histogram from catalog scan (one pass)."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    rows = build_catalog_rows(root)
    hist: dict[str, int] = {}
    feed_ids: set[str] = set()
    for r in rows:
        fid = normalize_feed_id(r.feed_id)
        if fid:
            feed_ids.add(fid)
        pd = r.publish_date
        if not pd or len(pd) < 7:
            continue
        month = pd[:7]
        hist[month] = hist.get(month, 0) + 1
    topic_bands = load_digest_topics()
    return CorpusStatsResponse(
        path=str(root),
        publish_month_histogram=hist,
        catalog_episode_count=len(rows),
        catalog_feed_count=len(feed_ids),
        digest_topics_configured=len(topic_bands),
    )


@router.get("/corpus/documents/manifest")
async def corpus_manifest_document(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
) -> JSONResponse:
    """Return ``corpus_manifest.json`` at corpus root if present."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    root_safe = resolved_corpus_root_str(root, anchor)
    fp = safe_fixed_file_under_root(Path(root_safe), "corpus_manifest.json")
    # codeql[py/path-injection] -- fp from normpath+startswith in safe_fixed_file.
    if not fp or not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="corpus_manifest.json not found.")
    try:
        # codeql[py/path-injection] -- fp sanitized above.
        with open(fp, encoding="utf-8") as fh:
            data_any: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON at %s", fp)
        raise HTTPException(
            status_code=500,
            detail="corpus_manifest.json is not valid JSON.",
        ) from exc
    if not isinstance(data_any, dict):
        raise HTTPException(
            status_code=500,
            detail="corpus_manifest.json must be a JSON object.",
        )
    return JSONResponse(content=data_any)


@router.get("/corpus/documents/run-summary")
async def corpus_run_summary_document(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
) -> JSONResponse:
    """Return ``corpus_run_summary.json`` at corpus root if present."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    root_safe = resolved_corpus_root_str(root, anchor)
    fp = safe_fixed_file_under_root(Path(root_safe), "corpus_run_summary.json")
    # codeql[py/path-injection] -- fp from normpath+startswith in safe_fixed_file.
    if not fp or not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="corpus_run_summary.json not found.")
    try:
        # codeql[py/path-injection] -- fp sanitized above.
        with open(fp, encoding="utf-8") as fh:
            data_any: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON at %s", fp)
        raise HTTPException(
            status_code=500,
            detail="corpus_run_summary.json is not valid JSON.",
        ) from exc
    if not isinstance(data_any, dict):
        raise HTTPException(
            status_code=500,
            detail="corpus_run_summary.json must be a JSON object.",
        )
    return JSONResponse(content=data_any)


@router.get("/corpus/runs/summary", response_model=CorpusRunsSummaryResponse)
async def corpus_runs_summary(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
) -> CorpusRunsSummaryResponse:
    """Discover ``run.json`` files (mtime order, capped) and return compact metrics."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    root_safe = resolved_corpus_root_str(root, anchor)
    discovered: list[str] = []
    safe_prefix = root_safe + os.sep
    for dirpath, _, filenames in os.walk(root_safe):
        if "run.json" not in filenames:
            continue
        candidate = os.path.normpath(os.path.join(dirpath, "run.json"))
        if candidate != root_safe and not candidate.startswith(safe_prefix):
            continue
        if os.path.isfile(candidate):
            discovered.append(candidate)
    discovered.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    discovered = discovered[:_MAX_RUN_JSON_FILES]
    runs: list[CorpusRunSummaryItem] = []
    for p in discovered:
        item = _parse_run_json(p, root_safe)
        if item is not None:
            runs.append(item)
    runs.sort(key=lambda x: (x.created_at or "", x.relative_path), reverse=True)
    return CorpusRunsSummaryResponse(path=str(root), runs=runs)
