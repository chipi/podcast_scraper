"""GET /api/corpus/theme-clusters — ``topic_theme_clusters.json`` overlay.

Theme clusters group topics *discussed together* (co-occurrence lift), as
opposed to ``/api/corpus/topic-clusters`` which serves the *semantic*
(embedding-similarity) clusters. The two are complementary and themed apart in
the consumer. Produced by the ``topic_theme_clusters`` enricher under
``enrichments/`` (not ``search/`` — different producer).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper import perf_cache
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])

_THEME_CLUSTERS_REL = "enrichments/topic_theme_clusters.json"

#: Smallest theme worth offering as a NAVIGATION destination (default; overridable per request).
#:
#: Themes are the browse surface — the operator's top-down zoom-out and the player's Storylines —
#: so a theme is a place a reader is sent, not merely a fact the corpus contains. Measured on the
#: 1,066-episode corpus (54 themes):
#:
#:     >= 2 members  54 themes  median  3 episodes     <- 27 of them are a single co-occurrence pair
#:     >= 3 members  27 themes  median  4 episodes
#:     >= 4 members  18 themes  median  6 episodes     <- 192 of 286 episodes still reachable
#:     >= 6 members   8 themes  median 14 episodes
#:
#: 4 drops 67% of themes but only 33% of episode coverage, because the discarded ones are tiny and
#: overlap what remains. A 2-member/3-episode theme is a co-occurrence pair, not a destination.
#:
#: Filtered at the SURFACING layer, deliberately, not in the enricher: the artifact keeps every
#: theme (the co-occurrence evidence is real, other consumers read it, and #1929's cluster-count
#: diagnostics need the full set), and this number can change without recomputing enrichment.
_DEFAULT_MIN_THEME_MEMBERS = 4


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


def _filter_by_min_members(payload: dict, min_members: int) -> dict:
    """Drop themes below *min_members* for the navigation surface (see the constant above).

    Never mutates the cached payload — ``perf_cache`` hands back the same object on every hit, so
    filtering in place would poison later requests (including ``min_members=0``) with whatever the
    first caller asked for.

    Reports what it withheld rather than silently shrinking the list: a consumer that sees
    ``clusters`` shorter than ``cluster_count`` should be able to tell filtering from an empty
    corpus, which is exactly the distinction #1929 is about.
    """
    clusters = payload.get("clusters")
    if min_members <= 0 or not isinstance(clusters, list):
        return payload

    def _size(c: object) -> int:
        if not isinstance(c, dict):
            return 0
        n = c.get("member_count")
        if isinstance(n, int):
            return n
        members = c.get("members")
        return len(members) if isinstance(members, list) else 0

    kept = [c for c in clusters if _size(c) >= min_members]
    if len(kept) == len(clusters):
        return payload
    out = dict(payload)
    out["clusters"] = kept
    out["surfaced_cluster_count"] = len(kept)
    out["withheld_below_min_members"] = len(clusters) - len(kept)
    out["min_members"] = min_members
    return out


@router.get("/corpus/theme-clusters")
async def corpus_theme_clusters(
    request: Request,
    path: str | None = Query(
        default=None,
        description=(
            "Corpus output dir (contains enrichments/). Omit to use server default output_dir."
        ),
    ),
    min_members: int = Query(
        default=_DEFAULT_MIN_THEME_MEMBERS,
        ge=0,
        description=(
            "Smallest theme to surface as a navigation destination. 0 returns the unfiltered "
            "artifact (diagnostics / #1929 cluster-count checks)."
        ),
    ),
) -> JSONResponse:
    """Return ``<corpus>/enrichments/topic_theme_clusters.json`` when present."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        raise HTTPException(
            status_code=400,
            detail="path query parameter is required when the server has no default output_dir.",
        )

    root_dir = safe_resolve_directory(root)
    if root_dir is None:
        raise HTTPException(status_code=400, detail="Invalid corpus path.")

    root_s = os.path.normpath(str(root_dir))
    safe_prefix = root_s + os.sep
    parts = [p for p in _THEME_CLUSTERS_REL.replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="Invalid corpus path.")
    joined = os.path.normpath(os.path.join(root_s, *parts))
    not_found = JSONResponse(
        status_code=404,
        content={
            "detail": "topic_theme_clusters.json not found under corpus enrichments/",
            "available": False,
        },
    )
    if joined != root_s and not joined.startswith(safe_prefix):
        return not_found
    # codeql[py/path-injection] -- joined under root_s (Type 1; CODEQL_DISMISSALS.md).
    if not os.path.isfile(joined):
        return not_found

    def _load() -> dict:
        try:
            # codeql[py/path-injection] -- joined sanitized above.
            with open(joined, encoding="utf-8") as fh:
                loaded = json.loads(fh.read())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("corpus_theme_clusters: failed to read %s: %s", joined, exc)
            raise HTTPException(
                status_code=500,
                detail="topic_theme_clusters.json is unreadable or invalid JSON.",
            ) from exc
        if not isinstance(loaded, dict):
            raise HTTPException(
                status_code=500,
                detail="topic_theme_clusters.json must be a JSON object.",
            )
        # The enrichment framework wraps enricher output in an envelope
        # ({derived, enricher_id, ..., data: {...}}). Serve the inner payload so the
        # response matches the un-enveloped semantic /api/corpus/topic-clusters shape
        # (clusters at top level). Tolerates an already-unwrapped file.
        inner = loaded.get("data")
        return inner if isinstance(inner, dict) else loaded

    # Whole-artifact read+parse, cached by the file's OWN mtime (the enricher rewrites it without
    # bumping corpus_run_summary.json, so a corpus-mtime token would go stale). compute() raises the
    # 500 on corrupt/non-object; get_or_compute never stores on exception, preserving the contract.
    # joined is normpath'd + startswith(safe_prefix)-guarded above; getmtime only stats it.
    # codeql[py/path-injection] -- joined startswith(safe_prefix)-guarded above (Type 1).
    artifact_mtime = os.path.getmtime(joined)
    payload = perf_cache.get_or_compute("corpus_theme_clusters", joined, artifact_mtime, _load)
    payload = _filter_by_min_members(payload, min_members)
    _clusters = payload.get("clusters")
    _n = len(_clusters) if isinstance(_clusters, list) else None
    logger.debug(
        "corpus_theme_clusters: serving schema_version=%s cluster_entries=%s min_members=%s",
        payload.get("schema_version"),
        _n,
        min_members,
    )
    return JSONResponse(content=payload)
