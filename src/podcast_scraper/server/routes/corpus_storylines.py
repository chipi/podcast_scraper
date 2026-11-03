"""GET /api/corpus/storylines — ``topic_theme_clusters.json`` overlay.

Storylines group topics *discussed together* (co-occurrence lift), as opposed to
``/api/corpus/topic-clusters``, which serves the *semantic*
(embedding-similarity) clusters the consumer calls **themes**. The two are
complementary and surfaced apart. Produced by the ``topic_theme_clusters``
enricher under ``enrichments/`` (not ``search/`` — different producer); the
artifact filename keeps its old name, which is why it still reads "theme".
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper import perf_cache
from podcast_scraper.search.storylines import DEFAULT_MIN_STORYLINE_MEMBERS
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.routes.app_auth import require_viewer_access
from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])

_STORYLINES_REL = "enrichments/topic_theme_clusters.json"


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


@router.get("/corpus/storylines")
def corpus_storylines(
    request: Request,
    path: str | None = Query(
        default=None,
        description=(
            "Corpus output dir (contains enrichments/). Omit to use server default output_dir."
        ),
    ),
    min_members: int = Query(
        default=DEFAULT_MIN_STORYLINE_MEMBERS,
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
    parts = [p for p in _STORYLINES_REL.replace("\\", "/").split("/") if p and p != "."]
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
            logger.warning("corpus_storylines: failed to read %s: %s", joined, exc)
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
    payload = perf_cache.get_or_compute("corpus_storylines", joined, artifact_mtime, _load)
    payload = _filter_by_min_members(payload, min_members)
    _clusters = payload.get("clusters")
    _n = len(_clusters) if isinstance(_clusters, list) else None
    logger.debug(
        "corpus_storylines: serving schema_version=%s cluster_entries=%s min_members=%s",
        payload.get("schema_version"),
        _n,
        min_members,
    )
    return JSONResponse(content=payload)


def _series(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    return [
        {"ts": r.get("ts"), "source": r.get("source"), **{k: r.get(k) for k in keys}} for r in rows
    ]


@router.get("/corpus/clustering/health")
def corpus_clustering_health(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output dir. Omit for default."),
) -> JSONResponse:
    """Storylines and themes over time, for the viewer dashboard (RFC-128, #2349).

    From ``enrichments/clustering_history*.jsonl`` (one row per run; the storyline backfill from
    the history seed's replay) and the storyline artifact's ``registry_health``: the latest run,
    its changes by name, the series to chart, and whether a history seed is pending or taken.
    """
    from podcast_scraper.search import clustering_history

    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    root_dir = safe_resolve_directory(root) if root is not None else None
    if root_dir is None:
        raise HTTPException(status_code=400, detail="Invalid corpus path.")
    root_s = os.path.normpath(str(root_dir))

    def under(rel: Path) -> Path:
        # Same guard as ``corpus_storylines`` above: a constant relative path, joined and
        # normalised, must stay under the validated corpus root (CODEQL_DISMISSALS.md Type 1).
        joined = os.path.normpath(os.path.join(root_s, *rel.parts))
        if not joined.startswith(root_s + os.sep):
            raise HTTPException(status_code=400, detail="Invalid corpus path.")
        return Path(joined)

    history = under(clustering_history.HISTORY)
    backfill = under(clustering_history.BACKFILL)
    artifact = under(Path(_STORYLINES_REL))
    story_rows = clustering_history.read_files(history, backfill, "storyline")
    theme_rows = clustering_history.read_files(history, backfill, "theme")
    health = None
    try:
        raw = json.loads(artifact.read_text(encoding="utf-8"))
        data = raw.get("data", raw) if isinstance(raw, dict) else {}
        health = data.get("registry_health") if isinstance(data, dict) else None
    except (OSError, ValueError):
        pass
    story_keys = (
        "storylines",
        "active",
        "quiet",
        "candidates",
        "episodes_covered",
        "episodes_total",
        "ok",
    )
    return JSONResponse(
        {
            "storylines": {
                "latest": story_rows[-1] if story_rows else None,
                "health": health,
                "seed": clustering_history.seed_status_files(
                    under(clustering_history.SEED), artifact
                ),
                "series": _series(story_rows, story_keys),
            },
            "themes": {
                "latest": theme_rows[-1] if theme_rows else None,
                "series": _series(theme_rows, ("themes", "topics", "kept")),
            },
        }
    )


@router.post(
    "/corpus/storylines/seed",
    responses={400: {"description": "Bad corpus path or mode"}},
)
async def storylines_history_seed(
    request: Request,
    path: str | None = Query(
        default=None, description="Corpus output dir. Omit for server default."
    ),
    mode: str = Query(default="preview", description="preview | apply | undo"),
    _user=Depends(require_viewer_access),
) -> JSONResponse:
    """The storyline history seed (RFC-128, #2349), from the dashboard: ``preview`` replays and
    reports without writing; ``apply`` writes the seed the next storyline run takes in (and the
    chart's past); ``undo`` withdraws it, or seeds back what a taken one replaced.

    Operator surface only, like the theme rebuild. Never writes the storyline artifact,
    so it cannot race a run. Seconds on production (no model).
    """
    import asyncio

    from podcast_scraper.search import storyline_history

    if mode not in ("preview", "apply", "undo"):
        raise HTTPException(status_code=400, detail="mode must be preview, apply or undo.")
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        raise HTTPException(status_code=400, detail="Corpus path is required (query or default).")
    if mode == "undo":
        result = await asyncio.to_thread(storyline_history.undo, root)
    else:
        result = await asyncio.to_thread(storyline_history.seed, root, apply=mode == "apply")
    return JSONResponse({"mode": mode, **result})
