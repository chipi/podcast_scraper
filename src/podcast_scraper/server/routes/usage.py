"""Operator surface for LLM token/cost usage — ``GET /api/usage``.

A normal read (like /api/health, /api/resilience): rolls up the ``llm_cost`` telemetry a run wrote
to disk and slices it by any dimension — model / operation / episode / run / provider — with the
full token breakdown (input / output / cached / cache-write) and de-dup by request_id (no double).
Self-contained: no Loki / Langfuse required. Queryable by the o11y MCP tool and dashboards.

Thin wrapper over ``utils.usage_status`` — the single source of truth.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.usage_status import usage_rollup_snapshot

router = APIRouter(tags=["usage"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    raise HTTPException(
        status_code=400,
        detail="path query parameter is required when the server has no default output_dir.",
    )


@router.get("/usage")
async def usage(
    request: Request,
    path: str | None = Query(
        default=None, description="Corpus root (defaults to server output_dir)."
    ),
    group_by: str = Query(
        default="provider,model",
        description="Comma-separated slice dimensions: provider, model, served_model, operation, "
        "stage, episode_id, run_id, feed_id.",
    ),
    run_id: str | None = Query(default=None, description="Restrict to one run_id."),
) -> dict:
    """Token/cost rollup for the corpus, sliced by ``group_by`` (de-duplicated by request_id)."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    dims = tuple(d.strip() for d in group_by.split(",") if d.strip())
    return usage_rollup_snapshot(root, group_by=dims, run_id=run_id)
