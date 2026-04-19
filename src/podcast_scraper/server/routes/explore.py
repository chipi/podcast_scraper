"""GET /api/explore — GI cross-episode explore + UC4 natural-language query."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.gi.explore import (
    explore_output_to_rfc_dict,
    ExploreValidationError,
    run_uc4_semantic_qa,
    run_uc5_insight_explorer,
    scan_artifact_paths,
)
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import ExploreApiResponse

router = APIRouter(tags=["explore"])

_NO_PATTERN_HINT = (
    "Try patterns like: “What insights about X?”, “What did Y say?”, "
    "“What did Y say about X?”, “Which topics have the most insights?”"
)


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/explore", response_model=ExploreApiResponse)
async def explore_corpus(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (metadata/*.gi.json). Omit to use server default.",
    ),
    question: str | None = Query(
        default=None,
        description="Natural-language question (UC4). If set (or ``q``), runs gi query.",
    ),
    q: str | None = Query(
        default=None,
        description="Alias for ``question``.",
    ),
    topic: str | None = Query(default=None),
    speaker: str | None = Query(default=None),
    grounded_only: bool = Query(default=False),
    min_confidence: float | None = Query(default=None),
    sort_by: Literal["confidence", "time"] = Query(default="confidence"),
    limit: int = Query(default=50, ge=1, le=500),
    strict: bool = Query(default=False),
) -> ExploreApiResponse:
    """Filter explore (topic/speaker) or pattern-matched UC4 question."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        return ExploreApiResponse(kind="explore", error="no_corpus_path")

    nl = (question or q or "").strip()
    if nl:
        if not scan_artifact_paths(root):
            return ExploreApiResponse(
                kind="natural_language",
                error="no_artifacts",
                question=nl,
                detail="No .gi.json files under this corpus path.",
            )
        try:
            result = run_uc4_semantic_qa(root, nl, limit=limit, strict=strict)
        except ExploreValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"path": str(exc.path), "message": str(exc)},
            ) from exc
        if result is None:
            return ExploreApiResponse(
                kind="natural_language",
                error="no_pattern_match",
                question=nl,
                detail=_NO_PATTERN_HINT,
            )
        return ExploreApiResponse(
            kind="natural_language",
            question=str(result.get("question") or nl),
            answer=result.get("answer") if isinstance(result.get("answer"), dict) else None,
            explanation=(
                str(result["explanation"]) if result.get("explanation") is not None else None
            ),
        )

    try:
        out = run_uc5_insight_explorer(
            root,
            topic=topic.strip() if topic and topic.strip() else None,
            speaker=speaker.strip() if speaker and speaker.strip() else None,
            grounded_only=grounded_only,
            min_confidence=min_confidence,
            sort_by=sort_by,
            limit=limit,
            strict=strict,
        )
    except ExploreValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"path": str(exc.path), "message": str(exc)},
        ) from exc

    return ExploreApiResponse(
        kind="explore",
        data=explore_output_to_rfc_dict(out),
    )
