"""GET /api/health — always available."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper import __version__
from podcast_scraper.corpus_version import (
    assess_corpus_version_compat,
    corpus_code_version,
    MIN_SUPPORTED_CORPUS_CODE_VERSION,
    read_produced_by,
)
from podcast_scraper.server.pathutil import CorpusPathRequestError, resolve_corpus_path_param
from podcast_scraper.server.schemas import CorpusProducedBy, HealthResponse

router = APIRouter(tags=["health"])


def _corpus_dir_for_health(
    path: str | None,
    default_output_dir: Path | None,
) -> Path | None:
    """Resolve corpus root for version preflight (optional ``path`` query)."""
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, default_output_dir)
    if isinstance(default_output_dir, Path) and default_output_dir.is_dir():
        return default_output_dir
    return None


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    path: str | None = Query(
        default=None,
        description=(
            "Optional corpus root for produced_by preflight. When omitted, uses the "
            "server default output_dir when configured."
        ),
    ),
) -> HealthResponse:
    """Server health check."""
    st = request.app.state
    produced_by_raw = None
    corpus_ver = None
    warning = None
    default_output_dir = getattr(st, "output_dir", None)
    anchor = default_output_dir if isinstance(default_output_dir, Path) else None
    corpus_dir: Path | None = None
    if path is not None and str(path).strip():
        try:
            corpus_dir = _corpus_dir_for_health(path, anchor)
        except CorpusPathRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    else:
        corpus_dir = _corpus_dir_for_health(None, anchor)
    if corpus_dir is not None and corpus_dir.is_dir():
        produced_by_raw = read_produced_by(corpus_dir)
        corpus_ver, warning = assess_corpus_version_compat(produced_by_raw)
    corpus_produced_by = None
    if produced_by_raw:
        cv = corpus_code_version(produced_by_raw)
        if cv is not None:
            try:
                corpus_produced_by = CorpusProducedBy.model_validate(produced_by_raw)
            except Exception:
                corpus_produced_by = None
    return HealthResponse().model_copy(
        update={
            "code_version": __version__,
            "min_supported_corpus_code_version": MIN_SUPPORTED_CORPUS_CODE_VERSION,
            "corpus_produced_by": corpus_produced_by,
            "corpus_code_version": corpus_ver,
            "corpus_version_warning": warning,
            "feeds_api": bool(getattr(st, "feeds_api_enabled", False)),
            "operator_config_api": bool(getattr(st, "operator_config_api_enabled", False)),
            "jobs_api": bool(getattr(st, "jobs_api_enabled", False)),
        }
    )
