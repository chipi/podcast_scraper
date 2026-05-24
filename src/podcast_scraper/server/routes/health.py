"""GET /api/health — always available."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from podcast_scraper import __version__
from podcast_scraper.corpus_version import (
    assess_corpus_version_compat,
    corpus_code_version,
    MIN_SUPPORTED_CORPUS_CODE_VERSION,
    read_produced_by,
)
from podcast_scraper.server.schemas import CorpusProducedBy, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Server health check."""
    st = request.app.state
    produced_by_raw = None
    corpus_ver = None
    warning = None
    output_dir = getattr(st, "output_dir", None)
    if isinstance(output_dir, Path) and output_dir.is_dir():
        produced_by_raw = read_produced_by(output_dir)
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
