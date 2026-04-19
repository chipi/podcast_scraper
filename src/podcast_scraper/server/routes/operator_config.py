"""GET/PUT /api/operator-config — non-secret operator YAML (path from serve flags / default)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, status

from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.operator_config_security import assert_operator_yaml_safe_for_persist
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import OperatorConfigGetResponse, OperatorConfigPutBody

router = APIRouter(tags=["operator-config"])


def _operator_file(request: Request) -> Path:
    raw = getattr(request.app.state, "operator_config_path", None)
    if raw is None:
        raise HTTPException(status_code=500, detail="operator_config_path is not configured.")
    return Path(raw)


@router.get("/operator-config", response_model=OperatorConfigGetResponse)
async def get_operator_config(
    request: Request,
    path: str = Query(
        ..., description="Corpus root (authorizes request; must resolve under anchor)."
    ),
) -> OperatorConfigGetResponse:
    """Return viewer-safe operator YAML from the configured resolved path."""
    anchor = getattr(request.app.state, "output_dir", None)
    corpus_root = resolve_corpus_path_param(path, anchor)
    cfg_path = _operator_file(request)
    if not cfg_path.is_file():
        return OperatorConfigGetResponse(
            corpus_path=str(corpus_root.resolve()),
            operator_config_path=str(cfg_path.resolve()),
            content="",
        )
    content = cfg_path.read_text(encoding="utf-8", errors="replace")
    try:
        assert_operator_yaml_safe_for_persist(content)
    except HTTPException as exc:
        detail = exc.detail
        if (
            exc.status_code == status.HTTP_400_BAD_REQUEST
            and isinstance(detail, dict)
            and detail.get("error") == "forbidden_operator_keys"
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="Existing operator YAML contains forbidden keys; fix the file out-of-band.",
            ) from exc
        raise
    return OperatorConfigGetResponse(
        corpus_path=str(corpus_root.resolve()),
        operator_config_path=str(cfg_path.resolve()),
        content=content,
    )


@router.put("/operator-config", response_model=OperatorConfigGetResponse)
async def put_operator_config(
    request: Request,
    body: OperatorConfigPutBody,
    path: str = Query(
        ..., description="Corpus root (authorizes request; must resolve under anchor)."
    ),
) -> OperatorConfigGetResponse:
    """Validate and atomically write operator YAML to the configured resolved path."""
    anchor = getattr(request.app.state, "output_dir", None)
    corpus_root = resolve_corpus_path_param(path, anchor)
    cfg_path = _operator_file(request)
    assert_operator_yaml_safe_for_persist(body.content)
    atomic_write_text(cfg_path, body.content)
    return OperatorConfigGetResponse(
        corpus_path=str(corpus_root.resolve()),
        operator_config_path=str(cfg_path.resolve()),
        content=body.content,
    )
