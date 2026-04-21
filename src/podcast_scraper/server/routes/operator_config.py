"""GET/PUT /api/operator-config — non-secret operator YAML (path from serve flags / default)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, status

from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.operator_config_security import (
    assert_operator_yaml_safe_for_persist,
    OperatorYamlUnsafeError,
)
from podcast_scraper.server.operator_paths import (
    packaged_viewer_operator_example_path,
    viewer_operator_yaml_path,
)
from podcast_scraper.server.operator_yaml_profile import expand_profile_only_with_packaged_example
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.profile_presets import list_packaged_profile_names
from podcast_scraper.server.schemas import OperatorConfigGetResponse, OperatorConfigPutBody
from podcast_scraper.utils.path_validation import normpath_if_under_root

router = APIRouter(tags=["operator-config"])


def _verified_operator_config_path(request: Request, corpus_root: Path, cfg_path: Path) -> Path:
    """Return ``cfg_path`` for filesystem use after matching server routing rules."""
    cfg_s = os.path.normpath(str(cfg_path.resolve()))
    fixed = getattr(request.app.state, "operator_config_fixed_path", None)
    if fixed is not None:
        fixed_s = os.path.normpath(str(Path(str(fixed)).resolve()))
        if cfg_s != fixed_s:
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Operator config path does not match server fixed operator path.",
            )
        return Path(cfg_s)
    root_s = os.path.normpath(str(corpus_root.resolve()))
    ok = normpath_if_under_root(cfg_s, root_s)
    if not ok:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Invalid operator config path.",
        )
    return Path(ok)


def _ensure_default_viewer_operator_yaml(cfg_path: Path) -> None:
    """Seed ``viewer_operator.yaml`` when absent or whitespace-only.

    Copies packaged ``config/examples/viewer_operator.example.yaml`` when present (no
    ``profile:`` in that file — operators choose preset in the viewer Profile menu or
    ``--profile`` on the CLI, same merge order as a thin ``--config`` file).

    When the file exists but is only a ``profile:`` line (viewer Save with empty overrides),
    merges packaged example overrides under that profile (same disk shape as a thin
    ``--profile`` + ``--config`` pair).
    """
    example_path = packaged_viewer_operator_example_path()
    # codeql[py/path-injection] -- cfg_path verified via _verified_operator_config_path (Type 1).
    if cfg_path.is_file():
        # codeql[py/path-injection] -- cfg_path verified before _ensure_default (Type 1).
        existing = cfg_path.read_text(encoding="utf-8", errors="replace")
        if existing.strip():
            expanded = expand_profile_only_with_packaged_example(
                existing, example_path=example_path
            )
            if expanded != existing:
                try:
                    assert_operator_yaml_safe_for_persist(expanded)
                except OperatorYamlUnsafeError:
                    return
                atomic_write_text(cfg_path, expanded)
            return
    if example_path is None:
        return
    # codeql[py/path-injection] -- packaged example under site-packages / repo tree only.
    text = example_path.read_text(encoding="utf-8", errors="replace")
    try:
        assert_operator_yaml_safe_for_persist(text)
    except OperatorYamlUnsafeError:
        return
    # codeql[py/path-injection] -- cfg_path verified via _verified_operator_config_path (Type 1).
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    # codeql[py/path-injection] -- cfg_path verified (Type 1).
    atomic_write_text(cfg_path, text)


def _operator_file(request: Request, corpus_root: Path) -> Path:
    if not (
        bool(getattr(request.app.state, "operator_config_api_enabled", False))
        or bool(getattr(request.app.state, "jobs_api_enabled", False))
    ):
        raise HTTPException(status_code=500, detail="operator config API is not enabled.")
    return viewer_operator_yaml_path(request.app, corpus_root)


@router.get("/operator-config", response_model=OperatorConfigGetResponse)
async def get_operator_config(
    request: Request,
    path: str = Query(
        ..., description="Corpus root (authorizes request; must resolve under anchor)."
    ),
) -> OperatorConfigGetResponse:
    """Return viewer-safe operator YAML from the configured resolved path.

    May create ``<corpus>/viewer_operator.yaml`` from the packaged overrides example when
    the file is missing or whitespace-only (see ``_ensure_default_viewer_operator_yaml``).
    ``profile:`` is not seeded; use the viewer Profile menu + Save (same idea as CLI
    ``--profile`` + ``--config``).
    """
    anchor = getattr(request.app.state, "output_dir", None)
    corpus_root = resolve_corpus_path_param(path, anchor)
    cfg_raw = _operator_file(request, corpus_root)
    cfg_path = _verified_operator_config_path(request, corpus_root, cfg_raw)
    profiles = list_packaged_profile_names()
    _ensure_default_viewer_operator_yaml(cfg_path)
    if not cfg_path.is_file():
        corpus_s = os.path.normpath(str(corpus_root.resolve()))
        op_s = os.path.normpath(str(cfg_path.resolve()))
        return OperatorConfigGetResponse(
            corpus_path=corpus_s,
            operator_config_path=op_s,
            content="",
            available_profiles=profiles,
        )
    # codeql[py/path-injection] -- cfg_path verified above.
    content = cfg_path.read_text(encoding="utf-8", errors="replace")
    try:
        assert_operator_yaml_safe_for_persist(content)
    except OperatorYamlUnsafeError as exc:
        detail = exc.detail
        if (
            exc.status_code == status.HTTP_400_BAD_REQUEST
            and isinstance(detail, dict)
            and detail.get("error") in ("forbidden_operator_keys", "forbidden_operator_feed_keys")
        ):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="Existing operator YAML contains forbidden keys; fix the file out-of-band.",
            ) from exc
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    corpus_s = os.path.normpath(str(corpus_root.resolve()))
    op_s = os.path.normpath(str(cfg_path.resolve()))
    return OperatorConfigGetResponse(
        corpus_path=corpus_s,
        operator_config_path=op_s,
        content=content,
        available_profiles=profiles,
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
    cfg_raw = _operator_file(request, corpus_root)
    cfg_path = _verified_operator_config_path(request, corpus_root, cfg_raw)
    to_write = expand_profile_only_with_packaged_example(
        body.content,
        example_path=packaged_viewer_operator_example_path(),
    )
    try:
        assert_operator_yaml_safe_for_persist(to_write)
    except OperatorYamlUnsafeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    atomic_write_text(cfg_path, to_write)
    profiles = list_packaged_profile_names()
    corpus_s = os.path.normpath(str(corpus_root.resolve()))
    op_s = os.path.normpath(str(cfg_path.resolve()))
    return OperatorConfigGetResponse(
        corpus_path=corpus_s,
        operator_config_path=op_s,
        content=to_write,
        available_profiles=profiles,
    )
