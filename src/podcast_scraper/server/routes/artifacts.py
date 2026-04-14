"""GET /api/artifacts — list and load GI/KG JSON artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import ArtifactItem, ArtifactListResponse
from podcast_scraper.utils.corpus_episode_paths import corpus_search_parent_hint

router = APIRouter(tags=["artifacts"])


def _is_under(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_artifact_target(base: Path, artifact_relpath: str) -> Path:
    """Join ``artifact_relpath`` under ``base`` with ``..`` / absolute segments rejected."""
    rel = artifact_relpath.strip().replace("\\", "/")
    if not rel or rel.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    segments = [p for p in rel.split("/") if p and p != "."]
    if any(p == ".." for p in segments):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")
    # lgtm[py/path-injection] -- basename-only segments; ``base`` is anchor-sanitized.
    target = base.joinpath(*segments).resolve()
    if not _is_under(base, target):
        raise HTTPException(status_code=400, detail="Path escapes corpus root.")
    return target


def _mtime_utc_iso(st_mtime: float) -> str:
    dt = datetime.fromtimestamp(st_mtime, tz=timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def _kind_for_suffix(name: str) -> Literal["gi", "kg", "bridge"] | None:
    if name.endswith(".gi.json"):
        return "gi"
    if name.endswith(".kg.json"):
        return "kg"
    if name.endswith(".bridge.json"):
        return "bridge"
    return None


@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    request: Request,
    path: str = Query(..., description="Corpus output directory to scan."),
) -> ArtifactListResponse:
    """List ``*.gi.json``, ``*.kg.json``, and ``*.bridge.json`` under the directory (recursive)."""
    anchor = getattr(request.app.state, "output_dir", None)
    base = resolve_corpus_path_param(path, anchor)
    items: list[ArtifactItem] = []
    seen: set[Path] = set()
    for pattern in ("**/*.gi.json", "**/*.kg.json", "**/*.bridge.json"):
        for p in sorted(base.glob(pattern)):
            if not p.is_file() or p in seen:
                continue
            seen.add(p)
            kind = _kind_for_suffix(p.name)
            if kind is None:
                continue
            try:
                rel = p.relative_to(base)
            except ValueError:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            items.append(
                ArtifactItem(
                    name=p.name,
                    relative_path=rel.as_posix(),
                    kind=kind,
                    size_bytes=int(st.st_size),
                    mtime_utc=_mtime_utc_iso(st.st_mtime),
                )
            )
    items.sort(key=lambda x: (x.relative_path, x.kind))
    hints = corpus_search_parent_hint(base)
    return ArtifactListResponse(path=str(base), artifacts=items, hints=hints)


@router.get("/artifacts/{artifact_path:path}")
async def get_artifact(
    request: Request,
    artifact_path: str,
    path: str = Query(..., description="Corpus output directory (root for relative path)."),
) -> JSONResponse:
    """Load and return a parsed artifact JSON by path relative to the corpus root."""
    anchor = getattr(request.app.state, "output_dir", None)
    base = resolve_corpus_path_param(path, anchor)
    target = _safe_artifact_target(base, artifact_path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    if _kind_for_suffix(target.name) is None:
        raise HTTPException(
            status_code=400,
            detail="Not a .gi.json, .kg.json, or .bridge.json file.",
        )
    try:
        # lgtm[py/path-injection] -- ``target`` built only under anchor-sanitized ``base``.
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    return JSONResponse(content=data)
