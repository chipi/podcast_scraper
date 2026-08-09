"""Operator corpus rollback — DELETE a run/episode to trash + full reindex (incremental-add P0.2).

The ONLY prod-SSH dependency in the cautious incremental-add loop was rollback: ``rm -rf`` a
``feeds/<slug>/run_*`` dir on the box, then a full reindex. This exposes that as two operator HTTP
endpoints so trigger→validate→rollback runs entirely over the tailnet operator API:

- ``DELETE /api/corpus/runs/{run_id}``     — primary rollback unit (a whole run's episodes).
- ``DELETE /api/corpus/episodes/{id}``     — finer-grained (``--append`` shares one stable
                                             ``run_append_<hash>`` dir, so run-scoped delete would
                                             remove all appended episodes; this removes just one).

Safety: gated by the operator middleware (``X-Operator-Key`` / admin session, same gate as
``/api/index/rebuild``); requires a typed ``confirm`` token (must equal the id being deleted);
supports ``dry_run=true`` (returns the plan, touches nothing); moves artifacts to
``<corpus>/.trash/<ts>/`` instead of hard-``rm`` so a mistaken delete is reversible. The delete
alone leaves stale vectors (the two-tier index upserts by id with no orphan sweep), so it triggers
a full ``index_corpus(..., rebuild=True)`` via the shared rebuild gate (409 if one is already
running); topic-clusters recompute after the index (finalize path).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper.server.index_rebuild import gate_for_corpus
from podcast_scraper.server.pathutil import CorpusPathRequestError, resolve_corpus_path_param
from podcast_scraper.server.routes.index_rebuild import (
    _spawn_rebuild_thread,
    resolve_topic_cluster_threshold,
)
from podcast_scraper.utils import filesystem

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])


def _corpus_root(request: Request, path: Optional[str]) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    try:
        return resolve_corpus_path_param(path or "", anchor)
    except CorpusPathRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _run_dirs_for_id(root: Path, run_id: str) -> List[Path]:
    """All on-disk run dirs matching ``run_<run_id>`` (corpus feeds layout + flat layout)."""
    name = f"run_{run_id}"
    found = [p for p in root.glob(f"feeds/*/{name}") if p.is_dir()]
    flat = root / name
    if flat.is_dir():
        found.append(flat)
    return sorted(found)


def _episode_files(root: Path, episode_id: str) -> Tuple[Optional[Path], List[Path]]:
    """Return (run_dir, files) for one episode, located by STABLE episode_id (drift-immune).

    Files are the idx-prefixed transcript/metadata/sidecars for that episode within its run's
    ``transcripts/`` + ``metadata/`` — the same NNNN prefix its metadata filename carries.
    """
    from podcast_scraper.workflow import run_index

    run_index.reset_corpus_metadata_index_cache_for_tests()
    entry = run_index.corpus_metadata_index(str(root))["by_id"].get(episode_id)
    if entry is None:
        return None, []
    meta_abs = (root / entry.metadata_rel).resolve()
    run_dir = meta_abs.parent.parent  # <run>/metadata/NNNN.json -> <run>
    prefix = f"{entry.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - "
    files: List[Path] = []
    for sub in (filesystem.TRANSCRIPTS_SUBDIR, filesystem.METADATA_SUBDIR):
        d = run_dir / sub
        if d.is_dir():
            files.extend(p for p in d.glob(f"{prefix}*") if p.is_file())
    return run_dir, sorted(files)


def _move_to_trash(root: Path, targets: List[Path], stamp: str) -> List[str]:
    """Move each target into ``<corpus>/.trash/<stamp>/`` (relative path preserved); return rels."""
    trash_root = root / ".trash" / stamp
    moved: List[str] = []
    for src in targets:
        rel = os.path.relpath(str(src), str(root))
        dst = trash_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(rel)
    return moved


def _count_metadata_files(paths: List[Path]) -> int:
    """Episodes represented by *paths* — count of ``*.metadata.*`` (one per episode)."""
    total = 0
    for p in paths:
        if p.is_dir():
            total += sum(1 for _ in (p / filesystem.METADATA_SUBDIR).glob("*.metadata.*"))
        elif ".metadata." in p.name:
            total += 1
    return total


def _reaggregate_manifest(root: Path) -> Optional[float]:
    """Re-write corpus_manifest.json so cost_rollup recomputes from remaining run metrics.

    Best-effort: returns the new ``total_cost_usd`` or None if there is no manifest to refresh.
    """
    from podcast_scraper.workflow import corpus_operations as cops

    manifest = root / cops.CORPUS_MANIFEST_FILE
    if not manifest.is_file():
        return None
    try:
        import json

        doc = json.loads(manifest.read_text(encoding="utf-8"))
        rows = doc.get("feeds") if isinstance(doc, dict) else None
        results = cops._manifest_feed_rows_to_results(rows) if isinstance(rows, list) else []
        cops.write_corpus_manifest(str(root), results)
        new_doc = json.loads(manifest.read_text(encoding="utf-8"))
        return float(new_doc.get("cost_rollup", {}).get("total_cost_usd", 0.0))
    except Exception as exc:  # noqa: BLE001 — best-effort refresh, never blocks the rollback
        logger.warning("corpus rollback: manifest re-aggregation failed for %s: %s", root, exc)
        return None


def _remaining_episode_count(root: Path) -> int:
    from podcast_scraper.workflow import run_index

    run_index.reset_corpus_metadata_index_cache_for_tests()
    return len(run_index.corpus_metadata_index(str(root))["by_id"])


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rollback(
    request: Request,
    *,
    scope: str,
    target_id: str,
    targets: List[Path],
    path: Optional[str],
    confirm: Optional[str],
    dry_run: bool,
    root: Path,
) -> JSONResponse:
    rels = [os.path.relpath(str(p), str(root)) for p in targets]
    episodes = _count_metadata_files(targets)

    if dry_run:
        return JSONResponse(
            status_code=200,
            content={
                "dry_run": True,
                "scope": scope,
                f"{scope}_id": target_id,
                "would_remove": rels,
                "episodes": episodes,
            },
        )

    if confirm != target_id:
        raise HTTPException(
            status_code=400,
            detail=f"confirm must equal the {scope}_id being deleted ({target_id!r}).",
        )

    # 409 BEFORE moving anything — never leave a half-deleted corpus with no rebuild running.
    gate = gate_for_corpus(request.app, root)
    if not gate.try_begin():
        raise HTTPException(status_code=409, detail="An index rebuild is already running.")
    try:
        moved = _move_to_trash(root, targets, _stamp())
        new_cost = _reaggregate_manifest(root)
        remaining = _remaining_episode_count(root)
    except Exception:
        gate.end("rollback aborted before rebuild")
        raise
    # Hand the already-acquired gate to the rebuild thread (don't re-acquire).
    corpus_key = os.path.normpath(os.path.realpath(str(root)))
    threading.Thread(
        target=_spawn_rebuild_thread,
        name=f"corpus-rollback-rebuild-{corpus_key}",
        kwargs={
            "corpus_key": corpus_key,
            "output_dir": corpus_key,
            "rebuild": True,
            "vector_index_path": None,
            "vector_embedding_model": None,
            "vector_index_types": None,
            "topic_cluster_threshold": resolve_topic_cluster_threshold(request),
            "gate": gate,
        },
        daemon=True,
    ).start()

    summary: Dict[str, Any] = {
        "removed": moved,
        "scope": scope,
        f"{scope}_id": target_id,
        "episodes_dropped": episodes,
        "catalog_episode_count": remaining,
        "cost_rollup_total_usd": new_cost,
        "rebuild": {"status": "in_progress", "poll": "/api/index/stats"},
    }
    return JSONResponse(status_code=202, content=summary)


@router.delete("/corpus/runs/{run_id}", responses={404: {}, 409: {}, 400: {}})
async def delete_corpus_run(
    request: Request,
    run_id: str,
    path: str | None = Query(default=None, description="Corpus root. Omit for server default."),
    confirm: str | None = Query(default=None, description="Must equal run_id to confirm delete."),
    dry_run: bool = Query(default=False, description="Return the plan without touching anything."),
) -> JSONResponse:
    """Roll back a whole run: move its episodes to ``.trash/`` then full reindex."""
    root = _corpus_root(request, path)
    targets = _run_dirs_for_id(root, run_id)
    if not targets:
        raise HTTPException(status_code=404, detail=f"No run dir found for run_id {run_id!r}.")
    return _rollback(
        request,
        scope="run",
        target_id=run_id,
        targets=targets,
        path=path,
        confirm=confirm,
        dry_run=dry_run,
        root=root,
    )


@router.delete("/corpus/episodes/{episode_id}", responses={404: {}, 409: {}, 400: {}})
async def delete_corpus_episode(
    request: Request,
    episode_id: str,
    path: str | None = Query(default=None, description="Corpus root. Omit for server default."),
    confirm: str | None = Query(
        default=None, description="Must equal episode_id to confirm delete."
    ),
    dry_run: bool = Query(default=False, description="Return the plan without touching anything."),
) -> JSONResponse:
    """Roll back a single episode (for ``--append`` runs that share one run dir)."""
    root = _corpus_root(request, path)
    _run_dir, files = _episode_files(root, episode_id)
    if not files:
        raise HTTPException(
            status_code=404, detail=f"No on-disk files found for episode_id {episode_id!r}."
        )
    return _rollback(
        request,
        scope="episode",
        target_id=episode_id,
        targets=files,
        path=path,
        confirm=confirm,
        dry_run=dry_run,
        root=root,
    )
