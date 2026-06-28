"""User-facing read routes for RFC-088 enrichment envelopes.

Separate from ``routes/enrichment.py`` (which serves the operator-facing
status / health / metrics / events / re-enable surface gated on
``enable_jobs_api``). This module is always mounted and serves the
on-disk envelopes the executor produces — the same shape the viewer
Topic Entity / Person Profile rails consume.

Routes:

* ``GET /api/corpus/enrichments/{enricher_id}`` — corpus-scope envelope
  read. Returns the parsed envelope (``schema_version``,
  ``enricher_id``, ``enricher_version``, ``data``, ...) or 404 when the
  enricher hasn't run yet.
* ``GET /api/corpus/episode/enrichments/{enricher_id}`` — episode-scope
  envelope read (caller supplies ``metadata_relpath``).
* ``GET /api/corpus/enrichments`` — list all enricher envelopes present
  under the corpus root (cheap availability probe for the viewer).

All routes resolve the corpus root from ``?path=`` or fall back to the
server's anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.path_validation import (
    safe_fixed_file_under_root,
    safe_relpath_under_corpus_root,
)

router = APIRouter(tags=["corpus_enrichments"])


# Enricher ids are stable strings — restrict to a safe identifier pattern so
# a stray ``..`` or ``/`` can't escape the corpus root.
_ENRICHER_ID_PATTERN = r"^[a-zA-Z0-9_]+$"


def _resolve_corpus(request: Request, path: str | None) -> Path:
    fallback = getattr(request.app.state, "output_dir", None)
    if path is not None and str(path).strip():
        return Path(resolve_corpus_path_param(path, fallback))
    if fallback is None:
        raise HTTPException(
            status_code=400, detail="No corpus path provided and no server default."
        )
    # Match the sibling /api/corpus/* routes — server.state.output_dir is
    # already resolved at create_app() time, but expanduser/resolve is
    # cheap + idempotent and survives a state mutation by middleware.
    return Path(fallback).expanduser().resolve()


def _read_envelope(envelope_path: Path) -> dict[str, Any]:
    if not envelope_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"enrichment envelope not found at {envelope_path.name}"
        )
    try:
        text = envelope_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"envelope read failed: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"envelope is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=500, detail="envelope is not a JSON object")
    return parsed


@router.get("/corpus/enrichments")
def list_corpus_enrichments(
    request: Request,
    path: str | None = Query(default=None, description="Corpus output dir."),
) -> dict[str, Any]:
    """List every corpus-scope envelope present under ``enrichments/``.

    Compact catalog — returns ``{enricher_id, file, size_bytes,
    schema_version, enricher_version}`` per envelope. The viewer uses
    this to render availability badges before a drill-down click.
    """
    root = _resolve_corpus(request, path)
    enrichments_dir = root / "enrichments"
    if not enrichments_dir.is_dir():
        return {"enrichments": []}
    items: list[dict[str, Any]] = []
    for envelope_path in sorted(enrichments_dir.glob("*.json")):
        # Skip the executor's own bookkeeping outputs.
        if envelope_path.name in ("run_summary.json",):
            continue
        try:
            parsed = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        items.append(
            {
                "enricher_id": parsed.get("enricher_id") or envelope_path.stem,
                "enricher_version": parsed.get("enricher_version"),
                "schema_version": parsed.get("schema_version"),
                "file": envelope_path.name,
                "size_bytes": envelope_path.stat().st_size,
            }
        )
    return {"enrichments": items}


@router.get("/corpus/enrichments/{enricher_id}")
def get_corpus_enrichment(
    enricher_id: str,
    request: Request,
    path: str | None = Query(default=None, description="Corpus output dir."),
) -> dict[str, Any]:
    """Read a single corpus-scope enrichment envelope (404 if absent)."""
    import re

    if not re.match(_ENRICHER_ID_PATTERN, enricher_id):
        raise HTTPException(status_code=400, detail="invalid enricher_id")
    root = _resolve_corpus(request, path)
    # codeql[py/path-injection] — enricher_id passes the strict
    # ^[a-zA-Z0-9_]+$ regex above; safe_relpath_under_corpus_root then
    # normalises and rejects anything that would escape the corpus root.
    safe = safe_relpath_under_corpus_root(root, f"enrichments/{enricher_id}.json")
    if safe is None:
        raise HTTPException(status_code=400, detail="invalid enricher_id")
    return _read_envelope(Path(safe))


@router.get("/corpus/episode/enrichments/{enricher_id}")
def get_episode_enrichment(
    enricher_id: str,
    request: Request,
    metadata_relpath: str = Query(
        ..., description="Episode metadata.json relpath (e.g. metadata/0001 - ep.metadata.json)."
    ),
    path: str | None = Query(default=None, description="Corpus output dir."),
) -> dict[str, Any]:
    """Read a single episode-scope enrichment envelope (404 if absent).

    Episode-scope envelopes live alongside the metadata file under
    ``<metadata_dir>/enrichments/<stem>.<enricher_id>.json``.
    """
    import re

    if not re.match(_ENRICHER_ID_PATTERN, enricher_id):
        raise HTTPException(status_code=400, detail="invalid enricher_id")
    root = _resolve_corpus(request, path)
    # codeql[py/path-injection] — metadata_relpath is fed through
    # safe_relpath_under_corpus_root which rejects absolute paths,
    # empty paths, and ``..`` segments before any filesystem access.
    safe_meta = safe_relpath_under_corpus_root(root, metadata_relpath)
    if safe_meta is None:
        raise HTTPException(status_code=400, detail="metadata_relpath must be a safe relpath")
    meta_path = Path(safe_meta)
    if not meta_path.name.endswith(".metadata.json"):
        raise HTTPException(
            status_code=400, detail="metadata_relpath must point at *.metadata.json"
        )
    stem = meta_path.name[: -len(".metadata.json")]
    # The enricher_id is the strict regex above; the filename is
    # ``{stem}.{enricher_id}.json`` — a single fixed segment.
    enrich_dir = meta_path.parent / "enrichments"
    rel_under_root = enrich_dir.relative_to(root)
    safe_envelope = safe_relpath_under_corpus_root(
        root, f"{rel_under_root.as_posix()}/{stem}.{enricher_id}.json"
    )
    if safe_envelope is None:
        raise HTTPException(status_code=400, detail="invalid envelope path")
    return _read_envelope(Path(safe_envelope))


_ = safe_fixed_file_under_root  # exported for future use


__all__ = ["router"]
