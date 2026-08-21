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
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper import perf_cache
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.routes.app_enrichment import filtered_entity_signals
from podcast_scraper.utils.path_validation import (
    safe_fixed_file_under_root,
    safe_relpath_under_corpus_root,
)

router = APIRouter(tags=["corpus_enrichments"])

# Parsed-catalogue cache for GET /corpus/enrichments (list). Every envelope is JSON-parsed to read
# its metadata, and topic_cooccurrence_corpus alone is multi-MB, so re-parsing them all on every
# viewer poll is the route's cost. Keyed by the enrichments dir + its mtime: envelopes and
# run_summary.json are written atomically (tmp + os.replace — see enrichment/health.py:283), so the
# DIRECTORY mtime bumps on every add / remove / rewrite — a precise, stat-only freshness signal.
#
# Hand-rolled (not perf_cache.get_or_compute) on purpose: get_or_compute needs a compute closure,
# and wrapping the list body in one would reindent the path-injection-sensitive glob below and
# re-attribute its (dismissed, false-positive) CodeQL alert to this diff — the exact churn the loop
# comment in the route warns against. The key is the absolute dir path, so entries never collide.
_CATALOGUE_LOCK = threading.Lock()
_CATALOGUE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


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

    Compact catalog — returns ``{enricher_id, file, size_bytes, schema_version,
    enricher_version, computed_at, produced_by_latest_run}`` per envelope. The viewer uses
    this to render availability badges before a drill-down click.

    ``produced_by_latest_run`` exists because presence on disk is not freshness (#1650). A
    DISABLED enricher's output stays in the directory forever and was indistinguishable from
    one produced seconds ago: ``topic_cooccurrence_corpus.json`` was listed here for days
    after that enricher stopped running, still 1.5 MB, still looking live. Silent staleness is
    the same failure mode that let a 3 ms enrichment no-op pass for a working pass.
    """
    root = _resolve_corpus(request, path)
    enrichments_dir = root / "enrichments"
    if not enrichments_dir.is_dir():
        return {"enrichments": []}

    # Serve the parsed catalogue from cache when the enrichments dir hasn't changed since (mtime
    # token; see the module-level cache above). Checked BEFORE the loop so a hit skips every parse.
    cache_key = os.path.normpath(str(enrichments_dir))
    # root is _resolve_corpus-validated (resolve_corpus_path_param raises on escape); "enrichments"
    # is a constant segment and getmtime only stats it.
    # codeql[py/path-injection] -- enrichments_dir under a validated corpus root (Type 1).
    dir_token = os.path.getmtime(enrichments_dir)
    with _CATALOGUE_LOCK:
        cached = _CATALOGUE_CACHE.get(cache_key)
        if cached is not None and cached[0] == dir_token:
            return cached[1]

    # One directory walk, read fully before anything is decided.
    #
    # The run summary is taken from this SAME glob rather than by joining
    # ``enrichments_dir / "run_summary.json"``. ``enrichments_dir`` derives from the ``path``
    # query parameter, so that join is a path-traversal sink — CodeQL flagged it high severity
    # and was right. Sourcing it from the glob removes the sink instead of sanitising it, costs
    # one fewer directory read, and matches how every other file here is reached.
    #
    # Two passes and not one: ``produced_by_latest_run`` needs the summary for EVERY row, and
    # ``run_summary.json`` sorts in the middle of the listing — deciding as we went would leave
    # alphabetically-earlier enrichers marked "unknown" purely because of their name.
    #
    # The loop variable keeps its original name deliberately. CodeQL carries a pre-existing
    # alert on this glob (``enrichments_dir`` descends from the ``path`` query parameter), and
    # renaming the variable re-attributed that alert to this PR as if it were new. The path IS
    # validated — ``resolve_corpus_path_param`` normalises against the trusted server anchor
    # and RAISES when the result escapes it — so the finding is a false positive that CodeQL's
    # taint model cannot see through. Fighting that model inside a shared resolver is not this
    # PR's business; touching the line only to rename a variable was.
    envelopes: list[tuple[Path, dict[str, Any]]] = []
    latest_run_ids: set[str] | None = None
    for envelope_path in sorted(enrichments_dir.glob("*.json")):
        try:
            parsed = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        if envelope_path.name == "run_summary.json":
            # The executor's own bookkeeping output — not an envelope, but it is what records
            # which enrichers the last run produced.
            latest_run_ids = _enricher_ids_from_summary(parsed)
            continue
        envelopes.append((envelope_path, parsed))

    items: list[dict[str, Any]] = []
    for envelope, parsed in envelopes:
        enricher_id = parsed.get("enricher_id") or envelope.stem
        items.append(
            {
                "enricher_id": enricher_id,
                "enricher_version": parsed.get("enricher_version"),
                "schema_version": parsed.get("schema_version"),
                "file": envelope.name,
                "size_bytes": envelope.stat().st_size,
                # When this artifact was computed — the field that makes staleness legible.
                "computed_at": parsed.get("computed_at"),
                # True / False / None (unknown: no run summary to compare against).
                "produced_by_latest_run": (
                    None if latest_run_ids is None else enricher_id in latest_run_ids
                ),
            }
        )
    result = {"enrichments": items}
    with _CATALOGUE_LOCK:
        _CATALOGUE_CACHE[cache_key] = (dir_token, result)
    return result


def _enricher_ids_from_summary(summary: dict[str, Any]) -> set[str] | None:
    """Enricher ids recorded in an already-parsed run summary; None when it says nothing.

    Takes the PARSED payload, not a path. The caller reads it from the directory glob it is
    already walking, so no path is constructed from the user-supplied ``path`` parameter — the
    traversal sink is absent rather than sanitised.

    None is deliberately distinct from an empty set: "we do not know what the last run did"
    must not render as "the last run produced nothing", or every artifact on a corpus that
    predates run summaries would be badged stale — noise that trains an operator to ignore the
    badge, which is worse than not having one.
    """
    per_enricher = summary.get("per_enricher")
    if not isinstance(per_enricher, dict):
        return None
    return set(per_enricher)


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
    envelope_path = Path(safe)
    # Corpus-scope envelopes (topic_cooccurrence_corpus is ~1.5 MB, and grows with the corpus) only
    # change when the enricher reruns, yet the viewer re-reads + re-parses the whole file on every
    # drill-down. Cache the parse, keyed by the envelope's OWN mtime — NOT corpus_mtime: enrichers
    # rewrite these files without bumping corpus_run_summary.json, so a corpus-mtime token would
    # serve a stale envelope until the next ingest. _read_envelope still raises (404 missing / 500
    # corrupt) inside compute(); get_or_compute runs compute outside its lock and never stores on
    # exception, so the error contract is preserved.
    try:
        # envelope_path from safe_relpath_under_corpus_root; enricher_id matches ^[a-zA-Z0-9_]+$.
        # codeql[py/path-injection] -- envelope_path via safe_relpath_under_corpus_root (Type 1).
        token = os.path.getmtime(envelope_path)
    except OSError:
        return _read_envelope(envelope_path)  # missing/unreadable → 404/500, uncached
    envelope: dict[str, Any] = perf_cache.get_or_compute(
        "corpus_enrichment_envelope",
        f"{root}::{enricher_id}",
        token,
        lambda: _read_envelope(envelope_path),
    )
    return envelope


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


@router.get("/corpus/entity-signals")
def get_corpus_entity_signals(
    request: Request,
    kind: str = Query(..., pattern="^(person|topic)$"),
    id: str = Query(..., min_length=1, description="Entity id — person:<...> or topic:<...>."),
    path: str | None = Query(default=None, description="Corpus output dir."),
) -> dict[str, Any]:
    """Corpus enrichers pre-filtered to ONE graph node's entity (``?path=``-scoped viewer plane).

    Operator sibling of ``/api/app/corpus/entity-signals`` (which serves the single platform
    corpus): the viewer's node panel represents one person/topic, so the server filters the full
    co-occurrence / co-appearance / grounding / consensus lists down to the rows touching that
    entity and returns a few KB instead of the whole multi-MB envelope. Shares the exact filter with
    the consumer route via :func:`filtered_entity_signals`; the graph OVERLAYS still read the full
    envelopes (they draw every edge), so this endpoint is only for the per-node card.
    """
    root = _resolve_corpus(request, path)
    return {"signals": filtered_entity_signals(root, kind, id)}


_ = safe_fixed_file_under_root  # exported for future use


__all__ = ["router"]
