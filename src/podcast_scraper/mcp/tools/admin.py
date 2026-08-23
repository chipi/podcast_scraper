"""Corpus admin MCP tools (RFC-118 §5) — freshness status + the re-derive levers.

``corpus_status`` is read-only. ``reenrich`` / ``reindex`` are WRITE tools, but they
never spawn work themselves: they append a QUEUED row to the shared, lock-guarded jobs
registry under ``.viewer/`` (the same enqueue seam the pipeline's post-run chain uses,
#1653) and the API server's drain promotes + spawns it. RUNNING is a promise only the
API server can keep, so these always enqueue with ``force_queued=True``. MCP and the
operator UI therefore drive identical machinery.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from ..context import CorpusContext


def _operator_yaml(corpus_root: Path) -> Path | None:
    candidate = corpus_root / "viewer_operator.yaml"
    return candidate if candidate.is_file() else None


def corpus_status(ctx: CorpusContext) -> Dict[str, Any]:
    """Corpus derivation freshness: enrichment staleness rows + index freshness facts.

    The MCP mirror of ``GET /api/enrichment/stats`` (+ the index sidecar facts),
    computed from on-disk state only. ``reenrich_recommended`` / per-enricher
    ``reasons`` say WHY; the ``reenrich`` / ``reindex`` tools are the levers.
    """
    from podcast_scraper.server.enrichment_staleness import compute_enrichment_staleness

    corpus_root = Path(ctx.corpus_dir)
    fields = compute_enrichment_staleness(corpus_root)
    lance_dir = corpus_root / "search" / "lance_index"
    index_present = lance_dir.is_dir() and any(lance_dir.iterdir())
    index_meta: Dict[str, Any] = {}
    if index_present:
        import json

        try:
            raw = json.loads((lance_dir / "index_meta.json").read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                index_meta = raw
        except (OSError, ValueError):
            index_meta = {}
    manifest_present = (corpus_root / "derivation_fingerprints.json").is_file()
    return {
        "enrichment": {
            "reenrich_recommended": fields.reenrich_recommended,
            "reenrich_reasons": fields.reenrich_reasons,
            "last_run_status": fields.last_run_status,
            "last_run_finished_at": fields.last_run_finished_at,
            "artifact_newest_mtime": fields.artifact_newest_mtime,
            "enrichers": [asdict(r) for r in fields.enrichers],
        },
        "index": {
            "present": index_present,
            "embedding_model": index_meta.get("embedding_model"),
            "embed_dim": index_meta.get("embed_dim"),
        },
        "delta_backbone": {
            "fingerprint_manifest_present": manifest_present,
        },
    }


def reenrich(ctx: CorpusContext, force: bool = False) -> Dict[str, Any]:
    """Enqueue a corpus enrichment pass; ``force=True`` = explicit FULL re-derive.

    ``force`` bypasses staleness gates AND the RFC-118 incremental caches (pair/vector
    caches + delta cursors) — use after a model or threshold change, or when
    ``corpus_status`` reports drift. Without it the run is incremental where supported.
    """
    from podcast_scraper.server.jobs import enqueue_enrichment_job

    corpus_root = Path(ctx.corpus_dir)
    rec = enqueue_enrichment_job(
        corpus_root,
        operator_yaml=_operator_yaml(corpus_root),
        force=force,
        force_queued=True,
    )
    return {
        "job_id": rec.get("job_id"),
        "status": rec.get("status"),
        "command_type": rec.get("command_type"),
        "force": force,
        "note": "queued; the API server's drain promotes and spawns it",
    }


def reindex(ctx: CorpusContext, rebuild: bool = False) -> Dict[str, Any]:
    """Enqueue a corpus vector reindex; ``rebuild=True`` = full drop-and-rebuild.

    Runs the standalone subprocess-isolated reindex entry point via the jobs queue.
    Requires ``viewer_operator.yaml`` at the corpus root (the child loads its config
    from it, exactly like a queued enrichment).
    """
    from podcast_scraper.server.jobs import enqueue_reindex_job

    corpus_root = Path(ctx.corpus_dir)
    operator_yaml = _operator_yaml(corpus_root)
    if operator_yaml is None:
        return {
            "job_id": None,
            "status": "rejected",
            "note": "no viewer_operator.yaml at the corpus root — the reindex child "
            "needs it as --config; run via POST /api/index/rebuild instead",
        }
    rec = enqueue_reindex_job(
        corpus_root,
        operator_yaml=operator_yaml,
        rebuild=rebuild,
    )
    return {
        "job_id": rec.get("job_id"),
        "status": rec.get("status"),
        "command_type": rec.get("command_type"),
        "rebuild": rebuild,
        "note": "queued; the API server's drain promotes and spawns it",
    }
