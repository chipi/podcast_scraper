"""GET /api/health — always available."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper import __version__
from podcast_scraper.corpus_version import (
    assess_corpus_version_compat,
    corpus_code_version,
    MIN_SUPPORTED_CORPUS_CODE_VERSION,
)
from podcast_scraper.server.pathutil import (
    CorpusPathRequestError,
    read_manifest_produced_by_under_anchor,
    resolve_corpus_path_param,
)
from podcast_scraper.server.schemas import CorpusProducedBy, HealthResponse

router = APIRouter(tags=["health"])


def _probe_enriched_search_available(corpus_dir: Path | None) -> bool:
    """Return True when the corpus has the enrichment output that the
    ``/api/search?enrich_results=true`` chain actually consumes.

    Ground truth: ``query_topic_relatedness`` (the chunk-5 QueryEnricher
    the /api/search chain runs) reads ``enrichments/topic_similarity.json``
    — its absence makes the enricher pass through unmodified, which is
    indistinguishable to the client from "no enrichment configured". This
    probe checks for the file so the S5 Enriched chip (which auto-adopts
    the capability) enables when there's actually something to show.
    """
    if corpus_dir is None:
        return False
    candidate = corpus_dir / "enrichments" / "topic_similarity.json"
    try:
        return candidate.is_file()
    except OSError:
        return False


def _auth_epoch(secret: str) -> str | None:
    """Non-secret fingerprint of the session signing key, or ``None`` when auth is unconfigured.

    ``HMAC-SHA256(secret, "auth-epoch")``, truncated. Deliberately an HMAC **keyed by the secret**
    rather than a plain hash OF the secret: a bare digest of a low-entropy secret is brute-forceable
    offline, and this value is served on an unauthenticated endpoint.

    Its only job is to CHANGE when the key changes. That is the moment every previously-issued token
    becomes unverifiable simultaneously — for a server-side reason — and a client that remembers the
    epoch it last saw can then distinguish "my session expired" from "this server rotated its keys
    and invalidated everyone", instead of concluding the user signed out and discarding their cached
    library (incident 2026-09-16).
    """
    if not secret:
        return None
    return hmac.new(secret.encode("utf-8"), b"auth-epoch", hashlib.sha256).hexdigest()[:16]


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
    if corpus_dir is not None and anchor is not None:
        produced_by_raw = read_manifest_produced_by_under_anchor(corpus_dir, anchor)
        corpus_ver, warning = assess_corpus_version_compat(produced_by_raw)
    corpus_produced_by = None
    if produced_by_raw:
        cv = corpus_code_version(produced_by_raw)
        if cv is not None:
            try:
                corpus_produced_by = CorpusProducedBy.model_validate(produced_by_raw)
            except Exception:
                corpus_produced_by = None
    # Readiness, not just liveness (incident 2026-09-16): platform auth needs BOTH a signing secret
    # and a user store, and without them every authed route fails while the process is perfectly
    # alive — a health check that only proves "the process answers" actively misleads the client.
    #
    # Scoped to deployments that ACTUALLY authenticate (an OAuth provider is configured). The
    # tailnet / operator modes deliberately run with no auth at all; reporting them "degraded"
    # would be crying wolf about a configuration that is working as intended.
    # `app_data_dir` is the same state `app_auth._data_dir` resolves against.
    auth_configured = getattr(st, "oauth_provider", None) is not None
    auth_ready = not auth_configured or (
        bool(getattr(st, "session_secret", "")) and getattr(st, "app_data_dir", None) is not None
    )
    return HealthResponse().model_copy(
        update={
            "status": "ok" if auth_ready else "degraded",
            "auth_ready": auth_ready,
            "auth_epoch": _auth_epoch(getattr(st, "session_secret", "")),
            "code_version": __version__,
            "player_version": getattr(st, "player_version", None),
            "min_supported_corpus_code_version": MIN_SUPPORTED_CORPUS_CODE_VERSION,
            "corpus_produced_by": corpus_produced_by,
            "corpus_code_version": corpus_ver,
            "corpus_version_warning": warning,
            "enriched_search_available": _probe_enriched_search_available(corpus_dir),
            "feeds_api": bool(getattr(st, "feeds_api_enabled", False)),
            "operator_config_api": bool(getattr(st, "operator_config_api_enabled", False)),
            "jobs_api": bool(getattr(st, "jobs_api_enabled", False)),
        }
    )
