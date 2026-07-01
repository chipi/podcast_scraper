"""FastAPI application factory for the GI/KG viewer API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import AsyncIterator, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from podcast_scraper import __version__
from podcast_scraper.server import app_roles
from podcast_scraper.server.app_access import policy_from_env
from podcast_scraper.server.app_oauth import provider_from_env
from podcast_scraper.server.app_operator_guard import OperatorWriteGuard
from podcast_scraper.server.app_user_seed import seed_from_env
from podcast_scraper.server.pathutil import CorpusPathRequestError
from podcast_scraper.server.routes import (
    app_admin,
    app_artwork,
    app_auth,
    app_capture,
    app_consolidation,
    app_discover,
    app_enrichment,
    app_episodes,
    app_relational,
    app_search,
    app_user_state,
    artifacts,
    cil,
    corpus_binary,
    corpus_coverage,
    corpus_digest,
    corpus_enrichments,
    corpus_library,
    corpus_media,
    corpus_metrics,
    corpus_persons,
    corpus_text_file,
    corpus_theme_clusters,
    corpus_topic_clusters,
    enrichment as enrichment_route,
    enrichment_config as enrichment_config_route,
    explore,
    feeds,
    health,
    index_rebuild,
    index_stats,
    jobs,
    operator_config,
    ops,
    query_activity,
    relational,
    scheduled_jobs as scheduled_jobs_route,
    search,
)

logger = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def serve_feature_kwargs_from_environ() -> dict[str, bool | str | None]:
    """Flags for ``create_app`` derived from ``PODCAST_SERVE_*`` (used by uvicorn --reload)."""
    raw_cfg = os.environ.get("PODCAST_SERVE_CONFIG_FILE", "").strip()
    return {
        "enable_feeds_api": _env_truthy("PODCAST_SERVE_ENABLE_FEEDS_API"),
        "enable_operator_config_api": _env_truthy("PODCAST_SERVE_ENABLE_OPERATOR_CONFIG_API"),
        "enable_jobs_api": _env_truthy("PODCAST_SERVE_ENABLE_JOBS_API"),
        "operator_config_file": raw_cfg or None,
    }


def _configure_platform_auth(app: FastAPI, resolved_output: Path | None) -> None:
    """Set consumer-platform auth/session state from env (RFC-098 §2; #1063).

    Auth stays inert until a session secret + OAuth creds are configured — the routes
    return 401/503 otherwise. Per-user data lives under ``APP_DATA_DIR`` (or
    ``<corpus>/.app``), kept outside the shared corpus tree.
    """
    app.state.session_secret = os.environ.get("APP_SESSION_SECRET", "")
    app.state.session_cookie_secure = _env_truthy("APP_SESSION_COOKIE_SECURE")
    raw = os.environ.get("APP_DATA_DIR", "").strip()
    if raw:
        app.state.app_data_dir = Path(raw).expanduser().resolve()
    elif resolved_output is not None:
        app.state.app_data_dir = resolved_output / ".app"
    else:
        app.state.app_data_dir = None
    app.state.oauth_provider = provider_from_env()
    app.state.access_policy = policy_from_env()
    app.state.admin_emails = app_roles.admin_emails_from_env()
    # Seed a fixed dev roster (1 admin / 2 creators / 2 listeners, mock identities) when
    # APP_SEED_USERS_FILE is set — so a fresh local platform has known users in the admin surface.
    seed_from_env(app.state.app_data_dir)
    # Personalized discovery ranking (PRD-043 FR4 / #1098) — OFF by default; the discovery feed
    # falls back to recency until this toggle is flipped (gated until the score is tuned).
    app.state.personalized_ranking = _env_truthy("APP_PERSONALIZED_RANKING")
    app.state.operator_api_key = os.environ.get("APP_OPERATOR_API_KEY", "")
    app.state.audit_path = (
        (app.state.app_data_dir / "audit.jsonl") if app.state.app_data_dir is not None else None
    )


def _default_static_dir() -> Path | None:
    """Built SPA assets under ``web/gi-kg-viewer/dist`` (repo root relative to this file)."""
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "web" / "gi-kg-viewer" / "dist"
    return dist if dist.is_dir() else None


def create_app(
    output_dir: Path | None = None,
    *,
    static_dir: Path | None | bool = None,
    enable_platform: bool = False,
    enable_feeds_api: bool = False,
    enable_operator_config_api: bool = False,
    enable_jobs_api: bool = False,
    operator_config_file: str | os.PathLike[str] | None = None,
) -> FastAPI:
    """Build the FastAPI app with viewer routes and optional static viewer assets.

    Args:
        output_dir: Default corpus directory (stored on ``app.state`` for future routes).
        static_dir: Directory of built Vue assets. ``True`` uses the default ``dist`` path
            when present; ``False`` skips static mounting; ``None`` auto-detects.
        enable_platform: Reserved legacy no-op (#50/#347). The consumer platform API
            (``/api/app/*``) now mounts **unconditionally** and is NOT gated by this flag;
            it is kept only for backward compatibility.
        enable_feeds_api: When ``True``, mount GET/PUT ``/api/feeds`` (requires ``output_dir``).
        enable_operator_config_api: When ``True``, mount GET/PUT ``/api/operator-config``
            (requires ``output_dir``). YAML defaults to ``<corpus>/viewer_operator.yaml``
            unless ``operator_config_file`` pins a single shared file.
        enable_jobs_api: When ``True``, mount ``/api/jobs`` pipeline job routes (requires
            ``output_dir``; uses the same operator path rules as operator-config).
        operator_config_file: Optional explicit operator YAML path when **either**
            ``enable_operator_config_api`` or ``enable_jobs_api`` is ``True``. When set,
            all corpora use this one file; otherwise each corpus has its own
            ``viewer_operator.yaml`` next to ``feeds.spec.yaml``.
    """
    # Sentry init runs first so any failure during app construction below
    # surfaces in Sentry. No-op when ``PODCAST_SENTRY_DSN_API`` is unset
    # (default — keeps dev / CI / offline boots silent).
    from podcast_scraper.utils.sentry_init import init_sentry

    init_sentry("api")

    @contextlib.asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Pin the event loop so the cron scheduler (running on a daemon
        # thread) can hand spawn callbacks back to FastAPI via
        # ``asyncio.run_coroutine_threadsafe``.
        app.state.event_loop = asyncio.get_running_loop()
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is not None:
            try:
                scheduler.start()
            except Exception as exc:
                logger.warning("scheduler startup failed: %s", exc)
        try:
            yield
        finally:
            scheduler = getattr(app.state, "scheduler", None)
            if scheduler is not None:
                with contextlib.suppress(Exception):
                    scheduler.shutdown()

    app = FastAPI(title="podcast_scraper", version=__version__, lifespan=_lifespan)

    @app.exception_handler(CorpusPathRequestError)
    async def _corpus_path_errors(
        _request: Request,
        exc: CorpusPathRequestError,
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:5174",
            "http://localhost:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Operator write-path authz (optional API key) + audit trail (#1071). Inert unless
    # APP_OPERATOR_API_KEY is set; consumer /api/app routes are never gated here.
    app.add_middleware(OperatorWriteGuard)

    # Prometheus /metrics endpoint, gated on ``PODCAST_METRICS_ENABLED``
    # so the default behaviour (no Grafana account, no agent running)
    # stays a no-op. Wired for the Grafana Cloud free-tier sink in
    # pre-prod (RFC-081, Phase 1B). The instrumentator emits the
    # standard FastAPI metrics: http_requests_total{method,route,status}
    # + http_request_duration_seconds histogram.
    if _env_truthy("PODCAST_METRICS_ENABLED"):
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            # ``should_group_status_codes=False`` keeps 2xx/4xx/5xx
            # distinguishable in dashboards. ``excluded_handlers`` keeps
            # the /metrics endpoint itself out of the request counter
            # (otherwise a Prometheus scrape inflates the count).
            Instrumentator(
                should_group_status_codes=False,
                excluded_handlers=["/metrics"],
            ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
        except ImportError:
            # ``prometheus-fastapi-instrumentator`` is listed under ``[dev]``.
            # If a deployment installs core only and sets the flag, fail
            # loud rather than silently shipping no metrics.
            raise RuntimeError(
                "PODCAST_METRICS_ENABLED is set but prometheus-fastapi-instrumentator "
                "is not installed. Install via ``pip install -e '.[dev]'`` "
                "(or add that package explicitly in minimal images)."
            )

    app.include_router(health.router, prefix="/api")
    app.include_router(artifacts.router, prefix="/api")
    app.include_router(index_stats.router, prefix="/api")
    app.include_router(index_rebuild.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(relational.router, prefix="/api")
    app.include_router(query_activity.router, prefix="/api")
    app.include_router(explore.router, prefix="/api")
    app.include_router(corpus_library.router, prefix="/api")
    app.include_router(corpus_binary.router, prefix="/api")
    app.include_router(corpus_media.router, prefix="/api")
    app.include_router(corpus_text_file.router, prefix="/api")
    app.include_router(corpus_metrics.router, prefix="/api")
    app.include_router(corpus_coverage.router, prefix="/api")
    app.include_router(corpus_persons.router, prefix="/api")
    app.include_router(corpus_digest.router, prefix="/api")
    app.include_router(corpus_enrichments.router, prefix="/api")
    app.include_router(corpus_topic_clusters.router, prefix="/api")
    app.include_router(corpus_theme_clusters.router, prefix="/api")
    app.include_router(cil.router, prefix="/api")
    app.include_router(ops.router, prefix="/api")
    # Consumer Learning Platform API (RFC-098): slug-addressed read routes under their
    # own /api/app namespace, separate from the operator routes. Read-only over the
    # shared corpus; access becomes auth-gated in later Epic-1 tasks (#1063/#1066).
    app.include_router(app_auth.router, prefix="/api/app")
    app.include_router(app_admin.router, prefix="/api/app")
    app.include_router(app_artwork.router, prefix="/api/app")
    app.include_router(app_episodes.router, prefix="/api/app")
    app.include_router(app_relational.router, prefix="/api/app")
    app.include_router(app_discover.router, prefix="/api/app")
    app.include_router(app_search.router, prefix="/api/app")
    app.include_router(app_user_state.router, prefix="/api/app")
    app.include_router(app_capture.router, prefix="/api/app")
    app.include_router(app_enrichment.router, prefix="/api/app")
    app.include_router(app_consolidation.router, prefix="/api/app")

    resolved_output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    app.state.output_dir = resolved_output
    _configure_platform_auth(app, resolved_output)

    app.state.feeds_api_enabled = bool(enable_feeds_api)
    app.state.operator_config_api_enabled = bool(enable_operator_config_api)
    app.state.jobs_api_enabled = bool(enable_jobs_api)
    app.state.enable_platform = bool(enable_platform)
    app.state.operator_config_fixed_path = None

    if enable_feeds_api and resolved_output is None:
        raise ValueError("enable_feeds_api requires output_dir (corpus anchor).")
    if enable_operator_config_api and resolved_output is None:
        raise ValueError("enable_operator_config_api requires output_dir (corpus anchor).")
    if enable_jobs_api and resolved_output is None:
        raise ValueError("enable_jobs_api requires output_dir (corpus anchor).")

    if (enable_operator_config_api or enable_jobs_api) and resolved_output is not None:
        if operator_config_file:
            app.state.operator_config_fixed_path = Path(operator_config_file).expanduser().resolve()
        else:
            raw = os.environ.get("PODCAST_SERVE_CONFIG_FILE", "").strip()
            if raw:
                app.state.operator_config_fixed_path = Path(raw).expanduser().resolve()

    if enable_feeds_api:
        app.include_router(feeds.router, prefix="/api")
    if enable_operator_config_api:
        app.include_router(operator_config.router, prefix="/api")
    if enable_jobs_api:
        app.include_router(jobs.router, prefix="/api")
        app.include_router(scheduled_jobs_route.router, prefix="/api")
        # Enrichment HTTP surface — same jobs_api gate (RFC-088 / Epic
        # #1101 chunk 1 sub-6). All routes gracefully degrade to a
        # "no run yet" payload when the corpus has no enrichment files.
        app.include_router(enrichment_route.router, prefix="/api")
        # RFC-088 v2 enrichment config surface: GET/PUT the enrichment
        # block + JSON Schema for UI form generation + provider-type
        # catalogue. Same jobs_api gate as the rest of the enrichment
        # routes.
        app.include_router(enrichment_config_route.router, prefix="/api")

    # #666 review item #8: resolve the pipeline exec mode ONCE at startup
    # and pin it on ``app.state``. Route handlers must read from
    # ``app.state.pipeline_exec_mode`` — never re-read ``PODCAST_PIPELINE_EXEC_MODE``
    # at request time. A rolling env-var change between startup and runtime
    # would otherwise silently bypass (or silently fall back from) the
    # Docker factory path.
    _pipe_mode = os.environ.get("PODCAST_PIPELINE_EXEC_MODE", "").strip().lower()
    app.state.pipeline_exec_mode = _pipe_mode
    if enable_jobs_api and _pipe_mode == "docker":
        from podcast_scraper.server.pipeline_docker_factory import attach_docker_jobs_factory

        attach_docker_jobs_factory(app)

    # In-process feed-sweep scheduler (#708). Only meaningful with jobs API
    # enabled (the scheduler reuses the same enqueue + post-submit path as
    # POST /api/jobs). Construction is cheap and pure — actual cron
    # registration happens in the lifespan hook above. No-op when
    # ``scheduled_jobs:`` is absent from the operator YAML.
    app.state.scheduler = None
    if enable_jobs_api and resolved_output is not None:
        from podcast_scraper.server.operator_paths import viewer_operator_yaml_path
        from podcast_scraper.server.scheduler import (
            make_app_spawn_callback,
            SchedulerService,
        )

        operator_yaml = viewer_operator_yaml_path(app, resolved_output)
        app.state.scheduler = SchedulerService(
            corpus_root=resolved_output,
            operator_yaml=operator_yaml,
            spawn=make_app_spawn_callback(app),
        )

    if static_dir is False:
        resolved_static = None
    elif static_dir is True or static_dir is None:
        resolved_static = _default_static_dir()
    else:
        resolved_static = static_dir if static_dir.is_dir() else None

    if resolved_static is not None:
        app.mount("/", StaticFiles(directory=str(resolved_static), html=True), name="viewer")

    return app


def create_app_for_uvicorn() -> FastAPI:
    """Factory entry point for ``uvicorn --factory`` (reload mode)."""
    raw = os.environ.get("PODCAST_SERVE_OUTPUT_DIR")
    if not raw:
        raise RuntimeError("PODCAST_SERVE_OUTPUT_DIR is not set")
    kw = serve_feature_kwargs_from_environ()
    return create_app(
        Path(raw),
        enable_feeds_api=bool(kw["enable_feeds_api"]),
        enable_operator_config_api=bool(kw["enable_operator_config_api"]),
        enable_jobs_api=bool(kw["enable_jobs_api"]),
        operator_config_file=cast(str | os.PathLike[str] | None, kw["operator_config_file"]),
    )
