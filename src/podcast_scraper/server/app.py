"""FastAPI application factory for the GI/KG viewer API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, cast

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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
    app_collections,
    app_comms,
    app_consolidation,
    app_corpus,
    app_discover,
    app_enrichment,
    app_episodes,
    app_export,
    app_graph_events,
    app_mcp,
    app_relational,
    app_search,
    app_user_preferences,
    app_user_state,
    app_your_week,
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
    corpus_rollback,
    corpus_text_file,
    corpus_theme_clusters,
    corpus_topic_clusters,
    corpus_trending,
    enrichment as enrichment_route,
    enrichment_config as enrichment_config_route,
    explore,
    feeds,
    health,
    index_rebuild,
    index_stats,
    internal_mcp,
    internal_outbox,
    jobs,
    llm_gateway,
    mcp_oauth,
    operator_config,
    ops,
    query_activity,
    relational,
    resilience as resilience_routes,
    scheduled_jobs as scheduled_jobs_route,
    search,
    usage as usage_routes,
)
from podcast_scraper.utils.correlation import current_trace_id

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
    # Derived interests (#1139) — when ON (and personalization is on), discovery also ranks by
    # interests inferred from what the user has heard/captured, not just explicit follows. OFF by
    # default so explicit-only stays the baseline until the derived signal is tuned.
    app.state.derived_interests = _env_truthy("APP_DERIVED_INTERESTS")
    # RFC-103 R2 momentum knobs (config block; see MomentumConfig.from_dict). Only the trend
    # inclusion floor is env-exposed today: a tiny corpus (e.g. the e2e validation fixture) can't
    # clear a "3 mentions in the recent window" floor, so trending rails would render empty — set
    # APP_MOMENTUM_MIN_TOTAL=1 there. Unset → the packaged default (3), correct for a real corpus.
    _mom_min = os.environ.get("APP_MOMENTUM_MIN_TOTAL", "").strip()
    app.state.momentum_config = (
        {"trend": {"min_total": int(_mom_min)}} if _mom_min.isdigit() else None
    )
    app.state.operator_api_key = os.environ.get("APP_OPERATOR_API_KEY", "")
    # Shared token for the internal outbox seam (#1415, RFC-110 §2) — the infra delivery worker
    # authenticates with it over the tailnet. Empty → the /internal/outbox endpoints 503 (disabled).
    app.state.internal_outbox_token = os.environ.get("INTERNAL_OUTBOX_TOKEN", "")
    # Public VAPID key for Web Push (RFC-110 §6). The private half lives with the infra worker; the
    # browser needs this public half to subscribe. Empty → GET /api/app/push/vapid-key 503s.
    app.state.vapid_public_key = os.environ.get("APP_VAPID_PUBLIC_KEY", "")
    # Shared token for the internal MCP verify seam (RFC-112 §4, #1471) — the MCP server process
    # authenticates with it over the tailnet. Empty → /internal/mcp/verify 503 (disabled).
    app.state.internal_mcp_token = os.environ.get("INTERNAL_MCP_TOKEN", "")
    app.state.audit_path = (
        (app.state.app_data_dir / "audit.jsonl") if app.state.app_data_dir is not None else None
    )


def _default_static_dir() -> Path | None:
    """Built SPA assets under ``web/gi-kg-viewer/dist`` (repo root relative to this file)."""
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "web" / "gi-kg-viewer" / "dist"
    return dist if dist.is_dir() else None


# Operator/read ``/api/*`` routers — mounted only when NOT app-only (#1163).
_OPERATOR_READ_ROUTES = (
    resilience_routes,
    usage_routes,
    artifacts,
    index_stats,
    index_rebuild,
    search,
    relational,
    query_activity,
    explore,
    corpus_library,
    corpus_binary,
    corpus_media,
    corpus_text_file,
    corpus_metrics,
    corpus_coverage,
    corpus_persons,
    corpus_digest,
    corpus_enrichments,
    corpus_topic_clusters,
    corpus_theme_clusters,
    corpus_trending,
    cil,
    ops,
    llm_gateway,
    # Destructive operator rollback (DELETE runs/episodes) — tailnet operator plane only, gated by
    # the OperatorWriteGuard middleware (X-Operator-Key / admin session) + a typed confirm token.
    # Deliberately NOT in _OPERATOR_PUBLIC_READ_ROUTES (mutates) and NOT mounted on the player.
    corpus_rollback,
)
# RFC-108: the CURATED subset of operator-read routers safe for the PUBLIC operator
# surface (operator.closelistening.app). This is ``_OPERATOR_READ_ROUTES`` MINUS the ones
# that mutate / control:
#   - ``index_rebuild`` (rebuilds the index — compute/write)
#   - ``ops`` (operational controls)
#   - ``resilience_routes`` (carries POST ``/api/ops/resilience/reset`` — a reset)
# Those stay on the tailnet-only operator serve. Everything kept here is read-only (the
# remaining POST routes are POST-for-query: search/compare, corpus resolve, topics/timeline
# — they compute + return, they do not mutate). Mounted with a router-level ≥creator gate.
# A new operator-read router MUST be consciously classified here (audit its non-GET routes).
_OPERATOR_PUBLIC_READ_ROUTES = (
    usage_routes,
    artifacts,
    index_stats,
    search,
    relational,
    query_activity,
    explore,
    corpus_library,
    corpus_binary,
    corpus_media,
    corpus_text_file,
    corpus_metrics,
    corpus_coverage,
    corpus_persons,
    corpus_digest,
    corpus_enrichments,
    corpus_topic_clusters,
    corpus_theme_clusters,
    corpus_trending,
    cil,
)
# Consumer Learning Platform API (RFC-098): slug-addressed routes under their own
# ``/api/app`` namespace, auth-gated (#1063/#1066). Always mounted.
_APP_ROUTES = (
    app_auth,
    app_admin,
    app_artwork,
    app_episodes,
    app_graph_events,
    app_relational,
    app_discover,
    app_search,
    app_user_state,
    app_user_preferences,
    app_capture,
    app_collections,
    app_comms,
    app_your_week,
    app_corpus,
    app_export,
    app_mcp,
    mcp_oauth,
    app_enrichment,
    app_consolidation,
)


def _mount_api_routers(app: FastAPI, *, app_only: bool, operator_public: bool = False) -> None:
    """Mount HTTP routers. ``health`` + the consumer ``/api/app/*`` plane always mount.

    Three serve postures for the operator/read ``/api/*`` plane:

    - **full** (tailnet operator, default): all ``_OPERATOR_READ_ROUTES``, **ungated** —
      tailnet privacy is the gate.
    - **app_only** (public player, #1163 / ADR-116): none of them — low-privilege backend.
    - **operator_public** (RFC-108, public operator surface): only the CURATED
      ``_OPERATOR_PUBLIC_READ_ROUTES`` subset, each mounted with a **router-level ≥creator
      gate** (``require_viewer_access``). ``index_rebuild`` / ``ops`` are NOT mounted here.
    """
    app.include_router(health.router, prefix="/api")
    if operator_public:
        gate = [Depends(app_auth.require_viewer_access)]
        for module in _OPERATOR_PUBLIC_READ_ROUTES:
            app.include_router(module.router, prefix="/api", dependencies=gate)
    elif not app_only:
        for module in _OPERATOR_READ_ROUTES:
            app.include_router(module.router, prefix="/api")
    for module in _APP_ROUTES:
        app.include_router(module.router, prefix="/api/app")
    # The internal delivery-outbox seam (#1415) — service-to-service, token-gated, tailnet-only.
    app.include_router(internal_outbox.router, prefix="/internal")
    # The internal MCP verify seam (#1471) — service-to-service, token-gated, tailnet-only.
    app.include_router(internal_mcp.router, prefix="/internal")
    # MCP OAuth 2.1 authorization-server metadata at the app ROOT (RFC 8414 discovery, #1471).
    app.include_router(mcp_oauth.wellknown_router)


class _AccessLogMiddleware:
    """Pure-ASGI request access log with trace correlation (ADR-119, G1 correlation).

    Deliberately NOT a Starlette ``BaseHTTPMiddleware`` (``@app.middleware("http")``): that
    wrapper buffers the response and BREAKS endpoint background tasks — which silently
    dropped queue-add / favourite persistence (caught by the auth-queue + library-saved
    e2e). A pure ASGI middleware wraps ``send`` without touching the body or the
    background-task lifecycle.

    Logs ONE line per request. The trace id is captured at ``http.response.start`` — the
    span is still active there (uvicorn's own access log runs AFTER the span closes, so it
    can only ever see ``"-"``) — so a VictoriaLogs line pivots to its VictoriaTraces span.
    ``trace=-`` when tracing is off, so it is a no-op-safe structured access log in every
    environment; ``/health`` + ``/metrics`` are skipped to keep the log readable.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._logger = logging.getLogger("podcast.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or path.endswith("/health") or path == "/metrics":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        captured = {"status": 0, "trace": "-"}

        async def _send(message: Message) -> None:
            if message["type"] == "http.response.start":
                captured["status"] = message["status"]
                captured["trace"] = current_trace_id()
            await send(message)

        await self.app(scope, receive, _send)
        self._logger.info(
            "%s %s -> %s in %.1fms trace=%s",
            scope.get("method", "?"),
            path,
            captured["status"],
            (time.perf_counter() - start) * 1000.0,
            captured["trace"],
        )


def _install_access_logging(app: FastAPI) -> None:
    """Attach the trace-correlated request access log (ADR-119, G1). See _AccessLogMiddleware."""
    app.add_middleware(_AccessLogMiddleware)


def _start_dev_metrics_pusher() -> None:
    """Start the dev-only metrics push (no-op unless PODCAST_METRICS_PUSH_URL is set)."""
    try:
        from ..obs.dev_push import start_metrics_pusher

        start_metrics_pusher()
    except Exception:  # noqa: BLE001 — telemetry must never break the app
        logger.debug("dev metrics pusher not started", exc_info=True)


def _guard_operator_public_open_signup(operator_public: bool) -> None:
    """RFC-108 hardening: the operator-public viewer self-grants ``creator`` via its
    ``?grant=creator`` login hint, so the email allowlist is the ONLY authZ boundary on
    the operator-read corpus. ``APP_SIGNUP_MODE=open`` drops that boundary and would expose
    the corpus to any authenticated Google account — refuse to boot rather than silently
    serve it wide open (one env flip should not open the whole surface)."""
    if operator_public and policy_from_env().mode == "open":
        raise RuntimeError(
            "PODCAST_SERVE_OPERATOR_PUBLIC=1 with APP_SIGNUP_MODE=open would expose the "
            "operator-read corpus to any authenticated Google account. Set "
            "APP_SIGNUP_MODE=allowlist with APP_ALLOWED_EMAILS."
        )


def _init_api_otel() -> None:
    """o11y P2: give the API surface a TracerProvider (the pipeline CLI already does this).

    Without it, API-originated errors/events carry no ``trace_id``, so Sentry↔trace and event↔trace
    correlation were permanently ``"-"`` for the server. True no-op unless ``OTEL_TRACES_EXPORTER``
    asks for it; never blocks API startup.
    """
    try:
        from podcast_scraper.utils.otel_init import init_otel

        init_otel()
    except Exception:  # pragma: no cover - never block API startup on tracing
        pass


async def _start_queue_sweeper_guarded(app: FastAPI) -> "asyncio.Task | None":
    """Start job-queue housekeeping (#1653); never let its failure block API startup.

    Deliberately NOT hung off the feed-sweep scheduler: that one only registers when
    ``scheduled_jobs:`` is present in the operator YAML, and the queue must keep moving on
    every deployment, configured or not. Without it, promotion is edge-triggered on another
    job finishing, so a row left ``running`` by a killed container holds a concurrency slot
    forever and everything behind it waits on an event that can no longer happen.

    Lives at module level rather than inside ``create_app``'s lifespan so the guard branches
    do not count against that function's complexity budget.
    """
    try:
        from podcast_scraper.server.queue_sweeper import start_queue_sweeper

        return await start_queue_sweeper(app)
    except Exception as exc:
        logger.warning("job queue: sweeper failed to start (%s); queue is edge-driven", exc)
        return None


async def _stop_queue_sweeper_guarded(task: "asyncio.Task | None") -> None:
    """Cancel the sweeper on shutdown; a failure here must not mask the real shutdown path."""
    if task is None:
        return
    try:
        from podcast_scraper.server.queue_sweeper import stop_queue_sweeper

        await stop_queue_sweeper(task)
    except Exception as exc:  # pragma: no cover - shutdown must not raise
        logger.warning("job queue: sweeper shutdown failed (%s)", exc)


def _start_cache_warmer_guarded(app: FastAPI) -> "threading.Event | None":
    """Start the consumer read-cache warmer (catalog / slug index / KG index): warm at startup +
    re-warm on ingest, on a daemon thread. Never blocks startup — a failure just means lazy fills.

    Module-level (like the queue sweeper) so its guard branches don't count against ``create_app``'s
    complexity budget. Disable with ``APP_CACHE_WARMING=0``.
    """
    root = getattr(app.state, "output_dir", None)
    if root is None or os.environ.get("APP_CACHE_WARMING", "1") == "0":
        return None
    try:
        from podcast_scraper.server.app_cache_warm import start_cache_warmer

        return start_cache_warmer(Path(root))
    except Exception as exc:  # pragma: no cover - never block startup on warming
        logger.warning("cache warmer failed to start: %s", exc)
        return None


def _stop_cache_warmer_guarded(stop: "threading.Event | None") -> None:
    """Signal the warmer loop to exit on shutdown (idempotent, never raises)."""
    if stop is not None:
        stop.set()


def _json_safe(value: Any) -> Any:
    """Recursively coerce a validation-error payload into something that can be serialised.

    ``inf`` / ``-inf`` / ``nan`` are Python floats with no JSON spelling (RFC 8259 has no literal
    for them), and Starlette renders with ``allow_nan=False``. Anywhere one reaches a response body
    it does not produce a wrong number — it produces a ``ValueError`` and a 500. Used on the
    validation-error path, where the offending value is echoed back to the caller by design.

    Non-JSON OBJECTS get the same treatment, and for the same reason. Pydantic puts the raised
    exception itself into an error's ``ctx`` (``{'error': ValueError(...)}``) whenever a
    ``model_validator`` rejects a body, so the moment this codebase gained its first cross-field
    validator (#34.8) the 422 handler started raising while REPORTING a 422 — rejecting the input
    correctly, then 500ing on the way out. Exactly the failure this function was written for, one
    type further along: the previous version only knew about floats and passed everything else
    through untouched.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)  # 'inf' / '-inf' / 'nan' — legible, and a string always serialises
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool, type(None))):
        return value
    if isinstance(value, float):
        return value
    return str(value)  # exceptions, enums, dataclasses — legible beats unserialisable


def _install_cors(app: FastAPI) -> None:
    """CORS allowlist: env-pinned web origins plus the fixed native-shell origins.

    Extracted from ``create_app``, which was over the complexity ceiling at 30. Self-contained
    configuration with its own branching and no coupling to the rest of the factory.
    """
    # CORS origins: default to the local Vue dev-server ports, but let prod pin
    # the real public hostname(s) via PODCAST_SERVE_CORS_ORIGINS (comma-separated)
    # — auth is cookie-based, so credentialed localhost origins must not be the
    # only allowlist on a public box (review 2026-07-17 M11).
    _default_cors = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ]
    _cors_env = os.environ.get("PODCAST_SERVE_CORS_ORIGINS", "").strip()
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or _default_cors
    # The Capacitor native shell's WebView serves the app from a FIXED local origin (not a network
    # host), so its cross-origin calls to this API need explicit CORS allowance (#1310). These are
    # constant app origins — safe to always allow, even when prod pins the web hostname above. Auth
    # rides a Bearer token on native (not the cookie), but allow_credentials stays on for the web.
    _native_origins = [
        "capacitor://localhost",  # iOS default
        "https://localhost",  # Android (androidScheme: https, our default)
        "http://localhost",  # Android (http scheme) / fallback
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[*_cors_origins, *_native_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _install_metrics(app: FastAPI) -> None:
    """Optional Prometheus instrumentation; a no-op unless ``PODCAST_METRICS_ENABLED``.

    Extracted from ``create_app`` for the same reason as :func:`_install_cors`.
    """
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
        except Exception:  # noqa: BLE001 — telemetry must never break the app
            # ``prometheus-fastapi-instrumentator`` is listed under ``[dev]``.
            # A missing package (or any instrument/expose failure) must NOT down
            # the app — log LOUDLY (an error surfaces in Sentry now) and run
            # WITHOUT metrics. Telemetry never breaks the app (ADR-120); an
            # app-up-without-metrics beats app-down. Was a fail-loud RuntimeError.
            logger.exception(
                "PODCAST_METRICS_ENABLED is set but metrics instrumentation failed "
                "— continuing WITHOUT metrics. If the package is missing, install "
                "via ``pip install -e '.[dev]'`` (or add it to the image)."
            )

        # Dev-only: push the metrics registry straight to VictoriaMetrics when
        # PODCAST_METRICS_PUSH_URL is set (no daemon/scraper on the dev box). True no-op
        # otherwise — the packaged image leaves it unset and Alloy scrapes /metrics instead.
        _start_dev_metrics_pusher()


def _install_exception_handlers(app: FastAPI) -> None:
    """App-wide error handlers: corpus-path failures, and a 422 body that can be serialised.

    Extracted from ``create_app`` alongside :func:`_install_cors` to bring the factory back
    under the complexity ceiling.
    """

    @app.exception_handler(CorpusPathRequestError)
    async def _corpus_path_errors(
        _request: Request,
        exc: CorpusPathRequestError,
    ) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def _validation_errors(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """FastAPI's default 422, but with a body that can actually be serialised.

        The default handler echoes the offending value back as ``input``. Starlette renders with
        ``allow_nan=False``, so a request carrying ``Infinity`` or ``NaN`` — non-standard tokens
        Python's ``json.loads`` accepts and RFC 8259 does not — made the 422 itself unrenderable:
        rejecting the value correctly, then 500ing while saying so. The rejection is the point, so
        the report has to survive being written.
        """
        return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})

    @app.exception_handler(PermissionError)
    async def _permission_errors(request: Request, exc: PermissionError) -> JSONResponse:
        """A ``PermissionError`` reaching a request handler means the server can't read/write a
        data file it needs — almost always an appdata file-lock whose file/dir is owned by another
        uid. The API runs as uid 1000; a root write into the bind-mounted appdata (a ``docker
        exec``, a backup restore, operator seeding) leaves files the app can neither lock nor
        ``chown`` back, so the ``filelock``/``open`` call raises ``PermissionError``. That is an
        environmental storage condition, not a client error and not an app-logic bug — so return a
        controlled 503 instead of an uncaught 500 that spams telemetry with a stack trace (GlitchTip
        #1483/#1485/#1859). Logged as a warning WITH the offending path so the cause stays visible.
        """
        logger.warning(
            "storage permission error on %s %s: %s", request.method, request.url.path, exc
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "Storage temporarily unavailable (permission denied)."},
        )


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
    _init_api_otel()

    def _warm_search(root: object) -> None:
        """Pay the search cold-start ONCE, at boot, off the request path.

        The embedding model loads lazily on the first query, and the LanceDB tables open with
        it — measured at ~39 s for that first request on a cold container (a warm one is ~4 s).
        Whoever searched first wore all of it: an e2e run's opening query blew a 30 s budget,
        and in production it is the first real user after every deploy or restart.

        Runs ONE genuine `hybrid_candidates` call rather than reaching for the loader directly.
        The model id is the one recorded in the index meta, and the singleton is keyed on the
        resolved id + device + cache folder — reconstructing that here would be a second copy of
        the resolution rules, free to drift and warm a model the query path never asks for. A
        real search cannot mis-key, because it IS the query path.

        Best-effort by construction: no index, no `[search]` extras, or no cached model all raise
        and are swallowed. It never downloads (`allow_download=False` lives in the search path)
        and it never blocks startup — the caller runs it on a daemon thread. Set
        ``PODCAST_SERVE_WARM_SEARCH=0`` to skip it.
        """
        try:
            from podcast_scraper.search.hybrid_search import hybrid_candidates

            hybrid_candidates(root, "warm", top_k=1)  # type: ignore[arg-type]
            logger.info("search warmup complete")
        except Exception as exc:  # noqa: BLE001 - warmup is advisory; the query path re-reports
            logger.debug("search warmup skipped: %s", exc)

    @contextlib.asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Pin the event loop so the cron scheduler (running on a daemon
        # thread) can hand spawn callbacks back to FastAPI via
        # ``asyncio.run_coroutine_threadsafe``.
        app.state.event_loop = asyncio.get_running_loop()
        root = getattr(app.state, "output_dir", None)
        if root is not None and os.environ.get("PODCAST_SERVE_WARM_SEARCH", "1") != "0":
            threading.Thread(
                target=_warm_search, args=(root,), name="search-warmup", daemon=True
            ).start()
        cache_warmer_stop = _start_cache_warmer_guarded(app)
        scheduler = getattr(app.state, "scheduler", None)
        if scheduler is not None:
            try:
                scheduler.start()
            except Exception as exc:
                logger.warning("scheduler startup failed: %s", exc)
        sweeper_task = await _start_queue_sweeper_guarded(app)
        try:
            yield
        finally:
            _stop_cache_warmer_guarded(cache_warmer_stop)
            await _stop_queue_sweeper_guarded(sweeper_task)
            scheduler = getattr(app.state, "scheduler", None)
            if scheduler is not None:
                with contextlib.suppress(Exception):
                    scheduler.shutdown()

    app = FastAPI(title="podcast_scraper", version=__version__, lifespan=_lifespan)

    _install_exception_handlers(app)

    _install_cors(app)

    # Operator write-path authz (optional API key) + audit trail (#1071). Inert unless
    # APP_OPERATOR_API_KEY is set; consumer /api/app routes are never gated here.
    app.add_middleware(OperatorWriteGuard)

    # Request access log with trace correlation (ADR-119, G1). See _install_access_logging.
    _install_access_logging(app)

    _install_metrics(app)

    # #1163 / ADR-116: app-only public serve mode. When ``PODCAST_SERVE_APP_ONLY``
    # is set, mount ONLY health + the consumer ``/api/app/*`` plane — none of the
    # operator/read ``/api/*`` routes. This is the low-privilege backend the public
    # consumer player proxies to (deployed with no ``docker.sock`` and no provider
    # keys). Operator flag-gated routers (jobs/operator-config/feeds) are forced off
    # here too, belt-and-suspenders, so the privileged surface can never mount on a
    # public deployment even if a flag env leaks in.
    app_only = _env_truthy("PODCAST_SERVE_APP_ONLY")
    # RFC-108: public operator surface — curated operator-read subset, each ≥creator-gated.
    # A separate least-privilege deployment (no docker.sock, no keys) like the player.
    operator_public = _env_truthy("PODCAST_SERVE_OPERATOR_PUBLIC")
    if app_only or operator_public:
        # Public deployments never mount the privileged flag-gated routers
        # (jobs/operator-config/feeds) — belt-and-suspenders even if a flag env leaks in.
        enable_feeds_api = False
        enable_operator_config_api = False
        enable_jobs_api = False

    # RFC-108 hardening — refuse operator-public boot under open signup (see helper).
    _guard_operator_public_open_signup(operator_public)

    _mount_api_routers(app, app_only=app_only, operator_public=operator_public)

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
