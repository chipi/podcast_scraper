"""FastAPI application factory for the GI/KG viewer API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from podcast_scraper import __version__
from podcast_scraper.server.pathutil import CorpusPathRequestError
from podcast_scraper.server.routes import (
    artifacts,
    cil,
    corpus_binary,
    corpus_coverage,
    corpus_digest,
    corpus_library,
    corpus_metrics,
    corpus_persons,
    corpus_text_file,
    corpus_topic_clusters,
    explore,
    feeds,
    health,
    index_rebuild,
    index_stats,
    jobs,
    operator_config,
    search,
)


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
        enable_platform: Reserved for v2.7 platform routes (#50, #347). When ``True``,
            platform route modules from ``routes/platform/`` will be mounted. Currently
            a no-op — stubs exist but no routers are implemented yet.
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
    app = FastAPI(title="podcast_scraper", version=__version__)

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
            # ``prometheus-fastapi-instrumentator`` is in [server] extras.
            # If a deployment installs core only and sets the flag, fail
            # loud rather than silently shipping no metrics.
            raise RuntimeError(
                "PODCAST_METRICS_ENABLED is set but prometheus-fastapi-instrumentator "
                "is not installed. Install via ``pip install '.[server]'``."
            )

    app.include_router(health.router, prefix="/api")
    app.include_router(artifacts.router, prefix="/api")
    app.include_router(index_stats.router, prefix="/api")
    app.include_router(index_rebuild.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(explore.router, prefix="/api")
    app.include_router(corpus_library.router, prefix="/api")
    app.include_router(corpus_binary.router, prefix="/api")
    app.include_router(corpus_text_file.router, prefix="/api")
    app.include_router(corpus_metrics.router, prefix="/api")
    app.include_router(corpus_coverage.router, prefix="/api")
    app.include_router(corpus_persons.router, prefix="/api")
    app.include_router(corpus_digest.router, prefix="/api")
    app.include_router(corpus_topic_clusters.router, prefix="/api")
    app.include_router(cil.router, prefix="/api")

    resolved_output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    app.state.output_dir = resolved_output

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
