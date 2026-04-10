"""FastAPI application factory for the GI/KG viewer API (RFC-062)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from podcast_scraper import __version__
from podcast_scraper.server.pathutil import CorpusPathRequestError
from podcast_scraper.server.routes import (
    artifacts,
    corpus_binary,
    corpus_digest,
    corpus_library,
    corpus_metrics,
    explore,
    health,
    index_rebuild,
    index_stats,
    search,
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
) -> FastAPI:
    """Build the FastAPI app with viewer routes and optional static viewer assets.

    Args:
        output_dir: Default corpus directory (stored on ``app.state`` for future routes).
        static_dir: Directory of built Vue assets. ``True`` uses the default ``dist`` path
            when present; ``False`` skips static mounting; ``None`` auto-detects.
        enable_platform: Reserved for v2.7 platform routes (#50, #347). When ``True``,
            platform route modules from ``routes/platform/`` will be mounted. Currently
            a no-op — stubs exist but no routers are implemented yet.
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

    app.include_router(health.router, prefix="/api")
    app.include_router(artifacts.router, prefix="/api")
    app.include_router(index_stats.router, prefix="/api")
    app.include_router(index_rebuild.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(explore.router, prefix="/api")
    app.include_router(corpus_library.router, prefix="/api")
    app.include_router(corpus_binary.router, prefix="/api")
    app.include_router(corpus_metrics.router, prefix="/api")
    app.include_router(corpus_digest.router, prefix="/api")

    resolved_output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    app.state.output_dir = resolved_output

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
    return create_app(Path(raw))
