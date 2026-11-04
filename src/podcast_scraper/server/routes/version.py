"""GET /api/version — what is running, without a sign-in; more for an admin.

Public: the deploy's release, the API build, whether the schema is at head, the corpus format.
Admin session: also every installed module, the database plane, revisions and server version.
Fast by design (no corpus walk): the snapshot is cached on the app for a few seconds.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import create_engine, Engine, text

from podcast_scraper import build_info
from podcast_scraper.corpus_version import assess_corpus_version_compat, corpus_code_version
from podcast_scraper.extensions import extension_versions
from podcast_scraper.server import app_roles
from podcast_scraper.server.pathutil import read_manifest_produced_by_under_anchor
from podcast_scraper.server.routes.app_auth import get_optional_user
from podcast_scraper.server.schemas import (
    CorpusProducedBy,
    VersionApi,
    VersionCorpus,
    VersionDatabase,
    VersionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["version"])

CACHE_SECONDS = 5.0


def _database_snapshot() -> dict[str, Any]:
    """Everything admins see about the database; the public block is a subset of it."""
    from podcast_scraper.db import engines, migrate
    from podcast_scraper.db.settings import DatabaseSettings, is_configured

    if not (engines.engines_open() or is_configured()):
        return {"configured": False}
    out: dict[str, Any] = {"configured": True, "schema_at_head": None}
    scratch: Engine | None = None
    try:
        settings = engines.current_settings() or DatabaseSettings.from_env()
        out["plane"] = settings.plane
        if engines.engines_open():
            engine = engines.ro_engine()
        else:
            scratch = engine = create_engine(settings.ro_url, connect_args={"connect_timeout": "5"})
        head = migrate.head_revision()
        with engine.connect() as conn:
            current = migrate.current_revision(conn, settings.plane)
            server_version = conn.execute(text("SHOW server_version")).scalar()
        out.update(
            current_revision=current,
            head_revision=head,
            server_version=server_version,
            schema_at_head=current == head,
        )
    except Exception as exc:  # noqa: BLE001 — an unreachable database is a null, not a 500
        logger.warning("version: database probe failed: %s", exc)
    finally:
        if scratch is not None:
            scratch.dispose()
    return out


def _corpus_snapshot(st: Any) -> dict[str, Any]:
    """The corpus stamp ``/api/health`` computes for the default output dir."""
    anchor = getattr(st, "output_dir", None)
    if not isinstance(anchor, Path) or not anchor.is_dir():
        return {"code_version": None, "produced_by": None}
    raw = read_manifest_produced_by_under_anchor(anchor, anchor)
    corpus_ver, _warning = assess_corpus_version_compat(raw)
    produced_by = None
    if raw and corpus_code_version(raw) is not None:
        try:
            produced_by = CorpusProducedBy.model_validate(raw)
        except Exception:  # noqa: BLE001 — same leniency as /api/health
            produced_by = None
    return {"code_version": corpus_ver, "produced_by": produced_by}


def snapshot(app: Any) -> dict[str, Any]:
    """The full (admin) view, cached on the app for :data:`CACHE_SECONDS`."""
    st = app.state
    cached = getattr(st, "_version_snapshot", None)
    now = time.monotonic()
    if cached is not None and now - cached[0] < CACHE_SECONDS:
        snap: dict[str, Any] = cached[1]
        return snap
    api = build_info.build_info()
    data = {
        "release": build_info.deploy_release(),
        "image_release": build_info.image_release(),
        "api": api,
        "database": _database_snapshot(),
        "corpus": _corpus_snapshot(st),
        "modules": extension_versions(),
    }
    st._version_snapshot = (now, data)
    return data


def _is_admin(request: Request) -> bool:
    user = get_optional_user(request)
    return user is not None and app_roles.is_admin(user.role)


_PUBLIC_DB_KEYS = ("configured", "schema_at_head")


def version_response(request: Request) -> VersionResponse:
    """The ``/version`` body for this request: public fields, plus admin fields for an admin."""
    data = snapshot(request.app)
    admin = _is_admin(request)
    db = data["database"]
    if not admin:
        db = {k: db[k] for k in _PUBLIC_DB_KEYS if k in db}
    # Admin-only fields stay unset for everyone else, so the route's exclude_unset drops them.
    extra: dict[str, Any] = {"modules": data["modules"]} if admin else {}
    return VersionResponse(
        release=data["release"],
        image_release=data["image_release"],
        api=VersionApi(**{k: data["api"][k] for k in ("version", "git_sha", "built_at")}),
        database=VersionDatabase(**db),
        corpus=VersionCorpus(**data["corpus"]),
        **extra,
    )


@router.get("/version", response_model=VersionResponse, response_model_exclude_unset=True)
def get_version(request: Request) -> VersionResponse:
    """What is running. Signed-out callers get the public fields; admins get more."""
    return version_response(request)


# --- Prometheus info metrics ------------------------------------------------------------------

_lock = threading.Lock()
_metrics_app: "weakref.ref[Any] | None" = None
_collector_registered = False


def install_version_metrics(app: Any) -> bool:
    """Publish ``podcast_build_info`` and ``podcast_db_schema_info`` on ``/metrics``.

    Read at scrape time from the cached :func:`snapshot`. One collector per process: the latest
    app wins, so rebuilding the app (tests) never double-registers. Never raises.
    """
    global _metrics_app, _collector_registered
    try:
        from prometheus_client import REGISTRY
        from prometheus_client.core import GaugeMetricFamily
    except Exception:  # noqa: BLE001 — metrics extras absent; run without these gauges
        return False

    class _VersionCollector:
        def collect(self):  # noqa: ANN202 — prometheus_client's duck-typed collector protocol
            build = GaugeMetricFamily(
                "podcast_build_info",
                "Build of each running component (value is always 1).",
                labels=["component", "version", "git_sha", "release"],
            )
            schema = GaugeMetricFamily(
                "podcast_db_schema_info",
                "Database schema revision of this process's plane (value is always 1).",
                labels=["plane", "revision", "at_head"],
            )
            target = _metrics_app() if _metrics_app is not None else None
            if target is not None:
                try:
                    data = snapshot(target)
                except Exception:  # noqa: BLE001 — telemetry never breaks the scrape
                    data = None
                if data is not None:
                    release = data["release"] or data["image_release"] or ""
                    api = data["api"]
                    build.add_metric(
                        [api["component"] or "api", api["version"], api["git_sha"] or "", release],
                        1,
                    )
                    for mod in data["modules"]:
                        build.add_metric(
                            [
                                mod["name"],
                                str(mod.get("version") or ""),
                                str(mod.get("git_sha") or ""),
                                release,
                            ],
                            1,
                        )
                    db = data["database"]
                    if db.get("configured"):
                        schema.add_metric(
                            [
                                db.get("plane") or "",
                                db.get("current_revision") or "",
                                {True: "true", False: "false", None: "unknown"}[
                                    db.get("schema_at_head")
                                ],
                            ],
                            1,
                        )
            yield build
            yield schema

    with _lock:
        _metrics_app = weakref.ref(app)
        if _collector_registered:
            return True
        try:
            REGISTRY.register(_VersionCollector())  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 — already registered by another import path
            return False
        _collector_registered = True
    return True


__all__ = ["install_version_metrics", "router", "snapshot", "version_response"]
