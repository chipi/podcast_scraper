"""The player as an extension (ADR-162): its routes, its lifecycle hooks, its scheduled digest and
the listener engagement trending blends in.

Moves to the private Player package at the cutover. Imports stay inside the functions (see
``app_mcp_extension``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

from podcast_scraper.extensions import Extension, RouterMount

if TYPE_CHECKING:
    from podcast_scraper.server.app_user_store import User

logger = logging.getLogger(__name__)


def _routers() -> Sequence[RouterMount]:
    from podcast_scraper.server.routes import (
        app_artwork,
        app_capture,
        app_collections,
        app_comms,
        app_consolidation,
        app_corpus,
        app_discover,
        app_enrichment,
        app_episodes,
        app_exits,
        app_export,
        app_key_voices,
        app_notifications,
        app_relational,
        app_search,
        app_user_state,
        app_your_week,
    )

    return tuple(
        RouterMount(module.router, "app")
        for module in (
            app_artwork,
            app_episodes,
            app_exits,
            app_relational,
            app_discover,
            app_search,
            app_user_state,
            app_capture,
            app_collections,
            app_comms,
            app_key_voices,
            app_notifications,
            app_your_week,
            app_corpus,
            app_export,
            app_enrichment,
            app_consolidation,
        )
    )


def _account_created(data_dir: Path, user: User, provider: str) -> None:
    from podcast_scraper.server import app_user_state

    app_user_state.append_account_created(data_dir, user.user_id, provider)


def _digest_health_metrics(app: Any) -> None:
    """The per-cadence digest delivery gauges (#2119), exported from the sidecar's state file.

    The digest sidecar runs ``network_mode: none`` and can neither expose nor push metrics. It
    writes a state file to the shared appdata volume and this API, already scraped as job ``api``,
    exports it at scrape time.
    """
    if os.environ.get("PODCAST_METRICS_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        return
    data_dir = getattr(app.state, "app_data_dir", None)
    if data_dir is None:
        return
    from podcast_scraper.server import app_digest_health

    app_digest_health.install_metrics(app, Path(data_dir))


def _start_cache_warmer(app: Any) -> Callable[[], None] | None:
    """Warm the read caches (catalog, slug index, KG index) at startup and on ingest, on a daemon
    thread. A failure just means lazy fills. Disable with ``APP_CACHE_WARMING=0``."""
    root = getattr(app.state, "output_dir", None)
    if root is None or os.environ.get("APP_CACHE_WARMING", "1") == "0":
        return None
    from podcast_scraper.server.app_cache_warm import start_cache_warmer

    return start_cache_warmer(Path(root)).set


def _digest_job(name: str, corpus_root: Path, app: Any) -> None:
    from podcast_scraper.server.app_digest_job import run_digest_job

    run_digest_job(name, corpus_root, app)


def _engagement_series(data_dir: Path, user_id: str | None) -> dict[str, Any]:
    from podcast_scraper.server.app_engagement_series import engagement_series

    return engagement_series(data_dir, user_id=user_id)


EXTENSION = Extension(
    name="player",
    routers=_routers,
    account_created=(_account_created,),
    app_configured=(_digest_health_metrics,),
    server_started=(_start_cache_warmer,),
    job_kinds={"digest": _digest_job},
    engagement_series=_engagement_series,
)
