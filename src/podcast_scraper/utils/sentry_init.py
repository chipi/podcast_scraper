"""Shared Sentry SDK initialisation for the api and the pipeline subprocess.

Both surfaces use the same helper so the configuration story is uniform:
DSN comes from a component-specific env var, environment + release tags
come from shared env vars, and a missing DSN turns init into a true
no-op so dev / CI / offline boots stay silent.

Wired from:

* :mod:`podcast_scraper.server.app` (api startup, FastAPI process)
* :mod:`podcast_scraper.cli` ``main()`` (pipeline subprocess, one-shot
  ``python -m podcast_scraper.cli`` invocations spawned by the api job
  factory inside ``docker compose run pipeline-llm`` / ``pipeline``)

Default behaviour: nothing happens unless the relevant DSN env var is
set. The api reads ``PODCAST_SENTRY_DSN_API``; the pipeline reads
``PODCAST_SENTRY_DSN_PIPELINE``. Different DSNs keep frontend / backend
event streams separable in the Sentry UI.

See: RFC-081 §Layer 2, issue #681.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

_LOGGER = logging.getLogger(__name__)

Component = Literal["api", "pipeline"]

_DSN_ENV = {
    "api": "PODCAST_SENTRY_DSN_API",
    "pipeline": "PODCAST_SENTRY_DSN_PIPELINE",
}

# Per-component traces sample rates. The api is long-lived and serves few
# requests at hobby scale → sampling 10 % stays well under Sentry free
# tier's 10 k transaction/month cap. The pipeline runs short, bursty
# subprocesses with many spans per run; we leave traces off by default
# because tracing the per-stage timings would burn the free tier on a
# single bulk run.
_DEFAULT_TRACES_SAMPLE_RATE: dict[Component, float] = {
    "api": 0.1,
    "pipeline": 0.0,
}


def init_sentry(component: Component) -> bool:
    """Initialise the Sentry SDK for the given component.

    Returns ``True`` if the SDK was initialised, ``False`` if the relevant
    DSN env var was unset (no-op path) or if ``sentry-sdk`` could not be
    imported. Callers do not need to act on the return value — it's
    informational so logs can record which component activated Sentry.

    The function is idempotent in practice: ``sentry_sdk.init`` replaces
    the global hub, so calling twice is fine but wasteful. Don't call
    twice from the same process unless you intend to swap DSNs.
    """
    dsn_var = _DSN_ENV[component]
    dsn = os.environ.get(dsn_var, "").strip()
    if not dsn:
        _LOGGER.debug(
            "sentry init skipped for component=%s (env %s unset)",
            component,
            dsn_var,
        )
        return False

    try:
        import sentry_sdk
    except ImportError:
        # sentry-sdk is in base deps so this should never trip in practice;
        # if it does, log loudly rather than silently swallow the misconfig.
        _LOGGER.error(
            "PODCAST_SENTRY_DSN_%s is set but sentry-sdk is not installed. "
            "Reinstall the package (sentry-sdk is in base dependencies).",
            component.upper(),
        )
        return False

    environment = os.environ.get("PODCAST_ENV", "dev").strip() or "dev"
    release = os.environ.get("PODCAST_RELEASE", "").strip() or None
    if not release:
        # Best-effort fallback to the package version so events at least
        # group by version.
        try:
            from podcast_scraper import __version__ as pkg_version

            release = f"podcast-scraper@{pkg_version}"
        except Exception:  # pragma: no cover - defensive
            release = None

    traces_rate = _DEFAULT_TRACES_SAMPLE_RATE[component]

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=traces_rate,
        # Keep `send_default_pii=False` so request bodies + headers don't
        # leak op data. Custom tags / contexts are still attached.
        send_default_pii=False,
        # Default integrations (Logging / Threading / Asyncio /
        # ExcepthookIntegration / DedupeIntegration / AtexitIntegration /
        # StdlibIntegration) cover the surfaces we care about. The FastAPI
        # integration auto-activates when FastAPI is imported, so the api
        # gets it without an explicit ``integrations=[...]`` argument.
    )

    # Tag every event with the component so the api / pipeline streams can
    # be filtered cleanly in Sentry's UI.
    sentry_sdk.set_tag("component", component)

    _LOGGER.info(
        "sentry init complete component=%s environment=%s release=%s " "traces_sample_rate=%s",
        component,
        environment,
        release,
        traces_rate,
    )
    return True
