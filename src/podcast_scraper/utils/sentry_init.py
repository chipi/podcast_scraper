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
from typing import Literal, Optional

_LOGGER = logging.getLogger(__name__)

Component = Literal["api", "pipeline", "mcp"]

_DSN_ENV = {
    "api": "PODCAST_SENTRY_DSN_API",
    "pipeline": "PODCAST_SENTRY_DSN_PIPELINE",
    # The remote MCP server (RFC-112) — its own GlitchTip project, per the per-surface DSN split.
    "mcp": "PODCAST_SENTRY_DSN_MCP",
}

# Dev rung of the env ladder (dev → prod; the backend has no staging). On the
# operator's laptop (``make serve`` with no DSN set), api + pipeline errors go to
# the dedicated ``operator-dev`` GlitchTip project via the Tailscale host
# ``homelab`` — NO fixed IP, so only a device on the tailnet reaches it; on any
# other machine the send silently fails. NEVER fires under pytest or CI (keeps the
# unset-DSN no-op contract green + airgapped CI silent), nor in prod
# (``PODCAST_ENV=production``). The Python SDK accepts the dashless key.
_DEV_DSN = "http://53a88592c99e48bc8d505d258597ab78@homelab:8090/9"
_DEV_ENVS = {"dev", "development", "local"}


def _use_dev_default() -> bool:
    """True only in genuine local dev — not under pytest, not in CI, not in prod."""
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
        return False
    env = (os.environ.get("PODCAST_ENV", "dev").strip() or "dev").lower()
    return env in _DEV_ENVS


# Per-component traces sample rates. The api is long-lived and serves few
# requests at hobby scale → sampling 10 % stays well under Sentry free
# tier's 10 k transaction/month cap. The pipeline runs short, bursty
# subprocesses with many spans per run; we leave traces off by default
# because tracing the per-stage timings would burn the free tier on a
# single bulk run.
_DEFAULT_TRACES_SAMPLE_RATE: dict[Component, float] = {
    "api": 0.1,
    "pipeline": 0.0,
    # Long-lived server like the api, low request volume at hobby scale — same 10% sample.
    "mcp": 0.1,
}

# Substrings that mark a key as sensitive — scrubbed from anything we send.
_SENSITIVE_SUBSTRINGS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "dsn",
)


def _redact_mapping(data: object) -> None:
    """In-place redact sensitive-looking keys of a dict (best-effort, shallow)."""
    if not isinstance(data, dict):
        return
    for key in list(data):
        if isinstance(key, str) and any(s in key.lower() for s in _SENSITIVE_SUBSTRINGS):
            data[key] = "[redacted]"


def _before_send(event: dict, hint: object) -> dict:
    """Scrub obviously-sensitive keys before an event leaves the app.

    GlitchTip/Sentry stores what we send, and a key-in-env app leaks secrets most
    easily through **stack-frame locals** (an api key bound to a variable in the
    failing frame). ``send_default_pii=False`` keeps request bodies/IPs out; this
    scrubs `extra`/`contexts`, request headers, and frame `vars`. Never raises —
    a scrub error must not drop the (already-useful) error event.
    """
    try:
        for section in ("extra", "contexts"):
            _redact_mapping(event.get(section))
        _redact_mapping((event.get("request") or {}).get("headers"))
        for exc in (event.get("exception") or {}).get("values", []) or []:
            for frame in (exc.get("stacktrace") or {}).get("frames", []) or []:
                _redact_mapping(frame.get("vars"))
    except Exception:  # noqa: BLE001 — a scrub failure must not drop the event
        _LOGGER.debug("sentry before_send scrub skipped", exc_info=True)
    # Tag with the active OTEL trace so an error jumps to its trace in VictoriaTraces
    # (correlation, ADR-119). Guarded — no [otel] / no span → no tag.
    try:
        from opentelemetry import trace as _otel_trace

        _ctx = _otel_trace.get_current_span().get_span_context()
        if getattr(_ctx, "is_valid", False):
            event.setdefault("tags", {})["trace_id"] = format(_ctx.trace_id, "032x")
    except Exception:  # noqa: BLE001 — no OTEL / no span
        pass
    return event


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
    if not dsn and _use_dev_default():
        dsn = _DEV_DSN
        _LOGGER.info(
            "sentry init using dev-rung default for component=%s → operator-dev "
            "(homelab, tailnet-gated) — %s unset in local dev",
            component,
            dsn_var,
        )
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

    try:
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_rate,
            # Keep `send_default_pii=False` so request bodies + headers don't
            # leak op data. Custom tags / contexts are still attached.
            send_default_pii=False,
            # Scrub sensitive keys (tokens/secrets) from extra/contexts/headers and,
            # critically, stack-frame locals — GlitchTip/Sentry stores what we send.
            # (SDK types before_send with its TypedDict Event; our plain-dict scrubber
            # is structurally compatible — ignore the narrow-type mismatch.)
            before_send=_before_send,  # type: ignore[arg-type]
            # Default integrations (Logging / Threading / Asyncio /
            # ExcepthookIntegration / DedupeIntegration / AtexitIntegration /
            # StdlibIntegration) cover the surfaces we care about. The FastAPI
            # integration auto-activates when FastAPI is imported, so the api
            # gets it without an explicit ``integrations=[...]`` argument.
        )
        # Tag every event with the component so the api / pipeline streams can
        # be filtered cleanly in Sentry's UI.
        sentry_sdk.set_tag("component", component)
        # The OTLP span exporter logs a transient "Failed to export span batch due to timeout" when
        # the endpoint is briefly unreachable at process shutdown (now bounded in otel_init). That
        # is best-effort tracing losing a batch, NOT an application error — keep it out of GlitchTip
        # so a deploy window doesn't spam the pipeline issue stream (73 events on 2026-07-28).
        try:
            from sentry_sdk.integrations.logging import ignore_logger

            for _otel_logger in (
                "opentelemetry.sdk.trace.export",
                "opentelemetry.exporter.otlp.proto.http.trace_exporter",
            ):
                ignore_logger(_otel_logger)
        except Exception:  # noqa: BLE001 — never break sentry init over a log filter
            pass
    except Exception:  # noqa: BLE001 — telemetry must NEVER break the app
        # A malformed DSN raises sentry_sdk.utils.BadDsn ("Unsupported scheme");
        # any other init failure is equally fatal if unguarded. A telemetry
        # misconfig must DISABLE Sentry, not crash the process (empty DSN is
        # already handled above; this guards a non-empty-but-invalid DSN — e.g.
        # a wrong scheme from a bad secret). The app continues without Sentry.
        _LOGGER.exception(
            "sentry init FAILED for component=%s (env %s) — Sentry disabled, "
            "app continues. Check the DSN value.",
            component,
            dsn_var,
        )
        return False

    _LOGGER.info(
        "sentry init complete component=%s environment=%s release=%s " "traces_sample_rate=%s",
        component,
        environment,
        release,
        traces_rate,
    )
    return True


def set_run_tag(run_id: Optional[str], episode_id: Optional[str] = None) -> None:
    """Tag the Sentry scope with the run/episode correlation ids (#1053).

    So a Sentry error carries the same ``run_id`` as the run's Loki cost events and
    Langfuse trace — the join key an agent uses to pull every signal for one run. A true
    no-op when ``sentry-sdk`` isn't installed or Sentry wasn't initialised (no DSN).
    """
    if not run_id:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        sentry_sdk.set_tag("run_id", run_id)
        if episode_id:
            sentry_sdk.set_tag("episode_id", episode_id)
    except Exception:  # pragma: no cover - never break the run for a tag
        _LOGGER.debug("sentry set_run_tag skipped", exc_info=True)


def set_correlation_tags(tags: dict) -> None:
    """Tag the Sentry scope with a correlation envelope (RFC-088 enrichment layer).

    Wider than ``set_run_tag``: accepts the full
    ``correlation.sentry_tags_for_context(ctx)`` dict (run_id,
    parent_run_id, enricher_id, enricher_version, tier, attempt).
    Tags must be strings — the correlation helper coerces ``attempt``
    and ``parent_run_id`` already.

    True no-op when ``sentry-sdk`` isn't installed or Sentry wasn't
    initialised — enrichment paths still work without the o11y extra.
    """
    if not tags:
        return
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        for key, value in tags.items():
            if isinstance(value, str):
                sentry_sdk.set_tag(key, value)
    except Exception:  # pragma: no cover - never break the run for a tag
        _LOGGER.debug("sentry set_correlation_tags skipped", exc_info=True)


def capture_stage_exception(exc: BaseException, *, stage: str, level: Optional[str] = None) -> None:
    """Send a CAUGHT pipeline exception to Sentry, tagged with the stage + run/episode ids (o11y).

    The pipeline swallows most stage failures (``except Exception: logger...``), so they never reach
    the SDK's automatic capture — a large error-observability blind spot. Call this at a
    stage-boundary catch to make a swallowed failure an issue in GlitchTip (tagged ``stage`` +
    ``run_id``/``episode_id`` for triage) while the pipeline still recovers. True no-op when
    ``sentry-sdk`` isn't installed or Sentry wasn't initialised (no DSN).

    ``level`` separates "the run broke" from "this episode degraded, by design". Without it every
    caught stage failure arrives at Sentry's default severity — ``error`` — so an outcome the code
    deliberately recovers from is indistinguishable from a crash. #1632 is the cost of that: a
    summary that failed its schema twice is a RECOVERABLE degradation (#1496 — the episode keeps
    its transcript, GI and KG), it is logged at WARNING, and it was still filed as an error-level
    bug because this function reported it as one. Pass ``level="warning"`` for a by-design
    degradation; leave it unset when the failure is genuinely an error.

    #1686 sharpened where that line falls, and corrected a claim this docstring used to make.
    "The stage ledger records it" was not true for the summary stage: the five paths ending in an
    episode with NO summary wrote nothing to the ledger, so the only trace was the warning. They
    now record ``summarization: failed`` with a cause slug — and because the degradation is
    recorded and retried elsewhere, a summary that is genuinely LOST is reported here at
    ``error``. By-design still means warning; permanently incomplete does not. The event is
    still captured either way — this changes how loud it is, not whether anyone can see it.
    """
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        from .correlation import get_episode_id, get_run_id

        with sentry_sdk.push_scope() as scope:
            scope.set_tag("stage", stage)
            run_id = get_run_id()
            if run_id:
                scope.set_tag("run_id", run_id)
            episode_id = get_episode_id()
            if episode_id:
                scope.set_tag("episode_id", episode_id)
            if level is not None:
                scope.level = level
            sentry_sdk.capture_exception(exc)
    except Exception:  # pragma: no cover - never break the pipeline to report an error
        _LOGGER.debug("sentry capture_stage_exception skipped", exc_info=True)


def emit_enrichment_breadcrumb(
    category: str,
    message: str,
    *,
    level: str = "info",
    data: Optional[dict] = None,
) -> None:
    """Fire a Sentry breadcrumb for an enrichment-layer event.

    Used by ``enrichment/resilience.py`` for circuit-open / auto-disable
    / stall-escalation event categories. Operators define their own
    threshold alert rules in Sentry (e.g. "more than 5
    enrichment.circuit_opened breadcrumbs per hour" → page).

    Categories used by the enrichment layer:

    * ``enrichment.circuit_opened`` — circuit breaker tripped for an
      enricher within a run.
    * ``enrichment.auto_disabled`` — N consecutive failed runs flipped
      the enricher to auto-disabled.
    * ``enrichment.stall_escalation`` — heartbeat watchdog escalated
      to cancel.

    True no-op when ``sentry-sdk`` isn't installed.
    """
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        sentry_sdk.add_breadcrumb(
            category=category,
            message=message,
            level=level,
            data=data or {},
        )
    except Exception:  # pragma: no cover - never break the run for a breadcrumb
        _LOGGER.debug("sentry emit_enrichment_breadcrumb skipped", exc_info=True)


def capture_enrichment_message(
    message: str,
    *,
    level: str = "warning",
    tags: Optional[dict] = None,
) -> None:
    """Send a one-off Sentry message for a notable enrichment event.

    Used for ``auto_disabled`` (warning level) and
    ``stall_escalation`` (error level) — events that warrant their
    own issue rather than just a breadcrumb on a future exception.

    True no-op when ``sentry-sdk`` isn't installed.
    """
    try:
        import sentry_sdk
    except ImportError:
        return
    try:
        with sentry_sdk.push_scope() as scope:
            for key, value in (tags or {}).items():
                if isinstance(value, str):
                    scope.set_tag(key, value)
            # ``level`` is typed as a narrow Literal in the SDK stubs; pass via
            # the underlying API which accepts a plain string.
            sentry_sdk.capture_message(message, level=level)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - never break the run for a message
        _LOGGER.debug("sentry capture_enrichment_message skipped", exc_info=True)
