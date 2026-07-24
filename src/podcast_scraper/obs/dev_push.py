"""Dev-only in-app push of logs + metrics straight to the homelab backends.

**DEV ONLY — and inert unless explicitly enabled.** Every entry point is a true
no-op unless the relevant push-URL env var is set:

* ``PODCAST_LOGS_PUSH_URL``    → VictoriaLogs jsonline ingest (logs, from ``emit_event``)
* ``PODCAST_METRICS_PUSH_URL`` → VictoriaMetrics import  (metrics, api background pusher)

Why this exists: a developer checks out a worktree and runs a server / CLI on some
port; they want the five signals in Grafana without installing or running any
collector daemon. So the process pushes its own logs + metrics (errors/LLM/traces
already self-push via their SDKs). Each process self-labels with
``instance=<worktree>-<port>`` so N servers on N ports across N worktrees never
collide.

**Packaged image:** the Docker/prod deploy does NOT set these env vars, so this
module does nothing — the Alloy collector ships everything exactly as before
(ADR-119 / ADR-121). No daemon, no new dependency (stdlib ``urllib`` only), one
bounded background queue for logs, one timer thread for metrics. Errors are
swallowed — telemetry must never break the app.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import threading
import urllib.request
from typing import Any, Optional
from urllib.parse import urlencode

_LOGGER = logging.getLogger(__name__)

_LOGS_URL_ENV = "PODCAST_LOGS_PUSH_URL"
_METRICS_URL_ENV = "PODCAST_METRICS_PUSH_URL"

# --- shared labels (service / environment / instance) --------------------------------


def _instance() -> str:
    """Stable per-process identity so multiple worktrees/ports don't collide.

    ``PODCAST_OBS_INSTANCE`` wins; else ``<worktree-dir>-<port>`` (the cwd basename is
    the worktree dir name, distinct per checkout; the port separates several servers in
    one worktree). Port from ``PODCAST_OBS_PORT`` / ``PORT`` if present.
    """
    explicit = os.environ.get("PODCAST_OBS_INSTANCE", "").strip()
    if explicit:
        return explicit
    worktree = os.path.basename(os.getcwd()) or "dev"
    port = (os.environ.get("PODCAST_OBS_PORT") or os.environ.get("PORT") or "").strip()
    return f"{worktree}-{port}" if port else worktree


def obs_labels() -> dict[str, str]:
    return {
        "service": (os.environ.get("OTEL_SERVICE_NAME", "").strip() or "pipeline"),
        "environment": (os.environ.get("PODCAST_ENV", "dev").strip() or "dev"),
        "instance": _instance(),
    }


def _post(url: str, body: bytes, content_type: str, timeout: float = 3.0) -> None:
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(req, timeout=timeout):  # noqa: S310 — tailnet-only dev URL
        pass


# --- logs: emit_event -> VictoriaLogs (bounded queue + one background sender) ---------

_log_q: "queue.Queue[Any]" = queue.Queue(maxsize=10_000)  # dict records + the _STOP sentinel
_log_worker: Optional[threading.Thread] = None
_log_worker_lock = threading.Lock()
_STOP = object()


def logs_push_enabled() -> bool:
    return bool(os.environ.get(_LOGS_URL_ENV, "").strip())


def _logs_endpoint() -> str:
    base = os.environ[_LOGS_URL_ENV].strip()
    # VictoriaLogs jsonline: label fields become the stream; ts/event_type map the envelope.
    q = urlencode(
        {
            "_stream_fields": "service,environment,instance",
            "_time_field": "ts",
            "_msg_field": "event_type",
        }
    )
    return f"{base}{'&' if '?' in base else '?'}{q}"


def _log_sender() -> None:
    endpoint = _logs_endpoint()
    while True:
        first = _log_q.get()
        if first is _STOP:
            return
        batch = [first]
        # Drain whatever else is queued (cap the batch so a burst can't build a huge body).
        while len(batch) < 500:
            try:
                nxt = _log_q.get_nowait()
            except queue.Empty:
                break
            if nxt is _STOP:
                batch.append(_STOP)  # honour stop after flushing what we have
                break
            batch.append(nxt)
        docs = [d for d in batch if d is not _STOP]
        try:
            body = "\n".join(json.dumps(d, default=str, ensure_ascii=False) for d in docs)
            _post(endpoint, body.encode("utf-8"), "application/stream+json")
        except Exception:  # noqa: BLE001 — never break the app for a log push
            _LOGGER.debug("dev_push logs POST failed (dropped %d)", len(docs), exc_info=True)
        if any(d is _STOP for d in batch):
            return


def _ensure_worker() -> None:
    global _log_worker
    if _log_worker is not None and _log_worker.is_alive():
        return
    with _log_worker_lock:
        if _log_worker is not None and _log_worker.is_alive():
            return
        _log_worker = threading.Thread(target=_log_sender, name="obs-logs-push", daemon=True)
        _log_worker.start()
        atexit.register(_flush_logs)


def push_event(record: dict[str, Any]) -> None:
    """Enqueue one ``emit_event`` record for VictoriaLogs. No-op unless enabled. Never blocks."""
    if not logs_push_enabled():
        return
    try:
        _ensure_worker()
        _log_q.put_nowait({**record, **obs_labels()})
    except queue.Full:
        _LOGGER.debug("dev_push logs queue full; dropping event")
    except Exception:  # noqa: BLE001
        _LOGGER.debug("dev_push push_event failed", exc_info=True)


def _flush_logs() -> None:
    if _log_worker is None or not _log_worker.is_alive():
        return
    try:
        _log_q.put_nowait(_STOP)
        _log_worker.join(timeout=4.0)
    except Exception:  # noqa: BLE001
        pass


# --- metrics: prometheus registry -> VictoriaMetrics (timer thread; api/server only) ---

_metrics_thread: Optional[threading.Thread] = None
_metrics_stop = threading.Event()


def metrics_push_enabled() -> bool:
    return bool(os.environ.get(_METRICS_URL_ENV, "").strip())


def _metrics_endpoint() -> str:
    base = os.environ[_METRICS_URL_ENV].strip()
    labels = obs_labels()
    # VM import/prometheus injects extra labels into every pushed series via query params.
    q = "&".join(f"extra_label={k}={v}" for k, v in labels.items())
    return f"{base}{'&' if '?' in base else '?'}{q}"


def push_metrics_once() -> None:
    """Serialize the default Prometheus registry and push it. No-op unless enabled."""
    if not metrics_push_enabled():
        return
    try:
        from prometheus_client import generate_latest, REGISTRY

        _post(_metrics_endpoint(), generate_latest(REGISTRY), "text/plain")
    except Exception:  # noqa: BLE001 — never break the app for a metrics push
        _LOGGER.debug("dev_push metrics POST failed", exc_info=True)


def start_metrics_pusher(interval_seconds: float = 15.0) -> bool:
    """Start a background timer that pushes metrics every ``interval_seconds``.

    For long-running surfaces (the api). No-op + returns False unless enabled. Idempotent.
    """
    global _metrics_thread
    if not metrics_push_enabled():
        return False
    if _metrics_thread is not None and _metrics_thread.is_alive():
        return True

    def _loop() -> None:
        while not _metrics_stop.wait(interval_seconds):
            push_metrics_once()

    _metrics_thread = threading.Thread(target=_loop, name="obs-metrics-push", daemon=True)
    _metrics_thread.start()
    atexit.register(push_metrics_once)  # final snapshot on shutdown
    _LOGGER.info(
        "dev_push metrics pusher started (interval=%ss, instance=%s)", interval_seconds, _instance()
    )
    return True
