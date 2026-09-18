"""Per-cadence delivery health — the signal that would have caught #2119.

THE GAP THIS FILLS
------------------
Before this, every alert on the digest/delivery chain was either a liveness check or needed
successful traffic to already exist:

===========================  ================================  =====================
alert                        watches                           caught #2119?
===========================  ================================  =====================
delivery-worker-down         ``up{job="delivery"}``            no — worker was up
delivery-scheduler-silent    tick log lines                    no — ticked hourly
delivery-events-stalled      cursor age                        no — cursor fine
delivery-high-bounce         *rate* of sends                   no — needs sends to exist
delivery-dead-letter         *increase* of sends               no — needs sends to exist
===========================  ================================  =====================

None of them can express "the expected output never arrived", which is why ``daily_recap`` and
the monthly ``recommendations`` digest ran dead for months behind green dashboards.

THE PRIMITIVE
-------------
Age of last success, PER CADENCE. Age beats rate here because it works from a single sample with
no traffic history, it catches *never happened* as well as *stopped happening*, and it needs no
interpretation — "daily_recap has produced nothing for 36 hours" is the finding.

Per cadence, not aggregate: the scheduler can be perfectly healthy while exactly one cadence is
broken, and an aggregate would mask it in both directions.

``digest_consenting_users`` is the denominator. Zero envelopes is CORRECT when nobody has
consented; without the denominator an alert cannot tell that from a fault, and an alert that
cries wolf on an empty roster gets muted, which is how you end up back here.

WRITE PATH
----------
The sidecar runs ``network_mode: none`` (it handles user content and must not reach the network),
so it can neither expose nor push metrics. It writes this state file to the shared appdata volume
instead; the player API — already scraped as job ``api`` — reads it and exports the gauges. No new
scrape target, no network for the sidecar.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

#: Written by the sidecar, read by the API exporter. Lives in appdata, which both mount.
STATE_FILENAME = "digest_health.json"


def _state_path(data_dir: Path) -> Path:
    return Path(data_dir) / STATE_FILENAME


def record_dispatch(
    data_dir: Path,
    result: Any,
    consenting: dict[str, int] | None = None,
    now: int | None = None,
) -> None:
    """Merge one dispatch pass into the state file.

    Carries forward the previous ``last_success_ts`` for any cadence that enqueued nothing this
    pass — "nobody was due this hour" must not look like "this cadence just succeeded", and it
    must not reset the age either.

    Never raises: health bookkeeping must not be able to break the delivery loop it measures.
    """
    now = int(time.time()) if now is None else now
    path = _state_path(data_dir)

    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent/corrupt state is a fresh start, not a failure
        prev = {}
    prev_cadences: dict[str, Any] = prev.get("cadences", {})

    cadences: dict[str, Any] = {}
    for label in result.ids.keys() | result.errors.keys():
        before = prev_cadences.get(label, {})
        ids = result.ids.get(label, [])
        errored = label in result.errors
        cadences[label] = {
            # Last time this cadence actually produced an envelope. Carried forward when idle.
            "last_success_ts": now if ids else before.get("last_success_ts"),
            "last_success_count": len(ids) if ids else before.get("last_success_count"),
            # Last time it RAN at all, successfully or not — distinguishes "never called"
            # (the #2119 failure) from "called and produced nothing".
            "last_run_ts": now,
            "last_error": result.errors.get(label),
            "last_error_ts": now if errored else before.get("last_error_ts"),
            "total_enqueued": int(before.get("total_enqueued") or 0) + len(ids),
        }

    payload = {
        "schema": 1,
        "updated_at": now,
        "cadences": cadences,
        "consenting": consenting if consenting is not None else prev.get("consenting", {}),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic: a torn read by the API exporter would export nonsense gauges.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".digest_health.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    except OSError:
        return  # a read-only or full volume must not take down the scheduler


def count_consenting_users(data_dir: Path) -> dict[str, int]:
    """How many users have each comms type enabled on the EMAIL channel.

    The denominator for the age alerts. Counted from the same store the enqueuers gate on, so a
    drift between "who the alert thinks is consenting" and "who the enqueuer serves" is not
    possible.
    """
    from podcast_scraper.server import app_comms_store

    counts: dict[str, int] = {t: 0 for t in app_comms_store.TYPES}
    try:
        from podcast_scraper.server.app_user_store import list_users
    except Exception:  # noqa: BLE001 — user store unavailable: report zeros, do not crash
        return counts

    try:
        users = list(list_users(data_dir))
    except Exception:  # noqa: BLE001
        return counts

    for user in users:
        try:
            comms = app_comms_store.get_comms(data_dir, user.user_id)
        except Exception:  # noqa: BLE001 — one unreadable user must not void the whole count
            continue
        for type_name in app_comms_store.TYPES:
            try:
                if app_comms_store.channel_enabled(comms, type_name, "email"):
                    counts[type_name] += 1
            except Exception:  # noqa: BLE001
                continue
    return counts


def read_state(data_dir: Path) -> dict[str, Any]:
    """Read the state file. Returns ``{}`` when absent — the exporter treats that as no data."""
    try:
        loaded = json.loads(_state_path(data_dir).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    # A truncated or hand-edited file could be any JSON type; the exporter indexes it as a
    # mapping, so anything else is treated as absent rather than allowed to raise at scrape time.
    return loaded if isinstance(loaded, dict) else {}


def install_metrics(app: Any, data_dir: Path) -> bool:
    """Publish the per-cadence gauges on the app's Prometheus registry.

    Gauges (all labelled by ``cadence`` so one broken cadence is visible while the rest are fine):

    ``podcast_digest_last_success_age_seconds`` — THE alert signal. Seconds since this cadence
        last produced an envelope, and **absent when it never has** (never zero — a zero would
        read as "succeeded just now" and invert the alert). The alert rules encode the "never"
        case explicitly as ``… or vector(999999) and on() (consenting_users > 0)``, so they run
        ``noDataState: OK``: an absent series with an EMPTY roster is correct silence, not a
        fault, and an alert that cries wolf on an empty roster gets muted.
    ``podcast_digest_last_run_age_seconds`` — seconds since it was last CALLED. Separates
        "never called" (the bug) from "called, nobody due" (correct).
    ``podcast_digest_enqueued_total`` — cumulative envelopes, for rate views once traffic exists.
    ``podcast_digest_last_error_age_seconds`` — seconds since it last raised.
    ``podcast_digest_consenting_users`` — labelled by ``type``; the denominator that keeps the
        age alerts from crying wolf on an empty roster.

    Returns True when the gauges were registered. Never raises: telemetry must not break the app
    (ADR-120).
    """
    try:
        from prometheus_client import Gauge, REGISTRY  # noqa: F401
        from prometheus_client.core import GaugeMetricFamily
    except Exception:  # noqa: BLE001 — metrics extras absent; run without these gauges
        return False

    class _DigestHealthCollector:
        """Collected at scrape time, so the gauges reflect the file as of the scrape rather
        than whenever the API last happened to run something."""

        def collect(self):  # noqa: ANN202 — prometheus_client's duck-typed collector protocol
            state = read_state(data_dir)
            now = int(time.time())

            success_age = GaugeMetricFamily(
                "podcast_digest_last_success_age_seconds",
                "Seconds since this cadence last enqueued an envelope (absent = never).",
                labels=["cadence"],
            )
            run_age = GaugeMetricFamily(
                "podcast_digest_last_run_age_seconds",
                "Seconds since this cadence's enqueuer was last called.",
                labels=["cadence"],
            )
            total = GaugeMetricFamily(
                "podcast_digest_enqueued_total",
                "Cumulative envelopes enqueued by this cadence.",
                labels=["cadence"],
            )
            error_age = GaugeMetricFamily(
                "podcast_digest_last_error_age_seconds",
                "Seconds since this cadence's enqueuer last raised (absent = never).",
                labels=["cadence"],
            )
            consenting = GaugeMetricFamily(
                "podcast_digest_consenting_users",
                "Users with this comms type enabled on the email channel.",
                labels=["type"],
            )

            for cadence, row in (state.get("cadences") or {}).items():
                if row.get("last_success_ts"):
                    success_age.add_metric([cadence], now - int(row["last_success_ts"]))
                if row.get("last_run_ts"):
                    run_age.add_metric([cadence], now - int(row["last_run_ts"]))
                if row.get("last_error_ts"):
                    error_age.add_metric([cadence], now - int(row["last_error_ts"]))
                total.add_metric([cadence], float(row.get("total_enqueued") or 0))

            for type_name, count in (state.get("consenting") or {}).items():
                consenting.add_metric([type_name], float(count))

            yield success_age
            yield run_age
            yield total
            yield error_age
            yield consenting

    try:
        REGISTRY.register(_DigestHealthCollector())  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — duplicate registration on app re-create is harmless
        return False
    return True
