#!/usr/bin/env python3
"""Player digest scheduler sidecar (#1412 / #1415).

The public player backend runs ``PODCAST_SERVE_APP_ONLY=1`` (ADR-116), which force-disables
the in-process job scheduler (``src/podcast_scraper/server/app.py``: ``enable_jobs_api = False``
under app_only) — so the "Your Week" digest would never auto-fire on the player. This tiny
sidecar owns that cadence instead: it wakes at the top of every interval and calls
``app_digest_dispatch.enqueue_all_due`` — literally the same function, over the same enqueuer
list, that the in-process scheduler calls. Each enqueuer gates every user on their own consent +
cadence slot and dedupes per period.

The shared dispatcher is load-bearing, not tidiness. This sidecar used to maintain its own copy
of the dispatch logic, and it drifted: ``daily_recap`` (#2039) and the monthly ``recommendations``
digest were added to ``scheduler.py`` and not here. Since the player runs ONLY this sidecar, both
shipped dead to production and stayed dead for their entire lives, while this loop logged a
healthy "enqueued 0 envelope(s)" every hour (#2119). There is now one list, in
``server/app_digest_dispatch.py``, and adding to it wires both callers at once.

It is pure filesystem work — reads the read-only corpus (``/app/output``) and the shared appdata
bind mount (``/app/appdata``), writes ``DeliveryEnvelope``s to the outbox. NO network (the
container runs ``network_mode: none``) and NO secrets. The homelab delivery worker drains the
outbox over the tailnet and does the actual send (it has its own idempotency ledger).

Hardening rationale (advisor review 2026-08-07):
- Interval-aligned wake (not a fixed ``sleep``): a fixed sleep drifts forward and eventually
  skips a whole clock hour, silently dropping a weekly user whose slot lands in the skipped
  hour. Aligning to the top of the interval fires each slot hour exactly once.
- Fire once immediately on start so a restart landing inside a user's slot still delivers
  (idempotent — the per-period envelope id dedupes).
- Per-cycle try/except so one bad cycle never kills the loop (the assembler is also
  per-user-guarded). A heartbeat file + one log line per cycle drive the container healthcheck
  and the homelab dead-man alert; the process must never go dark unnoticed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, NoReturn

HEARTBEAT = Path(os.environ.get("DIGEST_HEARTBEAT_FILE", "/heartbeat/tick"))
CORPUS_ROOT = Path(os.environ.get("DIGEST_CORPUS_ROOT", "/app/output"))
DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/app/appdata"))
INTERVAL_S = int(os.environ.get("DIGEST_INTERVAL_SECONDS", "3600"))
# Fire this many seconds AFTER the interval boundary so a fire never races the boundary itself.
OFFSET_S = int(os.environ.get("DIGEST_INTERVAL_OFFSET_SECONDS", "120"))


def _log(msg: str) -> None:
    print(f"[digest-scheduler] {msg}", flush=True)


def _log_json(**fields: object) -> None:
    """One structured line per event, alongside the human one.

    Why structured: the delivery worker already stamps ``correlation_id`` (= the envelope id) on
    every metric, log, span and the outbound Resend header, so a delivered email is traceable end
    to end — EXCEPT backwards. This side only ever wrote envelope ids as free text appended to a
    tick line, so "when was this envelope enqueued, and by which cadence?" was not answerable
    without regex over logs. Emitting the same ``correlation_id`` key here closes the join
    (#2119).

    The sidecar runs ``network_mode: none`` and cannot emit OTEL spans; a structured log line
    carrying the trace key is the correlation that isolation permits.
    """
    print("[digest-scheduler-json] " + json.dumps(fields, sort_keys=True, default=str), flush=True)


def _beat() -> None:
    """Record loop liveness for the container healthcheck (an errored cycle still beats — the
    process is alive; 'enqueues nothing despite consenting users' is a homelab outcome-alert)."""
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(str(int(time.time())), encoding="utf-8")
    except OSError as exc:
        _log(f"heartbeat write failed: {exc}")


def _run_once() -> None:
    # This sidecar deliberately owns NO list of enqueuers. It shares one with the in-process
    # scheduler (``app_digest_dispatch.ENQUEUERS``), because two hand-maintained lists is exactly
    # what shipped daily_recap and the monthly recommendations digest dead to prod (#2119).
    # Per-enqueuer isolation and per-enqueuer counts both come from the shared dispatcher.
    from podcast_scraper.server import app_digest_dispatch

    result = app_digest_dispatch.enqueue_all_due(CORPUS_ROOT, DATA_DIR)
    ts = int(time.time())

    for label, err in result.errors.items():
        _log(f"enqueuer {label} failed: {err}")
        _log_json(event="digest.enqueuer.failed", cadence=label, error=err, ts=ts)

    # One structured line PER ENVELOPE, keyed on correlation_id — the same key the delivery
    # worker stamps on its spans, logs, metrics and the Resend header. This is what makes an
    # envelope traceable backwards from "delivered" to "enqueued at T by cadence X".
    for label, ids in result.ids.items():
        for envelope_id in ids:
            _log_json(
                event="digest.envelope.enqueued",
                correlation_id=envelope_id,
                cadence=label,
                ts=ts,
            )

    # Per-cadence tick summary. Counts per enqueuer, not just a total: "enqueued 0" is
    # ambiguous across several enqueuers and that ambiguity is what hid #2119.
    _log_json(
        event="digest.tick",
        total=result.total,
        cadences={
            label: len(result.ids.get(label, [])) for label, _, _ in app_digest_dispatch.ENQUEUERS
        },
        errored=sorted(result.errors),
        ts=ts,
    )

    detail = ": " + ", ".join(result.all_ids) if result.all_ids else ""
    _log(f"tick: enqueued {result.total} envelope(s) [{result.summary()}]{detail}")


def _sleep_to_next_interval(sleep: Callable[[float], object] = time.sleep) -> None:
    now = time.time()
    delay = INTERVAL_S - (now % INTERVAL_S) + OFFSET_S
    sleep(delay)


def _cycle() -> None:
    """One iteration: enqueue, and beat regardless — a bad enqueue must never kill the loop, and
    the heartbeat proves the process is still alive (an empty-but-alive loop is a homelab alert)."""
    try:
        _run_once()
    except Exception as exc:  # noqa: BLE001 — a bad cycle must never kill the loop
        _log(f"cycle error: {exc!r}")
    _beat()


def main() -> NoReturn:
    _log(
        f"start: corpus={CORPUS_ROOT} data_dir={DATA_DIR} "
        f"interval={INTERVAL_S}s offset={OFFSET_S}s heartbeat={HEARTBEAT}"
    )
    while True:
        _cycle()
        _sleep_to_next_interval()


if __name__ == "__main__":
    main()
