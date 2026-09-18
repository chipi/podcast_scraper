#!/usr/bin/env python3
"""Player digest scheduler sidecar (#1412 / #1415).

The public player backend runs ``PODCAST_SERVE_APP_ONLY=1`` (ADR-116), which force-disables
the in-process job scheduler (``src/podcast_scraper/server/app.py``: ``enable_jobs_api = False``
under app_only) — so the "Your Week" digest would never auto-fire on the player. This tiny
sidecar owns that cadence instead: it wakes at the top of every interval and calls the SAME
assemblers the in-process scheduler would have (``_ENQUEUERS`` below), each of which gates every
user on their own consent + cadence slot and dedupes per period.

``_ENQUEUERS`` must mirror ``server/scheduler.py``'s digest branch. It did not, for the entire
life of ``daily_recap``: that enqueuer was added to ``scheduler.py`` and not here, so the only
code path prod runs never called it and no daily recap was ever produced (#2119).

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


def _beat() -> None:
    """Record loop liveness for the container healthcheck (an errored cycle still beats — the
    process is alive; 'enqueues nothing despite consenting users' is a homelab outcome-alert)."""
    try:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(str(int(time.time())), encoding="utf-8")
    except OSError as exc:
        _log(f"heartbeat write failed: {exc}")


#: Every enqueuer the in-process scheduler drives, so this sidecar cannot silently cover a
#: subset of them. Adding one to ``scheduler.py`` without adding it here is exactly how
#: ``daily_recap`` shipped dead to prod for its whole life (#2119) — the sidecar kept logging a
#: healthy "enqueued 0 envelope(s)" because the enqueuer that would have produced them was
#: never called. Keep this list in step with ``server/scheduler.py``'s digest branch.
_ENQUEUERS: tuple[tuple[str, str, str], ...] = (
    ("weekly", "app_digest_personal", "enqueue_due_digests"),
    ("recommendations", "app_digest_recommendations", "enqueue_due_recommendations"),
    ("daily_recap", "app_digest_daily_recap", "enqueue_due_daily_recaps"),
)


def _run_once() -> None:
    import importlib

    all_ids: list[str] = []
    counts: list[str] = []
    for label, module_name, func_name in _ENQUEUERS:
        # Isolated per enqueuer: a failure in one must not stop the others from running. They
        # are independent cadences and coupling them would let one broken assembler silence
        # every notification the product has.
        try:
            module = importlib.import_module(f"podcast_scraper.server.{module_name}")
            ids = list(getattr(module, func_name)(CORPUS_ROOT, DATA_DIR))
        except Exception as exc:  # noqa: BLE001 — one bad enqueuer must not mute the rest
            _log(f"enqueuer {label} failed: {exc!r}")
            counts.append(f"{label}=ERR")
            continue
        all_ids.extend(ids)
        counts.append(f"{label}={len(ids)}")

    detail = ": " + ", ".join(all_ids) if all_ids else ""
    # Per-enqueuer counts, not just the total: "enqueued 0" is ambiguous across several
    # enqueuers and that ambiguity is what hid #2119.
    _log(f"tick: enqueued {len(all_ids)} envelope(s) [{' '.join(counts)}]{detail}")


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
