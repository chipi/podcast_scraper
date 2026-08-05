"""The delivery outbox — the app side of the app↔infra seam (#1415, RFC-110 §2, ADR-145).

The app enqueues ``DeliveryEnvelope``s here; the infra delivery worker (#1412) drains them over
``/internal/outbox/*`` (see ``routes/internal_outbox``), renders + delivers, and reports terminal
status back. This store is the **single source of truth + the only suppression authority** — there
is no external queue (Listmonk was dropped, ADR-144 revision).

Layout: one file per envelope at ``<data_dir>/outbox/<id>.json`` =
``{envelope, status, detail, updated_at}``. ``enqueue`` dedupes on the envelope ``id`` (re-running
the same digest period is a no-op). Terminal statuses are final.

Seam v1.1 amendments implemented here:
- **(1) idempotent status** — a repeated terminal status is a no-op (``record_status``).
- **(2) current-consent filtering** — ``list_pending`` re-reads live ``comms`` (the enqueue-time
  ``consent_snapshot`` is informational only) and drops since-unsubscribed + expired envelopes.
- **(4) ``failed`` is always-terminal** — in ``_TERMINAL``; the worker dead-letters as ``failed``.
- **(5) push ``410/404`` → ``bounced``** — a ``bounced`` push terminal disables ``push.enabled``;
  an email ``bounced``/``complaint`` disables ``digest.enabled`` (the suppression write-back).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from filelock import FileLock

from podcast_scraper.server import app_comms_store
from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0

#: Statuses that end an envelope's life. ``failed`` is always-terminal (v1.1 amendment 4): the
#: worker dead-letters as ``failed`` after N retries, so the app never re-offers it.
_TERMINAL: frozenset[str] = frozenset({"delivered", "bounced", "complaint", "suppressed", "failed"})

#: Terminal statuses that mean "stop sending to this recipient on this channel" (v1.1 amendment 5).
_SUPPRESSING: frozenset[str] = frozenset({"bounced", "complaint", "suppressed"})


def _outbox_dir(data_dir: Path) -> Path:
    return data_dir / "outbox"


def _envelope_path(data_dir: Path, envelope_id: str) -> Path:
    # The filename is a HASH of the id, never the id itself. ``envelope_id`` arrives as a request
    # path param (POST /internal/outbox/{id}/status), so hashing removes the path-injection surface
    # entirely (a hex digest can't carry a separator; CodeQL py/path-injection). Dedupe still holds:
    # same id → same hash → same file. The real id lives inside the file, so ``list_pending`` (which
    # reads file contents, not names) is unaffected.
    digest = hashlib.sha256(envelope_id.encode("utf-8")).hexdigest()
    return _outbox_dir(data_dir) / f"{digest}.json"


def _lock(data_dir: Path, envelope_id: str) -> FileLock:
    path = _envelope_path(data_dir, envelope_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(path.with_suffix(".lock")), timeout=_LOCK_TIMEOUT_S)


def _read_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def enqueue(data_dir: Path, envelope: dict[str, Any]) -> bool:
    """Write an envelope to the outbox as ``pending``. Idempotent on ``envelope['id']``.

    Returns True when a new record was written, False when the id already existed (dedupe).
    """
    envelope_id = str(envelope.get("id") or "")
    if not envelope_id:
        raise ValueError("envelope must carry a non-empty id")
    with _lock(data_dir, envelope_id):
        path = _envelope_path(data_dir, envelope_id)
        if path.is_file():
            return False
        record = {
            "envelope": envelope,
            "status": "pending",
            "detail": None,
            "updated_at": int(time.time()),
        }
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2))
    return True


def _consent_allows(data_dir: Path, envelope: dict[str, Any]) -> bool:
    """Whether the user's *current* consent still permits delivering this envelope (amend. 2)."""
    user_id = str(envelope.get("user_id") or "")
    channel = envelope.get("channel")
    comms = app_comms_store.get_comms(data_dir, user_id)
    if channel == "email":
        digest = comms["digest"]
        return bool(digest["enabled"]) and not bool(digest["paused"])
    if channel == "push":
        return bool(comms["push"]["enabled"])
    return False


def list_pending(
    data_dir: Path, *, channel: str, limit: int = 50, now: int | None = None
) -> list[dict[str, Any]]:
    """Pending envelopes for a channel — oldest-first, current-consent-filtered, non-expired.

    Excludes any envelope whose user has since unsubscribed (live ``comms`` re-check) and any whose
    ``expires_at`` has passed (so a homelab-down window can't flush stale digests on recovery).
    """
    now = int(time.time()) if now is None else now
    outbox = _outbox_dir(data_dir)
    if not outbox.is_dir():
        return []
    rows: list[tuple[int, dict[str, Any]]] = []
    for path in outbox.glob("*.json"):
        record = _read_record(path)
        if record is None or record.get("status") != "pending":
            continue
        envelope = record.get("envelope")
        if not isinstance(envelope, dict) or envelope.get("channel") != channel:
            continue
        expires_at = envelope.get("expires_at")
        if isinstance(expires_at, str) and _expired(expires_at, now):
            continue
        if not _consent_allows(data_dir, envelope):
            continue
        rows.append((int(record.get("updated_at", 0)), envelope))
    rows.sort(key=lambda r: r[0])
    return [env for _, env in rows[: max(0, limit)]]


def _expired(expires_at: str, now: int) -> bool:
    import datetime as _dt

    try:
        dt = _dt.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError:
        return False  # unparsable TTL → treat as non-expiring rather than silently drop
    return dt.timestamp() < now


def record_status(data_dir: Path, envelope_id: str, status: str, detail: str | None = None) -> str:
    """Record a terminal status for an envelope. Idempotent per id (v1.1 amendment 1).

    A repeated terminal status is a no-op that returns the already-stored status (the worker may
    retry a status-write after a send whose ack failed). On a suppressing terminal status the
    matching consent channel is disabled — the app stops enqueuing to that recipient (amendment 5).
    Returns the effective (stored) status. Unknown ids return ``"unknown"``.
    """
    if status not in _TERMINAL:
        raise ValueError(f"not a terminal status: {status!r}")
    with _lock(data_dir, envelope_id):
        path = _envelope_path(data_dir, envelope_id)
        record = _read_record(path)
        if record is None:
            return "unknown"
        if record.get("status") in _TERMINAL:
            return str(record["status"])  # idempotent: already terminal
        record["status"] = status
        record["detail"] = detail
        record["updated_at"] = int(time.time())
        atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2))
        envelope = record.get("envelope") or {}
    if status in _SUPPRESSING:
        _suppress(data_dir, envelope)
    return status


def _suppress(data_dir: Path, envelope: dict[str, Any]) -> None:
    """Disable the channel the terminal status arrived on (bounce/complaint → stop sending)."""
    user_id = str(envelope.get("user_id") or "")
    if not user_id:
        return
    channel = envelope.get("channel")
    try:
        if channel == "email":
            app_comms_store.set_comms(data_dir, user_id, digest={"enabled": False})
        elif channel == "push":
            app_comms_store.set_comms(data_dir, user_id, push={"enabled": False})
    except ValueError:
        return
