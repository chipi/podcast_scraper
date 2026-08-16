"""Unit tests for the delivery outbox store (#1415, app_outbox_store) — the seam v1.1 invariants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import app_comms_store, app_outbox_store

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"


def _envelope(eid: str = "dgst_2026W31_" + _UID, channel: str = "email", **over: Any) -> dict:
    env = {
        "schema_version": "1",
        "id": eid,
        "user_id": _UID,
        "channel": channel,
        "template": "your-week-digest.v1",
        "recipient": {"email": "u@x.com", "email_verified": True},
        "consent_snapshot": {"digest_enabled": True, "cadence": "weekly", "unsubscribe_ref": "r"},
        "payload": {"sections": []},
        "created_at": "2026-08-04T00:00:00Z",
    }
    env.update(over)
    return env


def _enable_digest(data_dir: Path) -> None:
    app_comms_store.set_comms(data_dir, _UID, digest={"enabled": True})


def test_enqueue_is_idempotent_on_id(tmp_path: Path) -> None:
    assert app_outbox_store.enqueue(tmp_path, _envelope()) is True
    assert app_outbox_store.enqueue(tmp_path, _envelope()) is False  # dedupe


def test_enqueue_requires_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        app_outbox_store.enqueue(tmp_path, _envelope(eid=""))


def test_pending_filtered_by_current_consent(tmp_path: Path) -> None:
    app_outbox_store.enqueue(tmp_path, _envelope())
    # No consent yet → not returned (amendment 2: filter on CURRENT consent, not the snapshot).
    assert app_outbox_store.list_pending(tmp_path, channel="email") == []
    _enable_digest(tmp_path)
    got = app_outbox_store.list_pending(tmp_path, channel="email")
    assert [e["id"] for e in got] == [_envelope()["id"]]


def test_pending_excludes_paused(tmp_path: Path) -> None:
    app_outbox_store.enqueue(tmp_path, _envelope())
    app_comms_store.set_comms(tmp_path, _UID, digest={"enabled": True, "paused": True})
    assert app_outbox_store.list_pending(tmp_path, channel="email") == []


def test_pending_excludes_expired(tmp_path: Path) -> None:
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope(expires_at="2000-01-01T00:00:00Z"))
    assert app_outbox_store.list_pending(tmp_path, channel="email", now=10**9) == []


def test_pending_channel_scoped(tmp_path: Path) -> None:
    _enable_digest(tmp_path)
    app_comms_store.set_comms(tmp_path, _UID, push={"enabled": True})
    app_outbox_store.enqueue(tmp_path, _envelope(eid="a", channel="email"))
    app_outbox_store.enqueue(tmp_path, _envelope(eid="b", channel="push"))
    assert [e["id"] for e in app_outbox_store.list_pending(tmp_path, channel="email")] == ["a"]
    assert [e["id"] for e in app_outbox_store.list_pending(tmp_path, channel="push")] == ["b"]


def test_record_status_terminal_then_idempotent(tmp_path: Path) -> None:
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope())
    assert app_outbox_store.record_status(tmp_path, _envelope()["id"], "delivered") == "delivered"
    # A repeated / conflicting terminal status is a no-op returning the stored one (amendment 1).
    assert app_outbox_store.record_status(tmp_path, _envelope()["id"], "failed") == "delivered"
    # delivered → dropped from pending
    assert app_outbox_store.list_pending(tmp_path, channel="email") == []


def test_record_status_unknown_id(tmp_path: Path) -> None:
    assert app_outbox_store.record_status(tmp_path, "nope", "delivered") == "unknown"


def test_record_status_rejects_non_terminal(tmp_path: Path) -> None:
    app_outbox_store.enqueue(tmp_path, _envelope())
    with pytest.raises(ValueError):
        app_outbox_store.record_status(tmp_path, _envelope()["id"], "pending")


def test_email_bounce_suppresses_digest(tmp_path: Path) -> None:
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope())
    app_outbox_store.record_status(tmp_path, _envelope()["id"], "bounced")
    # Suppression write-back: the app stops enqueuing to this recipient (amendment 5).
    assert app_comms_store.get_comms(tmp_path, _UID)["digest"]["enabled"] is False


def test_push_bounce_suppresses_push(tmp_path: Path) -> None:
    app_comms_store.set_comms(tmp_path, _UID, push={"enabled": True})
    app_outbox_store.enqueue(tmp_path, _envelope(eid="p", channel="push"))
    app_outbox_store.record_status(tmp_path, "p", "bounced")
    assert app_comms_store.get_comms(tmp_path, _UID)["push"]["enabled"] is False


def test_delivered_does_not_suppress(tmp_path: Path) -> None:
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope())
    app_outbox_store.record_status(tmp_path, _envelope()["id"], "delivered")
    assert app_comms_store.get_comms(tmp_path, _UID)["digest"]["enabled"] is True


def test_pending_multi_user_consent_isolation(tmp_path: Path) -> None:
    # Two users, one consented one not — only the consented user's envelope is returned.
    other = "u_fedcba9876543210fedcba98"
    _enable_digest(tmp_path)  # _UID consents
    app_outbox_store.enqueue(tmp_path, _envelope(eid="mine"))
    app_outbox_store.enqueue(tmp_path, _envelope(eid="theirs", user_id=other))
    got = [e["id"] for e in app_outbox_store.list_pending(tmp_path, channel="email")]
    assert got == ["mine"]


def test_pending_malformed_expires_at_is_not_dropped(tmp_path: Path) -> None:
    # An unparsable TTL is treated as non-expiring (documented graceful behaviour), not dropped.
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope(expires_at="not-a-date"))
    got = app_outbox_store.list_pending(tmp_path, channel="email", now=10**12)
    assert [e["id"] for e in got] == [_envelope()["id"]]


def test_pending_excludes_expired_offset_form(tmp_path: Path) -> None:
    # A tz-aware expires_at WITHOUT the Z suffix (explicit +00:00) is parsed + expired correctly.
    _enable_digest(tmp_path)
    app_outbox_store.enqueue(tmp_path, _envelope(expires_at="2000-01-01T00:00:00+00:00"))
    assert app_outbox_store.list_pending(tmp_path, channel="email", now=10**9) == []


def test_path_injection_id_stays_in_outbox(tmp_path: Path) -> None:
    # A traversal-style id (worker path param) can't escape the outbox dir — the filename is a hash.
    _enable_digest(tmp_path)
    evil = "../../../etc/passwd"
    app_outbox_store.enqueue(tmp_path, _envelope(eid=evil))
    outbox = tmp_path / "outbox"
    files = list(outbox.glob("*.json"))
    assert len(files) == 1  # written inside outbox/, nowhere else
    assert not (tmp_path.parent / "passwd").exists()
    # and it still round-trips by the original id
    assert app_outbox_store.record_status(tmp_path, evil, "delivered") == "delivered"
