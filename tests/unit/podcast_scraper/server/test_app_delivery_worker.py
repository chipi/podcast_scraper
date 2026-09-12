"""Unit tests for the email renderer + delivery worker (RFC-122 #2039 / #1412).

Covers the daily-recap render (escaping, absolutized links, adaptive subject), template dispatch,
and the worker's drain → send → record-status path, dry-run, failure-leaves-pending, and the
type-aware one-click unsubscribe header — all with a fake transport (no network).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.server import (
    app_comms_store,
    app_delivery_worker,
    app_email_render,
    app_email_send,
    app_outbox_store,
)
from podcast_scraper.server.app_digest_common import iso
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = pytest.mark.unit

_NOW = 1_760_000_000

_EPISODE = {
    "slug": "x",
    "title": "Bigger <isn't> better",  # angle brackets to prove escaping
    "podcast_title": "The Signal",
    "artwork_url": None,
    "key_points": ["Grounding beats size.", "Evaluation is the bottleneck."],
    "signature_quote": {"text": "Show your work.", "speaker": "Dr. Elena Fischer"},
    "insights": ["The bottleneck moved to evaluation."],
    "topics": [{"id": "topic:ai", "label": "AI"}],
    "storylines": [],
    "deep_link": "/player/x",
}


class _FakeTransport:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls: list[dict[str, Any]] = []

    def send(self, *, to: str, email: app_email_render.RenderedEmail, from_addr: str, headers):
        self.calls.append({"to": to, "email": email, "from": from_addr, "headers": headers})
        return app_email_send.SendResult(self.ok, "fake")


# --- renderer ---------------------------------------------------------------------------------


def test_render_daily_recap_single_is_full_and_escaped() -> None:
    e = app_email_render.render_daily_recap(
        {"day": "2026-09-12", "count": 1, "episodes": [_EPISODE]},
        origin="https://closelistening.app",
        unsubscribe_url="https://cl.app/api/app/comms/unsubscribe?ref=R&type=daily_recap",
        settings_url="https://closelistening.app/profile",
    )
    assert e.subject == "Your recap: Bigger <isn't> better"
    assert "&lt;isn&#x27;t&gt;" in e.html  # user text is escaped, not injected
    assert "Grounding beats size." in e.html
    assert "Show your work." in e.html and "Dr. Elena Fischer" in e.html
    assert "https://closelistening.app/player/x" in e.html  # deep link absolutized
    assert "type=daily_recap" in e.html  # the unsubscribe link is the recap's
    assert "Show your work." in e.text  # a plain-text alternative exists


def test_render_daily_recap_many_is_compact_stack() -> None:
    two = {"day": "2026-09-12", "count": 2, "episodes": [_EPISODE, {**_EPISODE, "title": "Second"}]}
    e = app_email_render.render_daily_recap(
        two, origin="https://o", unsubscribe_url="u", settings_url="s"
    )
    assert e.subject == "Your day, recapped — 2 episodes"
    assert "Second" in e.html


def test_render_email_unknown_template_raises() -> None:
    with pytest.raises(app_email_render.UnknownTemplateError):
        app_email_render.render_email(
            {"template": "nope.v1", "payload": {}},
            origin="o",
            unsubscribe_url="u",
            settings_url="s",
        )


# --- worker -----------------------------------------------------------------------------------


def _enqueue_recap(data_dir: Path, uid: str, ref: str) -> str:
    env = {
        "schema_version": "1",
        "id": f"drcp_20260912_{uid}",
        "user_id": uid,
        "type": "daily_recap",
        "channel": "email",
        "template": "daily-recap.v1",
        "recipient": {"email": "u@gmail.com", "email_verified": True},
        "consent_snapshot": {"digest_enabled": True, "cadence": "daily", "unsubscribe_ref": ref},
        "payload": {"day": "2026-09-12", "count": 1, "episodes": [_EPISODE]},
        "not_before": iso(_NOW),
        "expires_at": iso(_NOW + 86_400),
        "created_at": iso(_NOW),
    }
    app_outbox_store.enqueue(data_dir, env)
    return str(env["id"])


def _opted_in_user(data_dir: Path) -> tuple[str, str]:
    uid = get_or_create_user(
        data_dir, provider="google", subject="s", email="u@gmail.com", name="U"
    ).user_id
    comms = app_comms_store.set_comms(data_dir, uid, types={"daily_recap": {"email": True}})
    return uid, comms["unsubscribe_ref"]


def test_worker_sends_and_records_delivered(tmp_path: Path) -> None:
    uid, ref = _opted_in_user(tmp_path)
    _enqueue_recap(tmp_path, uid, ref)
    fake = _FakeTransport(ok=True)

    summary = app_delivery_worker.deliver_pending_emails(
        tmp_path, transport=fake, now=_NOW, origin="https://closelistening.app"
    )
    assert summary.delivered == 1 and summary.failed == 0
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["to"] == "u@gmail.com"
    # The one-click unsubscribe header is type-aware — it targets the recap, not the weekly digest.
    unsub = call["headers"]["List-Unsubscribe"]
    assert f"ref={ref}" in unsub and "type=daily_recap" in unsub
    assert call["headers"]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    # The envelope is now terminal (delivered) — a second drain does nothing.
    assert (
        app_delivery_worker.deliver_pending_emails(tmp_path, transport=fake, now=_NOW).delivered
        == 0
    )


def test_worker_dry_runs_without_a_key_and_leaves_pending(tmp_path: Path) -> None:
    uid, ref = _opted_in_user(tmp_path)
    _enqueue_recap(tmp_path, uid, ref)
    # No transport, no api_key → dry-run.
    summary = app_delivery_worker.deliver_pending_emails(
        tmp_path, transport=None, api_key=None, now=_NOW
    )
    assert summary.dry_run == 1 and summary.delivered == 0
    # Still pending → a later real drain can deliver it.
    assert len(app_outbox_store.list_pending(tmp_path, channel="email", now=_NOW)) == 1


def test_worker_failure_leaves_pending(tmp_path: Path) -> None:
    uid, ref = _opted_in_user(tmp_path)
    _enqueue_recap(tmp_path, uid, ref)
    summary = app_delivery_worker.deliver_pending_emails(
        tmp_path, transport=_FakeTransport(ok=False), now=_NOW
    )
    assert summary.failed == 1 and summary.delivered == 0
    assert len(app_outbox_store.list_pending(tmp_path, channel="email", now=_NOW)) == 1
