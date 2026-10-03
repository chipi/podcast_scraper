"""Integration tests for email magic-link sign-in (#2272).

The flow has two halves and they are tested as one: requesting a link enqueues a delivery envelope,
and verifying that link creates-or-signs-in the account. Everything here runs against a real app
instance over a temp data dir — the outbox is the actual store, not a stub, because the single most
important property (an auth envelope SURVIVES the consent gate with no user) is a property of that
store and a stub would assert it away.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_magic_link, app_outbox_store, app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy

pytestmark = [pytest.mark.integration]

_SECRET = "test-secret"
_ALLOWED = "tester@example.test"
_STRANGER = "stranger@example.test"


def _client(tmp_path: Path, *, mode: str = "allowlist") -> tuple[TestClient, Path]:
    app = create_app(tmp_path, static_dir=False)
    data_dir = tmp_path / "appdata"
    app.state.session_secret = _SECRET
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy(mode, frozenset({_ALLOWED}), frozenset())
    return TestClient(app, follow_redirects=False), data_dir


def _link_token(data_dir: Path, email: str) -> str:
    """The token from the envelope enqueued for ``email``."""
    pending = [
        e
        for e in app_outbox_store.list_pending(data_dir, channel="email")
        if e["recipient"]["email"] == email
    ]
    assert pending, f"no envelope was enqueued for {email}"
    link = pending[0]["payload"]["link"]
    return link.split("token=", 1)[1].split("&", 1)[0]


# --- requesting a link ---------------------------------------------------------------------


def test_request_enqueues_an_envelope_that_survives_the_consent_gate(tmp_path: Path) -> None:
    """The headline property. A magic-link recipient has NO account, so no consent record exists.

    `app_outbox_store.list_pending` filters every envelope through `_consent_allows`, which looks a
    consent record up by `user_id`. Before the transactional class existed this envelope was
    silently dropped and the link was never sent — a failure with no error anywhere.
    """
    client, data_dir = _client(tmp_path)
    resp = client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    assert resp.status_code == 202

    pending = app_outbox_store.list_pending(data_dir, channel="email")
    assert len(pending) == 1
    envelope = pending[0]
    assert envelope["type"] == "auth_link"
    assert envelope["template"] == "magic-link.v1"
    assert envelope["user_id"] == ""  # no account yet, and the schema requires exactly this
    assert envelope["recipient"]["email"] == _ALLOWED
    assert envelope["recipient"]["email_verified"] is False
    assert "consent_snapshot" not in envelope
    assert envelope["payload"]["link"].startswith("http")


def test_request_answers_identically_for_allowed_and_unknown_addresses(tmp_path: Path) -> None:
    """No oracle. Any difference would report whether an address is on the operator's allowlist.

    That list is a small set of real people, so "is X a tester?" must not be answerable by a
    stranger. The access check therefore happens at VERIFY, where refusing tells nothing to anyone
    who does not already control the mailbox.
    """
    client, _ = _client(tmp_path)
    allowed = client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    stranger = client.post("/api/app/auth/email/request", json={"email": _STRANGER})
    assert allowed.status_code == stranger.status_code == 202
    assert allowed.json() == stranger.json() == {"ok": True}


def test_request_is_throttled_per_address(tmp_path: Path) -> None:
    client, data_dir = _client(tmp_path)
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    enqueued = [
        e
        for e in app_outbox_store.list_pending(data_dir, channel="email")
        if e["recipient"]["email"] == _ALLOWED
    ]
    assert len(enqueued) == 1, "a second request inside the window must not send a second email"


def test_request_normalises_the_address(tmp_path: Path) -> None:
    """One person must not collect several accounts because they typed their address differently."""
    client, data_dir = _client(tmp_path)
    client.post("/api/app/auth/email/request", json={"email": f"  {_ALLOWED.upper()}  "})
    assert _link_token(data_dir, _ALLOWED)


# --- verifying a link ----------------------------------------------------------------------


def test_new_account_lands_on_the_profile(tmp_path: Path) -> None:
    """A brand-new email identity arrives with no name and no picture, so it goes to the profile.

    Home would leave a half-built account the person never sees.
    """
    client, data_dir = _client(tmp_path)
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    resp = client.get(f"/api/app/auth/email/verify?token={_link_token(data_dir, _ALLOWED)}")
    assert resp.status_code == 307
    assert resp.headers["location"] == "/profile?welcome=1"
    assert app_sessions.SESSION_COOKIE in resp.headers.get("set-cookie", "")


def test_returning_account_lands_on_home(tmp_path: Path) -> None:
    client, data_dir = _client(tmp_path)
    first, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    assert client.get(f"/api/app/auth/email/verify?token={first}").status_code == 307
    second, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    resp = client.get(f"/api/app/auth/email/verify?token={second}")
    assert resp.status_code == 307
    assert resp.headers["location"] == "/"


def test_the_same_link_cannot_be_used_twice(tmp_path: Path) -> None:
    """Single use. A signature proves authenticity, NOT freshness-of-use.

    Mail clients, security scanners and corporate link-rewriters all fetch links, so a replayable
    link is a live credential sitting in an inbox and in every system that touched it.
    """
    client, data_dir = _client(tmp_path)
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    assert client.get(f"/api/app/auth/email/verify?token={token}").status_code == 307
    again = client.get(f"/api/app/auth/email/verify?token={token}")
    assert again.status_code == 400
    assert "already been used" in again.json()["detail"]


def test_a_disallowed_address_is_refused_at_verify_without_burning_the_token(
    tmp_path: Path,
) -> None:
    """Refused, but the link survives — the person did nothing wrong.

    If the operator then adds them to the allowlist, the link they already have must work. Consuming
    it on a policy refusal would make the fix invisible to the person it was for.
    """
    client, data_dir = _client(tmp_path)
    token, token_id = app_magic_link.issue(_STRANGER, _SECRET)
    refused = client.get(f"/api/app/auth/email/verify?token={token}")
    assert refused.status_code == 403

    used_marker = data_dir / "magic_link_used" / f"{token_id}.json"
    assert not used_marker.exists(), "a policy refusal must not consume the token"


def test_an_expired_link_is_refused(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    stale = int(time.time()) - app_magic_link.TOKEN_TTL_SECONDS - 60
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET, now=stale)
    resp = client.get(f"/api/app/auth/email/verify?token={token}")
    assert resp.status_code == 400


def test_a_session_cookie_cannot_be_replayed_as_a_sign_in_link(tmp_path: Path) -> None:
    """Purpose-binding. Both tokens are signed with the SAME secret, so only `purpose` separates
    them — without it, any leaked session cookie would also be a sign-in link for that account."""
    client, _ = _client(tmp_path)
    session = app_sessions.sign({"user_id": "u_" + "0" * 24, "iat": int(time.time())}, _SECRET)
    resp = client.get(f"/api/app/auth/email/verify?token={session}")
    assert resp.status_code == 400


def test_a_tampered_token_is_refused(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    body, _, sig = token.partition(".")
    resp = client.get(f"/api/app/auth/email/verify?token={body}.{sig[:-2]}xx")
    assert resp.status_code == 400


def test_native_platform_gets_a_deep_link_carrying_the_new_flag(tmp_path: Path) -> None:
    """The shell cannot see a cookie set in an external browser, so it takes the token by deep link
    — the same contract the OAuth callback uses, plus `new` so it knows which screen to open."""
    client, _ = _client(tmp_path)
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    resp = client.get(f"/api/app/auth/email/verify?token={token}&platform=native")
    assert resp.status_code == 307
    location = resp.headers["location"]
    assert location.startswith("closelistening://auth#token=")
    assert "&new=1" in location, "a first sign-in must tell the shell this is a new account"


def test_the_account_is_created_with_the_email_provider_and_a_signup_event(
    tmp_path: Path,
) -> None:
    """`user_id = sha256(provider || subject)`, so an email identity cannot collide with the Google
    identity for the same address — they are deliberately different accounts."""
    client, data_dir = _client(tmp_path)
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    client.get(f"/api/app/auth/email/verify?token={token}")

    profiles = list((data_dir / "users").glob("*/profile.json"))
    assert len(profiles) == 1
    profile = json.loads(profiles[0].read_text())
    assert profile["provider"] == "email"
    assert profile["email"] == _ALLOWED
    assert profile["analytics_id"], "every account gets a pseudonymous analytics id (#2265)"

    events = profiles[0].parent / "account_events.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    assert len(rows) == 1, "server-side signup truth, exactly once"
    assert rows[0]["provider"] == "email"


def test_open_mode_admits_any_address(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, mode="open")
    token, _ = app_magic_link.issue(_STRANGER, _SECRET)
    assert client.get(f"/api/app/auth/email/verify?token={token}").status_code == 307


# --- observability (ADR-119 events) ------------------------------------------------------------

_EVENT_LOGGER = "podcast_scraper.server.routes.app_auth"


def _magic_events(caplog: pytest.LogCaptureFixture) -> list[tuple[dict, str]]:
    """Every magic-link event captured, as (parsed record, the raw line that would ship)."""
    out = []
    for rec in caplog.records:
        if rec.name != _EVENT_LOGGER:
            continue
        line = rec.getMessage()
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if str(parsed.get("event_type", "")).startswith("magic_link_"):
            out.append((parsed, line))
    return out


def test_every_outcome_emits_an_event_that_names_nobody(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The events exist so "the tester never got the email" has an answer — and they must give that
    answer without becoming a list of who tried to sign in.

    The request endpoint answers identically for every address precisely so nobody can learn who has
    an account. A log line carrying the address would hand that back in the one place that is
    retained and shipped off the box, so: fingerprints only, and never any part of a token.
    """
    caplog.set_level("INFO", logger=_EVENT_LOGGER)
    client, data_dir = _client(tmp_path)

    client.post("/api/app/auth/email/request", json={"email": "no-at-sign"})
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})  # throttled
    link_token = _link_token(data_dir, _ALLOWED)
    client.get(f"/api/app/auth/email/verify?token={link_token}")  # created
    client.get(f"/api/app/auth/email/verify?token={link_token}")  # replayed
    again, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    client.get(f"/api/app/auth/email/verify?token={again}")  # returning
    stranger, _ = app_magic_link.issue(_STRANGER, _SECRET)
    client.get(f"/api/app/auth/email/verify?token={stranger}")  # refused_policy
    client.get("/api/app/auth/email/verify?token=garbage.token")  # invalid_or_expired

    events = _magic_events(caplog)
    seen = {(e["event_type"], e["outcome"]) for e, _ in events}
    assert seen >= {
        ("magic_link_requested", "rejected_shape"),
        ("magic_link_requested", "enqueued"),
        ("magic_link_requested", "throttled"),
        ("magic_link_verified", "created"),
        ("magic_link_verified", "replayed"),
        ("magic_link_verified", "returning"),
        ("magic_link_verified", "refused_policy"),
        ("magic_link_verified", "invalid_or_expired"),
    }, seen

    token_parts = {p for t in (link_token, again, stranger) for p in t.split(".") if p}
    for _, line in events:
        for address in (_ALLOWED, _STRANGER, "no-at-sign"):
            assert address not in line.lower(), f"plaintext address leaked: {line}"
            assert address.split("@")[0] not in line.lower(), f"local part leaked: {line}"
        for part in token_parts:
            assert part not in line, f"token material leaked: {line}"


def test_the_fingerprint_correlates_one_address_across_request_and_verify(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The operational question is "was the link I sent the one that got used?" — answerable only if
    the same address yields the same fingerprint on both halves, and a different one elsewhere."""
    caplog.set_level("INFO", logger=_EVENT_LOGGER)
    client, data_dir = _client(tmp_path, mode="open")
    client.post("/api/app/auth/email/request", json={"email": _ALLOWED})
    client.post("/api/app/auth/email/request", json={"email": _STRANGER})
    client.get(f"/api/app/auth/email/verify?token={_link_token(data_dir, _ALLOWED)}")

    by_outcome = {}
    for e, _ in _magic_events(caplog):
        by_outcome.setdefault((e["event_type"], e["outcome"]), []).append(e["email_fp"])
    requested = by_outcome[("magic_link_requested", "enqueued")]
    verified = by_outcome[("magic_link_verified", "created")]
    assert len(requested) == 2 and len(set(requested)) == 2, "different addresses, different fps"
    assert verified[0] in requested
    assert all(len(fp) == 16 for fp in requested + verified)


def test_a_refused_enqueue_is_reported_as_duplicate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`enqueue` answers False when the outbox already holds that envelope id. Ids are unique per
    token, so this should never happen — which is exactly why it must be visible if it does: the
    person was told 202 and no new email is coming."""
    caplog.set_level("INFO", logger=_EVENT_LOGGER)
    monkeypatch.setattr(app_outbox_store, "enqueue", lambda *_a, **_k: False)
    client, _ = _client(tmp_path)
    assert client.post("/api/app/auth/email/request", json={"email": _ALLOWED}).status_code == 202
    outcomes = [e["outcome"] for e, _ in _magic_events(caplog)]
    assert outcomes == ["duplicate"]


def test_verify_without_a_data_dir_is_reported_as_unavailable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=_EVENT_LOGGER)
    client, _ = _client(tmp_path)
    client.app.state.app_data_dir = None
    token, _ = app_magic_link.issue(_ALLOWED, _SECRET)
    assert client.get(f"/api/app/auth/email/verify?token={token}").status_code == 503
    events = [e for e, _ in _magic_events(caplog)]
    assert [e["outcome"] for e in events] == ["unavailable"]
    assert len(events[0]["email_fp"]) == 16
