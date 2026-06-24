"""Unit tests for the HMAC-signed session cookie (#1063).

Pure stdlib crypto — no HTTP, no disk.
"""

from __future__ import annotations

from podcast_scraper.server import app_sessions


def test_sign_verify_roundtrip() -> None:
    token = app_sessions.sign({"user_id": "u_1", "iat": 1000}, "secret")
    assert app_sessions.verify(token, "secret", max_age=0) == {"user_id": "u_1", "iat": 1000}


def test_wrong_secret_rejected() -> None:
    token = app_sessions.sign({"user_id": "u_1"}, "secret")
    assert app_sessions.verify(token, "other-secret") is None


def test_tampered_body_rejected() -> None:
    token = app_sessions.sign({"user_id": "u_1"}, "secret")
    _, _, sig = token.partition(".")
    forged = app_sessions._b64e(b'{"user_id":"hacker"}') + "." + sig
    assert app_sessions.verify(forged, "secret") is None


def test_expired_rejected() -> None:
    token = app_sessions.sign({"user_id": "u_1", "iat": 0}, "secret")
    assert app_sessions.verify(token, "secret", max_age=10) is None


def test_missing_iat_rejected_when_expiry_enforced() -> None:
    token = app_sessions.sign({"user_id": "u_1"}, "secret")  # no iat
    assert app_sessions.verify(token, "secret") is None  # default max_age > 0 → not immortal
    assert app_sessions.verify(token, "secret", max_age=0) == {"user_id": "u_1"}  # expiry off → ok


def test_garbage_and_empty_inputs() -> None:
    assert app_sessions.verify(None, "secret") is None
    assert app_sessions.verify("", "secret") is None
    assert app_sessions.verify("no-dot", "secret") is None
    assert app_sessions.verify("a.b", "secret") is None  # bad signature
    assert app_sessions.verify(app_sessions.sign({}, "secret"), "") is None  # no secret
