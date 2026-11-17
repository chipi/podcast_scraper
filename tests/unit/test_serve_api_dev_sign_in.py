"""``make serve-api`` defaults to the mock OAuth provider; its users must be able to sign in.

The access policy defaults to an allowlist with no addresses, and the recipe set none, so every
mock user (``<hint>@e2e.local``, or ``dev@localhost`` without a hint) got 403 "This account is not
allowed to sign in." until ``APP_SIGNUP_MODE=open`` was set by hand (public-only review,
2026-11-17).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from podcast_scraper.server.app_access import policy_from_env

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[2]


def _serve_api_default(var: str) -> str | None:
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\nserve-api:", 1)[1].split("\n\n", 1)[0]
    match = re.search(rf"export {var}=\$\${{{var}:-([^}}]*)}}", recipe)
    return match.group(1) if match else None


def test_the_mock_users_can_sign_in_with_the_serve_api_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _serve_api_default("APP_OAUTH_PROVIDER") == "mock"
    domains = _serve_api_default("APP_ALLOWED_DOMAINS")
    assert domains, "serve-api sets no APP_ALLOWED_DOMAINS default"
    monkeypatch.delenv("APP_SIGNUP_MODE", raising=False)
    monkeypatch.delenv("APP_ALLOWED_EMAILS", raising=False)
    monkeypatch.setenv("APP_ALLOWED_DOMAINS", domains)
    policy = policy_from_env()
    assert policy.mode == "allowlist"
    for email in ("ada-admin@e2e.local", "dev@localhost"):
        assert policy.is_allowed(email), email
    assert not policy.is_allowed("someone@gmail.com")
