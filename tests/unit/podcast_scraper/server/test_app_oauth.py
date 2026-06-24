"""Unit tests for OAuth provider selection + the local mock provider (#1079).

No network: the mock provider is offline by construction, and provider_from_env is
exercised purely through env vars. Google selection is asserted by type only.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest

from podcast_scraper.server.app_oauth import (
    GoogleProvider,
    MockOAuthProvider,
    OAuthIdentity,
    provider_from_env,
)


def test_mock_authorization_url_round_trips_code_and_state() -> None:
    p = MockOAuthProvider()
    url = p.authorization_url(state="xyz", redirect_uri="http://t/api/app/auth/callback")
    q = parse_qs(urlparse(url).query)
    assert q["code"] == [MockOAuthProvider.MOCK_CODE]
    assert q["state"] == ["xyz"]


def test_mock_authorization_url_preserves_existing_query() -> None:
    p = MockOAuthProvider()
    url = p.authorization_url(state="s", redirect_uri="http://t/cb?next=/home")
    q = parse_qs(urlparse(url).query)
    assert q["next"] == ["/home"]
    assert q["state"] == ["s"]
    assert q["code"] == [MockOAuthProvider.MOCK_CODE]


def test_mock_exchange_code_returns_fixed_identity_regardless_of_code() -> None:
    p = MockOAuthProvider()
    ident = p.exchange_code(code="anything", redirect_uri="http://t/cb")
    assert isinstance(ident, OAuthIdentity)
    assert ident.provider == "mock"
    assert ident.email == "dev@localhost"
    assert ident.subject == "dev-local"
    assert ident.name == "Dev User"


def test_mock_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_OAUTH_MOCK_EMAIL", "alice@dev.test")
    monkeypatch.setenv("APP_OAUTH_MOCK_SUBJECT", "alice-1")
    monkeypatch.setenv("APP_OAUTH_MOCK_NAME", "Alice Dev")
    ident = MockOAuthProvider.from_env().exchange_code(code="c", redirect_uri="http://t/cb")
    assert (ident.email, ident.subject, ident.name) == ("alice@dev.test", "alice-1", "Alice Dev")


def test_provider_from_env_selects_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_OAUTH_PROVIDER", "mock")
    assert isinstance(provider_from_env(), MockOAuthProvider)


def test_provider_from_env_mock_wins_over_google_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit mock selection takes precedence even when Google creds are present.
    monkeypatch.setenv("APP_OAUTH_PROVIDER", "mock")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", "csecret")
    assert isinstance(provider_from_env(), MockOAuthProvider)


def test_provider_from_env_selects_google_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_OAUTH_PROVIDER", raising=False)
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", "csecret")
    assert isinstance(provider_from_env(), GoogleProvider)


def test_provider_from_env_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_OAUTH_PROVIDER", raising=False)
    monkeypatch.delenv("APP_OAUTH_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", raising=False)
    assert provider_from_env() is None


def test_provider_from_env_ignores_unknown_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    # An unrecognised selector falls through to Google creds (here: absent -> None).
    monkeypatch.setenv("APP_OAUTH_PROVIDER", "bogus")
    monkeypatch.delenv("APP_OAUTH_GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("APP_OAUTH_GOOGLE_CLIENT_SECRET", raising=False)
    assert provider_from_env() is None
