"""OAuth identity providers for the consumer platform (#1063, RFC-098 §2).

A small protocol so the auth routes don't hard-code a vendor and tests can inject a stub
(no real OAuth call in CI). ``GoogleProvider`` implements the OAuth2 authorization-code
flow with ``httpx``; credentials come from env (``APP_OAUTH_GOOGLE_CLIENT_ID`` /
``APP_OAUTH_GOOGLE_CLIENT_SECRET``).

``MockOAuthProvider`` (#1079, RFC-099 §1) is a local, network-free provider for dev and
e2e: it self-completes the code flow with a fixed dev identity. It is selected **only**
when ``APP_OAUTH_PROVIDER=mock`` is set explicitly (never the default), so it can never
ship to production by accident — Google stays the production provider.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


@dataclass(frozen=True)
class OAuthIdentity:
    """Resolved identity from a provider's userinfo."""

    provider: str
    subject: str
    email: str
    name: str


class OAuthError(Exception):
    """Raised when an OAuth exchange fails (network, bad response, missing fields)."""


class OAuthProvider(Protocol):
    """Minimal provider contract the auth routes depend on."""

    name: str

    def authorization_url(self, *, state: str, redirect_uri: str) -> str:
        """Return the provider authorize URL for this ``state`` + ``redirect_uri``."""
        ...

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        """Exchange an authorization ``code`` for the resolved identity."""
        ...


class GoogleProvider:
    """Google OAuth2 authorization-code flow (openid email profile)."""

    name = "google"

    def __init__(self, client_id: str, client_secret: str, *, timeout: float = 10.0) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout

    def authorization_url(self, *, state: str, redirect_uri: str) -> str:
        """Build Google's OAuth2 consent URL (openid email profile) with CSRF ``state``."""
        query = urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
                "access_type": "online",
                "prompt": "select_account",
            }
        )
        return f"{GOOGLE_AUTH_URL}?{query}"

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        """Exchange the code for a token, fetch userinfo, return the identity (or OAuthError)."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                token_resp = client.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "code": code,
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "redirect_uri": redirect_uri,
                        "grant_type": "authorization_code",
                    },
                )
                token_resp.raise_for_status()
                access_token = token_resp.json().get("access_token")
                if not access_token:
                    raise OAuthError("token response missing access_token")
                info_resp = client.get(
                    GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                info_resp.raise_for_status()
                info = info_resp.json()
        except httpx.HTTPError as exc:
            raise OAuthError(f"OAuth exchange failed: {exc}") from exc

        subject = info.get("sub")
        email = info.get("email")
        if not subject or not email:
            raise OAuthError("userinfo missing sub/email")
        return OAuthIdentity(
            provider=self.name,
            subject=str(subject),
            email=str(email),
            name=str(info.get("name") or email),
        )


class MockOAuthProvider:
    """Local, network-free OAuth provider for dev + e2e — **never** production.

    ``authorization_url`` redirects the browser straight back to the callback with a
    fixed code, so the authorization-code flow self-completes offline (no external IdP).
    ``exchange_code`` returns a fixed dev identity regardless of the code. Selected only
    via ``APP_OAUTH_PROVIDER=mock``; the dev identity is overridable with
    ``APP_OAUTH_MOCK_EMAIL`` / ``APP_OAUTH_MOCK_SUBJECT`` / ``APP_OAUTH_MOCK_NAME``.
    """

    name = "mock"
    MOCK_CODE = "mock-auth-code"

    def __init__(
        self,
        *,
        email: str = "dev@localhost",
        subject: str = "dev-local",
        display_name: str = "Dev User",
    ) -> None:
        self._email = email
        self._subject = subject
        self._name = display_name

    def authorization_url(self, *, state: str, redirect_uri: str) -> str:
        """Redirect straight back to the callback with a fixed code (offline flow)."""
        query = urlencode({"code": self.MOCK_CODE, "state": state})
        sep = "&" if "?" in redirect_uri else "?"
        return f"{redirect_uri}{sep}{query}"

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        """Return the fixed dev identity (no network); ``code`` is ignored."""
        return OAuthIdentity(
            provider=self.name, subject=self._subject, email=self._email, name=self._name
        )

    @classmethod
    def from_env(cls) -> "MockOAuthProvider":
        """Build from optional ``APP_OAUTH_MOCK_*`` env overrides."""
        return cls(
            email=(os.environ.get("APP_OAUTH_MOCK_EMAIL", "").strip() or "dev@localhost"),
            subject=(os.environ.get("APP_OAUTH_MOCK_SUBJECT", "").strip() or "dev-local"),
            display_name=(os.environ.get("APP_OAUTH_MOCK_NAME", "").strip() or "Dev User"),
        )


def provider_from_env() -> OAuthProvider | None:
    """Build the configured provider from env, or ``None`` when unconfigured.

    Selection: ``APP_OAUTH_PROVIDER=mock`` → :class:`MockOAuthProvider` (dev/e2e only,
    logged loudly). Otherwise the Google provider when its creds are present.
    """
    selected = os.environ.get("APP_OAUTH_PROVIDER", "").strip().lower()
    if selected == "mock":
        logger.warning(
            "APP_OAUTH_PROVIDER=mock — using MockOAuthProvider (dev/e2e only). "
            "This MUST NOT be set in production."
        )
        return MockOAuthProvider.from_env()

    client_id = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return GoogleProvider(client_id, client_secret)
    return None
