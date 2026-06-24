"""OAuth identity providers for the consumer platform (#1063, RFC-098 §2).

A small protocol so the auth routes don't hard-code a vendor and tests can inject a stub
(no real OAuth call in CI). ``GoogleProvider`` implements the OAuth2 authorization-code
flow with ``httpx``; credentials come from env (``APP_OAUTH_GOOGLE_CLIENT_ID`` /
``APP_OAUTH_GOOGLE_CLIENT_SECRET``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import httpx

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


def provider_from_env() -> OAuthProvider | None:
    """Build the configured provider from env, or ``None`` when creds are absent."""
    client_id = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        return GoogleProvider(client_id, client_secret)
    return None
