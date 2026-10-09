"""The real sign-in providers: Google and Sign in with Apple (#1063, #2275).

Common ``identity`` (ADR-162): the platform keeps the :class:`OAuthProvider` protocol, the provider
registry and the mock; installed extensions supply these by name. ``GoogleProvider`` implements the
OAuth2 authorization-code flow with ``httpx``. ``AppleProvider`` is Sign in with Apple, which App
Store guideline 4.8 requires next to Google Sign-In; it sits BESIDE the primary provider
(``APP_OAUTH_PROVIDERS=google,apple``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
import jwt

from podcast_scraper.server.app_oauth import OAuthError, OAuthIdentity

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


class GoogleProvider:
    """Google OAuth2 authorization-code flow (openid email profile)."""

    name = "google"

    def __init__(self, client_id: str, client_secret: str, *, timeout: float = 10.0) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._timeout = timeout

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        """Build Google's OAuth2 consent URL (openid email profile) with CSRF ``state``.

        ``login_hint`` is ignored — real identity comes from Google's userinfo, never a caller hint.
        """
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
        picture = info.get("picture")
        return OAuthIdentity(
            provider=self.name,
            subject=str(subject),
            email=str(email),
            name=str(info.get("name") or email),
            image=str(picture) if picture else None,
        )

    @classmethod
    def from_env(cls) -> "GoogleProvider | None":
        """Build from ``APP_OAUTH_GOOGLE_CLIENT_ID`` / ``_SECRET``, or ``None`` (with a warning)."""
        client_id = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_SECRET", "").strip()
        if client_id and client_secret:
            return cls(client_id, client_secret)
        logger.warning(
            "APP_OAUTH_PROVIDER=google but APP_OAUTH_GOOGLE_CLIENT_ID/SECRET are missing "
            "— consumer auth disabled."
        )
        return None


class SigningKeySource(Protocol):
    """What :class:`AppleProvider` needs from a JWKS client: the key that signed a token."""

    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the JWKS key that signed ``token`` (matched on its ``kid`` header)."""
        ...


APPLE_ISSUER = "https://appleid.apple.com"
APPLE_AUTH_URL = f"{APPLE_ISSUER}/auth/authorize"
APPLE_TOKEN_URL = f"{APPLE_ISSUER}/auth/token"
APPLE_KEYS_URL = f"{APPLE_ISSUER}/auth/keys"
APPLE_REVOKE_URL = f"{APPLE_ISSUER}/auth/revoke"


class AppleProvider:
    """Sign in with Apple, web flow (#2275).

    Three things differ from Google, and each is handled here rather than in the route:

    * **The client secret is a JWT**, signed ES256 with the team's ``.p8`` key (iss = team id,
      sub = Services ID, aud = Apple). Apple accepts one up to six months old; this mints a short
      one and reuses it until it is close to expiry.
    * **Identity comes from the ``id_token``**, verified against Apple's published keys (RS256,
      audience = the Services ID, issuer = Apple). There is no userinfo endpoint.
    * **The name is sent ONCE**, in the ``user`` form field of the very first authorization, never
      again. ``get_or_create_user`` only uses the name when it creates the account, which is that
      same first call — so it lands; later sign-ins fall back to the email, as Google's do.

    ``response_mode=form_post`` is required whenever ``name``/``email`` are requested, so Apple's
    callback is a cross-site POST that carries no SameSite=lax state cookie; the route's signed-
    state fallback (#1977) is what accepts it.
    """

    name = "apple"
    _SECRET_TTL_S = 24 * 3600

    def __init__(
        self,
        *,
        team_id: str,
        key_id: str,
        services_id: str,
        private_key_pem: str,
        timeout: float = 10.0,
        jwks_client: SigningKeySource | None = None,
    ) -> None:
        self._team_id = team_id
        self._key_id = key_id
        self._services_id = services_id
        self._private_key = private_key_pem
        self._timeout = timeout
        self._jwks = jwks_client or jwt.PyJWKClient(APPLE_KEYS_URL, timeout=int(timeout))
        self._secret: tuple[str, float] | None = None

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        """Apple's consent URL, asking for name + email (hence ``form_post``)."""
        query = urlencode(
            {
                "client_id": self._services_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "response_mode": "form_post",
                "scope": "name email",
                "state": state,
            }
        )
        return f"{APPLE_AUTH_URL}?{query}"

    def client_secret(self, now: float | None = None) -> str:
        """The ES256 client-secret JWT, reused until an hour before it expires."""
        now = time.time() if now is None else now
        if self._secret is not None and self._secret[1] - now > 3600:
            return self._secret[0]
        exp = now + self._SECRET_TTL_S
        token = jwt.encode(
            {
                "iss": self._team_id,
                "iat": int(now),
                "exp": int(exp),
                "aud": APPLE_ISSUER,
                "sub": self._services_id,
            },
            self._private_key,
            algorithm="ES256",
            headers={"kid": self._key_id},
        )
        self._secret = (token, exp)
        return token

    def verify_id_token(self, id_token: str) -> dict:
        """Verify Apple's ``id_token`` (signature, issuer, audience, expiry); return the claims."""
        try:
            key = self._jwks.get_signing_key_from_jwt(id_token)
            return jwt.decode(
                id_token,
                key.key,
                algorithms=["RS256"],
                audience=self._services_id,
                issuer=APPLE_ISSUER,
            )
        except jwt.PyJWTError as exc:
            raise OAuthError(f"Apple id_token rejected: {exc}") from exc

    def exchange_code(
        self, *, code: str, redirect_uri: str, user_json: str | None = None
    ) -> OAuthIdentity:
        """Exchange the code, verify the ``id_token``, and resolve the identity."""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    APPLE_TOKEN_URL,
                    data={
                        "client_id": self._services_id,
                        "client_secret": self.client_secret(),
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": redirect_uri,
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                id_token = body.get("id_token")
                refresh_token = body.get("refresh_token")
        except httpx.HTTPError as exc:
            raise OAuthError(f"Apple token exchange failed: {exc}") from exc
        if not id_token:
            raise OAuthError("Apple token response missing id_token")
        claims = self.verify_id_token(id_token)
        subject = claims.get("sub")
        email = claims.get("email")
        if not subject or not email:
            raise OAuthError("Apple id_token missing sub/email")
        return OAuthIdentity(
            provider=self.name,
            subject=str(subject),
            email=str(email),
            name=_apple_name(user_json) or str(email),
            refresh_token=str(refresh_token) if refresh_token else None,
        )

    def revoke(self, refresh_token: str) -> None:
        """Revoke the user's Apple tokens — required when they delete their account (#2273).

        Apple answers 200 for a token it has already invalidated, so this is safe to repeat. Raises
        :class:`OAuthError` on anything else; the caller decides whether that blocks deletion.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    APPLE_REVOKE_URL,
                    data={
                        "client_id": self._services_id,
                        "client_secret": self.client_secret(),
                        "token": refresh_token,
                        "token_type_hint": "refresh_token",
                    },
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OAuthError(f"Apple token revocation failed: {exc}") from exc

    @classmethod
    def from_env(cls) -> "AppleProvider | None":
        """Build from ``APP_OAUTH_APPLE_*``, or ``None`` (with a warning) when any part is missing.

        The key is the ``.p8`` file's PEM text in ``APP_OAUTH_APPLE_PRIVATE_KEY`` (``\\n`` escapes
        accepted, since env files are one line per value) or a path in
        ``APP_OAUTH_APPLE_PRIVATE_KEY_FILE``.
        """
        team_id = os.environ.get("APP_OAUTH_APPLE_TEAM_ID", "").strip()
        key_id = os.environ.get("APP_OAUTH_APPLE_KEY_ID", "").strip()
        services_id = os.environ.get("APP_OAUTH_APPLE_SERVICES_ID", "").strip()
        pem = os.environ.get("APP_OAUTH_APPLE_PRIVATE_KEY", "").strip().replace("\\n", "\n")
        key_file = os.environ.get("APP_OAUTH_APPLE_PRIVATE_KEY_FILE", "").strip()
        if not pem and key_file:
            try:
                with open(key_file, encoding="utf-8") as fh:
                    pem = fh.read().strip()
            except OSError:
                pem = ""
        if team_id and key_id and services_id and pem:
            return cls(team_id=team_id, key_id=key_id, services_id=services_id, private_key_pem=pem)
        logger.warning(
            "Apple sign-in requested but APP_OAUTH_APPLE_TEAM_ID/KEY_ID/SERVICES_ID/PRIVATE_KEY "
            "are not all set — the Apple button stays hidden."
        )
        return None


def _apple_name(user_json: str | None) -> str:
    """The display name from Apple's first-authorization ``user`` field (empty when absent)."""
    if not user_json:
        return ""
    try:
        name = (json.loads(user_json) or {}).get("name") or {}
    except (ValueError, AttributeError):
        return ""
    parts = [str(name.get(k) or "").strip() for k in ("firstName", "lastName")]
    return " ".join(p for p in parts if p)[:60]
