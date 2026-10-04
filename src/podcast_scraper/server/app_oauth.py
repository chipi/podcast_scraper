"""OAuth identity providers for the consumer platform (#1063, RFC-098 §2).

A small protocol so the auth routes don't hard-code a vendor and tests can inject a stub
(no real OAuth call in CI). ``GoogleProvider`` implements the OAuth2 authorization-code
flow with ``httpx``; credentials come from env (``APP_OAUTH_GOOGLE_CLIENT_ID`` /
``APP_OAUTH_GOOGLE_CLIENT_SECRET``).

``MockOAuthProvider`` (#1079, RFC-099 §1) is a local, network-free provider for dev and
e2e: it self-completes the code flow with a fixed dev identity. It is selected **only**
when ``APP_OAUTH_PROVIDER=mock`` is set explicitly (never the default), so it can never
ship to production by accident — Google stays the production provider.

``AppleProvider`` (#2275) — Sign in with Apple, which App Store guideline 4.8 requires next to
Google Sign-In. It sits BESIDE the primary provider (``APP_OAUTH_PROVIDERS=google,apple``); see
:func:`providers_from_env`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx
import jwt

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
    image: str | None = None  # provider avatar URL (Google `picture`); None when absent
    #: Apple's refresh token (#2273). Kept ONLY so account deletion can revoke it, which Apple
    #: requires of every app that offers Sign in with Apple. Never logged or returned.
    refresh_token: str | None = field(default=None, repr=False)


class OAuthError(Exception):
    """Raised when an OAuth exchange fails (network, bad response, missing fields)."""


class OAuthProvider(Protocol):
    """Minimal provider contract the auth routes depend on."""

    name: str

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        """Return the provider authorize URL for this ``state`` + ``redirect_uri``.

        ``login_hint`` is an optional identity hint; real providers may ignore it. The mock
        provider uses it to self-complete as a distinct identity (dev/e2e isolation).
        """
        ...

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        """Exchange an authorization ``code`` for the resolved identity.

        A provider that receives extra callback fields (Apple's ``user``) takes them as optional
        keyword arguments; the route passes them only when present, so the other providers — and
        test stubs — keep this two-argument signature.
        """
        ...


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


def _safe_hint(raw: str | None) -> str:
    """Sanitise a mock identity hint to a short ``[a-z0-9-]`` token (empty when unusable)."""
    if not raw:
        return ""
    cleaned = re.sub(r"[^a-z0-9-]", "", raw.strip().lower())
    return cleaned[:32]


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

    def authorization_url(
        self, *, state: str, redirect_uri: str, login_hint: str | None = None
    ) -> str:
        """Redirect straight back to the callback with a fixed code (offline flow).

        When ``login_hint`` is given (dev/e2e), it is baked into the code (``mock-auth-code:<h>``)
        so ``exchange_code`` self-completes as a **distinct** identity — letting parallel e2e specs
        run as isolated users instead of one shared mock user.
        """
        hint = _safe_hint(login_hint)
        code = f"{self.MOCK_CODE}:{hint}" if hint else self.MOCK_CODE
        query = urlencode({"code": code, "state": state})
        sep = "&" if "?" in redirect_uri else "?"
        return f"{redirect_uri}{sep}{query}"

    def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthIdentity:
        """Return the dev identity (no network). A ``mock-auth-code:<hint>`` code yields a distinct
        per-hint identity (``<hint>`` subject); a bare code yields the configured fixed identity."""
        prefix = f"{self.MOCK_CODE}:"
        if code.startswith(prefix):
            hint = _safe_hint(code[len(prefix) :])
            if hint:
                return OAuthIdentity(
                    provider=self.name,
                    subject=f"e2e-{hint}",
                    email=f"{hint}@e2e.local",
                    name=hint,
                )
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


class SigningKeySource(Protocol):
    """What :class:`AppleProvider` needs from a JWKS client: the key that signed a token."""

    def get_signing_key_from_jwt(self, token: str) -> Any: ...


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


def providers_from_env() -> dict[str, OAuthProvider]:
    """Every configured provider, by name; the PRIMARY (``provider_from_env``) first.

    ``APP_OAUTH_PROVIDERS`` (e.g. ``google,apple``) adds providers beside the primary — still
    explicit, never inferred from credentials being present. Only ``apple`` can be added this way:
    the primary already covers google/mock, and the mock must never ride along with a real one.
    """
    out: dict[str, OAuthProvider] = {}
    primary = provider_from_env()
    if primary is not None:
        out[primary.name] = primary
    extra = {
        p.strip().lower() for p in os.environ.get("APP_OAUTH_PROVIDERS", "").split(",") if p.strip()
    }
    if "apple" in extra and primary is not None and primary.name != "mock":
        apple = AppleProvider.from_env()
        if apple is not None:
            out[apple.name] = apple
    return out


def provider_from_env() -> OAuthProvider | None:
    """Build the configured provider from env, or ``None`` when unconfigured.

    Selection is one explicit switch — ``APP_OAUTH_PROVIDER`` — set by the deployment
    (its compose/profile), never inferred:

    * ``mock``   → :class:`MockOAuthProvider` (dev/e2e only, logged loudly; never prod).
    * ``google`` → :class:`GoogleProvider`; needs ``APP_OAUTH_GOOGLE_CLIENT_ID`` +
      ``APP_OAUTH_GOOGLE_CLIENT_SECRET`` (``None`` + a warning when they are absent).
    * anything else / unset → ``None`` (consumer auth disabled).

    Explicit by design: Google creds alone never silently enable a real provider — the
    deployment names its provider in one place, so switching mock↔google is a single
    config change and a half-set env can't accidentally go live.
    """
    selected = os.environ.get("APP_OAUTH_PROVIDER", "").strip().lower()
    if selected == "mock":
        logger.warning(
            "APP_OAUTH_PROVIDER=mock — using MockOAuthProvider (dev/e2e only). "
            "This MUST NOT be set in production."
        )
        return MockOAuthProvider.from_env()
    if selected == "google":
        client_id = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_ID", "").strip()
        client_secret = os.environ.get("APP_OAUTH_GOOGLE_CLIENT_SECRET", "").strip()
        if client_id and client_secret:
            return GoogleProvider(client_id, client_secret)
        logger.warning(
            "APP_OAUTH_PROVIDER=google but APP_OAUTH_GOOGLE_CLIENT_ID/SECRET are missing "
            "— consumer auth disabled."
        )
        return None
    return None
