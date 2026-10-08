"""OAuth identity providers for the consumer platform (#1063, RFC-098 §2).

A small protocol so the auth routes don't hard-code a vendor and tests can inject a stub
(no real OAuth call in CI). The real providers (Google, Apple) live in Common ``identity`` and
reach the registry below through installed extensions (ADR-158).

``MockOAuthProvider`` (#1079, RFC-099 §1) is a local, network-free provider for dev and
e2e: it self-completes the code flow with a fixed dev identity. It is selected **only**
when ``APP_OAUTH_PROVIDER=mock`` is set explicitly (never the default), so it can never
ship to production by accident.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

#: Names ``APP_OAUTH_PROVIDER`` may select as the primary provider (besides ``mock``).
PRIMARY_PROVIDERS = frozenset({"google"})
#: Names ``APP_OAUTH_PROVIDERS`` may add beside a real primary. Apple sits beside Google, which App
#: Store guideline 4.8 requires; it is never the only way in.
BESIDE_PROVIDERS = frozenset({"apple"})


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


def providers_from_env() -> dict[str, OAuthProvider]:
    """Every configured provider, by name; the PRIMARY (``provider_from_env``) first.

    ``APP_OAUTH_PROVIDERS`` (e.g. ``google,apple``) adds providers beside the primary — still
    explicit, never inferred from credentials being present. Only a :data:`BESIDE_PROVIDERS` name
    (``apple``) can be added this way: the primary already covers google/mock, and the mock must
    never ride along with a real one.
    """
    out: dict[str, OAuthProvider] = {}
    primary = provider_from_env()
    if primary is not None:
        out[primary.name] = primary
    extra = {
        p.strip().lower() for p in os.environ.get("APP_OAUTH_PROVIDERS", "").split(",") if p.strip()
    }
    if primary is None or primary.name == "mock":
        return out
    builders = installed_oauth_providers()
    for name in sorted(extra & BESIDE_PROVIDERS):
        build = builders.get(name)
        provider = build() if build is not None else None
        if provider is not None:
            out[provider.name] = provider
    return out


def provider_from_env() -> OAuthProvider | None:
    """Build the configured provider from env, or ``None`` when unconfigured.

    Selection is one explicit switch — ``APP_OAUTH_PROVIDER`` — set by the deployment
    (its compose/profile), never inferred:

    * ``mock``   → :class:`MockOAuthProvider` (dev/e2e only, logged loudly; never prod).
    * ``google`` → the provider an installed extension registers under that name (Common
      identity, ADR-158), built from its own env; ``None`` + a warning when no extension provides
      it or its credentials are absent.
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
    if selected in PRIMARY_PROVIDERS:
        build = installed_oauth_providers().get(selected)
        if build is None:
            logger.warning(
                "APP_OAUTH_PROVIDER=%s but no installed extension provides it "
                "— consumer auth disabled.",
                selected,
            )
            return None
        return build()
    return None


def installed_oauth_providers() -> dict[str, Callable[[], "OAuthProvider | None"]]:
    """Provider builders by name, from installed extensions (the first to register a name wins)."""
    from podcast_scraper.extensions import load_extensions

    out: dict[str, Callable[[], OAuthProvider | None]] = {}
    for ext in load_extensions():
        for name, build in ext.oauth_providers.items():
            out.setdefault(name, build)
    return out
