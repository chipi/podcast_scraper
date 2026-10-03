"""Consumer platform auth routes + ``get_current_user`` (#1063, RFC-098 §2).

``/api/app/auth/{login,callback,logout}`` runs a single-provider OAuth code flow and sets
a stdlib HMAC-signed session cookie; ``get_current_user`` is the dependency that gates the
per-user routes. Provider, session secret, and per-user data dir come from ``app.state``
(set in ``create_app`` from env) so tests can inject a stub provider + temp data dir.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from podcast_scraper.server import (
    app_access_store,
    app_magic_link,
    app_outbox_store,
    app_roles,
    app_sessions,
    app_user_state,
)
from podcast_scraper.server.app_oauth import OAuthError, OAuthProvider
from podcast_scraper.server.app_user_store import get_or_create_user, get_user, set_role, User

logger = logging.getLogger(__name__)

#: Upper bound on ``return_to`` — it travels inside the signed OAuth ``state`` (#1977).
_MAX_RETURN_TO_CHARS = 512

router = APIRouter(tags=["app"])

# Custom URL scheme the native shell registers for the OAuth deep-link callback (#1310). The app
# opens login in an external browser; on success the callback redirects here with the signed token.
NATIVE_AUTH_SCHEME = "closelistening"


def _native_scheme(request: Request) -> str:
    return getattr(request.app.state, "native_auth_scheme", None) or NATIVE_AUTH_SCHEME


def _secret(request: Request) -> str:
    return getattr(request.app.state, "session_secret", "") or ""


def _data_dir(request: Request) -> Path | None:
    raw = getattr(request.app.state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


def _provider(request: Request) -> OAuthProvider | None:
    return getattr(request.app.state, "oauth_provider", None)


def _secure(request: Request) -> bool:
    return bool(getattr(request.app.state, "session_cookie_secure", False))


def _callback_uri(request: Request) -> str:
    return str(request.url_for("app_auth_callback"))


def _bearer_token(request: Request) -> str | None:
    """The token from an ``Authorization: Bearer <token>`` header, or ``None``.

    The native shell (#1310) can't use the session cookie (OAuth completes in an external browser
    whose cookie jar the WebView can't see), so it carries the SAME signed session token as a Bearer
    header instead. Web clients keep sending the cookie; both verify identically.
    """
    header = request.headers.get("Authorization") or request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return value.strip() or None
    return None


def get_current_user(request: Request) -> User:
    """Resolve the signed session (cookie OR Bearer token) to a ``User``.

    Raises **503** when the server cannot authenticate *anyone* (no signing secret / no user
    store), and **401** only when this specific caller's credential is bad.
    """
    secret = _secret(request)
    data_dir = _data_dir(request)
    if not secret or data_dir is None:
        # Two very different situations share this condition, and they must not share a status code.
        #
        # (a) Auth was never configured on this deployment (no provider either) — the tailnet /
        #     operator modes run this way. Nothing is broken; the route simply requires a session
        #     that cannot exist here, and 401 "deny" is correct and is what the login-first route
        #     matrix asserts.
        #
        # (b) This deployment DOES authenticate — a provider is configured — but it currently
        #     cannot: the signing secret or the user store is gone. That is a SERVER fault, it is
        #     identical for every user, and answering 401 tells every client that every credential
        #     went bad at the same instant. The player believed exactly that on 2026-09-16, when a
        #     reboot lost ``APP_SESSION_SECRET``: it discarded the device snapshot AND the cached
        #     content, or — when no user had been painted yet — sat in a half-broken signed-out
        #     state that never self-healed, while ``/api/health`` kept answering 200.
        #
        # ``/auth/login`` already returns 503 for (b); this makes the rest of the API agree, so a
        # client can separate "my token is bad" from "this server can't authenticate anyone"
        # without heuristics.
        if _provider(request) is not None:
            raise HTTPException(status_code=503, detail="Auth is not configured.")
        raise HTTPException(status_code=401, detail="Not authenticated.")
    # Cookie is the web path; the Bearer token is the native-shell path (#1310). Same signer/secret,
    # same payload shape — try the cookie first, then fall back to the header.
    payload = app_sessions.verify(request.cookies.get(app_sessions.SESSION_COOKIE), secret)
    if not payload:
        payload = app_sessions.verify(_bearer_token(request), secret)
    user_id = payload.get("user_id") if payload else None
    user = get_user(data_dir, str(user_id)) if user_id else None
    if user is None or user.disabled:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user


def get_optional_user(request: Request) -> User | None:
    """Resolve the session to a ``User``, or ``None`` when unauthenticated (no 401).

    For read surfaces that personalize *when* signed in but stay open otherwise (e.g. the
    discovery feed): an anonymous request simply gets the un-personalized response.
    """
    try:
        return get_current_user(request)
    except HTTPException:
        return None


def get_admin_user(request: Request) -> User:
    """Like :func:`get_current_user` but requires the ``admin`` role (403 otherwise)."""
    user = get_current_user(request)
    if not app_roles.is_admin(user.role):
        raise HTTPException(status_code=403, detail="Admin role required.")
    return user


def require_viewer_access(request: Request) -> User:
    """Require a signed-in user with **at least ``creator``** (RFC-108 operator surfaces).

    Mounted as a router-level dependency on the operator-read routers **only** in the
    public operator serve mode (``PODCAST_SERVE_OPERATOR_PUBLIC``); the tailnet-only
    operator serve leaves them ungated (tailnet privacy is the gate). A signed-in
    ``listener`` gets 403 — the operator surface is creator/admin only.
    """
    user = get_current_user(request)
    if not app_roles.can_use_viewer(user.role):
        raise HTTPException(status_code=403, detail="Creator or admin role required.")
    return user


def _safe_return_to(value: str | None) -> str | None:
    """Open-redirect guard for the post-login ``return_to``.

    Allow ONLY a same-origin *relative* path (single leading ``/``). Rejects protocol-relative
    (``//host``), absolute URLs, backslashes, and CRLF so a poisoned ``return_to`` can't bounce
    the post-login redirect off-site. Used by the MCP ``/authorize`` bounce (RFC-112): an
    unauthenticated remote-client authorize is sent through Google sign-in and back here.
    """
    if not value or not isinstance(value, str):
        return None
    if not value.startswith("/") or value.startswith("//"):
        return None
    if "://" in value or "\\" in value or "\n" in value or "\r" in value:
        return None
    # #1977: this value now rides inside the signed `state` sent to the provider, and providers
    # cap `state` length. A pathological return_to would silently break login rather than merely
    # redirect oddly, so bound it here — nothing legitimate is near this.
    if len(value) > _MAX_RETURN_TO_CHARS:
        return None
    return value


@router.get("/auth/login")
async def app_auth_login(
    request: Request,
    as_: str | None = Query(default=None, alias="as", description="Mock identity hint (dev/e2e)."),
    grant: str | None = Query(
        default=None, description="Role hint for new users; only 'creator' is honoured."
    ),
    platform: str | None = Query(
        default=None, description="'native' → callback returns a deep-link token, not a cookie."
    ),
    return_to: str | None = Query(
        default=None,
        description="Same-origin path to return to after login (open-redirect-guarded).",
    ),
) -> RedirectResponse:
    """Begin the OAuth flow: redirect to the provider with a CSRF state cookie.

    ``?as=<name>`` is an optional identity hint honoured **only by the mock provider** (dev/e2e) so
    parallel e2e specs can sign in as isolated users; real providers ignore it.

    ``?grant=creator`` is the viewer's login hint: a *new* (or ``listener``) user is promoted to
    ``creator`` on callback. Only ``creator`` is ever granted this way — never ``admin``.
    """
    provider = _provider(request)
    secret = _secret(request)
    if provider is None or not secret:
        raise HTTPException(status_code=503, detail="Auth is not configured.")
    nonce = secrets.token_urlsafe(24)
    # #1977: the SIGNED payload is what goes to the provider, not the bare nonce. It is
    # tamper-evident and URL-safe (`{b64(json)}.{hmac}`), so the callback can validate the flow
    # from the echoed `state` alone when the cookie does not come back — which is what happens
    # when the sign-in is handed off to another browser context (the claude.ai mobile app's
    # connector flow, 2026-09-05). The cookie is still set and still preferred; see the callback.
    signed_state = app_sessions.sign(
        {
            "state": nonce,
            "iat": int(time.time()),
            "grant": grant or "",
            "platform": "native" if platform == "native" else "",
            "return_to": _safe_return_to(return_to) or "",
        },
        secret,
    )
    url = provider.authorization_url(
        state=signed_state, redirect_uri=_callback_uri(request), login_hint=as_
    )
    resp = RedirectResponse(url, status_code=307)
    resp.set_cookie(
        app_sessions.STATE_COOKIE,
        signed_state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    return resp


@router.get("/auth/callback", name="app_auth_callback")
async def app_auth_callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code."),
    state: str = Query(..., description="CSRF state echoed by the provider."),
) -> RedirectResponse:
    """Complete the OAuth flow: verify state, exchange code, upsert user, set session."""
    provider = _provider(request)
    secret = _secret(request)
    data_dir = _data_dir(request)
    if provider is None or not secret or data_dir is None:
        raise HTTPException(status_code=503, detail="Auth is not configured.")
    raw_state_cookie = request.cookies.get(app_sessions.STATE_COOKIE)
    from_cookie = app_sessions.verify(raw_state_cookie, secret, max_age=600)
    # The provider echoes back exactly what we sent: our own signed, unexpired payload. An
    # attacker cannot forge the HMAC, so this is a complete CSRF check on its own (#1977).
    from_state = app_sessions.verify(state, secret, max_age=600)

    saved = None
    if from_cookie is not None and from_cookie.get("state") == (from_state or {}).get("state"):
        # Normal path, unchanged strength: cookie present AND agreeing with the echoed state.
        saved = from_cookie
    elif from_cookie is None and from_state is not None:
        # The cookie did not come back — a cross-context handoff, not an attack signal we can
        # act on. The echoed state is signed by us and inside its 600s window, so the flow is
        # authentic; proceed on that. Logged so the frequency stays visible.
        saved = from_state
        logger.info("OAuth callback: state cookie absent; proceeding on the signed state (#1977)")

    if saved is None:
        # Say WHICH check failed — these have completely different causes and a bare 400 cannot
        # tell them apart, which cost an hour of log archaeology on 2026-09-05. No state VALUE is
        # logged, only the reason.
        if from_state is None and not raw_state_cookie:
            reason = "no usable state: cookie absent and the echoed state is not validly signed"
        elif from_state is None:
            reason = "echoed state is not validly signed (forged, corrupted, or older than 600s)"
        elif from_cookie is None:
            reason = "state cookie absent and the signed-state fallback did not apply"
        else:
            reason = "state cookie and echoed state are both valid but disagree"
        logger.warning("OAuth callback rejected: %s", reason)
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")
    try:
        identity = provider.exchange_code(code=code, redirect_uri=_callback_uri(request))
    except OAuthError as exc:
        raise HTTPException(status_code=502, detail="OAuth exchange failed.") from exc
    # Resolved PER SIGN-IN, not once at startup: the persisted policy (admin endpoint, #2190) wins
    # over the env one, so admitting a beta tester is an API call rather than a production
    # redeploy. Absent file -> the env policy, i.e. exactly the previous behaviour.
    policy = app_access_store.effective_policy(
        getattr(request.app.state, "app_data_dir", None),
        getattr(request.app.state, "access_policy", None),
    )
    if policy is not None and not policy.is_allowed(identity.email):
        raise HTTPException(status_code=403, detail="This account is not allowed to sign in.")
    user = get_or_create_user(
        data_dir,
        provider=identity.provider,
        subject=identity.subject,
        email=identity.email,
        name=identity.name,
        image=identity.image,
        # #2266: server-side truth for signups, independent of whether the Umami script ever
        # loaded, and the anchor for the 24-hour "Activated" window. Fires from inside the store's
        # creation branch so it is exactly once per account even when an OAuth callback double-fires
        # — the alternative, checking existence here first, races and would report two signups for
        # one account.
        on_created=lambda created: app_user_state.append_account_created(
            data_dir, created.user_id, identity.provider
        ),
    )
    # Apply the role policy: admin allowlist > creator grant > existing role (never downgraded).
    admin_emails: frozenset[str] = getattr(request.app.state, "admin_emails", frozenset())
    effective = app_roles.resolve_login_role(
        user.role, email=user.email, grant=saved.get("grant"), admin_emails=admin_emails
    )
    if effective != user.role:
        set_role(data_dir, user.user_id, effective)
        user = replace(user, role=effective)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, secret)
    # Native shell (#1310): the OAuth completed in an external browser, so a cookie can't reach the
    # WebView. Hand the SAME signed token back via the app's custom-scheme deep link; the app stores
    # it and sends it as a Bearer header. The token rides the URL fragment (never logged/cached by
    # proxies the way a query string is), and the state cookie is cleared either way.
    if saved.get("platform") == "native":
        deep_link = f"{_native_scheme(request)}://auth#token={token}"
        resp = RedirectResponse(deep_link, status_code=307)
        resp.delete_cookie(app_sessions.STATE_COOKIE)
        return resp
    # Return to where login was initiated (e.g. the MCP /authorize consent, RFC-112) when a
    # guarded same-origin return_to rode the state cookie; otherwise the player home.
    dest = _safe_return_to(saved.get("return_to")) or "/"
    resp = RedirectResponse(dest, status_code=307)
    resp.set_cookie(
        app_sessions.SESSION_COOKIE,
        token,
        max_age=app_sessions.DEFAULT_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_secure(request),
    )
    resp.delete_cookie(app_sessions.STATE_COOKIE)
    return resp


@router.post("/auth/logout")
async def app_auth_logout() -> Response:
    """Clear the session cookie."""
    resp = Response(status_code=204)
    resp.delete_cookie(app_sessions.SESSION_COOKIE)
    return resp


def _user_dict(user: User) -> dict[str, object]:
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "username": user.username,  # immutable handle (Area E)
        "image": user.image,  # avatar URL (OAuth-captured or uploaded)
        "role": user.role,
        "disabled": user.disabled,
        "mcp_access": user.mcp_access,  # RFC-112: gates the MCP connection UI
        # #2267: which OAuth provider this identity came from, for the `auth_completed` analytics
        # event. The client cannot know it otherwise — the provider is server-configured, and the
        # login UI only knows whether the MOCK one is active.
        "provider": user.provider,
        # #2265: the pseudonymous analytics identity the client passes to Umami's `identify`, and
        # which Settings › About shows so the operator can note it per beta participant. Empty
        # only for an account whose backfill has not run yet (see `_backfill_analytics_id`); the
        # client treats empty as "do not identify" rather than identifying with a blank id.
        "analytics_id": user.analytics_id,
    }


@router.get("/me")
def app_me(user: User = Depends(get_current_user)) -> dict[str, object]:
    """Return the signed-in user's basic profile + role (401 when not authenticated)."""
    return _user_dict(user)


@router.get("/auth/dev-users")
def app_auth_dev_users(request: Request) -> dict[str, object]:
    """Predefined dev identities for the sign-in picker — only when the MOCK provider is active.

    With the fake (mock) OAuth provider on, the sign-in UI lets you pick a seeded user (or type a
    custom name) and signs in as ``?as=<hint>``. With a real provider (Google), ``enabled`` is
    ``False`` and the UI shows the normal provider button instead.
    """
    provider = _provider(request)
    is_mock = getattr(provider, "name", "") == "mock"
    users: list[dict[str, str]] = []
    if is_mock:
        from podcast_scraper.server.app_oauth import _safe_hint
        from podcast_scraper.server.app_user_seed import seeds_from_env

        for seed in seeds_from_env():
            hint = _safe_hint(str(seed.get("hint", "")))
            if not hint:
                continue
            users.append(
                {
                    "hint": hint,
                    "name": str(seed.get("name") or hint),
                    "role": app_roles.normalize_role(seed.get("role")),
                }
            )
    return {"enabled": is_mock, "users": users}


@router.get("/auth/status")
def app_auth_status(request: Request) -> dict[str, object]:
    """Whether platform auth is *configured*, plus the signed-in user (if any) — never 401s.

    The viewer gates its UI on this: when auth is not configured (no session secret / provider /
    data dir — e.g. a bare deployment or a backend-less e2e), the app renders **open**, preserving
    the pre-auth behaviour. Only when auth is enabled does an anonymous request get the login gate.
    """
    enabled = bool(_secret(request) and _provider(request) is not None and _data_dir(request))
    user = get_optional_user(request) if enabled else None
    return {"enabled": enabled, "user": _user_dict(user) if user is not None else None}


# ---------------------------------------------------------------------------------------------
# Email magic-link sign-in (#2272)
#
# A second front door, for people who will not create a Google account. No password is stored and
# no credential exists: the proof of identity is demonstrated control of a mailbox, which is what a
# password reset already relies on.
#
# Registration and login are the SAME mechanism — the link either creates the account or signs the
# person in — and differ only in where they land afterwards. The UI offers two entry points over
# this one flow.
# ---------------------------------------------------------------------------------------------

#: Where a brand-new account lands: the profile, to fill in the things we could not learn from an
#: OAuth payload. An email identity arrives with nothing but an address — no name, no picture — so
#: dropping them on Home would leave a half-built profile they never see.
_NEW_ACCOUNT_DEST = "/profile?welcome=1"

#: Where a returning account lands.
_RETURNING_DEST = "/"

#: Per-address send throttle. Low, because the cost of being wrong is someone else's inbox.
_REQUEST_MIN_INTERVAL_S = 60

#: DeliveryEnvelope schema version (RFC-110) — matches `app_digest_personal.SCHEMA_VERSION`.
_ENVELOPE_SCHEMA_VERSION = "1"


def _email_fingerprint(email: str) -> str:
    """A stable, short hash of an address — for logs, never the address itself.

    The whole flow is built so that nobody can learn whether an address has an account here: the
    request endpoint answers identically for every address, and the throttle keys on a hash for the
    same reason. A log line carrying the plain address would hand that back, in the one place that
    is retained, searchable and shipped off the box.

    A fingerprint still answers every operational question — "did THIS person's link get sent, and
    was it the same person who then verified?" — by correlating two events, without naming anyone.
    """
    import hashlib

    return hashlib.sha256(app_magic_link.normalise_email(email).encode("utf-8")).hexdigest()[:16]


def _magic_event(event_type: str, **fields: Any) -> None:
    """Emit one ADR-119 event for the magic-link flow. Best-effort; never raises.

    `sink="log"` (stdout), not a per-user file: these events happen BEFORE an account exists, which
    is the entire point of the flow, so there is no per-user path to write to. `emit_event` attaches
    the current trace context, so a log line and the request's span correlate without extra work.
    """
    from podcast_scraper.obs.events import emit_event

    emit_event(event_type, sink="log", logger=logger, **fields)


class MagicLinkRequest(BaseModel):
    """A request for a sign-in link. ``platform`` mirrors the OAuth login route's native switch."""

    email: str = Field(min_length=3, max_length=254)
    platform: str | None = None
    return_to: str | None = Field(default=None, max_length=_MAX_RETURN_TO_CHARS)


def _magic_link_url(request: Request, token: str) -> str:
    """Absolute URL of the verify route carrying ``token``."""
    return f"{request.url_for('app_auth_magic_verify')}?token={token}"


def _recent_request_marker(data_dir: Path, email: str) -> Path:
    import hashlib

    digest = hashlib.sha256(email.encode("utf-8")).hexdigest()[:32]
    return data_dir / "magic_link_recent" / f"{digest}.txt"


def _throttled(data_dir: Path | None, email: str, *, now: int) -> bool:
    """True when a link was already sent to this address within the throttle window.

    Keyed by a HASH of the address, not the address: this directory would otherwise become a list of
    everyone who has ever asked for a link, including people who never had an account.
    """
    if data_dir is None:
        return False
    marker = _recent_request_marker(data_dir, email)
    try:
        if marker.is_file() and (now - int(marker.stat().st_mtime)) < _REQUEST_MIN_INTERVAL_S:
            return True
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(now), encoding="utf-8")
    except OSError:
        # Throttle state is best-effort; failing to record it must not block a legitimate sign-in.
        return False
    return False


@router.post("/auth/email/request", status_code=202)
async def app_auth_magic_request(
    body: MagicLinkRequest, request: Request, response: Response
) -> dict[str, bool]:
    """Send a sign-in link to ``email``. ALWAYS answers 202, whatever the address is.

    The uniform answer is the point. Any variation — 404 for unknown, 403 for not-allowed, a slower
    path for one of them — turns this into an oracle that reports whether an address has an account
    and whether it is on the operator's allowlist. Both are things a stranger should not be able to
    ask, and the allowlist in particular is a small, guessable set of real people.

    So the access-policy check does NOT happen here. It happens at verify, where it already has to
    happen anyway (the gate runs on every sign-in), and where refusing reveals nothing to anyone who
    did not already control the mailbox.
    """
    secret = _secret(request)
    data_dir = _data_dir(request)
    email = app_magic_link.normalise_email(body.email)
    now = int(time.time())

    # Shape check only — deliverability is the mail system's business, and a stricter local rule
    # would reject real addresses.
    if not secret or "@" not in email or email.startswith("@") or email.endswith("@"):
        # Still reported, because "nothing was sent" is the single hardest state to diagnose from
        # the outside: the person sees the same "check your inbox" either way.
        _magic_event(
            "magic_link_requested", outcome="rejected_shape", email_fp=_email_fingerprint(email)
        )
        return {"ok": True}
    if _throttled(data_dir, email, now=now):
        _magic_event(
            "magic_link_requested", outcome="throttled", email_fp=_email_fingerprint(email)
        )
        return {"ok": True}

    token, token_id = app_magic_link.issue(email, secret, now=now)
    link = _magic_link_url(request, token)
    if body.platform == "native":
        link = f"{link}&platform=native"
    safe_return = _safe_return_to(body.return_to)
    if safe_return:
        from urllib.parse import quote

        link = f"{link}&return_to={quote(safe_return, safe='')}"

    if data_dir is not None:
        envelope = {
            # The shared DeliveryEnvelope schema version (RFC-110), same value
            # `app_digest_personal.SCHEMA_VERSION` stamps. Written literally rather than imported:
            # auth has no business depending on the digest module, and the envelope contract belongs
            # to the outbox, not to either producer.
            "schema_version": _ENVELOPE_SCHEMA_VERSION,
            # Unique per token, so the outbox's id-dedupe cannot collapse two genuine requests.
            "id": f"auth_{token_id}",
            # No account exists yet — that is the whole point of this message. The transactional
            # class in `app_outbox_store` is what lets an envelope with no user_id through the
            # consent gate.
            "user_id": "",
            "type": "auth_link",
            "channel": "email",
            "template": "magic-link.v1",
            "recipient": {"email": email, "email_verified": False},
            "payload": {
                "link": link,
                "expires_minutes": app_magic_link.TOKEN_TTL_SECONDS // 60,
            },
            "not_before": _iso_now(now),
            # Pointless to deliver a link that has already expired: the outbox drops expired
            # envelopes rather than flushing stale ones after an outage.
            "expires_at": _iso_now(now + app_magic_link.TOKEN_TTL_SECONDS),
            "created_at": _iso_now(now),
        }
        enqueued = app_outbox_store.enqueue(data_dir, envelope)
        _magic_event(
            "magic_link_requested",
            outcome="enqueued" if enqueued else "duplicate",
            email_fp=_email_fingerprint(email),
            envelope_id=envelope["id"],
            expires_minutes=app_magic_link.TOKEN_TTL_SECONDS // 60,
            platform=body.platform or "web",
        )
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


def _iso_now(epoch: int) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc).isoformat().replace("+00:00", "Z")


@router.get("/auth/email/verify", name="app_auth_magic_verify")
async def app_auth_magic_verify(
    request: Request,
    token: str = Query(...),
    platform: str | None = Query(default=None),
    return_to: str | None = Query(default=None),
) -> Response:
    """Consume a sign-in link: create or sign in the account, then redirect.

    Mirrors the OAuth callback deliberately — same access-policy check, same ``get_or_create_user``,
    same role resolution, same session token, same native deep-link branch — because any divergence
    between the two doors is a difference in who can get in and with what rights.

    WHERE IT LANDS is the one real difference, and it is driven by whether the account was CREATED:
    a new account goes to the profile to fill in what an email identity cannot supply (no name, no
    picture), a returning one goes home.
    """
    secret = _secret(request)
    data_dir = _data_dir(request)
    payload = app_magic_link.parse(token, secret)
    if payload is None:
        # No fingerprint: the token did not parse, so there is no address to attribute this to.
        # A burst of these is the signal that links are arriving after their TTL.
        _magic_event("magic_link_verified", outcome="invalid_or_expired")
        raise HTTPException(status_code=400, detail="This sign-in link is invalid or has expired.")
    email = str(payload["email"])

    # The gate runs on EVERY sign-in, exactly as it does for OAuth. Checked BEFORE the token is
    # consumed: a person refused by the allowlist has done nothing wrong, and burning their link
    # would also deny them the retry that an operator adding them would make work.
    policy = app_access_store.effective_policy(
        data_dir, getattr(request.app.state, "access_policy", None)
    )
    if policy is not None and not policy.is_allowed(email):
        # The operationally important one: a tester who was never added to the allowlist looks, from
        # their side, exactly like a broken link.
        _magic_event(
            "magic_link_verified", outcome="refused_policy", email_fp=_email_fingerprint(email)
        )
        raise HTTPException(status_code=403, detail="This account is not allowed to sign in.")

    if data_dir is None:
        _magic_event(
            "magic_link_verified", outcome="unavailable", email_fp=_email_fingerprint(email)
        )
        raise HTTPException(status_code=503, detail="Sign-in is not available.")
    if not app_magic_link.consume(data_dir, str(payload["jti"])):
        # Usually benign — a prefetching mail client or a double tap — but a sustained rate means
        # something is fetching links before people do, which changes what the TTL is protecting.
        _magic_event("magic_link_verified", outcome="replayed", email_fp=_email_fingerprint(email))
        # Single-use. The ordinary cause is a mail client prefetching the link or the person
        # clicking twice — so the message says what to do rather than implying wrongdoing.
        raise HTTPException(
            status_code=400,
            detail="This sign-in link has already been used. Request a new one.",
        )

    # `on_created` fires from INSIDE the store's creation branch, so it is exactly once per account
    # even if a link is somehow verified twice — and it is also how we learn whether to land the
    # person on the profile or on home. Checking existence beforehand would race and could send a
    # returning user to the new-account screen.
    was_created = False

    def _on_created(fresh: User) -> None:
        nonlocal was_created
        was_created = True
        app_user_state.append_account_created(data_dir, fresh.user_id, "email")

    user = get_or_create_user(
        data_dir,
        provider="email",
        subject=email,
        email=email,
        # An email identity supplies no display name. The local part is a placeholder, better
        # than an empty masthead; a new account lands on the profile, whose welcome card asks
        # for a real name and saves it through POST /profile/name.
        name=email.partition("@")[0],
        image=None,
        on_created=_on_created,
    )
    admin_emails: frozenset[str] = getattr(request.app.state, "admin_emails", frozenset())
    effective = app_roles.resolve_login_role(
        user.role, email=user.email, grant=None, admin_emails=admin_emails
    )
    if effective != user.role:
        set_role(data_dir, user.user_id, effective)
        user = replace(user, role=effective)

    is_new = was_created
    _magic_event(
        "magic_link_verified",
        outcome="created" if is_new else "returning",
        email_fp=_email_fingerprint(email),
        platform=platform or "web",
    )
    session = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, secret)

    if platform == "native":
        # Same contract as the OAuth callback: the token rides the fragment (never logged or cached
        # the way a query string is). `new` tells the shell which screen to open.
        deep_link = f"{_native_scheme(request)}://auth#token={session}&new={'1' if is_new else '0'}"
        return RedirectResponse(deep_link, status_code=307)

    dest = _safe_return_to(return_to) or (_NEW_ACCOUNT_DEST if is_new else _RETURNING_DEST)
    resp = RedirectResponse(dest, status_code=307)
    resp.set_cookie(
        app_sessions.SESSION_COOKIE,
        session,
        max_age=app_sessions.DEFAULT_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )
    return resp
