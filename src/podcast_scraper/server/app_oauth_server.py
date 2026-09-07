"""OAuth 2.1 authorization server for the MCP (RFC-112 slice 3, #1471).

The **primary** auth path — required so claude.ai per-user connectors work (per-user sign-in ⇒
OAuth; the MCP spec mandates OAuth 2.1 + PKCE for public servers). We are the authorization
server: a client (claude.ai) self-registers (Dynamic Client Registration), the user approves on a
consent screen backed by the existing platform session, and we issue an access token bound to that
user. The access token is verified the same way a PAT is (see ``routes/internal_mcp``), so the MCP
transport (slice 2) accepts both.

**Public clients + PKCE only** (no client secret — MCP clients are public). Tokens are opaque and
stored **hashed**; auth codes are single-use + short-lived. State lives in a few lock-serialised
files under ``<data_dir>`` (low volume: a handful of clients + tokens per user).
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock

from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_CLIENTS_FILE = "oauth_clients.json"
_GRANTS_FILE = "oauth_grants.json"  # auth codes + access/refresh tokens, keyed by hash
_CONSENTS_FILE = "oauth_consents.json"  # remembered (user, client, scope) approvals
_USE_FILE = "oauth_client_use.json"  # {user\x00client: unix_ts} — last time a grant was used

_CODE_TTL_S = 60  # authorization codes are single-use + short-lived
_ACCESS_TTL_S = 3600  # 1h access tokens
_REFRESH_TTL_S = 30 * 86400  # 30d refresh tokens
_SCOPE = "mcp:read"
_SUPPORTED_SCOPES = frozenset({"mcp:read"})  # v1: read-only; room for mcp:export etc.

_MAX_CLIENTS = 2000  # disk-fill guard on unauthenticated DCR (H4)
_MAX_REDIRECT_URIS = 10  # per-client cap (L2)
# Prune DCR clients that never completed a flow (no live grant) after this age, so an
# unauthenticated registration flood can't PERMANENTLY fill the store at _MAX_CLIENTS (advisor LOW).
_CLIENT_UNUSED_TTL_S = 7 * 24 * 3600

_ACCESS_PREFIX = "clp_mcpat_"
_REFRESH_PREFIX = "clp_mcprt_"


def _lock(data_dir: Path, name: str) -> FileLock:
    data_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str((data_dir / name).with_suffix(".lock")), timeout=_LOCK_TIMEOUT_S)


def _read(data_dir: Path, name: str) -> dict[str, Any]:
    path = data_dir / name
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _write(data_dir: Path, name: str, doc: dict[str, Any]) -> None:
    atomic_write_text(data_dir / name, json.dumps(doc, ensure_ascii=False, indent=2))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _now() -> int:
    return int(time.time())


# --- Dynamic Client Registration (DCR) ---


def register_client(
    data_dir: Path, *, redirect_uris: list[str], client_name: str
) -> dict[str, Any]:
    """Register a public client. Returns the client metadata incl. the new ``client_id``."""
    uris = [u for u in redirect_uris if isinstance(u, str) and u.startswith("https://")]
    # Loopback http is allowed for native/CLI clients (RFC 8252).
    uris += [
        u
        for u in redirect_uris
        if isinstance(u, str)
        and (u.startswith("http://127.0.0.1") or u.startswith("http://localhost"))
    ]
    if not uris:
        raise ValueError("at least one https (or loopback) redirect_uri is required")
    if len(uris) > _MAX_REDIRECT_URIS:
        raise ValueError(f"at most {_MAX_REDIRECT_URIS} redirect_uris allowed")
    client_id = f"mcpc_{uuid.uuid4().hex}"
    client: dict[str, Any] = {
        "client_id": client_id,
        "client_name": (client_name or "").strip()[:120] or "agent",
        "redirect_uris": list(dict.fromkeys(uris)),
        "token_endpoint_auth_method": "none",  # public client + PKCE
        "created_at": _now(),
    }
    with _lock(data_dir, _CLIENTS_FILE):
        clients = _read(data_dir, _CLIENTS_FILE)
        # Drop stale never-used clients first (advisor LOW): a client with NO live grant that is
        # older than the unused-TTL is a registration that never completed a flow — reclaim it so a
        # DCR flood can't permanently wedge the store at the cap. Clients with a live grant (active
        # code/access/refresh) are always kept.
        clients = _prune_stale_clients(clients, _read(data_dir, _GRANTS_FILE))
        if len(clients) >= _MAX_CLIENTS:
            # Unauthenticated DCR: refuse once the store is full rather than grow without bound.
            raise ValueError("client registration limit reached")
        clients[client_id] = client
        _write(data_dir, _CLIENTS_FILE, clients)
    return client


def get_client(data_dir: Path, client_id: str) -> dict[str, Any] | None:
    """The registered client metadata for ``client_id``, or None if unknown."""
    client = _read(data_dir, _CLIENTS_FILE).get(client_id)
    return client if isinstance(client, dict) else None


def is_scope_supported(scope: str) -> bool:
    """Whether ``scope`` is one we mint (v1: only ``mcp:read``)."""
    return scope in _SUPPORTED_SCOPES


# --- Remembered consent (skip re-prompting once a user has approved a client + scope) ---


def _consent_key(user_id: str, client_id: str, scope: str) -> str:
    return f"{user_id}\x00{client_id}\x00{scope}"


def remember_consent(data_dir: Path, *, user_id: str, client_id: str, scope: str) -> None:
    """Record that ``user_id`` approved ``client_id`` for ``scope`` (future authorizes silent)."""
    with _lock(data_dir, _CONSENTS_FILE):
        consents = _read(data_dir, _CONSENTS_FILE)
        consents[_consent_key(user_id, client_id, scope)] = _now()
        _write(data_dir, _CONSENTS_FILE, consents)


def has_consent(data_dir: Path, *, user_id: str, client_id: str, scope: str) -> bool:
    """Whether ``user_id`` has already approved ``client_id`` for ``scope``."""
    return _consent_key(user_id, client_id, scope) in _read(data_dir, _CONSENTS_FILE)


def revoke_consent(data_dir: Path, *, user_id: str, client_id: str) -> bool:
    """Forget a user's approval of a client (all scopes). Returns True when anything was removed."""
    prefix = f"{user_id}\x00{client_id}\x00"
    with _lock(data_dir, _CONSENTS_FILE):
        consents = _read(data_dir, _CONSENTS_FILE)
        keys = [k for k in consents if k.startswith(prefix)]
        for k in keys:
            consents.pop(k, None)
        if keys:
            _write(data_dir, _CONSENTS_FILE, consents)
    return bool(keys)


def revoke_client_grants(data_dir: Path, *, user_id: str, client_id: str) -> int:
    """Invalidate a user's live grants for a client (codes + access + refresh); return the count.

    A true *disconnect*: forgetting consent alone leaves an already-issued access token valid until
    expiry (and its refresh token usable for 30d). It also leaves any **un-exchanged authorization
    code** live for its 60s TTL — and since the code→token exchange does not re-check consent, that
    code would resurrect a fresh 30-day grant *after* the user clicked Disconnect. So we drop codes
    too. Pair with :func:`revoke_consent`.
    """
    with _lock(data_dir, _GRANTS_FILE):
        grants = _read(data_dir, _GRANTS_FILE)
        doomed = [
            h
            for h, rec in grants.items()
            if isinstance(rec, dict)
            and rec.get("user_id") == user_id
            and rec.get("client_id") == client_id
            and rec.get("kind") in ("code", "access", "refresh")
        ]
        for h in doomed:
            grants.pop(h, None)
        if doomed:
            _write(data_dir, _GRANTS_FILE, grants)
    return len(doomed)


# --- Last use (#2004 item 14) ---
#
# "Connected agents" listed six opaque client ids and the date each was approved. Approval date
# answers "did I click yes", which nobody doubts; it does not answer the question a person actually
# has in front of this screen — **is this thing still talking to my account?** A connection last
# used nine months ago is the one worth revoking, and there was no way to tell it from the one that
# ran an hour ago.
#
# Kept in its OWN file rather than folded into the consents map. That map's values are bare ints
# (the consent timestamp); widening them to a dict would mean every reader handling two shapes
# forever, for a field that is decoration rather than an authorisation input. A separate file is
# additive: absent means never used, which is exactly right for connections that predate this.

#: Only write when the stored value is at least this stale.
#:
#: `verify_access_token` runs on EVERY authenticated MCP request, and a naive stamp would mean a
#: lock + read + write of a JSON file per request — turning a display nicety into the hottest write
#: path in the server. A human reading "last used" cannot tell 08:31 from 08:34, so buying two
#: orders of magnitude fewer writes with five minutes of precision is free.
_USE_STAMP_INTERVAL_S = 300


def _use_key(user_id: str, client_id: str) -> str:
    return f"{user_id}\x00{client_id}"


def record_client_use(data_dir: Path, *, user_id: str, client_id: str) -> None:
    """Stamp that ``client_id`` just used ``user_id``'s grant.

    Coalesced to one write per client per ``_USE_STAMP_INTERVAL_S``; see the note above.
    """
    if not user_id or not client_id:
        return
    key = _use_key(user_id, client_id)
    now = _now()
    # Read WITHOUT the lock first: the common case is "stamped recently, nothing to do", and taking
    # the lock to discover that would reintroduce the per-request contention the interval exists to
    # avoid. The re-check inside the lock is what makes the decision correct.
    if now - int(_read(data_dir, _USE_FILE).get(key, 0)) < _USE_STAMP_INTERVAL_S:
        return
    with _lock(data_dir, _USE_FILE):
        uses = _read(data_dir, _USE_FILE)
        if now - int(uses.get(key, 0)) < _USE_STAMP_INTERVAL_S:
            return
        uses[key] = now
        _write(data_dir, _USE_FILE, uses)


def forget_client_use(data_dir: Path, *, user_id: str, client_id: str) -> None:
    """Drop the use record on disconnect, so a later reconnect does not inherit an old timestamp."""
    key = _use_key(user_id, client_id)
    with _lock(data_dir, _USE_FILE):
        uses = _read(data_dir, _USE_FILE)
        if key in uses:
            uses.pop(key, None)
            _write(data_dir, _USE_FILE, uses)


def list_consents(data_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """The connected OAuth clients.

    Rows are ``{client_id, client_name, scopes, connected_at, last_used_at}``.

    Joins remembered consents (per user+client+scope) against the client registry for display names,
    newest first. This is what the 'Connected agents' UI lists so a user can revoke a connection.

    ``last_used_at`` is None for a connection that has never made a request, and for every
    connection that predates the use log — the two are indistinguishable here, and the UI says
    "not used yet" rather than inventing a date.
    """
    consents = _read(data_dir, _CONSENTS_FILE)
    clients = _read(data_dir, _CLIENTS_FILE)
    uses = _read(data_dir, _USE_FILE)
    prefix = f"{user_id}\x00"
    by_client: dict[str, dict[str, Any]] = {}
    for key, ts in consents.items():
        if not key.startswith(prefix):
            continue
        _uid, _, rest = key.partition("\x00")
        client_id, _, scope = rest.partition("\x00")
        entry = by_client.setdefault(
            client_id,
            {
                "client_id": client_id,
                "client_name": str((clients.get(client_id) or {}).get("client_name") or client_id),
                "scopes": [],
                "connected_at": 0,
                "last_used_at": (
                    int(uses[_use_key(user_id, client_id)])
                    if _use_key(user_id, client_id) in uses
                    else None
                ),
            },
        )
        if scope and scope not in entry["scopes"]:
            entry["scopes"].append(scope)
        entry["connected_at"] = max(int(entry["connected_at"]), int(ts))
    return sorted(by_client.values(), key=lambda e: e["connected_at"], reverse=True)


# --- Authorization codes (PKCE) ---


def create_authorization_code(
    data_dir: Path,
    *,
    user_id: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    scope: str = _SCOPE,
    resource: str = "",
) -> str:
    """Mint a single-use, short-lived authorization code bound to the user + PKCE challenge.

    ``resource`` (RFC 8707) is recorded on the code so the token minted from it is audienced to the
    resource the user actually consented to, not to a server-wide default (#1979).
    """
    code = secrets.token_urlsafe(32)
    with _lock(data_dir, _GRANTS_FILE):
        grants = _read(data_dir, _GRANTS_FILE)
        grants[_hash(code)] = {
            "kind": "code",
            "user_id": user_id,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "scope": scope,
            "resource": (resource or "").strip().rstrip("/"),
            "expires_at": _now() + _CODE_TTL_S,
        }
        _write(data_dir, _GRANTS_FILE, grants)
    return code


def _pkce_ok(verifier: str, challenge: str) -> bool:
    """S256: base64url(sha256(verifier)) == challenge (no padding)."""
    # RFC 7636 §4.1: the verifier is 43–128 chars. Enforce the length bounds so a lazy/malicious
    # client can't self-downgrade to a low-entropy verifier (advisor MED #1505).
    if not (43 <= len(verifier) <= 128):
        return False
    try:
        raw = verifier.encode("ascii")  # PKCE verifiers are ASCII (RFC 7636); non-ASCII → reject
    except UnicodeEncodeError:
        return False
    digest = hashlib.sha256(raw).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(computed, challenge)


def _prune_expired(grants: dict[str, Any]) -> dict[str, Any]:
    """Drop expired codes/access/refresh records so ``oauth_grants.json`` stays bounded (M2)."""
    now = _now()
    return {h: rec for h, rec in grants.items() if int(rec.get("expires_at", 0)) >= now}


def _prune_stale_clients(clients: dict[str, Any], grants: dict[str, Any]) -> dict[str, Any]:
    """Drop DCR clients with no LIVE grant that are older than the unused-TTL (advisor LOW).

    A client is kept if it has any non-expired grant (code/access/refresh → an active connection)
    OR it was registered within the unused-TTL (give an in-flight authorize time to complete).
    Everything else is a registration that never completed a flow — reclaim its slot.
    """
    now = _now()
    active = {
        str(rec.get("client_id"))
        for rec in _prune_expired(grants).values()
        if isinstance(rec, dict) and rec.get("client_id")
    }
    return {
        cid: c
        for cid, c in clients.items()
        if cid in active or (now - int(c.get("created_at", 0))) < _CLIENT_UNUSED_TTL_S
    }


def _resource_aud() -> str:
    """The DEFAULT audience (RFC 8707) — the first configured MCP resource URL, or ``""``.

    Used only when a client does not name a ``resource``. Multi-resource deployments must send one;
    see :func:`allowed_resources`.
    """
    allowed = allowed_resources()
    return allowed[0] if allowed else ""


def allowed_resources() -> list[str]:
    """Every MCP resource URL this authorization server may mint tokens for.

    ``APP_MCP_RESOURCE_URLS`` (comma-separated) is the multi-resource form; it falls back to the
    single ``APP_MCP_RESOURCE_URL`` so existing single-resource deployments are unchanged.

    #1979: this used to be one fixed value stamped on EVERY token, which meant an app acting as the
    AS for two MCP resources could only ever serve one of them — a client registering for the other
    completed the whole flow, received a token audienced to the wrong resource, and (correctly, per
    RFC 8707) refused to send it. The resource server then never saw a single request.
    """
    import os

    raw = os.environ.get("APP_MCP_RESOURCE_URLS", "").strip()
    if not raw:
        raw = os.environ.get("APP_MCP_RESOURCE_URL", "").strip()
    out: list[str] = []
    for part in raw.split(","):
        v = part.strip().rstrip("/")
        if v and v not in out:
            out.append(v)
    return out


def resolve_resource(requested: str) -> str | None:
    """Validate a client-supplied ``resource`` against the allowlist.

    Returns the canonical (trailing-slash-stripped) value, or ``None`` when the client named a
    resource this AS does not serve — the caller turns that into RFC 8707 ``invalid_target``.
    An empty request resolves to the default resource, preserving pre-#1979 behaviour.
    """
    allowed = allowed_resources()
    want = (requested or "").strip().rstrip("/")
    if not want:
        return allowed[0] if allowed else ""
    return want if want in allowed else None


def _issue_tokens(
    data_dir: Path, *, user_id: str, client_id: str, scope: str, aud: str | None = None
) -> dict[str, Any]:
    access = _ACCESS_PREFIX + secrets.token_urlsafe(32)
    refresh = _REFRESH_PREFIX + secrets.token_urlsafe(32)
    now = _now()
    # #1979: the audience is an ARGUMENT now. It comes from the authorization code's recorded
    # resource (or the refresh token being rotated), never from a re-read of the environment —
    # re-reading is what bound every token to one resource regardless of what was requested.
    aud = _resource_aud() if aud is None else (aud or "").strip().rstrip("/")
    with _lock(data_dir, _GRANTS_FILE):
        grants = _prune_expired(_read(data_dir, _GRANTS_FILE))
        grants[_hash(access)] = {
            "kind": "access",
            "user_id": user_id,
            "client_id": client_id,
            "scope": scope,
            "aud": aud,
            "expires_at": now + _ACCESS_TTL_S,
        }
        grants[_hash(refresh)] = {
            "kind": "refresh",
            "user_id": user_id,
            "client_id": client_id,
            "scope": scope,
            "aud": aud,
            "expires_at": now + _REFRESH_TTL_S,
        }
        _write(data_dir, _GRANTS_FILE, grants)
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": _ACCESS_TTL_S,
        "scope": scope,
    }


def exchange_authorization_code(
    data_dir: Path,
    *,
    code: str,
    code_verifier: str,
    client_id: str,
    redirect_uri: str,
    is_entitled: Callable[[str], bool] | None = None,
    resource: str = "",
) -> dict[str, Any] | None:
    """Verify code + PKCE + client/redirect binding, consume, issue tokens (None on fail).

    The issued audience is the resource recorded on the CODE. A ``resource`` supplied on the token
    request must agree with it (RFC 8707 §2.2) — a mismatch is a failed exchange, not a silent
    re-binding.

    ``is_entitled(user_id)`` (injected by the route) is re-checked at exchange time so a user whose
    ``mcp_access`` was pulled cannot mint fresh tokens from a still-live code (H2).
    """
    code_hash = _hash(code)
    with _lock(data_dir, _GRANTS_FILE):
        grants = _read(data_dir, _GRANTS_FILE)
        rec = grants.get(code_hash)
        if not isinstance(rec, dict) or rec.get("kind") != "code":
            return None
        # Single-use: consume regardless of outcome.
        grants.pop(code_hash, None)
        _write(data_dir, _GRANTS_FILE, grants)
    if rec["expires_at"] < _now():
        return None
    if rec["client_id"] != client_id or rec["redirect_uri"] != redirect_uri:
        return None
    if not _pkce_ok(code_verifier, str(rec["code_challenge"])):
        return None
    if is_entitled is not None and not is_entitled(str(rec["user_id"])):
        return None
    bound = str(rec.get("resource") or "")
    asked = (resource or "").strip().rstrip("/")
    if asked and asked != bound:
        return None
    # A code with NO recorded resource is a client that never sent one (or a code minted before
    # #1979). Fall back to the configured default via the `None` sentinel — passing "" straight
    # through would mint an unaudienced token and silently drop the RFC 8707 binding that
    # single-resource deployments already rely on.
    return _issue_tokens(
        data_dir,
        user_id=str(rec["user_id"]),
        client_id=client_id,
        scope=str(rec["scope"]),
        aud=bound or None,
    )


def refresh_access_token(
    data_dir: Path,
    *,
    refresh_token: str,
    client_id: str,
    is_entitled: Callable[[str], bool] | None = None,
) -> dict[str, Any] | None:
    """Rotate a refresh token → new access+refresh. Consumes the old refresh. None on fail.

    ``is_entitled(user_id)`` is re-checked so the refresh chain dies the moment ``mcp_access`` is
    pulled — the old refresh is already consumed above, so a de-entitled user's chain terminates
    (not just its reads) rather than lingering re-activatable for 30 days (H2).
    """
    rt_hash = _hash(refresh_token)
    with _lock(data_dir, _GRANTS_FILE):
        grants = _read(data_dir, _GRANTS_FILE)
        rec = grants.get(rt_hash)
        if not isinstance(rec, dict) or rec.get("kind") != "refresh":
            return None
        grants.pop(rt_hash, None)  # rotate: old refresh is invalidated
        _write(data_dir, _GRANTS_FILE, grants)
    if rec["expires_at"] < _now() or rec["client_id"] != client_id:
        return None
    if is_entitled is not None and not is_entitled(str(rec["user_id"])):
        return None
    # Rotation must PRESERVE the audience the grant was made for; re-reading env here would
    # silently re-point a long-lived refresh chain at a different resource (#1979).
    return _issue_tokens(
        data_dir,
        user_id=str(rec["user_id"]),
        client_id=client_id,
        scope=str(rec["scope"]),
        aud=str(rec.get("aud") or "") or None,
    )


def verify_access_token(data_dir: Path, token: str) -> dict[str, Any] | None:
    """Resolve an OAuth access token → ``{user_id, scope, aud}`` (unexpired), else None."""
    if not token:
        return None
    rec = _read(data_dir, _GRANTS_FILE).get(_hash(token))
    if not isinstance(rec, dict) or rec.get("kind") != "access":
        return None
    if int(rec.get("expires_at", 0)) < _now():
        return None
    # The one place every authenticated MCP request passes through, so the one place "this
    # connection is alive" can be observed (#2004 item 14). Coalesced to one write per client per
    # five minutes; see `record_client_use`.
    record_client_use(
        data_dir, user_id=str(rec["user_id"]), client_id=str(rec.get("client_id") or "")
    )
    return {
        "user_id": str(rec["user_id"]),
        "scope": str(rec.get("scope", _SCOPE)),
        "aud": str(rec.get("aud", "")),
    }
