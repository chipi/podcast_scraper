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
from typing import Any

from filelock import FileLock

from podcast_scraper.server.atomic_write import atomic_write_text

_LOCK_TIMEOUT_S = 5.0
_CLIENTS_FILE = "oauth_clients.json"
_GRANTS_FILE = "oauth_grants.json"  # auth codes + access/refresh tokens, keyed by hash

_CODE_TTL_S = 60  # authorization codes are single-use + short-lived
_ACCESS_TTL_S = 3600  # 1h access tokens
_REFRESH_TTL_S = 30 * 86400  # 30d refresh tokens
_SCOPE = "mcp:read"

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
        clients[client_id] = client
        _write(data_dir, _CLIENTS_FILE, clients)
    return client


def get_client(data_dir: Path, client_id: str) -> dict[str, Any] | None:
    client = _read(data_dir, _CLIENTS_FILE).get(client_id)
    return client if isinstance(client, dict) else None


# --- Authorization codes (PKCE) ---


def create_authorization_code(
    data_dir: Path,
    *,
    user_id: str,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    scope: str = _SCOPE,
) -> str:
    """Mint a single-use, short-lived authorization code bound to the user + PKCE challenge."""
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
            "expires_at": _now() + _CODE_TTL_S,
        }
        _write(data_dir, _GRANTS_FILE, grants)
    return code


def _pkce_ok(verifier: str, challenge: str) -> bool:
    """S256: base64url(sha256(verifier)) == challenge (no padding)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(computed, challenge)


def _issue_tokens(data_dir: Path, *, user_id: str, client_id: str, scope: str) -> dict[str, Any]:
    access = _ACCESS_PREFIX + secrets.token_urlsafe(32)
    refresh = _REFRESH_PREFIX + secrets.token_urlsafe(32)
    now = _now()
    with _lock(data_dir, _GRANTS_FILE):
        grants = _read(data_dir, _GRANTS_FILE)
        grants[_hash(access)] = {
            "kind": "access",
            "user_id": user_id,
            "client_id": client_id,
            "scope": scope,
            "expires_at": now + _ACCESS_TTL_S,
        }
        grants[_hash(refresh)] = {
            "kind": "refresh",
            "user_id": user_id,
            "client_id": client_id,
            "scope": scope,
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
) -> dict[str, Any] | None:
    """Verify code + PKCE + client/redirect binding, consume, issue tokens (None on fail)."""
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
    return _issue_tokens(
        data_dir, user_id=str(rec["user_id"]), client_id=client_id, scope=str(rec["scope"])
    )


def refresh_access_token(
    data_dir: Path, *, refresh_token: str, client_id: str
) -> dict[str, Any] | None:
    """Rotate a refresh token → new access+refresh. Consumes the old refresh. None on fail."""
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
    return _issue_tokens(
        data_dir, user_id=str(rec["user_id"]), client_id=client_id, scope=str(rec["scope"])
    )


def verify_access_token(data_dir: Path, token: str) -> dict[str, Any] | None:
    """Resolve an OAuth access token → ``{user_id, scope}`` (unexpired), else None."""
    if not token:
        return None
    rec = _read(data_dir, _GRANTS_FILE).get(_hash(token))
    if not isinstance(rec, dict) or rec.get("kind") != "access":
        return None
    if int(rec.get("expires_at", 0)) < _now():
        return None
    return {"user_id": str(rec["user_id"]), "scope": str(rec.get("scope", _SCOPE))}
