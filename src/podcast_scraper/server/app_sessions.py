"""Stateless HMAC-signed cookie sessions for the consumer platform (#1063, RFC-098 §2).

A minimal signed cookie carrying the user id — no server-side session store and no new
dependency (stdlib ``hmac``/``hashlib``/``base64``). Tamper-evident: a wrong or invalid
signature yields no session. The payload is signed, **not encrypted** — store only
non-sensitive ids (the user id), never secrets.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

SESSION_COOKIE = "lp_session"
STATE_COOKIE = "lp_oauth_state"
DEFAULT_MAX_AGE = 30 * 24 * 3600  # 30 days


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sig(body: str, secret: str) -> str:
    return _b64e(hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())


def sign(payload: dict[str, Any], secret: str) -> str:
    """Return ``{base64(json)}.{hmac}`` for ``payload`` signed with ``secret``."""
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{body}.{_sig(body, secret)}"


def verify(
    token: str | None, secret: str, *, max_age: int = DEFAULT_MAX_AGE
) -> dict[str, Any] | None:
    """Return the payload if ``token`` is validly signed and unexpired, else ``None``."""
    if not token or not secret or "." not in token:
        return None
    body, _, sig = token.partition(".")
    if not hmac.compare_digest(sig, _sig(body, secret)):
        return None
    try:
        payload = json.loads(_b64d(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    iat = payload.get("iat")
    if max_age > 0 and isinstance(iat, (int, float)) and (time.time() - float(iat)) > max_age:
        return None
    return payload
