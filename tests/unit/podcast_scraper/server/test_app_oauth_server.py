"""Unit tests for the MCP OAuth 2.1 authorization server store (RFC-112 slice 3, #1471)."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from podcast_scraper.server import app_oauth_server as oa

pytestmark = pytest.mark.unit

_UID = "u_0123456789abcdef01234567"
_REDIRECT = "https://claude.ai/api/mcp/callback"


def _pkce() -> tuple[str, str]:
    verifier = "verifier-" + "a" * 43
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )
    return verifier, challenge


def test_register_client_requires_https_or_loopback(tmp_path: Path) -> None:
    c = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="claude.ai")
    assert c["client_id"].startswith("mcpc_")
    assert c["token_endpoint_auth_method"] == "none"
    with pytest.raises(ValueError):
        oa.register_client(tmp_path, redirect_uris=["http://evil.example/cb"], client_name="x")
    # loopback is allowed (RFC 8252)
    lb = oa.register_client(tmp_path, redirect_uris=["http://127.0.0.1:1234/cb"], client_name="cli")
    assert lb["redirect_uris"] == ["http://127.0.0.1:1234/cb"]


def test_full_code_exchange_with_pkce(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    tokens = oa.exchange_authorization_code(
        tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    assert tokens is not None
    assert tokens["access_token"].startswith("clp_mcpat_")
    assert tokens["refresh_token"].startswith("clp_mcprt_")
    assert tokens["token_type"] == "Bearer" and tokens["scope"] == "mcp:read"
    # the access token verifies to the user
    assert oa.verify_access_token(tmp_path, tokens["access_token"]) == {
        "user_id": _UID,
        "scope": "mcp:read",
    }


def test_code_is_single_use(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    first = oa.exchange_authorization_code(
        tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    second = oa.exchange_authorization_code(
        tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    assert first is not None and second is None  # consumed


def test_wrong_pkce_verifier_rejected(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    _verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    assert (
        oa.exchange_authorization_code(
            tmp_path,
            code=code,
            code_verifier="wrong-verifier",
            client_id=cid,
            redirect_uri=_REDIRECT,
        )
        is None
    )


def test_redirect_uri_binding_enforced(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    assert (
        oa.exchange_authorization_code(
            tmp_path,
            code=code,
            code_verifier=verifier,
            client_id=cid,
            redirect_uri="https://evil.example/cb",
        )
        is None
    )


def test_refresh_rotates(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    t1 = oa.exchange_authorization_code(
        tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    assert t1 is not None
    t2 = oa.refresh_access_token(tmp_path, refresh_token=t1["refresh_token"], client_id=cid)
    assert t2 is not None and t2["access_token"] != t1["access_token"]
    # old refresh is now invalid (rotated)
    assert (
        oa.refresh_access_token(tmp_path, refresh_token=t1["refresh_token"], client_id=cid) is None
    )


def test_verify_unknown_or_non_access(tmp_path: Path) -> None:
    assert oa.verify_access_token(tmp_path, "clp_mcpat_nope") is None
    assert oa.verify_access_token(tmp_path, "") is None
