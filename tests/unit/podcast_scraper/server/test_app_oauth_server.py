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
    # the access token verifies to the user (aud empty — no APP_MCP_RESOURCE_URL configured here)
    assert oa.verify_access_token(tmp_path, tokens["access_token"]) == {
        "user_id": _UID,
        "scope": "mcp:read",
        "aud": "",
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


def test_tokens_bind_audience_when_resource_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com/")  # trailing slash trimmed
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
    resolved = oa.verify_access_token(tmp_path, tokens["access_token"])
    assert resolved is not None and resolved["aud"] == "https://mcp.example.com"


def test_scope_support_predicate() -> None:
    assert oa.is_scope_supported("mcp:read") is True
    assert oa.is_scope_supported("mcp:admin") is False
    assert oa.is_scope_supported("") is False


def test_non_ascii_verifier_rejected(tmp_path: Path) -> None:
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    _verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    # A non-ASCII verifier must fail closed (no 500).
    assert (
        oa.exchange_authorization_code(
            tmp_path,
            code=code,
            code_verifier="vérifïer-ünicode",
            client_id=cid,
            redirect_uri=_REDIRECT,
        )
        is None
    )


def test_register_rejects_too_many_redirect_uris(tmp_path: Path) -> None:
    uris = [f"https://ex{i}.example/cb" for i in range(oa._MAX_REDIRECT_URIS + 1)]
    with pytest.raises(ValueError):
        oa.register_client(tmp_path, redirect_uris=uris, client_name="greedy")


def test_client_registration_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oa, "_MAX_CLIENTS", 2)
    oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="a")
    oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="b")
    with pytest.raises(ValueError):
        oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")  # store full


def test_expired_grants_pruned_on_issue(tmp_path: Path) -> None:
    import json

    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    # Inject a long-expired access record straight into the grants file.
    grants_path = tmp_path / oa._GRANTS_FILE
    stale_hash = oa._hash("clp_mcpat_stale")
    grants_path.write_text(
        json.dumps(
            {stale_hash: {"kind": "access", "user_id": _UID, "scope": "mcp:read", "expires_at": 1}}
        ),
        encoding="utf-8",
    )
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    oa.exchange_authorization_code(
        tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    remaining = json.loads(grants_path.read_text(encoding="utf-8"))
    assert stale_hash not in remaining  # the expired record was purged when new tokens were issued


def test_list_consents_joins_client_names(tmp_path: Path) -> None:
    c1 = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="claude.ai")
    c2 = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="Cursor")
    oa.remember_consent(tmp_path, user_id=_UID, client_id=c1["client_id"], scope="mcp:read")
    oa.remember_consent(tmp_path, user_id=_UID, client_id=c2["client_id"], scope="mcp:read")
    # another user's consent must not leak into _UID's list
    oa.remember_consent(tmp_path, user_id="u_other", client_id=c1["client_id"], scope="mcp:read")

    rows = oa.list_consents(tmp_path, _UID)
    assert {r["client_id"] for r in rows} == {c1["client_id"], c2["client_id"]}
    names = {r["client_name"] for r in rows}
    assert names == {"claude.ai", "Cursor"}
    assert all(r["scopes"] == ["mcp:read"] and r["connected_at"] > 0 for r in rows)


def test_revoke_client_grants_kills_live_tokens(tmp_path: Path) -> None:
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
    assert oa.verify_access_token(tmp_path, tokens["access_token"]) is not None  # live

    dropped = oa.revoke_client_grants(tmp_path, user_id=_UID, client_id=cid)
    assert dropped == 2  # the access + refresh grant
    # the access token no longer verifies AND the refresh token can't rotate
    assert oa.verify_access_token(tmp_path, tokens["access_token"]) is None
    assert (
        oa.refresh_access_token(tmp_path, refresh_token=tokens["refresh_token"], client_id=cid)
        is None
    )


def test_revoke_client_grants_kills_unexchanged_code(tmp_path: Path) -> None:
    # H1: a disconnect must also drop a live (un-exchanged) authorization code, else it resurrects a
    # fresh 30-day grant within the code's 60s TTL after the user clicked Disconnect.
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    assert oa.revoke_client_grants(tmp_path, user_id=_UID, client_id=cid) == 1  # the code
    assert (
        oa.exchange_authorization_code(
            tmp_path, code=code, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
        )
        is None  # the code can no longer be exchanged
    )


def test_exchange_and_refresh_reject_de_entitled_user(tmp_path: Path) -> None:
    # H2: a user whose mcp_access was pulled cannot mint or rotate tokens (the AS re-checks).
    client = oa.register_client(tmp_path, redirect_uris=[_REDIRECT], client_name="c")
    cid = client["client_id"]
    verifier, challenge = _pkce()
    code = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )

    def denied(_uid: str) -> bool:
        return False  # user is no longer entitled

    assert (
        oa.exchange_authorization_code(
            tmp_path,
            code=code,
            code_verifier=verifier,
            client_id=cid,
            redirect_uri=_REDIRECT,
            is_entitled=denied,
        )
        is None
    )
    # the entitled path still issues, but a refresh once de-entitled dies (old refresh consumed)
    code2 = oa.create_authorization_code(
        tmp_path, user_id=_UID, client_id=cid, redirect_uri=_REDIRECT, code_challenge=challenge
    )
    t = oa.exchange_authorization_code(
        tmp_path, code=code2, code_verifier=verifier, client_id=cid, redirect_uri=_REDIRECT
    )
    assert t is not None
    assert (
        oa.refresh_access_token(
            tmp_path, refresh_token=t["refresh_token"], client_id=cid, is_entitled=denied
        )
        is None
    )


def test_consent_remember_and_revoke(tmp_path: Path) -> None:
    cid = "mcpc_abc"
    assert oa.has_consent(tmp_path, user_id=_UID, client_id=cid, scope="mcp:read") is False
    oa.remember_consent(tmp_path, user_id=_UID, client_id=cid, scope="mcp:read")
    assert oa.has_consent(tmp_path, user_id=_UID, client_id=cid, scope="mcp:read") is True
    # scoped: a different scope is NOT implicitly consented
    assert oa.has_consent(tmp_path, user_id=_UID, client_id=cid, scope="mcp:write") is False
    # scoped: a different user is NOT consented
    assert oa.has_consent(tmp_path, user_id="u_other", client_id=cid, scope="mcp:read") is False
    # revoke forgets all scopes for that (user, client)
    assert oa.revoke_consent(tmp_path, user_id=_UID, client_id=cid) is True
    assert oa.has_consent(tmp_path, user_id=_UID, client_id=cid, scope="mcp:read") is False
    assert oa.revoke_consent(tmp_path, user_id=_UID, client_id=cid) is False  # nothing left


def _challenge(verifier: str) -> str:
    return (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )


def test_pkce_verifier_length_enforced() -> None:
    """RFC 7636 §4.1: reject a verifier outside 43–128, even if the challenge matches (MED)."""
    short = "abc"  # < 43
    assert oa._pkce_ok(short, _challenge(short)) is False
    valid = "v" * 50  # 43–128
    assert oa._pkce_ok(valid, _challenge(valid)) is True
    toolong = "v" * 200  # > 128
    assert oa._pkce_ok(toolong, _challenge(toolong)) is False


def test_prune_stale_clients() -> None:
    """Old clients with no live grant are reclaimed; granted/recent ones kept (advisor LOW)."""
    now = oa._now()
    old = now - oa._CLIENT_UNUSED_TTL_S - 10
    clients = {
        "old_unused": {"client_id": "old_unused", "created_at": old},
        "old_active": {"client_id": "old_active", "created_at": old},
        "recent": {"client_id": "recent", "created_at": now - 5},
    }
    grants = {"h1": {"kind": "refresh", "client_id": "old_active", "expires_at": now + 3600}}
    kept = oa._prune_stale_clients(clients, grants)
    assert set(kept) == {"old_active", "recent"}  # old_unused pruned


def test_prune_ignores_expired_grant_when_keeping_client() -> None:
    """A client whose only grant has EXPIRED is treated as un-granted → pruned if old."""
    now = oa._now()
    clients = {"c": {"client_id": "c", "created_at": now - oa._CLIENT_UNUSED_TTL_S - 10}}
    grants = {"h": {"kind": "access", "client_id": "c", "expires_at": now - 1}}  # expired
    assert oa._prune_stale_clients(clients, grants) == {}
