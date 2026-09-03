"""Unit tests for the remote MCP auth (RFC-112 slice 2, #1471)."""

from __future__ import annotations

import pytest

from podcast_scraper.mcp import auth

pytestmark = pytest.mark.unit


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_urlopen(monkeypatch, payload: bytes | None, *, raise_exc: Exception | None = None):
    def _open(req, timeout=5.0):
        if raise_exc is not None:
            raise raise_exc
        return _FakeResp(payload or b"")

    monkeypatch.setattr(auth.urllib.request, "urlopen", _open)


def _config(monkeypatch) -> None:
    monkeypatch.setenv("APP_MCP_VERIFY_URL", "http://app/internal/mcp/verify")
    monkeypatch.setenv("INTERNAL_MCP_TOKEN", "tok")


def test_verify_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    assert auth.verify_bearer("") is None


def test_verify_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_VERIFY_URL", raising=False)
    monkeypatch.delenv("INTERNAL_MCP_TOKEN", raising=False)
    assert auth.verify_bearer("clp_mcp_x") is None  # fails closed


def test_verify_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": true}')
    # Returns identity AND the granted scopes now (#1916): the app has always sent `scope`, and
    # this server used to discard it — so every tool ran on the entitlement alone.
    assert auth.verify_bearer("clp_mcp_x") == ("u_1", frozenset())


def test_verify_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": false}')
    assert auth.verify_bearer("clp_mcp_x") is None


def test_verify_no_entitlement(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": false}')
    assert auth.verify_bearer("clp_mcp_x") is None


def test_verify_audience_mismatch_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://ours")

    def _resp(aud: str) -> bytes:
        return (
            '{"authenticated": true, "user_id": "u_1", "mcp_access": true, "aud": "%s"}' % aud
        ).encode()

    # A token aud-bound to a DIFFERENT resource must not be replayable here.
    _stub_urlopen(monkeypatch, _resp("https://other"))
    assert auth.verify_bearer("clp_mcpat_x") is None
    # matching aud (and an empty-aud PAT) are accepted
    _stub_urlopen(monkeypatch, _resp("https://ours"))
    assert auth.verify_bearer("clp_mcpat_x") == ("u_1", frozenset())
    _stub_urlopen(monkeypatch, _resp(""))
    assert auth.verify_bearer("clp_mcp_pat") == ("u_1", frozenset())


def test_verify_audience_bound_but_no_resource_configured_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config(monkeypatch)
    monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)  # this server can't identify itself
    # An aud-BOUND token must be rejected when we have no resource to check against (M1) …
    _stub_urlopen(
        monkeypatch,
        b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "aud": "https://x"}',
    )
    assert auth.verify_bearer("clp_mcpat_x") is None
    # … but an unbound PAT (empty aud) still works.
    _stub_urlopen(
        monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "aud": ""}'
    )
    assert auth.verify_bearer("clp_mcp_pat") == ("u_1", frozenset())


def test_verify_network_error_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, None, raise_exc=OSError("boom"))
    assert auth.verify_bearer("clp_mcp_x") is None


# --- ASGI middleware ---


def _run_asgi(app, headers: list[tuple[bytes, bytes]]):
    """Drive an ASGI app once for an http scope; return (status, captured_user)."""
    import asyncio

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    sent: list[dict] = []
    captured: dict[str, str | None] = {"user": "sentinel"}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    async def inner(s, r, sd):
        captured["user"] = auth.current_mcp_user.get()
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    mw = auth.McpAuthMiddleware(inner, verifier=lambda t: "u_1" if t == "good" else None)
    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, captured["user"]


def test_middleware_401_without_token() -> None:
    status, user = _run_asgi(_app_unused(), [])
    assert status == 401 and user == "sentinel"  # inner never ran


def test_middleware_401_bad_token() -> None:
    status, _ = _run_asgi(_app_unused(), [(b"authorization", b"Bearer bad")])
    assert status == 401


def test_middleware_passes_and_sets_user() -> None:
    status, user = _run_asgi(_app_unused(), [(b"authorization", b"Bearer good")])
    assert status == 200 and user == "u_1"


def _app_unused():
    return None  # the middleware's inner is provided inside _run_asgi


# --- RFC 9728 protected-resource discovery ---


def _drive(app, scope: dict) -> tuple[int, dict[bytes, bytes], bytes]:
    """Drive an ASGI app once; return (status, headers-dict, body)."""
    import asyncio

    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = {k: v for k, v in start.get("headers", [])}
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], headers, body


def _mw(verifier=lambda t: None):
    async def inner(s, r, sd):
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    return auth.McpAuthMiddleware(inner, verifier=verifier)


def test_protected_resource_metadata_public(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com")
    monkeypatch.setenv("APP_MCP_ISSUER_URL", "https://app.example.com")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/oauth-protected-resource",
        "headers": [],
    }
    status, _headers, body = _drive(_mw(), scope)  # no bearer → still served
    assert status == 200
    import json

    meta = json.loads(body)
    assert meta["resource"] == "https://mcp.example.com"
    assert meta["authorization_servers"] == ["https://app.example.com"]


def test_protected_resource_metadata_503_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_MCP_RESOURCE_URL", raising=False)
    monkeypatch.delenv("APP_MCP_ISSUER_URL", raising=False)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/.well-known/oauth-protected-resource",
        "headers": [],
    }
    status, _headers, _body = _drive(_mw(), scope)
    assert status == 503


def test_401_carries_resource_metadata_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_RESOURCE_URL", "https://mcp.example.com")
    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": []}
    status, headers, _body = _drive(_mw(), scope)
    assert status == 401
    www = headers[b"www-authenticate"].decode()
    assert 'resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"' in www


def test_origin_guard_blocks_disallowed_browser_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_MCP_ALLOWED_ORIGINS", "https://claude.ai")
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"origin", b"https://evil.example"), (b"authorization", b"Bearer good")],
    }
    status, _h, _b = _drive(_mw(lambda t: "u_1"), scope)  # even a valid bearer is rejected
    assert status == 403


def test_origin_guard_allows_listed_origin_and_serverside_no_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_MCP_ALLOWED_ORIGINS", "https://claude.ai")
    listed = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"origin", b"https://claude.ai"), (b"authorization", b"Bearer good")],
    }
    no_origin = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer good")],  # server-to-server sends no Origin
    }
    assert _drive(_mw(lambda t: "u_1"), listed)[0] == 200
    assert _drive(_mw(lambda t: "u_1"), no_origin)[0] == 200


# --- scope enforcement (#1916) -------------------------------------------------
#
# Scope was plumbed end to end and thrown away: the app's verify endpoint has always returned the
# token's granted `scope`, and this module never read it. Every tool ran on the `mcp_access`
# entitlement alone — including `reenrich` and `reindex`, which are corpus WRITES.


def test_granted_scopes_are_parsed_from_the_verify_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config(monkeypatch)
    _stub_urlopen(
        monkeypatch,
        b'{"authenticated": true, "user_id": "u_1", "mcp_access": true, "scope": "mcp:read"}',
    )
    assert auth.verify_bearer("clp_mcp_x") == ("u_1", frozenset({"mcp:read"}))


def test_scope_parsing_tolerates_the_shapes_a_client_might_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OAuth scope is space-delimited (RFC 6749 §3.3); commas and junk must not become a scope named
    # "mcp:read,mcp:write" that matches nothing and silently denies everything.
    _config(monkeypatch)
    _stub_urlopen(
        monkeypatch,
        b'{"authenticated": true, "user_id": "u_1", "mcp_access": true,'
        b' "scope": " mcp:read,  mcp:write "}',
    )
    _, scopes = auth.verify_bearer("clp_mcp_x")  # type: ignore[misc]
    assert scopes == frozenset({"mcp:read", "mcp:write"})


def test_a_missing_scope_authenticates_but_authorises_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _config(monkeypatch)
    _stub_urlopen(monkeypatch, b'{"authenticated": true, "user_id": "u_1", "mcp_access": true}')
    _, scopes = auth.verify_bearer("clp_mcp_x")  # type: ignore[misc]
    assert scopes == frozenset()


def test_require_scope_refuses_when_the_token_did_not_grant_it() -> None:
    token = auth.current_mcp_scopes.set(frozenset({"mcp:read"}))
    try:
        auth.require_scope(auth.SCOPE_READ)  # granted → passes
        with pytest.raises(auth.McpScopeError) as exc:
            auth.require_scope(auth.SCOPE_WRITE)
        # The message names what was needed and what was presented — an agent has to be able to
        # tell "I need a different token" from "this tool is broken".
        assert "mcp:write" in str(exc.value)
    finally:
        auth.current_mcp_scopes.reset(token)


def test_an_http_request_with_no_scopes_is_the_CLOSED_state() -> None:
    token = auth.current_mcp_scopes.set(frozenset())
    try:
        with pytest.raises(auth.McpScopeError):
            auth.require_scope(auth.SCOPE_READ)
    finally:
        auth.current_mcp_scopes.reset(token)


def test_an_undeclared_transport_with_no_scopes_FAILS_CLOSED() -> None:
    """The fail-open default, closed (advisor 2.3).

    This test used to assert that an unset contextvar passes — which is the default in ANY fresh
    context, including one produced by an HTTP propagation break. It restated the default rather
    than testing anything. Absence of an authorisation context is now a refusal.
    """
    auth.__reset_transport_trust()
    assert auth.current_mcp_scopes.get() is None
    with pytest.raises(auth.McpScopeError):
        auth.require_scope(auth.SCOPE_WRITE)


def test_stdio_passes_only_once_it_has_DECLARED_itself_stdio() -> None:
    auth.__reset_transport_trust()
    try:
        auth.mark_stdio_transport()
        auth.require_scope(auth.SCOPE_WRITE)  # must not raise
    finally:
        auth.__reset_transport_trust()


def test_the_middleware_binds_the_session_to_the_credential() -> None:
    """The SDK refuses a session driven by a different credential — but only when we give it a
    principal. We never set `scope["user"]`, so its check compared None against None and any
    holder of any valid token who learned a session id could drive it (advisor 2.3).
    """
    scope: dict = {"type": "http", "method": "POST", "path": "/mcp", "headers": []}

    import asyncio

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        pass

    async def inner(s, r, sd):
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    scope["headers"] = [(b"authorization", b"Bearer good")]
    mw = auth.McpAuthMiddleware(inner, verifier=lambda t: ("u_1", frozenset({"mcp:read"})))
    asyncio.run(mw(scope, receive, send))

    principal = scope.get("user")
    assert principal is not None, "no principal set — the SDK session binding stays inert"
    # `subject` is what the binding compares, and it must be the USER: two users must never be
    # able to share a session, whatever client they connected with.
    assert principal.access_token.subject == "u_1"


def _run_with_verifier(verifier, headers):
    """Drive the middleware and report (status, user, scopes) as the inner app saw them."""
    import asyncio

    scope = {"type": "http", "method": "POST", "path": "/mcp", "headers": headers}
    sent: list[dict] = []
    seen: dict = {"user": "sentinel", "scopes": "sentinel"}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    async def inner(s, r, sd):
        seen["user"] = auth.current_mcp_user.get()
        seen["scopes"] = auth.current_mcp_scopes.get()
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    mw = auth.McpAuthMiddleware(inner, verifier=verifier)
    asyncio.run(mw(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, seen["user"], seen["scopes"]


def test_middleware_puts_the_granted_scopes_where_tools_can_see_them() -> None:
    status, user, scopes = _run_with_verifier(
        lambda t: ("u_1", frozenset({"mcp:read"})) if t == "good" else None,
        [(b"authorization", b"Bearer good")],
    )
    assert status == 200 and user == "u_1"
    assert scopes == frozenset({"mcp:read"})


def test_a_verifier_returning_a_bare_user_id_authorises_NOTHING() -> None:
    """Back-compat that fails CLOSED.

    A verifier that only reports identity (an older injected double, or any caller that does not
    care about scope) must not accidentally grant everything. It authenticates, and every scoped
    tool then refuses.
    """
    status, user, scopes = _run_with_verifier(
        lambda t: "u_1" if t == "good" else None, [(b"authorization", b"Bearer good")]
    )
    assert status == 200 and user == "u_1"
    assert scopes == frozenset()


def test_scopes_do_not_leak_out_of_the_request() -> None:
    # The contextvar is reset in a finally, so a later stdio call cannot inherit an HTTP request's
    # scopes — which would be an authorisation leak between transports.
    _run_with_verifier(
        lambda t: ("u_1", frozenset({"mcp:write"})), [(b"authorization", b"Bearer good")]
    )
    assert auth.current_mcp_scopes.get() is None


def test_run_stdio_declares_local_trust_before_serving(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_stdio` must mark the transport, or every scoped tool refuses on stdio.

    This exists because the marker was added to `run_stdio` without importing it: the function
    would have raised NameError on the first local run, and the whole unit suite still passed —
    nothing exercised the entrypoint, only the helper it calls. Lint caught it; a test should.
    """
    from podcast_scraper.mcp import server as mcp_server

    auth.__reset_transport_trust()
    served: list[str] = []

    class _FakeServer:
        def run(self) -> None:
            # Trust must already be declared by the time the server actually serves.
            served.append("ran")
            # Passes by NOT raising — require_scope returns None by design.
            auth.require_scope(auth.SCOPE_WRITE)

    monkeypatch.setattr(mcp_server, "build_server", lambda _dir: _FakeServer())
    try:
        mcp_server.run_stdio("/tmp/does-not-matter")
        assert served == ["ran"]
    finally:
        auth.__reset_transport_trust()


def test_a_missing_SDK_binding_refuses_the_request_rather_than_unbinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one seam in the hijack fix that could fail OPEN (advisor-2, low).

    `_principal` returning None leaves `scope["user"]` unset, which is the exact state that made
    the SDK compare None to None and let any valid token drive another user's session. If the SDK
    renames those types, the transport must FAIL — quietly not binding is how the hole reopens.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name.startswith("mcp.server.auth"):
            raise ImportError("simulated SDK rename")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(auth.McpSessionBindingUnavailable):
        auth._principal("u_1", frozenset({"mcp:read"}))


def test_the_binding_failure_is_not_silently_swallowed_by_the_middleware() -> None:
    """A 500 is the correct outcome — never a 200 with an unbound session."""
    import asyncio

    scope: dict = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"authorization", b"Bearer good")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        pass

    async def inner(s, r, sd):
        await sd({"type": "http.response.start", "status": 200, "headers": []})
        await sd({"type": "http.response.body", "body": b"ok"})

    def _boom(_user_id, _scopes):
        raise auth.McpSessionBindingUnavailable("simulated")

    original = auth._principal
    auth._principal = _boom  # type: ignore[assignment]
    try:
        mw = auth.McpAuthMiddleware(inner, verifier=lambda t: ("u_1", frozenset({"mcp:read"})))
        with pytest.raises(auth.McpSessionBindingUnavailable):
            asyncio.run(mw(scope, receive, send))
    finally:
        auth._principal = original  # type: ignore[assignment]
