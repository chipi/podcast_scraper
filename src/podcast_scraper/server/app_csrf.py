"""Refuse cross-site writes made with the session cookie (CSRF), by checking the request origin.

The API authenticates browsers with the ``lp_session`` cookie, and until this check the ONLY
cross-site defence on a state-changing request was the cookie's ``SameSite=Lax``. Lax stops the
classic cross-site form POST on modern browsers, but it is a browser policy, not a server check:
it does nothing for an older browser, for a same-site-but-cross-origin page (another subdomain),
or for a future cookie change. An admin write — re-roling users, changing the access policy,
writing overrides — must not rest on it alone.

THE RULE. A request is refused (403) when ALL of these hold:

* its method changes state (POST, PUT, PATCH, DELETE);
* it carries the session cookie (whatever else it carries: the cookie is what authenticates);
* it states where it came from (``Origin``, else ``Referer``), and that origin is neither THIS
  host nor one of the trusted origins (the CORS allowlist, ``app.state.trusted_origins``).

A request that states no origin at all passes. Every browser sends ``Origin`` on a cross-site
POST, so the attack always carries one; what arrives without it is a non-browser client (curl, a
script, the test client), which a forged page cannot drive. OWASP's "verify the origin with
standard headers" recommendation, without a token round-trip the SPA and native shell would both
have to learn.

Exempt: the OAuth callback (Sign in with Apple POSTs it from appleid.apple.com by design).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app_audit import append_audit

_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Paths another site is SUPPOSED to post to. Sign in with Apple uses `response_mode=form_post`,
#: so its callback is a cross-site POST from appleid.apple.com by design.
_EXEMPT_PATHS = frozenset({"/api/app/auth/callback"})


def _origin_of(url: str) -> str:
    """``scheme://host[:port]`` of *url*, lowercased; empty for anything unparsable."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _request_origin(request: Request) -> str:
    """Where the request says it came from: ``Origin``, else the origin of ``Referer``."""
    origin = request.headers.get("origin", "")
    if origin and origin.lower() != "null":
        return _origin_of(origin)
    if origin.lower() == "null":
        # An opaque origin (sandboxed iframe, file://, some redirects) is never trusted.
        return "null"
    return _origin_of(request.headers.get("referer", ""))


def _hostname(netloc: str) -> str:
    """``host[:port]`` -> ``host``, lowercased (IPv6-safe); empty for anything unparsable."""
    try:
        return (urlsplit(f"//{netloc.strip()}").hostname or "").lower()
    except ValueError:
        return ""


def _own_hosts(request: Request) -> set[str]:
    """The host NAMES this request was addressed to, including through the reverse proxy.

    Names, not ``host:port``: nginx's ``$host`` drops the port, so behind nginx on a non-default
    port (the local app stack on :8081) the page's Origin carries ``:8081`` while Host does not,
    and every same-origin cookie write was refused. Cookies are not port-scoped, so comparing the
    port added no protection.
    """
    hosts = {_hostname(request.headers.get("host", ""))}
    forwarded = request.headers.get("x-forwarded-host", "")
    hosts.update(_hostname(h) for h in forwarded.split(",") if h.strip())
    return {h for h in hosts if h}


def is_trusted_origin(origin: str, request: Request, trusted: Sequence[str]) -> bool:
    """True when *origin* is this host or one of the trusted (CORS-allowed) origins."""
    if not origin or origin == "null":
        return False
    if origin in {_origin_of(t) for t in trusted}:
        return True
    return _hostname(urlsplit(origin).netloc) in _own_hosts(request)


def is_cross_site_cookie_write(request: Request, trusted: Sequence[str]) -> bool:
    """True for a state-changing, cookie-authenticated write from a foreign site."""
    if request.method.upper() not in _STATE_CHANGING:
        return False
    if request.url.path in _EXEMPT_PATHS:
        return False
    if app_sessions.SESSION_COOKIE not in request.cookies:
        return False
    # NO exemption for an Authorization / X-Operator-Key header once the cookie is present: the
    # cookie is checked FIRST by `get_current_user`, so it is what authenticates, and a browser
    # attaches cached Basic credentials (the /preview gate) on its own — an "any Authorization
    # header" exemption switched this check off for every preview user. A Bearer or key client
    # sends no session cookie, so it never reaches this line.
    origin = _request_origin(request)
    if not origin:
        return False
    return not is_trusted_origin(origin, request, trusted)


class CrossSiteWriteGuard(BaseHTTPMiddleware):
    """403 for a cookie-authenticated state-changing request from a foreign origin."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse a cross-site cookie write; pass everything else through untouched."""
        trusted = getattr(request.app.state, "trusted_origins", None) or []
        if is_cross_site_cookie_write(request, trusted):
            append_audit(
                getattr(request.app.state, "audit_path", None),
                {
                    "action": "cross_site_write_refused",
                    "method": request.method,
                    "path": request.url.path,
                    "origin": _request_origin(request),
                },
            )
            return JSONResponse(status_code=403, content={"detail": "Cross-site request refused."})
        return await call_next(request)
