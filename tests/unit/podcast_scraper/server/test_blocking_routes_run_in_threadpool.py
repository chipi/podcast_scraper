"""Routes whose work is synchronous file IO must be plain ``def`` so FastAPI runs them in its
threadpool. As ``async def`` with no ``await`` they run ON the event loop and stall every other
request for their duration (prod, 2026-10-03: the corpus digest's cold build froze the api;
/api/usage held /api/health at 1.08 s instead of 0.008 s).
"""

from __future__ import annotations

import inspect

import pytest

from podcast_scraper.server.routes.corpus_digest import corpus_digest
from podcast_scraper.server.routes.usage import usage

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("handler", [corpus_digest, usage], ids=["corpus_digest", "usage"])
def test_a_blocking_route_is_not_a_coroutine(handler) -> None:
    assert not inspect.iscoroutinefunction(handler)


#: Handlers that stay ``async def`` on purpose: instant, no IO beyond memory, and they must answer
#: even when every worker thread is busy (a liveness probe stuck behind slow reads restarts the
#: container).
_KEEP_ASYNC = {
    ("health.py", "health"),
    ("internal_mcp.py", "verify"),
    ("mcp_oauth.py", "authorization_server_metadata"),
    ("app_comms.py", "vapid_key"),
    ("app_auth.py", "app_auth_logout"),
}


def test_no_route_handler_is_async_without_awaiting() -> None:
    """An ``async def`` route with no ``await`` runs its body on the event loop. 148 of them did
    (2026-10-03); a new one must be a plain ``def`` or be listed above with its reason."""
    import ast
    from pathlib import Path

    import podcast_scraper.server.routes as routes_pkg

    offenders = []
    for path in sorted(Path(routes_pkg.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            is_route = any(
                isinstance(d, ast.Call)
                and getattr(d.func, "attr", "") in ("get", "post", "put", "delete", "patch")
                for d in node.decorator_list
            )
            awaits = any(
                isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in ast.walk(node)
            )
            if is_route and not awaits and (path.name, node.name) not in _KEEP_ASYNC:
                offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert offenders == []
