"""Every GET route of the API answers the same over a mixed corpus as over its podcasts alone
(core release §3; ontology v3 "Validation": a mixed corpus read by every route that reads the
graph).

The mixed corpus is the guards' (``test_mixed_corpus_guards.corpora``: the podcast validation
corpus plus a text batch in ``sources/`` naming a podcast guest, a person only it names and a
podcast topic). The baseline is the same tree with ``sources/`` removed, so both carry the root
stamps the batch wrote (manifest, run summary, produced-by) and differ only in the other type's
items. Each route the app registers is called on both, signed in, with the parameters a route
requires filled from ids both hold. The answers are compared with the corpus path and ISO times
masked.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

pytest.importorskip("fastapi")

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user
from tests.integration.test_mixed_corpus_guards import corpora  # noqa: F401 - the fixture

pytestmark = [pytest.mark.integration]

#: Values for the path and required query parameters: ids both corpora hold.
_PARAMS: Dict[str, str] = {
    "q": "trail building",
    "query": "trail building",
    "topic": "topic:trail-building",
    "topic_id": "topic:trail-building",
    "person": "person:liam-verbeek",
    "person_id": "person:liam-verbeek",
    "entity": "person:liam-verbeek",
    "entity_id": "person:liam-verbeek",
    "id": "person:liam-verbeek",
    "node_id": "person:liam-verbeek",
}

_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")


@pytest.fixture(scope="module")
def baseline_and_mixed(corpora: Tuple[Path, Path]) -> Tuple[Path, Path]:  # noqa: F811
    _podcast, mixed = corpora
    baseline = mixed.parent / "baseline"
    shutil.copytree(mixed, baseline, dirs_exist_ok=False)
    shutil.rmtree(baseline / "sources")
    return baseline, mixed


def _client(root: Path, data: Path) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _iter_routes(routes: Any, prefix: str = "") -> Iterator[Tuple[str, APIRoute]]:
    """``(full path, route)`` for every route, walking into included routers: ``app.routes`` holds
    ``_IncludedRouter`` wrappers, not the routes (see ``test_login_first_matrix._iter_routes``)."""
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            ctx = getattr(route, "include_context", None)
            yield from _iter_routes(included.routes, prefix + (getattr(ctx, "prefix", "") or ""))
        elif isinstance(route, APIRoute):
            yield prefix + route.path, route


def _get_routes(root: Path) -> List[Tuple[str, APIRoute]]:
    app = create_app(root, static_dir=False)
    return sorted(
        ((p, r) for p, r in _iter_routes(app.routes) if "GET" in (r.methods or set())),
        key=lambda pr: pr[0],
    )


def _url(path: str, route: APIRoute) -> Optional[str]:
    for name in re.findall(r"{(\w+)(?::path)?}", path):
        if name not in _PARAMS:
            return None
        path = re.sub(r"{%s(?::path)?}" % name, _PARAMS[name], path)
    query = [
        f"{p.name}={_PARAMS[p.name]}"
        for p in route.dependant.query_params
        if p.field_info.is_required() and p.name in _PARAMS
    ]
    return path + ("?" + "&".join(query) if query else "")


def _masked(resp: Any, roots: Tuple[Path, ...]) -> Tuple[int, str]:
    # Both roots: the baseline is a copy of the mixed tree, so its stored documents (the manifest,
    # the run summary) name the mixed corpus's path.
    body = resp.text
    for root in roots:
        body = body.replace(str(root.resolve()), "<ROOT>").replace(str(root), "<ROOT>")
    return resp.status_code, _ISO_TIME.sub("<TIME>", body)


def test_every_get_route_answers_the_same(
    baseline_and_mixed: Tuple[Path, Path], tmp_path: Path
) -> None:
    baseline, mixed = baseline_and_mixed
    clients = {
        "baseline": _client(baseline, tmp_path / "baseline-data"),
        "mixed": _client(mixed, tmp_path / "mixed-data"),
    }
    compared: List[str] = []
    differ: List[str] = []
    for path, route in _get_routes(baseline):
        url = _url(path, route)
        if url is None:
            continue
        b = _masked(clients["baseline"].get(url), (baseline, mixed))
        m = _masked(clients["mixed"].get(url), (baseline, mixed))
        compared.append(f"{b[0]} {url}")
        if b != m:
            differ.append(f"{url}: baseline {b[0]} {b[1][:400]!r}\n   mixed {m[0]} {m[1][:400]!r}")
    print("\n".join(compared))
    assert not differ, "\n".join(differ)
    ok = [c for c in compared if c.startswith("200 ")]
    assert len(ok) > 60, f"only {len(ok)} routes answered 200: {compared}"
