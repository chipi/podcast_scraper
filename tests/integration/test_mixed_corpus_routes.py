"""Every GET route of the API answers the same over a mixed corpus as over its podcasts alone
(core release §3; ontology v3 "Validation": a mixed corpus read by every route that reads the
graph).

The mixed corpus is the guards' (``test_mixed_corpus_guards.corpora``: the podcast validation
corpus plus a text batch in ``sources/`` naming a podcast guest, a person only it names and a
podcast topic). The baseline is the same tree with ``sources/`` removed, so both carry the root
stamps the batch wrote (manifest, run summary, produced-by) and differ only in the other type's
items. Each corpus builds a search index (the real indexer, a fake embedder); the two must hold
the same rows, and then share one (see ``baseline_and_mixed``). Each route the app registers is
called on both, signed in, with the parameters a route requires filled from ids both hold (about
130 answer 200). The answers are compared with the corpus path and ISO times masked.
"""

from __future__ import annotations

import hashlib
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

_RUN = "feeds/p01/run_20260101-000000"

#: Values for the path and required query parameters: ids both corpora hold (the validation
#: corpus's p01_e01 and its guest). ``slug`` is added at run time from the app's own episode list.
#: ``ref`` is added at run time too (an artwork ref the app hands out); ``path`` is each side's own
#: corpus root. Left out, as they read no corpus, so those routes answer 422 on both sides: auth
#: and OAuth flow parameters, an unsubscribe token, the outbox ``channel``.
_PARAMS: Dict[str, str] = {
    "q": "trail building",
    "query": "trail building",
    "topic": "topic:trail-building",
    "topic_id": "topic:trail-building",
    "subject": "topic:trail-building",
    "person": "person:liam-verbeek",
    "person_id": "person:liam-verbeek",
    "entity": "person:liam-verbeek",
    "entity_id": "person:liam-verbeek",
    "id": "person:liam-verbeek",
    "node_id": "person:liam-verbeek",
    "kind": "person",
    "feed_id": "p01",
    "podcast": "p01",
    "episode": "ep-db8f44d2e07a22bd",
    "insight": "insight:bf9b3bf4a21ee6f8",
    "metadata_relpath": f"{_RUN}/metadata/p01_e01.metadata.json",
    "relpath": f"{_RUN}/transcripts/p01_e01.txt",
    "enricher_id": "insight_sentiment",
    "path": "{CORPUS}",
}

_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")


_EMBED_DIM = 8


def _fake_encode(text: Any, *_args: Any, **_kwargs: Any) -> List[float]:
    """A fixed, model-free embedding (no download): the same text gives the same vector."""
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:_EMBED_DIM]]


def _index_rows(root: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Every row of the corpus's search index, by tier and id, without its vector."""
    from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend

    backend = LanceDBBackend(str(root / "search" / "lance_index"))
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    for tier in backend.TABLES:
        table = backend._open_if_exists(tier)
        if table is None:
            continue
        frame = table.to_pandas()
        cols = [c for c in frame.columns if c not in ("vector", "embedding")]
        for rec in frame[cols].to_dict("records"):
            rows[(tier, str(rec.get("id")))] = {k: str(v) for k, v in rec.items()}
    return rows


@pytest.fixture(scope="module")
def baseline_and_mixed(
    corpora: Tuple[Path, Path],  # noqa: F811
) -> Iterator[Tuple[Path, Path]]:
    """The two corpora with a search index, built by the real indexer over a fake embedder
    (patched for the whole module, so search queries embed the same way).

    Two builds over identical rows still rank ties differently (shown with two podcast-only
    copies, 2026-11-12), so the index is checked at build time and shared at query time: each
    corpus builds its own and the two must hold the same rows; then the mixed corpus serves the
    baseline's, so the routes compare what the corpus adds at query time, on one index.
    """
    from podcast_scraper.providers.ml import embedding_loader
    from podcast_scraper.search.two_tier_indexer import build_two_tier_index

    _podcast, mixed = corpora
    baseline = mixed.parent / "baseline"
    shutil.copytree(mixed, baseline, dirs_exist_ok=False)
    shutil.rmtree(baseline / "sources")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(embedding_loader, "encode", _fake_encode)
        for root in (baseline, mixed):
            stats = build_two_tier_index(root, root / "search" / "lance_index", drop_existing=True)
            assert stats.episodes > 50, stats
        built = _index_rows(baseline)
        assert len(built) > 1000, len(built)
        assert _index_rows(mixed) == built, "the mixed corpus's index holds other rows"
        shutil.rmtree(mixed / "search" / "lance_index")
        shutil.copytree(baseline / "search" / "lance_index", mixed / "search" / "lance_index")
        yield baseline, mixed


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


def _url(path: str, route: APIRoute, params: Dict[str, str]) -> Optional[str]:
    for name in re.findall(r"{(\w+)(?::path)?}", path):
        if name not in params:
            return None
        path = re.sub(r"{%s(?::path)?}" % name, params[name], path)
    query = [
        f"{p.name}={params[p.name]}"
        for p in route.dependant.query_params
        if p.field_info.is_required() and p.name in params
    ]
    return path + ("?" + "&".join(query) if query else "")


def _slug(client: TestClient) -> str:
    """A podcast episode's slug, from the app's own episode list (its first entry)."""
    items = client.get("/api/app/episodes").json()["items"]
    assert items, "the episode list is empty"
    return str(items[0]["slug"])


def _artwork_ref(client: TestClient) -> str:
    """An artwork ``ref`` the app itself hands out, from its podcast list."""
    from urllib.parse import parse_qs, urlsplit

    for item in client.get("/api/app/podcasts").json()["items"]:
        url = str(item.get("artwork_url") or "")
        ref = parse_qs(urlsplit(url).query).get("ref")
        if ref:
            return ref[0]
    raise AssertionError("no podcast has an artwork ref")


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
    # The app's own slug and artwork ref; ``ref`` here is an artwork's (``collections/containing``
    # takes a person id with ``kind=person``, and gets the same value only by name).
    params = {
        **_PARAMS,
        "slug": _slug(clients["baseline"]),
        "ref": _artwork_ref(clients["baseline"]),
    }
    compared: List[str] = []
    differ: List[str] = []
    for path, route in _get_routes(baseline):
        url = _url(path, route, params)
        if url is None:
            continue
        # ``path`` is the corpus root itself (``/api/artifacts``): each side is asked for its own.
        b_url = url.replace("{CORPUS}", str(baseline))
        m_url = url.replace("{CORPUS}", str(mixed))
        b = _masked(clients["baseline"].get(b_url), (baseline, mixed))
        m = _masked(clients["mixed"].get(m_url), (baseline, mixed))
        compared.append(f"{b[0]} {url}")
        if b != m:
            at = next(
                (i for i, (x, y) in enumerate(zip(b[1], m[1])) if x != y), min(len(b[1]), len(m[1]))
            )
            lo = max(0, at - 150)
            differ.append(
                f"{url} (status {b[0]}/{m[0]}, first difference at {at}):\n"
                f"   baseline …{b[1][lo:at + 250]!r}\n   mixed    …{m[1][lo:at + 250]!r}"
            )
    print("\n".join(compared))
    assert not differ, "\n".join(differ)
    ok = [c for c in compared if c.startswith("200 ")]
    assert len(ok) > 130, f"only {len(ok)} routes answered 200: {compared}"
