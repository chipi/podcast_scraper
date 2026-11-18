"""No route serves a news body (RFC-134 §10), checked across every read route, not only the news ones.

Every article body in the app validation corpus holds ``NEWS-BODY-MARKER-<n>``. Every GET route in
the app's OpenAPI schema is called with each required parameter filled from :data:`_VALUES` (news
item ids, news file paths, the topic and person news shares with the podcasts, queries that match
the body text and the marker itself), plus once with ``content_types=news_article`` where a route
takes it; the read-only POST routes are called with news-shaped bodies. No response may contain the
marker, nor the withheld item's title.

A GET route whose required parameter has no entry in :data:`_VALUES` fails the test, so a new
route cannot escape the guard. Not called: the routes that change state (admin, auth, preferences,
profile, graph events, deletes, the outbox, feed edits); they answer with what was sent or a
status, not with corpus text. Public has no MCP server; the MCP tools live with the private
extension and need their own guard there.
"""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest

pytest.importorskip("lancedb")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.hybrid_search import lance_index_dir  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from tests.integration.conftest import requires  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus"
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_MARKER = "NEWS-BODY-MARKER"
_WITHHELD = "WITHHELD-ITEM-TITLE"
_TOPIC = "topic:risk-management"
_PERSON = "person:skanda-amarnath"
_SOURCE = "source:125361805bbcfcb84b90"
_ITEM = "item:news_article:a8d6e89ce92ef7521ed5"
_WITHHELD_ITEM = "item:news_article:2043d8230c86beddfb95"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("guard") / "corpus"
    shutil.copytree(_FIXTURE / "v3", root, ignore=shutil.ignore_patterns("lance_index"))
    tti.build_two_tier_index(root, lance_index_dir(root), model_id=_MODEL, limit_episodes=2)
    return root


@pytest.fixture()
def client(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_FIXTURE / "publishers.fixture.yaml"))
    return TestClient(create_app(corpus, static_dir=False))


def _news_files(root: Path) -> List[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in (root / "sources" / "news_article").rglob("*")
        if p.is_file()
    )


def _values(root: Path) -> Dict[str, List[str]]:
    files = _news_files(root)
    metas = [f for f in files if f.endswith(".metadata.json")]
    queries = ["Risk desks learn to sit still", f"{_MARKER}-1", "first paragraph of the article"]
    nodes = [_TOPIC, _PERSON, _SOURCE, _ITEM, _WITHHELD_ITEM, "org:daily-example"]
    return {
        "path": [str(root)],
        "relpath": files,
        "artifact_path": files,
        "metadata_relpath": metas,
        "item_id": [_ITEM, _WITHHELD_ITEM],
        "q": queries,
        "topic": [_TOPIC],
        "person": [_PERSON],
        "entity": [_PERSON, "org:daily-example"],
        "insight": ["insight:bf9b3bf4a21ee6f8"],
        "episode": [_ITEM, _WITHHELD_ITEM, "ep-db8f44d2e07a22bd"],
        "podcast": [_SOURCE, "p01"],
        "feed_id": [_SOURCE, "p01"],
        "kind": ["topic", "person", "organization", "theme", "storyline", "episode"],
        "id": nodes,
        "ident": [_TOPIC, _PERSON, "tc:risk-management", "thc:managing-risk", _ITEM],
        "route": ["topic", "person", "theme", "storyline", "episode"],
        "enricher_id": ["topic_theme_clusters", "news_connection", "temporal_velocity"],
        # Auth, outbox and per-user routes: no corpus text behind them, called so the coverage
        # check holds; any value answers with a status.
        "code": ["x"],
        "state": ["x"],
        "token": ["x"],
        "channel": ["email"],
        "session_id": ["x"],
        "user_id": ["x"],
    }


def _get_calls(client: TestClient, values: Dict[str, List[str]]) -> Iterator[Tuple[str, Dict]]:
    schema = client.app.openapi()  # type: ignore[attr-defined]
    for template, ops in schema["paths"].items():
        op = ops.get("get")
        if op is None:
            continue
        params = op.get("parameters", [])
        required = [p for p in params if p.get("required")]
        missing = [p["name"] for p in required if p["name"] not in values]
        assert not missing, f"GET {template}: no guard values for {missing}"
        names = [p["name"] for p in required]
        for combo in itertools.product(*(values[n] for n in names)):
            args = dict(zip(names, combo))
            url = template
            query: Dict[str, Any] = {}
            for p in required:
                if p["in"] == "path":
                    url = url.replace("{" + p["name"] + "}", args[p["name"]])
                else:
                    query[p["name"]] = args[p["name"]]
            yield url, query
            if any(p["name"] == "content_types" for p in params):
                yield url, {**query, "content_types": "news_article"}


def _served(text: str, sent: Any) -> bool:
    """Whether *text* holds a body marker or the withheld title, beyond echoes of what was sent
    (a search response repeats its query, which may be the marker itself)."""
    for value in sent:
        if isinstance(value, str) and value:
            text = text.replace(value, "")
    return _MARKER in text or _WITHHELD in text


@requires("sentence_transformers")
def test_no_read_route_serves_a_news_body(client: TestClient, corpus: Path) -> None:
    values = _values(corpus)
    seen: List[str] = []
    leaks: List[str] = []
    for url, query in _get_calls(client, values):
        res = client.get(url, params=query)
        seen.append(url)
        if _served(res.text, query.values()):
            leaks.append(f"GET {url} {query} -> {res.status_code}")
    posts: List[Tuple[str, Dict[str, Any]]] = [
        ("/api/corpus/node-episodes", {"node_id": n, "content_types": ["news_article"]})
        for n in values["id"] + ["tc:risk-management", "thc:managing-risk"]
    ]
    posts += [
        ("/api/corpus/resolve-episode-artifacts", {"episode_ids": values["episode"]}),
        (
            "/api/search/compare",
            {
                "subject_a": {"kind": "topic", "id": _TOPIC},
                "subject_b": {"kind": "person", "id": _PERSON},
                "q": values["q"][1],
            },
        ),
    ]
    for url, body in posts:
        res = client.post(url, json=body)
        seen.append(url)
        assert res.status_code == 200, f"POST {url} {res.status_code}: {res.text[:200]}"
        if _served(res.text, [values["q"][1]]):
            leaks.append(f"POST {url} {body} -> {res.status_code}")
    assert "/api/corpus/text-file" in seen and "/api/search" in seen
    assert not leaks, "news body or withheld item served:\n" + "\n".join(leaks)
