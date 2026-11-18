"""News search (RFC-134 §9, §10) over the app validation corpus's invented news items."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.hybrid_search import lance_index_dir  # noqa: E402
from podcast_scraper.search.news_index import news_rows, search_news, TABLE  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from tests.integration.conftest import requires  # noqa: E402

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus"
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_INDEXED = {"risk-desks", "mercados", "wire-reliability"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("news") / "corpus"
    shutil.copytree(_FIXTURE / "v3", root, ignore=shutil.ignore_patterns("lance_index"))
    tti.build_two_tier_index(root, lance_index_dir(root), model_id=_MODEL, limit_episodes=2)
    return root


@pytest.fixture()
def client(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_FIXTURE / "publishers.fixture.yaml"))
    return TestClient(create_app(corpus, static_dir=False))


def _stem(row: dict) -> str:
    return Path(row["metadata_relative_path"]).name[: -len(".metadata.json")]


def test_only_full_text_shown_items_are_chunked() -> None:
    stems = {_stem(r) for r in news_rows(_FIXTURE / "v3")}
    assert stems == _INDEXED


@requires("sentence_transformers")
def test_the_news_table_holds_the_indexed_items(corpus: Path) -> None:
    import lancedb

    table = lancedb.connect(str(lance_index_dir(corpus))).open_table(TABLE)
    stems = {_stem(r) for r in table.to_arrow().to_pylist()}
    assert stems == _INDEXED


@requires("sentence_transformers")
def test_search_news_finds_the_risk_article(corpus: Path) -> None:
    hits = search_news(lance_index_dir(corpus), "risk management behaviour", _MODEL, top_k=3)
    meta = next((_FIXTURE / "v3" / "sources").rglob("risk-desks.metadata.json"))
    assert hits and hits[0][0] == json.loads(meta.read_text(encoding="utf-8"))["item_id"]


@requires("sentence_transformers")
def test_a_search_without_content_types_returns_no_news(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "risk management", "top_k": 20}).json()
    assert body["results"] and all(
        r["content_type"] == "podcast_episode" and r["news_item"] is None for r in body["results"]
    )


@requires("sentence_transformers")
def test_news_only_and_merged_searches(client: TestClient) -> None:
    news = client.get(
        "/api/search", params={"q": "risk management", "content_types": "news_article"}
    ).json()["results"]
    assert news and all(r["content_type"] == "news_article" for r in news)
    assert all(r["text"] == "" and r["source_tier"] == "news" for r in news)
    assert len({r["doc_id"] for r in news}) == len(news)
    both = client.get(
        "/api/search",
        params=[
            ("q", "risk management"),
            ("content_types", "podcast_episode"),
            ("content_types", "news_article"),
            ("top_k", "20"),
        ],
    ).json()["results"]
    assert {r["content_type"] for r in both} == {"podcast_episode", "news_article"}
    scores = [r["score"] for r in both]
    assert scores == sorted(scores, reverse=True)


@requires("sentence_transformers")
def test_no_search_response_carries_a_body(client: TestClient) -> None:
    for q in ("risk", "reliability at scale", "first paragraph of the article body", "marker"):
        text = client.get(
            "/api/search",
            params=[
                ("q", q),
                ("content_types", "podcast_episode"),
                ("content_types", "news_article"),
            ],
        ).text
        assert "NEWS-BODY-MARKER" not in text


@requires("sentence_transformers")
def test_an_item_withdrawn_after_indexing_is_not_returned(corpus: Path, client: TestClient) -> None:
    from podcast_scraper import perf_cache

    meta = next((corpus / "sources").rglob("risk-desks.metadata.json"))
    original = meta.read_text(encoding="utf-8")
    doc = json.loads(original)
    try:
        meta.write_text(json.dumps({**doc, "serveable": "none"}), encoding="utf-8")
        perf_cache.clear()
        ids = [
            r["doc_id"]
            for r in client.get(
                "/api/search", params={"q": "risk desks", "content_types": "news_article"}
            ).json()["results"]
        ]
        assert doc["item_id"] not in ids
    finally:
        meta.write_text(original, encoding="utf-8")
        perf_cache.clear()


@requires("sentence_transformers")
def test_an_index_without_the_news_table_answers_with_no_news(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lancedb

    old = tmp_path / "old"
    shutil.copytree(corpus, old)
    lancedb.connect(str(lance_index_dir(old))).drop_table(TABLE)
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_FIXTURE / "publishers.fixture.yaml"))
    body = (
        TestClient(create_app(old, static_dir=False))
        .get("/api/search", params={"q": "risk", "content_types": "news_article"})
        .json()
    )
    assert body["results"] == [] and body.get("error") is None


@requires("sentence_transformers")
def test_a_news_only_corpus_gets_a_searchable_index(tmp_path: Path) -> None:
    """With no podcast episodes there is no podcast index; the news table is built anyway."""
    root = tmp_path / "news-only"
    shutil.copytree(_FIXTURE / "v3" / "sources", root / "sources")
    tti.build_two_tier_index(root, lance_index_dir(root), model_id=_MODEL)
    assert lance_index_dir(root).is_dir()
    hits = search_news(lance_index_dir(root), "risk management behaviour", _MODEL, top_k=3)
    assert hits, "a news-only corpus must be searchable"


@requires("sentence_transformers")
def test_news_topic_rows_share_the_podcast_aux_shape(corpus: Path) -> None:
    """Piece B (news joins themes): one kg_topic row per (item, Topic) in news_aux, the aux schema
    unchanged, outside the backend's tiers so podcast search never reads it."""
    import lancedb

    from podcast_scraper.search.backends.lancedb_backend import _aux_schema, LanceDBBackend
    from podcast_scraper.search.indexer import _kg_embed_text_topic, kg_topic_embed_text
    from podcast_scraper.search.news_index import AUX_TABLE, news_topic_rows

    table = lancedb.connect(str(lance_index_dir(corpus))).open_table(AUX_TABLE)
    rows = table.to_arrow().to_pylist()
    assert {r["id"] for r in rows} == {r["id"] for r in news_topic_rows(corpus)}
    assert table.schema == _aux_schema(len(rows[0]["embedding"]))
    risk = next(r for r in rows if r["source_id"] == "topic:risk-management")
    assert risk["doc_type"] == "kg_topic" and risk["episode_id"].startswith("item:news_article:")
    assert risk["show_id"].startswith("source:") and risk["text"] == "Risk Management"
    assert AUX_TABLE not in LanceDBBackend.TABLES.values()
    assert _kg_embed_text_topic is kg_topic_embed_text
