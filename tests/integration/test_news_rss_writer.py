"""The news writer over an RSS source, end to end on invented fixtures (E05, News 1.0).

A feeds spec names a news source; the multi-feed loop dispatches it to the text-in path, the RSS
transport lists the feed, the news type fetches and extracts each page, and each item is written
exactly as ``docs/architecture/corpus/news-item-1.0.md`` has it. The network is a fake that serves
``tests/fixtures/news/``; nothing reaches audio, transcription, ad cleaning or GI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.news import document, fetch
from podcast_scraper.publishers import Publisher, Registry
from podcast_scraper.rss import downloader

pytestmark = [pytest.mark.integration]

_SPEC = importlib.util.spec_from_file_location(
    "build_news_fixtures",
    Path(__file__).resolve().parents[1] / "fixtures" / "news" / "build_news_fixtures.py",
)
assert _SPEC is not None and _SPEC.loader is not None
fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fixtures)

NEWS = "news_article"

#: The top-level fields the contract lists, in its order (summary absent: no enrichment ran).
CONTRACT_FIELDS = [
    "content_type",
    "item_id",
    "url",
    "source_id",
    "title",
    "published_at",
    "ingested_at",
    "language",
    "licence",
    "paywalled",
    "serveable",
    "mineable",
    "publisher_id",
    "fetched_at",
    "body_sha256",
    "news_article",
]


class FakeNet:
    """Serves the fixture files by URL; counts every request."""

    def __init__(self) -> None:
        self.routes = fixtures.routes()
        self.requests: List[str] = []

    def _body(self, url: str) -> Optional[bytes]:
        self.requests.append(url)
        path = self.routes.get(url)
        return path.read_bytes() if path else None

    def fetch(self, url: str, user_agent: str, timeout: int, **_: Any) -> Optional[httpx.Response]:
        body = self._body(url)
        if body is None:
            return None
        return httpx.Response(200, content=body, request=httpx.Request("GET", url))

    def any_status(self, url: str, user_agent: str, timeout: int) -> Tuple[int, bytes]:
        body = self._body(url)
        return (200, body) if body is not None else (404, b"")


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> FakeNet:
    fake = FakeNet()
    monkeypatch.setattr(downloader, "fetch_rss_feed_url", fake.fetch)
    monkeypatch.setattr(downloader, "fetch_url", fake.fetch)
    monkeypatch.setattr(downloader, "http_get_any_status", fake.any_status)
    fetch.reset_robots_cache()
    return fake


@pytest.fixture
def no_podcast_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(what: str) -> Any:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(f"a news run must never reach {what}")

        return boom

    from podcast_scraper import builtin_stage_hooks
    from podcast_scraper.workflow import orchestration

    monkeypatch.setattr(
        builtin_stage_hooks,
        "BUILTIN",
        dataclasses.replace(
            builtin_stage_hooks.BUILTIN,
            adfree_base=never("ad cleaning"),
            excise_ads=never("ad cleaning"),
            remove_commercials=never("ad cleaning"),
            excise_crosspromo=never("ad cleaning"),
            episode_insights=never("GI"),
        ),
    )
    monkeypatch.setattr(orchestration, "_create_transcription_provider", never("transcription"))
    monkeypatch.setattr(orchestration, "_fetch_and_prepare_episodes", never("the podcast feed"))


def _source_id() -> str:
    return ct.source_id(ct.content_type(NEWS), fixtures.FEED_URL)


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    reg = Registry(
        (
            Publisher(
                id="org:daily-example",
                name="Daily Example",
                domains=("dailyexample.test",),
                sources=(_source_id(),),
            ),
        )
    )
    monkeypatch.setattr(document, "load_registry", lambda: reg)
    return reg


def _run(tmp_path: Path) -> Path:
    from podcast_scraper import cli

    corpus = tmp_path / "corpus"
    spec = tmp_path / "feeds.spec.yaml"
    spec.write_text(f"feeds:\n  - url: {fixtures.FEED_URL}\n    content_type: {NEWS}\n", "utf-8")

    def podcast(cfg: Any) -> Tuple[int, str]:
        raise AssertionError("the podcast pipeline must not run for a news source")

    argv = ["--feeds-spec", str(spec), "--output-dir", str(corpus), "--no-transcript-cache"]
    assert cli.main(argv, run_pipeline_fn=podcast) == 0
    return corpus


def _docs(corpus: Path) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
    out = {}
    for path in sorted(corpus.rglob("*.metadata.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[doc["news_article"]["transport_ids"]["rss"]] = (path, doc)
    return out


def _body_path(meta_path: Path) -> Path:
    stem = meta_path.name[: -len(".metadata.json")]
    return meta_path.parent.parent / "transcripts" / f"{stem}.txt"


def test_each_item_is_written_as_the_contract_has_it(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    corpus = _run(tmp_path)
    docs = _docs(corpus)
    assert sorted(docs) == sorted(i["stem"] for i in fixtures.ITEMS)

    ctype = ct.content_type(NEWS)
    for item in fixtures.ITEMS:
        meta_path, doc = docs[item["stem"]]
        assert meta_path.relative_to(corpus).parts[:2] == ("sources", NEWS)
        assert list(doc) == CONTRACT_FIELDS, item["stem"]
        assert doc["content_type"] == NEWS
        assert doc["item_id"] == ct.item_id(ctype, ct.SourceItem(item_id="x", url=item["url"]))
        assert doc["item_id"] == f"item:{NEWS}:{ct.url_identity(item['url'])}"
        assert doc["url"] == item["url"]
        assert doc["source_id"] == _source_id()
        assert doc["title"] == item["title"]
        assert doc["published_at"] == item["published_at"]
        assert doc["language"] == item["lang"]
        assert doc["serveable"] == "metadata"  # no rights stated for the source: the narrowest
        assert doc["mineable"] is True
        assert doc["publisher_id"] == "org:daily-example"
        assert doc["fetched_at"].endswith("Z") and doc["ingested_at"].endswith("Z")

        body = _body_path(meta_path).read_text(encoding="utf-8")
        assert f"{fixtures.BODY_MARKER}-{item['n']}" in body
        assert doc["body_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()

        own = doc["news_article"]
        assert own["transport_ids"] == {"rss": item["stem"]}
        assert own["byline_raw"] == item["author"]
        assert own["editorial_type"] is None
        assert own["text_status"] == ("teaser" if item["stem"] == "soil-report" else "full")
        assert "live_blog" not in own

    assert docs["soil-report"][1]["news_article"]["byline"] == ["Ana Example", "Cy Example"]
    assert docs["withheld"][1]["news_article"]["byline"] == []


def test_the_body_is_paragraphs_between_blank_lines_with_its_page_beside_it(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """Ontology v3 "Segments": an untimed body's paragraphs are the text between blank lines, and
    no segments file is written. The fetched page is the raw cache (RFC-134 §5)."""
    from podcast_scraper.news.extract import paragraphs

    corpus = _run(tmp_path)
    for item in fixtures.ITEMS:
        meta_path, _ = _docs(corpus)[item["stem"]]
        body_path = _body_path(meta_path)
        body = body_path.read_text(encoding="utf-8")
        stem = str(body_path)[: -len(".txt")]
        assert not Path(stem + ".segments.json").exists()
        paras = paragraphs(body)
        assert paras[0] == item["title"]
        assert paras[1].startswith(f"{fixtures.BODY_MARKER}-{item['n']} ")
        assert paras[2:] == item["paragraphs"]  # a paywall's "Subscribe…" line is not the article
        page = (fixtures.OUT / "pages" / f"{item['stem']}.html").read_text(encoding="utf-8")
        assert Path(stem + ".raw.html").read_text(encoding="utf-8") == page
        assert "All rights reserved" not in body  # the footer is not the article


def test_a_second_run_lists_nothing_new(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    corpus = _run(tmp_path)
    first = len(net.requests)
    _run(tmp_path)
    assert net.requests[first:] == [fixtures.FEED_URL]  # the feed only: no page fetched again
    assert len(_docs(corpus)) == len(fixtures.ITEMS)


def test_each_items_language_says_where_it_came_from(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """The page's <html lang> decides over the feed's <language> (one English feed carries the
    Spanish article), and the item records where its language came from."""
    corpus = _run(tmp_path)
    for item in fixtures.ITEMS:
        _, doc = _docs(corpus)[item["stem"]]
        assert doc["language"] == item["lang"]
        assert doc["news_article"]["language_source"] == "page"
