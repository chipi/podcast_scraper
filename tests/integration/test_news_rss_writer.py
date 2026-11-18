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

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

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


def _run(tmp_path: Path, *extra: str) -> Path:
    from podcast_scraper import cli

    corpus = tmp_path / "corpus"
    spec = tmp_path / "feeds.spec.yaml"
    spec.write_text(f"feeds:\n  - url: {fixtures.FEED_URL}\n    content_type: {NEWS}\n", "utf-8")

    def podcast(cfg: Any) -> Tuple[int, str]:
        raise AssertionError("the podcast pipeline must not run for a news source")

    argv = ["--feeds-spec", str(spec), "--output-dir", str(corpus), "--no-transcript-cache"]
    assert cli.main([*argv, *extra], run_pipeline_fn=podcast) == 0
    return corpus


def test_each_items_kg_holds_its_publisher_and_its_authors(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """KG 3.0 per item (the contract): the item and its source, the publisher PUBLISHED_BY from
    the source, and each byline name a person with the role author on the item."""
    from podcast_scraper.graph_id_utils import person_node_id
    from podcast_scraper.kg.schema import validate_artifact

    corpus = _run(tmp_path, "--generate-kg", "--kg-extraction-source", "metadata_only")
    for item in fixtures.ITEMS:
        meta_path, doc = _docs(corpus)[item["stem"]]
        kg_path = meta_path.with_name(meta_path.name.replace(".metadata.json", ".kg.json"))
        if item["lang"] != "en":
            # As a non-English episode: no analysis until it is translated, and this run has
            # no translator (RFC-124 §5.3's completeness gate).
            assert not kg_path.exists()
            continue
        art = json.loads(kg_path.read_text(encoding="utf-8"))
        validate_artifact(art, strict=True)
        assert art["item_id"] == doc["item_id"]
        types = {n["id"]: n["type"] for n in art["nodes"]}
        assert types[doc["item_id"]] == "ContentItem" and types[doc["source_id"]] == "Source"
        assert types["org:daily-example"] == "Organization"
        assert {"type": "PUBLISHED_BY", "from": doc["source_id"], "to": "org:daily-example"} in art[
            "edges"
        ]
        authors = {
            e["from"]
            for e in art["edges"]
            if e["type"] == "PARTICIPATES_IN"
            and (e.get("properties") or {}).get("role") == "author"
        }
        assert authors == {person_node_id(n) for n in doc["news_article"]["byline"]}


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


def test_an_article_two_feeds_carry_or_one_feed_lists_twice_is_written_once(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """One article is one item (the contract): a second feed carrying it (BBC World and Business
    share articles), or one feed listing it twice under two guids, writes no second copy. The
    first source to write it keeps it."""
    feed = (fixtures.OUT / "feed.xml").read_text(encoding="utf-8")
    twice = feed.replace(
        '<guid isPermaLink="false">risk-desks</guid>',
        '<guid isPermaLink="false">risk-desks#0</guid>',
    ).replace("  </channel>", _risk_desks_again() + "\n  </channel>")
    second = tmp_path / "second.xml"
    second.write_text(twice, encoding="utf-8")
    other_feed = "https://www.dailyexample.test/world/feed.xml"
    net.routes[other_feed] = second

    from podcast_scraper import cli

    corpus = tmp_path / "corpus"
    spec = tmp_path / "two.feeds.spec.yaml"
    spec.write_text(
        f"feeds:\n  - url: {fixtures.FEED_URL}\n    content_type: {NEWS}\n"
        f"  - url: {other_feed}\n    content_type: {NEWS}\n",
        "utf-8",
    )
    argv = ["--feeds-spec", str(spec), "--output-dir", str(corpus), "--no-transcript-cache"]
    assert cli.main(argv, run_pipeline_fn=lambda cfg: (0, "")) == 0

    ids = [
        json.loads(p.read_text(encoding="utf-8"))["item_id"]
        for p in corpus.rglob("*.metadata.json")
    ]
    assert len(ids) == len(set(ids)) == len(fixtures.ITEMS)
    first_source = ct.source_id(ct.content_type(NEWS), fixtures.FEED_URL)
    assert {
        json.loads(p.read_text(encoding="utf-8"))["source_id"]
        for p in corpus.rglob("*.metadata.json")
    } == {first_source}


def test_rederive_reruns_the_stored_items_without_fetching_anything(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """``--pipeline-stage rederive_only`` re-runs a news source's stored items under the run's
    profile: no feed or page is fetched, each item is written again in a new run with the same
    id, body, byline and rights, and the source's cursor is left as it was."""
    corpus = _run(tmp_path)
    first = {stem: doc for stem, (_p, doc) in _docs(corpus).items()}
    cursors = {p: p.read_text(encoding="utf-8") for p in corpus.rglob("source_cursor.json")}
    fetched = len(net.requests)

    _run(tmp_path, "--pipeline-stage", "rederive_only")

    assert net.requests[fetched:] == []  # nothing fetched
    assert {p: p.read_text(encoding="utf-8") for p in corpus.rglob("source_cursor.json")} == cursors
    metas = sorted(corpus.rglob("*.metadata.json"))
    assert len(metas) == 2 * len(fixtures.ITEMS)  # one new run beside the first
    newest: Dict[str, Tuple[str, Path, Dict[str, Any]]] = {}
    for p in metas:
        doc = json.loads(p.read_text(encoding="utf-8"))
        run = p.parent.parent.name
        stem = doc["news_article"]["transport_ids"]["rss"]
        if stem not in newest or newest[stem][0] < run:
            newest[stem] = (run, p, doc)
    for stem, (_run_dir, path, doc) in newest.items():
        before = first[stem]
        for key in (
            "item_id",
            "url",
            "source_id",
            "title",
            "published_at",
            "language",
            "serveable",
            "mineable",
            "publisher_id",
            "body_sha256",
            "fetched_at",
        ):
            assert doc[key] == before[key], (stem, key)
        for key in ("transport_ids", "byline", "byline_raw", "text_status", "language_source"):
            assert doc["news_article"][key] == before["news_article"][key], (stem, key)
        assert _body_path(path).read_text(encoding="utf-8")  # the body is written again
        raw = Path(str(_body_path(path))[: -len(".txt")] + ".raw.html")
        assert raw.exists()  # and its raw page


def test_a_source_that_may_not_be_mined_gets_no_kg(
    tmp_path: Path, net: FakeNet, no_podcast_stages: None, registry: Registry
) -> None:
    """``mineable: false`` on the source: the items are written (title, link, metadata), but no
    enrichment stage runs on them, so no kg.json; the same source minable gets its KG."""
    from podcast_scraper import cli

    def run(corpus: Path, mineable: bool) -> List[Path]:
        spec = tmp_path / f"spec-{mineable}.yaml"
        spec.write_text(
            f"feeds:\n  - url: {fixtures.FEED_URL}\n    content_type: {NEWS}\n"
            f"    news: {{mineable: {str(mineable).lower()}}}\n",
            "utf-8",
        )
        argv = [
            "--feeds-spec",
            str(spec),
            "--output-dir",
            str(corpus),
            "--no-transcript-cache",
            "--generate-kg",
            "--kg-extraction-source",
            "metadata_only",
        ]
        fetch.reset_robots_cache()
        assert cli.main(argv, run_pipeline_fn=lambda cfg: (0, "")) == 0
        return sorted(corpus.rglob("*.metadata.json"))

    closed = run(tmp_path / "closed", False)
    assert len(closed) == len(fixtures.ITEMS)
    assert not list((tmp_path / "closed").rglob("*.kg.json"))
    for meta in closed:
        assert json.loads(meta.read_text(encoding="utf-8"))["mineable"] is False

    run(tmp_path / "open", True)
    assert list((tmp_path / "open").rglob("*.kg.json"))


def _risk_desks_again() -> str:
    item = fixtures.ITEMS[0]
    return (
        "    <item>\n"
        f"      <title>{item['title']}</title>\n"
        f"      <link>{item['url']}</link>\n"
        '      <guid isPermaLink="false">risk-desks#3</guid>\n'
        "      <pubDate>Tue, 17 Nov 2026 07:30:00 +0000</pubDate>\n"
        "    </item>"
    )


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
