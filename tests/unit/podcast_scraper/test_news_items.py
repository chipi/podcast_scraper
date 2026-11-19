"""News items as surfaces show them (news-item-1.0.md; ontology v3 rights; ADR-179)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from podcast_scraper.news_items import iter_news_metadata, served_item, served_items
from podcast_scraper.publishers import load_registry, Registry

pytestmark = pytest.mark.unit

BODY_MARKER = "BODY-MARKER-7f3c"


def _registry(tmp_path: Path) -> Registry:
    path = tmp_path / "publishers.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "publishers": [
                    {
                        "id": "org:daily-example",
                        "name": "Daily Example",
                        "domains": ["dailyexample.test"],
                        "sources": ["source:aaaaaaaaaaaaaaaaaaaa"],
                    },
                    {"id": "org:other-wire", "name": "Other Wire", "domains": ["otherwire.test"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_registry(path)


def _doc(**over: Any) -> Dict[str, Any]:
    doc: Dict[str, Any] = {
        "content_type": "news_article",
        "item_id": "item:news_article:0123456789abcdef0123",
        "url": "https://www.dailyexample.test/2026/11/a-story",
        "source_id": "source:aaaaaaaaaaaaaaaaaaaa",
        "title": "A story",
        "published_at": "2026-11-17T08:00:00Z",
        "language": "en",
        "serveable": "summary",
        "summary": "What the story says.",
        "paywalled": False,
        "news_article": {
            "byline": ["Ana Example", "Bo Example"],
            "editorial_type": "analysis",
            "text_status": "full",
        },
    }
    doc.update(over)
    return doc


def test_a_summary_item_shows_its_summary_and_its_source_publisher(tmp_path: Path) -> None:
    item = served_item(_doc(), _registry(tmp_path))
    assert item is not None
    assert item["publisher"] == {
        "id": "org:daily-example",
        "name": "Daily Example",
        "domain": "dailyexample.test",
    }
    assert item["summary"] == "What the story says."
    assert item["byline"] == ["Ana Example", "Bo Example"]
    assert item["editorial_type"] == "analysis"


@pytest.mark.parametrize("value", ["metadata", None, "bogus"])
def test_metadata_or_absent_serveable_shows_no_summary(tmp_path: Path, value: Any) -> None:
    doc = _doc(serveable=value) if value is not None else _doc()
    if value is None:
        doc.pop("serveable")
    item = served_item(doc, _registry(tmp_path))
    assert item is not None and item["serveable"] == "metadata" and item["summary"] is None


def test_a_none_item_is_not_shown(tmp_path: Path) -> None:
    assert served_item(_doc(serveable="none"), _registry(tmp_path)) is None


def test_an_item_its_writer_marked_unusable_is_not_shown(tmp_path: Path) -> None:
    """news-item-1.0: ``unusable`` at the top level (D-44: not in English, not translated)."""
    doc = _doc(unusable=True, unusable_reason="translation_off")
    assert served_item(doc, _registry(tmp_path)) is None
    assert served_item(_doc(unusable=False), _registry(tmp_path)) is not None


def test_a_podcast_document_is_not_a_news_item(tmp_path: Path) -> None:
    assert served_item({"title": "An episode"}, _registry(tmp_path)) is None


def test_publisher_order_item_then_source_then_domain(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    own = served_item(_doc(publisher_id="org:other-wire"), reg)
    assert own is not None and own["publisher"]["id"] == "org:other-wire"
    aggregated = served_item(
        _doc(source_id="source:hn00000000000000000", url="https://m.otherwire.test/x"), reg
    )
    assert aggregated is not None and aggregated["publisher"] == {
        "id": "org:other-wire",
        "name": "Other Wire",
        "domain": "otherwire.test",
    }
    unknown = served_item(
        _doc(source_id="source:hn00000000000000000", url="https://nowhere.test/x"), reg
    )
    assert unknown is not None and unknown["publisher"] is None


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    run = root / "sources/news_article/daily-example/run_20261117-080000"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    for stem, over in (
        ("a", {}),
        ("b", {"item_id": "item:news_article:b", "published_at": "2026-11-18T08:00:00Z"}),
        ("c", {"item_id": "item:news_article:c", "serveable": "none"}),
    ):
        doc = _doc(**over)
        (run / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
        (run / "transcripts" / f"{stem}.txt").write_text(f"{BODY_MARKER} body", encoding="utf-8")
    pod = root / "feeds/show/run_20261117-000000/metadata"
    pod.mkdir(parents=True)
    (pod / "e1.metadata.json").write_text(json.dumps({"title": "An episode"}), encoding="utf-8")
    return root


def test_the_corpus_read_lists_shown_news_newest_first_and_no_body(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    items = served_items(root, _registry(tmp_path))
    assert [i["item_id"] for i in items] == [
        "item:news_article:b",
        "item:news_article:0123456789abcdef0123",
    ]
    assert items[0]["metadata_relative_path"].startswith("sources/news_article/daily-example/")
    assert BODY_MARKER not in json.dumps(items)


def test_the_walk_reads_no_podcast_episode(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    assert all("feeds/" not in f.relative_path for f in iter_news_metadata(root))


_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus"


def test_the_fixture_items_list_their_kg_nodes_and_author() -> None:
    from podcast_scraper.news_items import item_kg

    registry = load_registry(_FIXTURE / "publishers.fixture.yaml")
    items = {i["title"]: i for i in served_items(_FIXTURE / "v3", registry)}
    # trail-video is a video page: hidden (the operator, 2026-11-19).
    assert "WITHHELD-ITEM-TITLE" not in items and len(items) == 4
    risk = items["Risk desks learn to sit still"]
    kg = item_kg(_FIXTURE / "v3", risk["metadata_relative_path"])
    assert kg["kg_relative_path"].endswith("risk-desks.kg.json")
    by_id = {n["id"]: n for n in kg["kg_nodes"]}
    assert by_id["topic:risk-management"]["label"] == "Risk Management"
    assert by_id["person:ana-example"]["role"] == "author"
    assert "role" not in by_id["person:skanda-amarnath"]
    assert by_id["org:daily-example"]["type"] == "Organization"
    assert risk["publisher"]["id"] == "org:daily-example"
    assert items["Reliability at scale, a field report"]["publisher"]["domain"] == "otherwire.test"
    assert items["Reliability at scale, a field report"]["roundup"] is True
    assert "Trail building, in pictures" not in items  # a video page


def test_one_item_over_two_sources_is_served_once(tmp_path: Path) -> None:
    """Contract: one article reached by two sources (or two runs) is one item."""
    root = tmp_path / "corpus"
    for source, run in (
        ("bbc-business", "run_1"),
        ("bbc-world", "run_2"),
        ("bbc-business", "run_3"),
    ):
        meta = root / f"sources/news_article/{source}/{run}/metadata"
        meta.mkdir(parents=True)
        (meta / "a.metadata.json").write_text(json.dumps(_doc()), encoding="utf-8")
    items = served_items(root, _registry(tmp_path))
    assert [i["item_id"] for i in items] == ["item:news_article:0123456789abcdef0123"]


def test_a_relative_corpus_root_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _corpus(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert len(served_items(Path("corpus"), _registry(tmp_path))) == 2


def test_roundup_is_served_as_a_boolean(tmp_path: Path) -> None:
    """A newsletter issue covering several stories carries ``news_article.roundup: true`` (set at
    ingest from the source's roundup_title_pattern); every served item says true or false."""
    reg = _registry(tmp_path)
    issue = _doc()
    issue["news_article"] = {**issue["news_article"], "roundup": True}
    served = served_item(issue, reg)
    assert served is not None and served["roundup"] is True
    plain = served_item(_doc(), reg)
    assert plain is not None and plain["roundup"] is False


def _copies(tmp_path: Path, copies: list) -> Path:
    """Write one item's copies: (source, run, fetched_at, overrides) each."""
    root = tmp_path / "corpus"
    for source, run, fetched_at, over in copies:
        meta = root / f"sources/news_article/{source}/{run}/metadata"
        meta.mkdir(parents=True, exist_ok=True)
        doc = _doc(fetched_at=fetched_at, **over)
        (meta / "a.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


def test_a_newer_copy_that_withholds_the_item_withdraws_it(tmp_path: Path) -> None:
    """The current copy decides: a later run marking the item none (or unusable) withdraws the
    older shown copy too (found by review 2026-11-19: the older copy was still served)."""
    for over in ({"serveable": "none"}, {"unusable": True}):
        root = _copies(
            tmp_path / str(len(over)) / next(iter(over)),
            [
                ("daily-example", "run_20261117-080000", "2026-11-17T08:00:00Z", {}),
                ("daily-example", "run_20261118-080000", "2026-11-18T08:00:00Z", over),
            ],
        )
        assert served_items(root, _registry(tmp_path)) == []


def test_a_newer_copy_that_shows_the_item_restores_it(tmp_path: Path) -> None:
    root = _copies(
        tmp_path,
        [
            ("daily-example", "run_20261117-080000", "2026-11-17T08:00:00Z", {"serveable": "none"}),
            ("daily-example", "run_20261118-080000", "2026-11-18T08:00:00Z", {}),
        ],
    )
    assert len(served_items(root, _registry(tmp_path))) == 1


def test_across_sources_the_newer_fetch_decides_and_a_tie_withholds(tmp_path: Path) -> None:
    """Across sources the run path says nothing about time; fetched_at does. On a tie, the copy
    that may not be shown wins."""
    newer_shown = _copies(
        tmp_path / "a",
        [
            ("zz-source", "run_20261117-080000", "2026-11-17T08:00:00Z", {"serveable": "none"}),
            ("aa-source", "run_20261117-090000", "2026-11-18T08:00:00Z", {}),
        ],
    )
    assert len(served_items(newer_shown, _registry(tmp_path))) == 1
    tie = _copies(
        tmp_path / "b",
        [
            ("aa-source", "run_20261117-080000", "2026-11-17T08:00:00Z", {"serveable": "none"}),
            ("zz-source", "run_20261117-080000", "2026-11-17T08:00:00Z", {}),
        ],
    )
    assert served_items(tie, _registry(tmp_path)) == []


def test_news_topic_rows_leave_out_withheld_and_unusable_items() -> None:
    """news_aux holds shown items' topics only: the theme builder copies a news-only topic's label
    into the theme file whatever its items' rights (review 2026-11-19, the database session)."""
    from podcast_scraper.search.news_index import news_topic_rows

    ids = {
        r["episode_id"] for r in news_topic_rows(Path("tests/fixtures/app-validation-corpus/v3"))
    }
    assert "item:news_article:2043d8230c86beddfb95" not in ids  # withheld (serveable none)
    assert "item:news_article:772f365b10f0cb5aeae2" not in ids  # unusable (not translated)
    assert "item:news_article:a8d6e89ce92ef7521ed5" in ids  # risk-desks, shown


def test_the_fixture_items_keep_the_contract() -> None:
    """The app validation corpus's news items obey news-item-1.0 (review 2026-11-19 found the
    teaser and video items carrying summaries and a Spanish item served with translation off)."""
    root = Path("tests/fixtures/app-validation-corpus/v3/sources/news_article")
    translates = {"daily-example"}  # build_news_items.py: Daily Example has translate: true
    docs = {
        p.name: (p, json.loads(p.read_text(encoding="utf-8")))
        for p in root.rglob("*.metadata.json")
    }
    assert len(docs) == 7
    for name, (path, doc) in docs.items():
        own = doc["news_article"]
        assert "summary" not in doc or doc["summary"] is not None, name
        if own["text_status"] in ("teaser", "video"):
            assert "summary" not in doc, f"{name}: a {own['text_status']} item has no summary"
        if doc["language"] != "en" and path.parts[-4] not in translates:
            assert doc.get("unusable") is True, f"{name}: untranslated and not marked unusable"
    assert any(d["news_article"].get("roundup") is True for _p, d in docs.values())


@pytest.mark.parametrize(
    "text_status, item_class, shown",
    [
        ("full", None, True),
        ("teaser", "news_summary", True),
        ("teaser", None, False),  # written before the classifier: hidden until rewritten
        ("teaser", "small_teaser", False),
        ("teaser", "day_brief", False),
        ("teaser", "not_article", False),
        ("video", None, False),
        ("video", "news_summary", False),
    ],
)
def test_a_teaser_shows_only_as_a_news_summary_and_a_video_never(
    tmp_path: Path, text_status: str, item_class: Any, shown: bool
) -> None:
    """The operator, 2026-11-19: teasers hidden unless the writer classed them a news summary
    (news_classify); video pages hidden for now."""
    own = {"text_status": text_status, **({"item_class": item_class} if item_class else {})}
    doc = _doc(news_article=own)
    assert (served_item(doc, _registry(tmp_path)) is not None) is shown
