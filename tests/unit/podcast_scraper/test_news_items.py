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
    assert "WITHHELD-ITEM-TITLE" not in items and len(items) == 5
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
    assert items["Trail building, in pictures"]["publisher"] is None


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
