"""Theme and storyline cards with their news articles (the news release's piece F).

The grouping files are Common's; here a copy of the app validation corpus gets ``news_item_ids`` on
one theme member and one storyline member, and the groupings an extension provides are stood in
for by maps read from the same files.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.search import groupings
from podcast_scraper.server import relational_view as rv
from podcast_scraper.server.schemas import AppClusterCard, AppClusterCardWithNews

_FIXTURE = Path("tests/fixtures/app-validation-corpus")
_THEME = "tc:alerting-actionability"
_STORYLINE = "thc:managing-risk"
_SUMMARY_NEW = "item:news_article:a8d6e89ce92ef7521ed5"  # serveable summary, 2026-11-17
_METADATA_OLD = "item:news_article:10f6e9ad61532e120b60"  # serveable metadata, 2026-11-14
_NONE = "item:news_article:2043d8230c86beddfb95"  # serveable none


def _clusters(doc: Dict[str, Any]) -> list:
    holder = doc["data"] if isinstance(doc.get("data"), dict) else doc
    return list(holder["clusters"])


def _add_news(path: Path, cluster_id: str, per_member: list) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    cluster = next(c for c in _clusters(doc) if c["graph_compound_parent_id"] == cluster_id)
    for member, ids in zip(cluster["members"], per_member):
        member["news_item_ids"] = sorted(ids)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _map(path: Path, id_key: str, label_key: str) -> Dict[str, Dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for c in _clusters(doc):
        for m in c["members"]:
            out[m["topic_id"]] = {
                id_key: c["graph_compound_parent_id"],
                label_key: c.get("canonical_label") or "",
            }
    return out


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "v3"
    shutil.copytree(_FIXTURE / "v3", root)
    themes = root / "search" / "topic_clusters.json"
    storylines = root / "enrichments" / "topic_theme_clusters.json"
    # Two members name the same item; the card lists it once.
    _add_news(themes, _THEME, [[_METADATA_OLD, _NONE], [_SUMMARY_NEW, _METADATA_OLD]])
    _add_news(storylines, _STORYLINE, [[_SUMMARY_NEW], []])
    stand_in = groupings.TopicGroupings(
        theme_map_by_topic=lambda r: _map(
            r / "search" / "topic_clusters.json", "cluster_id", "cluster_label"
        ),
        storyline_map_by_topic=lambda r: _map(
            r / "enrichments" / "topic_theme_clusters.json", "storyline_id", "storyline_label"
        ),
    )
    monkeypatch.setattr(groupings, "installed", lambda: stand_in)
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_FIXTURE / "publishers.fixture.yaml"))
    return root


def test_without_news_the_card_is_unchanged(corpus: Path) -> None:
    card = rv.build_theme_card(corpus, _THEME)
    assert type(card) is AppClusterCard
    assert "articles" not in card.model_dump() and "article_count" not in card.model_dump()


def test_a_theme_card_lists_its_members_articles_newest_first_without_none(corpus: Path) -> None:
    plain = rv.build_theme_card(corpus, _THEME)
    card = rv.build_theme_card(corpus, _THEME, news=True)
    assert isinstance(card, AppClusterCardWithNews) and plain is not None
    assert [a.item_id for a in card.articles] == [_SUMMARY_NEW, _METADATA_OLD]
    assert card.article_count == 2
    assert card.episode_count == plain.episode_count
    assert card.model_dump(exclude={"articles", "article_count"}) == plain.model_dump()
    by_id = {a.item_id: a for a in card.articles}
    assert by_id[_SUMMARY_NEW].summary and not by_id[_METADATA_OLD].summary
    assert "body" not in json.dumps(card.model_dump())


def test_a_storyline_card_lists_its_articles(corpus: Path) -> None:
    card = rv.build_storyline_card(corpus, _STORYLINE, news=True)
    assert isinstance(card, AppClusterCardWithNews)
    assert [a.item_id for a in card.articles] == [_SUMMARY_NEW] and card.article_count == 1


def test_a_file_without_news_item_ids_gives_no_articles(corpus: Path) -> None:
    card = rv.build_storyline_card(corpus, "thc:" + "nothing", news=True)
    assert card is None
    other = next(
        c["graph_compound_parent_id"]
        for c in _clusters(json.loads((corpus / "search" / "topic_clusters.json").read_text()))
        if c["graph_compound_parent_id"] != _THEME
    )
    card = rv.build_theme_card(corpus, other, news=True)
    assert card is not None and card.articles == [] and card.article_count == 0


def test_a_theme_lists_the_articles_of_its_news_only_topics(corpus: Path) -> None:
    """Themes schema 3: news-only topics that joined by meaning sit in ``news_members`` with their
    own ``news_item_ids``; the card takes the union with the podcast members' items."""
    path = corpus / "search" / "topic_clusters.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    other = next(c for c in _clusters(doc) if c["graph_compound_parent_id"] != _THEME)
    other["news_members"] = [
        {
            "topic_id": "topic:only-in-the-news",
            "label": "Only in the news",
            "news_item_ids": [_SUMMARY_NEW, _NONE],
            "similarity": 0.9,
        }
    ]
    path.write_text(json.dumps(doc), encoding="utf-8")
    card = rv.build_theme_card(corpus, other["graph_compound_parent_id"], news=True)
    assert isinstance(card, AppClusterCardWithNews)
    assert [a.item_id for a in card.articles] == [_SUMMARY_NEW]
