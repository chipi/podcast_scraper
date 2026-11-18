"""The news routes over the app validation corpus's invented news items (UXS-019; RFC-134 §10)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus"
_RISK = "Risk desks learn to sit still"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    root = tmp_path / "corpus"
    shutil.copytree(_FIXTURE / "v3", root)
    monkeypatch.setenv("PODCAST_PUBLISHERS_FILE", str(_FIXTURE / "publishers.fixture.yaml"))
    return TestClient(create_app(root, static_dir=False))


def test_health_lists_news_when_the_corpus_holds_it(client: TestClient) -> None:
    assert client.get("/api/health").json()["content_types"] == ["podcast_episode", "news_article"]


def test_health_lists_podcasts_only_without_news(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    shutil.copytree(_FIXTURE / "v3", root, ignore=shutil.ignore_patterns("sources"))
    client = TestClient(create_app(root, static_dir=False))
    assert client.get("/api/health").json()["content_types"] == ["podcast_episode"]


def test_items_are_newest_first_and_never_the_withheld_one(client: TestClient) -> None:
    body = client.get("/api/corpus/news/items").json()
    titles = [i["title"] for i in body["items"]]
    assert body["total"] == 5 and "WITHHELD-ITEM-TITLE" not in titles
    dates = [i["published_at"] for i in body["items"]]
    assert dates == sorted(dates, reverse=True)
    risk = next(i for i in body["items"] if i["title"] == _RISK)
    assert risk["publisher"] == {
        "id": "org:daily-example",
        "name": "Daily Example",
        "domain": "dailyexample.test",
    }
    soil = next(i for i in body["items"] if i["serveable"] == "metadata" and i["paywalled"])
    assert soil["summary"] is None


def test_a_publisher_lists_its_own_items(client: TestClient) -> None:
    body = client.get("/api/corpus/news/items", params={"publisher_id": "org:other-wire"}).json()
    assert [i["title"] for i in body["items"]] == ["Reliability at scale, a field report"]


def test_the_item_read_carries_its_kg_nodes(client: TestClient) -> None:
    items = client.get("/api/corpus/news/items").json()["items"]
    risk = next(i for i in items if i["title"] == _RISK)
    body = client.get("/api/corpus/news/item", params={"item_id": risk["item_id"]}).json()
    assert body["item"]["item_id"] == risk["item_id"]
    assert body["kg_relative_path"].endswith("risk-desks.kg.json")
    authors = [n["id"] for n in body["kg_nodes"] if n.get("role") == "author"]
    assert authors == ["person:ana-example"]


def test_the_withheld_item_is_not_found(client: TestClient) -> None:
    meta = next((_FIXTURE / "v3" / "sources").rglob("withheld.metadata.json"))
    item_id = json.loads(meta.read_text(encoding="utf-8"))["item_id"]
    assert client.get("/api/corpus/news/item", params={"item_id": item_id}).status_code == 404


def test_no_news_response_carries_the_body(client: TestClient) -> None:
    items = client.get("/api/corpus/news/items").json()["items"]
    texts = [client.get("/api/corpus/news/items").text]
    texts += [
        client.get("/api/corpus/news/item", params={"item_id": i["item_id"]}).text for i in items
    ]
    assert all("NEWS-BODY-MARKER" not in t for t in texts)


def test_node_episodes_lists_news_only_when_asked(client: TestClient) -> None:
    body = {"node_id": "k:topic:risk-management"}
    plain = client.post("/api/corpus/node-episodes", json=body).json()
    assert plain["episodes"] and plain["news_items"] == []
    both = client.post(
        "/api/corpus/node-episodes",
        json={**body, "content_types": ["podcast_episode", "news_article"]},
    ).json()
    assert both["episodes"] == plain["episodes"]
    titles = [i["title"] for i in both["news_items"]]
    assert "Risk desks learn to sit still" in titles
    assert "WITHHELD-ITEM-TITLE" not in titles
    assert all(i["content_type"] == "news_article" for i in both["news_items"])


def test_node_episodes_lists_a_persons_articles(client: TestClient) -> None:
    got = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "person:skanda-amarnath", "content_types": ["news_article"]},
    ).json()
    assert sorted(i["title"] for i in got["news_items"]) == [
        "Reliability at scale, a field report",
        "Risk desks learn to sit still",
    ]
    assert "NEWS-BODY-MARKER" not in json.dumps(got)


def test_since_keeps_items_published_on_or_after_the_day(client: TestClient) -> None:
    body = client.get("/api/corpus/news/items", params={"since": "2026-11-17"}).json()
    assert body["items"] and all(i["published_at"][:10] >= "2026-11-17" for i in body["items"])
    everything = client.get("/api/corpus/news/items").json()["total"]
    assert body["total"] < everything


def test_summary_counts_the_shown_items(client: TestClient) -> None:
    body = client.get("/api/corpus/news/summary").json()
    # Six fixture items, the withheld one never counted; Daily Example and Other Wire publish them.
    assert body["items"] == 5 and body["publishers"] == 2
    assert body["teasers"] == 2 and body["paywalled"] == 1


def _name_news(path: Path, cluster_id: str, item_ids: list) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    holder = doc["data"] if isinstance(doc.get("data"), dict) else doc
    cluster = next(c for c in holder["clusters"] if c["graph_compound_parent_id"] == cluster_id)
    cluster["members"][0]["news_item_ids"] = sorted(item_ids)
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.parametrize(
    "relpath, cluster_id",
    [
        ("search/topic_clusters.json", "tc:alerting-actionability"),
        ("enrichments/topic_theme_clusters.json", "thc:managing-risk"),
    ],
)
def test_node_episodes_lists_a_groupings_articles(
    client: TestClient, tmp_path: Path, relpath: str, cluster_id: str
) -> None:
    """A theme or storyline rail lists its members' news_item_ids (the news release's piece H),
    served items only: the withheld item named there never shows."""
    items = client.get("/api/corpus/news/items").json()["items"]
    risk = next(i["item_id"] for i in items if i["title"] == _RISK)
    withheld = "item:news_article:2043d8230c86beddfb95"
    _name_news(tmp_path / "corpus" / relpath, cluster_id, [risk, withheld])
    res = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "g:" + cluster_id, "content_types": ["news_article"]},
    )
    assert res.status_code == 200
    assert [i["item_id"] for i in res.json()["news_items"]] == [risk]


@pytest.mark.parametrize("cluster_id", ["tc:risk-management", "thc:managing-risk"])
def test_the_fixture_groupings_carry_the_shared_topics_articles(
    client: TestClient, cluster_id: str
) -> None:
    """The checked-in grouping files (Common's builders over this corpus, news fields only; piece
    J): the theme and the storyline on topic:risk-management list its two shown articles, newest
    first. The theme file also names the withheld item, which never shows."""
    res = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": cluster_id, "content_types": ["news_article"]},
    )
    assert res.status_code == 200
    assert [i["item_id"] for i in res.json()["news_items"]] == [
        "item:news_article:a8d6e89ce92ef7521ed5",
        "item:news_article:b626e4b9538458cb6235",
    ]
