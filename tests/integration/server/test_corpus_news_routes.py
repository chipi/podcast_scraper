"""The news routes over the app validation corpus's invented news items (UXS-019; RFC-134 §10)."""

from __future__ import annotations

import json
import os
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
    assert body["total"] == 4 and "WITHHELD-ITEM-TITLE" not in titles  # video hidden
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
    # Seven fixture items: the withheld, the unusable and the video item are never counted;
    # Daily Example and Other Wire publish the four shown. The one teaser is classed a news summary.
    assert body["items"] == 4 and body["publishers"] == 2
    assert body["teasers"] == 1 and body["paywalled"] == 1


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


def test_the_text_file_route_refuses_every_news_file(client: TestClient, tmp_path: Path) -> None:
    """RFC-134 §10: the corpus text-file route refuses a news body, and every other file of a news
    item (its metadata names a withheld item's title and summary; nothing here serves them). Found
    2026-11-18: the route served a body whole."""
    root = tmp_path / "corpus"
    files = sorted(p for p in (root / "sources" / "news_article").rglob("*") if p.is_file())
    assert any(p.suffix == ".txt" for p in files) and any(p.suffix == ".json" for p in files)
    for p in files:
        res = client.get(
            "/api/corpus/text-file", params={"relpath": p.relative_to(root).as_posix()}
        )
        assert res.status_code == 404, p.name
        assert "NEWS-BODY-MARKER" not in res.text and "WITHHELD-ITEM-TITLE" not in res.text


def test_a_symlink_to_a_news_body_is_refused(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    body = next((root / "sources" / "news_article").rglob("risk-desks.txt"))
    link = root / "notes.txt"
    link.symlink_to(body)
    res = client.get("/api/corpus/text-file", params={"relpath": "notes.txt"})
    assert res.status_code == 404 and "NEWS-BODY-MARKER" not in res.text


def test_the_artifact_route_serves_shown_news_kgs_only(client: TestClient, tmp_path: Path) -> None:
    """A withheld item's KG names its title; the artifact route served it (found 2026-11-18 by
    the every-route guard). The graph's expand loads a shown item's KG through this route to draw
    its article node, so those still load; the withheld item's does not."""
    root = tmp_path / "corpus"
    kgs = sorted((root / "sources" / "news_article").rglob("*.kg.json"))
    assert len(kgs) == 7
    for p in kgs:
        res = client.get(
            "/api/artifacts/" + p.relative_to(root).as_posix(), params={"path": str(root)}
        )
        withheld = p.name in ("withheld.kg.json", "untranslated.kg.json", "trail-video.kg.json")
        assert res.status_code == (404 if withheld else 200), p.name
        assert "WITHHELD-ITEM-TITLE" not in res.text and "NEWS-BODY-MARKER" not in res.text


def test_the_graph_expand_loads_the_articles_it_is_given(client: TestClient) -> None:
    """The Workbench graph's expand (GraphCanvas): node-episodes with news, then each item's KG
    beside its metadata through /api/artifacts. Every KG it asks for loads."""
    res = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "k:topic:risk-management", "content_types": ["news_article"]},
    ).json()
    paths = [
        i["metadata_relative_path"].replace(".metadata.json", ".kg.json") for i in res["news_items"]
    ]
    assert paths
    for rel in paths:
        got = client.get("/api/artifacts/" + rel, params={"path": res["path"]})
        assert got.status_code == 200, rel
        assert any(n.get("type") == "ContentItem" for n in got.json()["nodes"])


def test_an_item_without_a_publisher_names_its_site(client: TestClient, tmp_path: Path) -> None:
    """An aggregator's link to an unregistered site has no publisher; the item names the URL's
    host (without www.) as its origin, so a surface never falls back to a source id. The fixture's
    only such item is a video page, hidden since 2026-11-19, so this copy makes it full text."""
    from podcast_scraper import perf_cache

    meta = next((tmp_path / "corpus").rglob("trail-video.metadata.json"))
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["news_article"]["text_status"] = "full"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    perf_cache.clear()
    items = {i["title"]: i for i in client.get("/api/corpus/news/items").json()["items"]}
    unknown = [i for i in items.values() if i["publisher"] is None]
    assert unknown, "the fixture's trail-video item links to an unknown domain"
    for i in unknown:
        assert i["site"] and not i["site"].startswith("www.") and i["site"] in i["url"]
    assert all(i["site"] for i in items.values())


def test_an_older_runs_kg_is_refused_once_a_newer_run_withholds_the_item(
    client: TestClient, tmp_path: Path
) -> None:
    """The artifact route asks whether the item is served now, not whether this copy was: an
    older run's KG of an item a newer run withholds is refused (review 2026-11-19)."""
    root = tmp_path / "corpus"
    meta = next((root / "sources" / "news_article").rglob("risk-desks.metadata.json"))
    kg_rel = meta.with_name("risk-desks.kg.json").relative_to(root).as_posix()
    assert client.get("/api/artifacts/" + kg_rel, params={"path": str(root)}).status_code == 200
    newer = meta.parents[2] / "run_20261119-090000" / "metadata"
    newer.mkdir(parents=True)
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc.update(serveable="none", fetched_at="2026-11-19T09:00:00Z")
    (newer / meta.name).write_text(json.dumps(doc), encoding="utf-8")
    from podcast_scraper import perf_cache

    perf_cache.clear()
    titles = [i["title"] for i in client.get("/api/corpus/news/items").json()["items"]]
    assert "Risk desks learn to sit still" not in titles
    assert client.get("/api/artifacts/" + kg_rel, params={"path": str(root)}).status_code == 404


def test_a_rights_change_on_one_item_shows_without_a_restart(
    client: TestClient, tmp_path: Path
) -> None:
    """The writer replaces a metadata file by renaming a new one into place; the news caches see
    it without an ingest or a restart (review 2026-11-19: they keyed only on the corpus summary)."""
    root = tmp_path / "corpus"
    meta = next((root / "sources" / "news_article").rglob("risk-desks.metadata.json"))

    def titles() -> list:
        return [i["title"] for i in client.get("/api/corpus/news/items").json()["items"]]

    assert "Risk desks learn to sit still" in titles()
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["serveable"] = "none"
    tmp = meta.with_name(meta.name + ".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    old = meta.parent.stat().st_mtime
    os.replace(tmp, meta)
    os.utime(meta.parent, (old + 5, old + 5))  # the rename's own time can equal the read's
    assert "Risk desks learn to sit still" not in titles()


def test_a_persons_items_say_where_the_person_is_the_author(client: TestClient) -> None:
    """node-episodes carries node_role, so the rail can mark "By" (UXS-019; review 2026-11-19)."""
    res = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "k:person:ana-example", "content_types": ["news_article"]},
    ).json()
    roles = {i["title"]: i["node_role"] for i in res["news_items"]}
    assert roles["Risk desks learn to sit still"] == "author"
    topic = client.post(
        "/api/corpus/node-episodes",
        json={"node_id": "k:topic:risk-management", "content_types": ["news_article"]},
    ).json()
    assert all(i["node_role"] is None for i in topic["news_items"])
