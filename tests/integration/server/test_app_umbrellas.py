"""An umbrella subject is left out of every list of what something is about (operator 2026-10-12).

The committed fixture corpus once carried two topics stamped on every episode, and they led Home's
week, a show's topics, "Also about" and the interests picker. The corpus is natural now and real
data has no such topic (prod: 1.0% at most), so these build the stamped case on a COPY: two topics
on every episode, plus a theme made of them. ``app_umbrellas`` must name exactly those, and each
surface must leave them out while still listing the episode's own subjects.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_graph_refs import refs_for_slug
from podcast_scraper.server.app_slugs import _slug_index
from podcast_scraper.server.app_umbrellas import UMBRELLA_MIN_EPISODES, umbrella_tokens
from podcast_scraper.server.app_user_store import get_or_create_user
from podcast_scraper.server.feed_signals import compute_feed_signals

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
STAMPED = {"topic:lifelong-learning": "lifelong learning", "topic:expert-interviews": "experts"}
STAMPED_THEME = "tc:stamped-umbrella"


def _stamp(corpus: Path) -> None:
    for kg_path in corpus.glob("feeds/*/run_*/metadata/*.kg.json"):
        kg = json.loads(kg_path.read_text("utf-8"))
        episode = next(n["id"] for n in kg["nodes"] if n.get("type") == "Episode")
        for tid, label in STAMPED.items():
            kg["nodes"].append({"id": tid, "type": "Topic", "properties": {"label": label}})
            kg["edges"].append({"from": episode, "to": tid, "type": "MENTIONS"})
        kg_path.write_text(json.dumps(kg), "utf-8")
    clusters_path = corpus / "search" / "topic_clusters.json"
    doc = json.loads(clusters_path.read_text("utf-8"))
    doc["clusters"].append(
        {
            "graph_compound_parent_id": STAMPED_THEME,
            "canonical_label": "Stamped umbrella",
            # The widest theme, as an umbrella's would be: unguarded it would lead the picker.
            "member_count": 99,
            "members": [{"topic_id": t, "label": lbl} for t, lbl in STAMPED.items()],
        }
    )
    clusters_path.write_text(json.dumps(doc), "utf-8")


@pytest.fixture(scope="module")
def stamped(tmp_path_factory: pytest.TempPathFactory) -> Path:
    corpus = tmp_path_factory.mktemp("umbrellas") / "v3"
    shutil.copytree(CORPUS, corpus, ignore=shutil.ignore_patterns("lance_index"))
    _stamp(corpus)
    return corpus


def _client(root: Path) -> TestClient:
    data_dir = root.parent / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _first_episode(root: Path) -> tuple[str, dict]:
    slug, row = sorted(_slug_index(root).items())[0]
    return slug, json.loads((root / row.kg_relative_path).read_text("utf-8"))


def test_the_natural_corpus_has_no_umbrella() -> None:
    assert umbrella_tokens(CORPUS) == frozenset()


def test_the_stamped_topics_and_their_theme_are_umbrellas_and_nothing_else(stamped: Path) -> None:
    assert umbrella_tokens(stamped) == frozenset(STAMPED) | {STAMPED_THEME}


def test_a_small_corpus_has_no_umbrellas(tmp_path: Path) -> None:
    """In a corpus under ``UMBRELLA_MIN_EPISODES`` a topic on every episode is what it is about."""
    small = tmp_path / "v3"
    shutil.copytree(CORPUS, small, ignore=shutil.ignore_patterns("lance_index"))
    feeds = sorted((small / "feeds").iterdir())
    kept = 0
    for feed in feeds:
        n = len(list(feed.glob("run_*/metadata/*.kg.json")))
        if kept + n >= UMBRELLA_MIN_EPISODES:
            shutil.rmtree(feed)
        else:
            kept += n
    _stamp(small)
    assert 0 < kept < UMBRELLA_MIN_EPISODES
    assert umbrella_tokens(small) == frozenset()


def test_every_list_of_what_an_episode_is_about_leaves_them_out(stamped: Path) -> None:
    slug, kg = _first_episode(stamped)
    own = {n["id"] for n in kg["nodes"] if n.get("type") == "Topic"} - set(STAMPED)
    assert own, "the episode has subjects of its own to list"
    client = _client(stamped)

    entities = client.get(f"/api/app/episodes/{slug}/entities").json()
    ids = {t["id"] for t in entities["topics"]}
    assert ids == own
    assert not {t.get("cluster_id") for t in entities["topics"]} & {STAMPED_THEME}

    recap = client.get(f"/api/app/episodes/{slug}/recap").json()
    assert recap["topics"] and not {t["id"] for t in recap["topics"]} & set(STAMPED)

    refs = {r["id"] for r in refs_for_slug(stamped, slug, limit=100) if r["kind"] == "topic"}
    assert refs == own


def test_a_person_card_and_a_show_do_not_lead_with_them(stamped: Path) -> None:
    slug, kg = _first_episode(stamped)
    person = next(n["id"] for n in kg["nodes"] if n.get("type") == "Person")
    card = _client(stamped).get(f"/api/app/persons/{person}").json()
    assert card["related_topics"]
    assert not {t["id"] for t in card["related_topics"]} & set(STAMPED)

    feed_id = _slug_index(stamped)[slug].feed_id
    signals = compute_feed_signals(stamped, str(feed_id))
    assert signals.top_topics
    assert not {t.topic_id for t in signals.top_topics} & set(STAMPED)
    assert STAMPED_THEME not in {t.theme_id for t in signals.dominant_themes}


def test_discover_lists_no_umbrella_theme(stamped: Path) -> None:
    from podcast_scraper.search.topic_clusters import top_themes_by_member_count

    ranked = [c["id"] for c in top_themes_by_member_count(stamped, 12)]
    assert STAMPED_THEME in ranked, "unguarded, the umbrella theme would be in the picker"
    themes = _client(stamped).get("/api/app/themes", params={"limit": 12}).json()
    ids = [c["id"] for c in themes["items"]]
    assert len(ids) == 12 and STAMPED_THEME not in ids


def test_the_week_behind_home_does_not_lead_with_them(stamped: Path, tmp_path: Path) -> None:
    from podcast_scraper.server import app_recap, app_user_state

    now = int(time.time())
    for i, slug in enumerate(sorted(_slug_index(stamped))[:4]):
        app_user_state.append_listen_event(tmp_path, "u", slug, f"f{i}", now - 3600 + i)
    recap = app_recap.build_recap(tmp_path, "u", "week", now, root=stamped)
    topic_ids = {t["token"] for t in recap["topics"]}
    assert topic_ids and not topic_ids & set(STAMPED)


def test_the_digest_never_calls_them_heating_up(stamped: Path, tmp_path: Path) -> None:
    from podcast_scraper.server import app_digest_sections, app_user_state

    path = stamped / "enrichments" / "temporal_velocity.json"
    doc = json.loads(path.read_text("utf-8"))
    doc["data"]["topics"].append(
        {"topic_id": "topic:lifelong-learning", "velocity_last_over_6mo": 99.0, "total": 55}
    )
    path.write_text(json.dumps(doc), "utf-8")
    assert "topic:lifelong-learning" in app_digest_sections._topic_velocity(stamped)
    now = int(time.time())
    for slug in sorted(_slug_index(stamped)):  # heard: the position is past the end
        app_user_state.set_playback(tmp_path, "u", slug, 1e6, now, finished=True)
    items = app_digest_sections.trending_items(stamped, tmp_path, "u", limit=20)
    ids = {r["id"] for it in items for r in it["graph_refs"]}
    assert ids and not ids & set(STAMPED)
