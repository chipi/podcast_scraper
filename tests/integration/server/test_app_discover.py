"""Integration tests for personalized discovery (#1098).

GET /api/app/clusters (interests picker) and GET /api/app/discover (flag-gated ranking):
- flag OFF (default) → recency, identical to the catalog;
- flag ON + signed-in user with interests → significance × interest-affinity re-ranking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_ranking_telemetry, app_sessions, app_user_state
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    published: str,
    with_gi: bool = False,
    persons: list[tuple[str, str]] | None = None,
    feed_id: str = "myfeed",
    with_kg: bool = True,
    bullets: list[str] | None = None,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {
            "feed_id": feed_id,
            "title": f"Show {feed_id}",
            "url": f"https://pod.example/{feed_id}.xml",
        },
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": list(bullets if bullets is not None else ["a"])},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    nodes += [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in (persons or [])
    ]
    if with_kg:
        (root / "metadata" / f"{stem}.kg.json").write_text(
            json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
        )
    if with_gi:
        gi = {"episode_id": episode_id, "nodes": [], "edges": []}
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # epOld is older but about AI; epNew is newer but about Health.
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
        with_gi=True,
        persons=[("person:jane", "Jane")],
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "AI",
                "member_count": 3,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            },
            {
                "graph_compound_parent_id": "tc:health",
                "canonical_label": "Health",
                "member_count": 1,
                "members": [{"topic_id": "topic:health", "label": "Health"}],
            },
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_theme_clusters(root: Path) -> None:
    """One theme cluster ("storyline") over topic:ai — the co-occurrence overlay for the corpus."""
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    payload = {
        "data": {
            "clusters": [
                {
                    "cluster_type": "theme",
                    "canonical_label": "AI safety",
                    "graph_compound_parent_id": "thc:ai-safety",
                    "member_count": 2,
                    "members": [
                        {"topic_id": "topic:ai", "label": "AI", "lift_to_cluster": 3.1},
                        {"topic_id": "topic:ethics", "label": "Ethics", "lift_to_cluster": 1.2},
                    ],
                }
            ]
        }
    }
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _client(root: Path, *, personalized: bool, derived: bool = False) -> TestClient:
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = root / "appdata"
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    app.state.personalized_ranking = personalized
    app.state.derived_interests = derived
    return TestClient(app)


def _sign_in_heard(client: TestClient, root: Path, heard_episode_ids: list[str]) -> None:
    """Sign in a user with NO explicit interests, but who has *heard* the given episodes."""
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _sign_in(client, root, [])  # same user (subject 's1'), no explicit interests
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    slugs = {r.episode_id: slug_for_row(r) for r in build_catalog_rows_cumulative(root)}
    for eid in heard_episode_ids:
        app_user_state.set_playback(data_dir, user.user_id, slugs[eid], 400.0, 1)  # 40% → heard


def _sign_in(client: TestClient, root: Path, interests: list[str]) -> None:
    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    app_user_state.set_interests(data_dir, user.user_id, interests)
    signed_cookie = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, "test-secret"
    )
    client.cookies.set(app_sessions.SESSION_COOKIE, signed_cookie)


def test_clusters_endpoint_returns_top_by_prevalence(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = (
        _client(tmp_path, personalized=False).get("/api/app/clusters", params={"limit": 5}).json()
    )
    ids = [c["id"] for c in body["items"]]
    assert ids == ["tc:ai", "tc:health"]  # ranked by member_count desc
    assert body["items"][0] == {"id": "tc:ai", "label": "AI", "size": 3}


def test_theme_clusters_endpoint_returns_storylines(tmp_path: Path) -> None:
    _corpus(tmp_path)
    _write_theme_clusters(tmp_path)
    body = _client(tmp_path, personalized=False).get("/api/app/theme-clusters").json()
    assert body["items"] == [
        {"id": "thc:ai-safety", "label": "AI safety", "size": 2, "anchor_topic_id": "topic:ai"}
    ]


def test_theme_clusters_endpoint_empty_without_artifact(tmp_path: Path) -> None:
    _corpus(tmp_path)  # no enrichments/topic_theme_clusters.json → empty items, not 404
    body = _client(tmp_path, personalized=False).get("/api/app/theme-clusters").json()
    assert body["items"] == []


def test_discover_personalizes_by_followed_storyline(tmp_path: Path) -> None:
    # Following a storyline (thc: token) re-ranks like any other interest: epOld's topic:ai is in
    # the theme cluster, so epOld leads despite being older.
    _corpus(tmp_path)
    _write_theme_clusters(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["thc:ai-safety"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_recency_when_flag_off(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=False)
    _sign_in(client, tmp_path, ["tc:ai"])  # interests present but flag off → still recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]  # newest-first


def test_discover_personalizes_when_flag_on_and_interests_set(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["tc:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    # epOld (about AI, the user's interest) now leads despite being older.
    assert titles == ["Episode old", "Episode new"]


def test_discover_personalizes_by_followed_topic(tmp_path: Path) -> None:
    # Following the topic itself (topic: token) — not its cluster — also re-ranks.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_personalizes_by_followed_person(tmp_path: Path) -> None:
    # Following a person (person: token) boosts episodes that feature them.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["person:jane"])  # Jane appears only in (older) epOld
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_derives_interests_from_heard_episodes(tmp_path: Path) -> None:
    # #1139: NO explicit interests, but the user has *heard* the (older) AI episode.
    # Its entities (topic:ai / person:jane) become derived interests → epOld is lifted
    # above the newer Health episode, personalizing from behaviour alone.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True, derived=True)
    _sign_in_heard(client, tmp_path, ["old"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode old", "Episode new"]


def test_discover_derived_off_by_default_stays_recency(tmp_path: Path) -> None:
    # Personalization on, but the derived-interests flag is OFF and there are no explicit
    # interests → recency, unchanged. Guards the new signal behind its own toggle.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # derived defaults off
    _sign_in_heard(client, tmp_path, ["old"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]


def test_discover_recency_when_flag_on_but_anonymous(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # no sign-in → no interests → recency
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles == ["Episode new", "Episode old"]


def _user_id(tmp_path: Path) -> str:
    data_dir = tmp_path / "appdata"
    return get_or_create_user(
        data_dir, provider="stub", subject="s1", email="j@x.com", name="J"
    ).user_id


def test_discover_records_impressions_for_signed_in_user(tmp_path: Path) -> None:
    # #11 telemetry: a signed-in discover call logs the shown slugs (rank order) + the variant.
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    resp = client.get("/api/app/discover", params={"limit": 5})
    assert resp.status_code == 200
    shown = [it["slug"] for it in resp.json()["items"]]
    events = app_ranking_telemetry.read_events(tmp_path / "appdata", _user_id(tmp_path))
    imps = [e for e in events if e["kind"] == "impression"]
    assert imps, "expected an impression event"
    assert imps[-1]["shown"] == shown
    assert imps[-1]["variant"] == "personalized"


def test_discover_click_records_event(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    resp = client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 2})
    assert resp.status_code == 204
    clicks = [
        e
        for e in app_ranking_telemetry.read_events(tmp_path / "appdata", _user_id(tmp_path))
        if e["kind"] == "click"
    ]
    assert clicks and clicks[-1]["slug"] == "some-slug" and clicks[-1]["position"] == 2


def test_discover_click_signed_out_is_noop_204(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)  # not signed in
    resp = client.post("/api/app/discover/click", json={"slug": "x", "position": 0})
    assert resp.status_code == 204


def _sign_in_admin(client: TestClient, root: Path) -> None:
    from podcast_scraper.server.app_user_store import set_role

    data_dir = root / "appdata"
    user = get_or_create_user(data_dir, provider="stub", subject="s1", email="j@x.com", name="J")
    set_role(data_dir, user.user_id, "admin")
    signed_cookie = app_sessions.sign(
        {"user_id": user.user_id, "iat": int(time.time())}, "test-secret"
    )
    client.cookies.set(app_sessions.SESSION_COOKIE, signed_cookie)


def test_ranking_config_get_requires_admin(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, [])  # signed in but not admin
    assert client.get("/api/app/ranking-config").status_code == 403


def test_ranking_config_get_returns_signals_for_admin(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    resp = client.get("/api/app/ranking-config")
    assert resp.status_code == 200
    names = [s["name"] for s in resp.json()["signals"]]
    assert "significance" in names and "trend_velocity" in names


def test_ranking_config_put_persists_and_reads_back(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    put = client.put(
        "/api/app/ranking-config",
        json={"signals": [{"name": "trend_velocity", "enabled": True, "weight": 5.0}]},
    )
    assert put.status_code == 200
    got = client.get("/api/app/ranking-config").json()
    trend = next(s for s in got["signals"] if s["name"] == "trend_velocity")
    assert trend["enabled"] is True and trend["weight"] == 5.0
    # untouched signals survive the merge
    assert any(s["name"] == "significance" for s in got["signals"])


# --------------------------------------------------------------------------- #
# RFC-103 momentum: GET /api/app/trending + GET /api/corpus/trending
# --------------------------------------------------------------------------- #
_TRENDING_NOW = "2026-07-01T00:00:00Z"


def _write_content_series(root: Path, topics: list[dict]) -> None:
    """A temporal_velocity envelope carrying only the RFC-103 content_series (topics)."""
    (root / "enrichments").mkdir(parents=True, exist_ok=True)
    env = {
        "enricher_id": "temporal_velocity",
        "status": "ok",
        "data": {"content_series": {"topics": topics, "persons": []}},
    }
    (root / "enrichments" / "temporal_velocity.json").write_text(json.dumps(env), encoding="utf-8")


def _rising_topics(root: Path, topic_id: str) -> None:
    from podcast_scraper.server.app_momentum import _weeks_ending, resolve_as_of_week

    weeks = _weeks_ending(resolve_as_of_week(_TRENDING_NOW))
    _write_content_series(
        root, [{"topic_id": topic_id, "weekly_counts": {weeks[-2]: 4, weeks[-1]: 7}}]
    )


def test_trending_endpoint_ranks_rising_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", _TRENDING_NOW)
    _corpus(tmp_path)
    _rising_topics(tmp_path, "topic:ai")
    body = (
        _client(tmp_path, personalized=False)
        .get("/api/app/trending", params={"kind": "topic"})
        .json()
    )
    assert body["kind"] == "topic" and body["scope"] == "corpus"
    assert body["as_of_week"].startswith("2026-W")
    assert body["items"][0]["entity_id"] == "topic:ai"
    assert body["items"][0]["heating_up"] is True
    assert body["items"][0]["series"]  # sparkline present


def test_trending_endpoint_rejects_unknown_kind(tmp_path: Path) -> None:
    _corpus(tmp_path)
    r = _client(tmp_path, personalized=False).get("/api/app/trending", params={"kind": "banana"})
    assert r.status_code == 400


def test_corpus_trending_operator_global_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", _TRENDING_NOW)
    _corpus(tmp_path)
    _rising_topics(tmp_path, "topic:ai")
    body = _client(tmp_path, personalized=False).get("/api/corpus/trending").json()
    assert body["as_of_week"].startswith("2026-W")
    assert "topic" in body["kinds"] and "episode" in body["kinds"]  # every kind present
    assert body["kinds"]["topic"][0]["entity_id"] == "topic:ai"


# --- #11 telemetry: impressions and clicks must be labelled by the SAME rule -------------------
#
# They were not. The impression used `personalized and interests`; the click used the
# personalized_ranking flag alone. A flag-on user with NO interests therefore logged `recency`
# impressions and `personalized` clicks, so any CTR-by-variant comparison over
# ranking_events.jsonl was wrong before an experiment could start. The click now reads back the
# variant of the feed that produced it (app_ranking_telemetry.last_impression_variant).


def _variants(tmp_path: Path) -> tuple[list[str], list[str]]:
    events = app_ranking_telemetry.read_events(tmp_path / "appdata", _user_id(tmp_path))
    imps = [e["variant"] for e in events if e["kind"] == "impression"]
    clicks = [e["variant"] for e in events if e["kind"] == "click"]
    return imps, clicks


def test_click_variant_matches_impression_when_flag_on_but_no_interests(tmp_path: Path) -> None:
    """The regression case: personalisation enabled, user has followed nothing."""
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, [])  # no interests
    client.get("/api/app/discover?limit=3")
    client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 0})
    imps, clicks = _variants(tmp_path)
    assert imps and clicks
    assert imps[-1] == "recency", imps
    assert clicks[-1] == imps[-1], (
        f"click logged variant={clicks[-1]!r} against impression variant={imps[-1]!r} — "
        "the two sides are labelled by different rules, which corrupts CTR-by-variant analysis"
    )


def test_click_variant_matches_impression_when_personalized(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    client.get("/api/app/discover?limit=3")
    client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 1})
    imps, clicks = _variants(tmp_path)
    assert imps[-1] == "personalized"
    assert clicks[-1] == imps[-1]


def test_click_variant_matches_impression_when_flag_off(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=False)
    _sign_in(client, tmp_path, ["topic:ai"])  # interests exist but the flag gates them
    client.get("/api/app/discover?limit=3")
    client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 0})
    imps, clicks = _variants(tmp_path)
    assert imps[-1] == "recency"
    assert clicks[-1] == imps[-1]


def test_click_without_a_preceding_impression_still_logs(tmp_path: Path) -> None:
    """Deep link or cleared log: fall back to the flag rather than dropping the event."""
    _corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    client.post("/api/app/discover/click", json={"slug": "some-slug", "position": 0})
    _imps, clicks = _variants(tmp_path)
    assert clicks == ["personalized"], clicks


# --- the stored ranking config must reach RANKING, not just the store (#21) ----------------------
#
# The pre-existing tests prove PUT /ranking-config round-trips through GET. Neither they nor the
# offline eval proved it reached the FEED — the eval hardcoded DEFAULT_RANKING_CONFIG — so an admin
# tuning change could ship while every check went on measuring a system nobody ran.
#
# These need a corpus the shared `_corpus` cannot provide. Scoring is
# `significance x (1 + SUM weight_i * signal_i)`: significance is the BASE, not one of the weighted
# boosts, so its weight cannot silence it. And in `_corpus` significance and affinity both favour
# the same episode ("old" has a GI and matches the interest), so they never disagree and no weight
# change can flip the order. Both facts cost me a wrong test first — asserting a flip that the
# fixture makes impossible looks like a product bug and is not one.
#
# `_affinity_decides_corpus` gives both episodes an EQUAL base (neither has a GI) and lets only the
# interest differ, on the OLDER one. Affinity is then the sole thing standing between the feed and
# recency order, so turning it down has an unambiguous, attributable effect.


def _affinity_decides_corpus(root: Path) -> None:
    """Two episodes, equal significance, only the OLDER matching the user's interest."""
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )


def test_turning_affinity_down_reaches_the_feed(tmp_path: Path) -> None:
    _affinity_decides_corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    assert [e["title"] for e in client.get("/api/app/discover").json()["items"]] == [
        "Episode old",
        "Episode new",
    ], "affinity is not lifting the matching episode — the premise of this test is broken"

    _sign_in_admin(client, tmp_path)
    put = client.put(
        "/api/app/ranking-config",
        json={"signals": [{"name": "interest_affinity", "enabled": True, "weight": 0.0}]},
    )
    assert put.status_code == 200, put.text

    _sign_in(client, tmp_path, ["topic:ai"])  # same user, same interest, same corpus
    after = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert after == ["Episode new", "Episode old"], (
        f"the feed ignored the stored ranking config ({after}) — an operator tuning change would "
        "be invisible in production"
    )


def test_disabling_affinity_reaches_the_feed_too(tmp_path: Path) -> None:
    """`enabled: false` is the other way an operator silences a signal; same effect required."""
    _affinity_decides_corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    client.put(
        "/api/app/ranking-config",
        json={"signals": [{"name": "interest_affinity", "enabled": False, "weight": 2.0}]},
    )
    _sign_in(client, tmp_path, ["topic:ai"])
    assert [e["title"] for e in client.get("/api/app/discover").json()["items"]] == [
        "Episode new",
        "Episode old",
    ]


def test_an_untouched_config_still_personalises(tmp_path: Path) -> None:
    """The control: writing a config that does NOT touch affinity must leave the feed alone, or the
    two tests above would pass for any config write at all."""
    _affinity_decides_corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    _sign_in_admin(client, tmp_path)
    client.put(
        "/api/app/ranking-config",
        json={"signals": [{"name": "trend_velocity", "enabled": False, "weight": 0.4}]},
    )
    _sign_in(client, tmp_path, ["topic:ai"])
    assert [e["title"] for e in client.get("/api/app/discover").json()["items"]] == [
        "Episode old",
        "Episode new",
    ]


# --- enrichment COVERAGE must not outrank INTEREST (#23) ------------------------------------------
#
# `_significance` scores has_gi / has_kg / bullet count — whether ENRICHMENT RAN, not how good the
# episode is. That is harmless on a uniformly-enriched corpus, and the committed fixture IS
# uniformly enriched, which is precisely why it cannot reveal the problem. This corpus is built
# uneven on purpose: without per-feed normalisation, the dense show's OFF-interest episodes outscore
# the sparse show's ON-interest one, and every user's feed is led by whichever show the pipeline
# happened to process more thoroughly.


def _uneven_coverage_corpus(root: Path) -> None:
    """Two shows: one fully enriched and irrelevant, one bare and exactly what the user follows."""
    for i in range(3):
        _write_episode(
            root,
            stem=f"010{i}-dense",
            episode_id=f"dense{i}",
            topics=[("topic:health", "Health")],
            published=f"2024-0{i + 1}-01T00:00:00",
            with_gi=True,
            with_kg=True,
            bullets=["a", "b", "c", "d", "e"],  # everything the pipeline can add: sig 5.0
            feed_id="dense",
            persons=[("person:pat", "Pat")],
        )
    _write_episode(
        root,
        stem="0200-sparse",
        episode_id="sparse0",
        topics=[("topic:ai", "AI")],  # the user's actual interest
        published="2024-01-15T00:00:00",
        # KEEPS its KG — that is where the topic lives, so dropping it to lower significance
        # would also drop the interest match and the scenario would be about something else
        # entirely (the first attempt did exactly that, and the episode simply stopped matching).
        # Depth is lowered the only way that leaves the match intact: no GI, no bullets.
        with_kg=True,
        bullets=[],
        feed_id="sparse",
    )


def test_a_sparsely_enriched_show_still_wins_on_interest(tmp_path: Path) -> None:
    """The user follows AI. Only the bare show covers AI. It must lead anyway.

    The coverage gap has to EXCEED what affinity can absorb, or the scenario never bites — the
    first version of this test used a 4.2-vs-2.2 gap and passed with normalisation switched off,
    proving nothing. Measured values now: dense 5.0 (GI + KG + 5 bullets), sparse 1.0 (neither, no
    bullets), a 5x gap against affinity's 3x multiplier.
    """
    _uneven_coverage_corpus(tmp_path)
    client = _client(tmp_path, personalized=True)
    # TWO follows, one of which this corpus covers. Affinity is matched/len(interests), so a user
    # with more than one interest gets a SMALLER boost per match — which is the realistic case and
    # the one where a coverage gap can actually overpower the follow.
    _sign_in(client, tmp_path, ["topic:ai", "topic:quantum"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles[0] == "Episode sparse0", (
        f"the feed led with {titles[0]!r} — a show is outranking the user's interest because the "
        "pipeline enriched it more thoroughly, not because it is more relevant"
    )


def test_depth_still_orders_episodes_WITHIN_a_show(tmp_path: Path) -> None:
    """Normalisation must not flatten significance away — it only stops it crossing show
    boundaries. Inside one feed, the richer episode still ranks above the barer one."""
    _write_episode(
        root=tmp_path,
        stem="0001-thin",
        episode_id="thin",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
        feed_id="solo",
    )
    _write_episode(
        root=tmp_path,
        stem="0002-rich",
        episode_id="rich",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",  # older, so only depth can lift it
        with_gi=True,
        feed_id="solo",
    )
    client = _client(tmp_path, personalized=True)
    _sign_in(client, tmp_path, ["topic:ai"])
    titles = [e["title"] for e in client.get("/api/app/discover").json()["items"]]
    assert titles[0] == "Episode rich", titles
