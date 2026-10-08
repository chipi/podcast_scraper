"""Integration tests for the consumer knowledge-card routes (#1095/#1096/#1097).

GET /api/app/persons/{id}, /api/app/topics/{id} (KG co-occurrence cards) and
/api/app/entities/search (exact/near-exact entity resolution) over a real fixture corpus via
TestClient. Mounted at /api/app, separate from the operator API.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]],
    topics: list[tuple[str, str]],
    published: str = "2024-03-10T00:00:00",
) -> None:
    """Write one episode (metadata + KG with the given person/topic nodes)."""
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons]
    nodes += [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    kg = {"episode_id": episode_id, "nodes": nodes}
    (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _write_clusters(root: Path) -> None:
    """A topic_clusters.json making topic:ai and topic:ml siblings of one 'AI' cluster."""
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "Artificial Intelligence",
                "members": [
                    {"topic_id": "topic:ai", "label": "AI"},
                    {"topic_id": "topic:ml", "label": "Machine Learning"},
                ],
            }
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _two_episode_corpus(root: Path) -> None:
    # ep1: Jane + Bob discuss AI & ML.  ep2: Jane + Carol discuss AI.
    _write_episode(
        root,
        stem="0001-a",
        episode_id="ep1",
        persons=[("person:jane-doe", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002-b",
        episode_id="ep2",
        persons=[("person:jane-doe", "Jane Doe"), ("person:carol", "Carol")],
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )


def _client(root: Path) -> TestClient:
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_person_card_aggregates_episodes_and_related(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:jane-doe")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "person:jane-doe"
    assert body["label"] == "Jane Doe"
    assert body["episode_count"] == 2
    # newest-first: ep2 (June) before ep1 (Jan).
    assert [e["title"] for e in body["episodes"]] == ["Episode ep2", "Episode ep1"]
    related_ids = {p["id"] for p in body["related_people"]}
    assert related_ids == {"person:bob", "person:carol"}  # self excluded
    topic_ids = {t["id"] for t in body["related_topics"]}
    assert topic_ids == {"topic:ai", "topic:ml"}


def test_person_card_related_topics_carry_cluster_info(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    body = _client(tmp_path).get("/api/app/persons/person:jane-doe").json()
    ai = next(t for t in body["related_topics"] if t["id"] == "topic:ai")
    assert ai["cluster_id"] == "tc:ai"
    assert ai["cluster_label"] == "Artificial Intelligence"
    assert ai["cluster_size"] == 2


def test_person_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    resp = _client(tmp_path).get("/api/app/persons/person:nobody")
    assert resp.status_code == 404


def test_topic_card_episodes_siblings_and_people(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    resp = _client(tmp_path).get("/api/app/topics/topic:ai")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "topic:ai"
    assert body["label"] == "AI"
    assert body["cluster_id"] == "tc:ai"
    assert body["cluster_size"] == 2
    assert body["episode_count"] == 2  # both episodes discuss AI
    sibling_ids = {s["id"] for s in body["sibling_topics"]}
    assert sibling_ids == {"topic:ml"}  # cluster co-member, self excluded
    people_ids = {p["id"] for p in body["related_people"]}
    assert people_ids == {"person:jane-doe", "person:bob", "person:carol"}


def test_card_episodes_are_paged_on_the_server_and_the_count_stays_the_total(
    tmp_path: Path,
) -> None:
    """2026-10-08: cards returned every episode (a storyline 99, 1.29 MB on prod) and the client
    sliced five. Paged on the server; with no limit the full list still comes back (1.0.2)."""
    _two_episode_corpus(tmp_path)
    client = _client(tmp_path)
    full = client.get("/api/app/topics/topic:ai").json()
    assert full["episode_count"] == 2 and len(full["episodes"]) == 2
    first = client.get("/api/app/topics/topic:ai?episodes_limit=1").json()
    second = client.get("/api/app/topics/topic:ai?episodes_limit=1&episodes_offset=1").json()
    assert first["episode_count"] == 2 and len(first["episodes"]) == 1
    assert first["episodes_total"] == 2
    # The sparkline and top shows are over ALL episodes, not the page of one.
    assert sum(m["count"] for m in first["episode_months"]) == sum(
        m["count"] for m in full["episode_months"]
    )
    assert first["top_shows"] == full["top_shows"]
    assert sum(sh["count"] for sh in first["top_shows"]) == 2
    assert full.get("episodes_total") is None  # the full list is not "paged"
    assert len(second["episodes"]) == 1
    assert first["episodes"][0]["slug"] != second["episodes"][0]["slug"]
    person = client.get("/api/app/persons/person:jane-doe?episodes_limit=1").json()
    assert len(person["episodes"]) == 1 and person["episode_count"] >= 1


def test_topic_card_without_clusters_has_no_siblings(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/topics/topic:ml").json()
    assert body["episode_count"] == 1  # only ep1 discusses ML
    assert body["sibling_topics"] == []
    assert body["cluster_id"] is None


def test_topic_card_unknown_is_404(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert _client(tmp_path).get("/api/app/topics/topic:nope").status_code == 404


def test_entity_search_resolves_person_exact_and_near_exact(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    client = _client(tmp_path)
    exact = client.get("/api/app/entities/search", params={"q": "Jane Doe"}).json()
    assert exact["entity"] == {"id": "person:jane-doe", "kind": "person", "label": "Jane Doe"}
    # Case/punctuation-insensitive ("near-exact"): hyphen + lowercase still resolves.
    near = client.get("/api/app/entities/search", params={"q": "jane-doe"}).json()
    assert near["entity"]["id"] == "person:jane-doe"


def test_entity_search_resolves_topic(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = (
        _client(tmp_path).get("/api/app/entities/search", params={"q": "machine learning"}).json()
    )
    assert body["entity"] == {"id": "topic:ml", "kind": "topic", "label": "Machine Learning"}


def test_entity_search_no_match_returns_null(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    body = (
        _client(tmp_path)
        .get("/api/app/entities/search", params={"q": "quantum chromodynamics"})
        .json()
    )
    assert body["query"] == "quantum chromodynamics"
    assert body["entity"] is None


def test_entity_search_prefers_person_over_topic_on_collision(tmp_path: Path) -> None:
    # A person and a topic share the name "Focus" → the person card wins.
    _write_episode(
        tmp_path,
        stem="0001-c",
        episode_id="ep1",
        persons=[("person:focus", "Focus")],
        topics=[("topic:focus", "Focus")],
    )
    body = _client(tmp_path).get("/api/app/entities/search", params={"q": "focus"}).json()
    assert body["entity"]["kind"] == "person"
    assert body["entity"]["id"] == "person:focus"


def _write_arc_bundle(root: Path, stem: str, publish: str, insight_id: str, label: str) -> None:
    """One bridge+gi+kg episode + an insight_sentiment sidecar (drives the CIL conversation arc)."""
    md = root / "metadata"
    md.mkdir(parents=True, exist_ok=True)
    bridge = {
        "schema_version": "3.0",
        "episode_id": f"ep-{stem}",
        "identities": [
            {"id": "person:a", "type": "person", "sources": {"gi": True}, "display_name": "A"},
            {"id": "topic:ai", "type": "topic", "sources": {"gi": True}, "display_name": "AI"},
        ],
    }
    gi = {
        "schema_version": "3.0",
        "episode_id": f"ep-{stem}",
        "nodes": [
            {"id": insight_id, "type": "Insight", "properties": {"text": "an AI take"}},
            {"id": f"q-{stem}", "type": "Quote", "properties": {"text": "q"}},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": f"q-{stem}", "to": "person:a"},
            {"type": "SUPPORTED_BY", "from": insight_id, "to": f"q-{stem}"},
            {"type": "ABOUT", "from": insight_id, "to": "topic:ai"},
        ],
    }
    kg = {"nodes": [{"id": "kg:ep", "type": "Episode", "properties": {"publish_date": publish}}]}
    (md / f"{stem}.bridge.json").write_text(json.dumps(bridge), encoding="utf-8")
    (md / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
    (md / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")
    (md / "enrichments").mkdir(parents=True, exist_ok=True)
    (md / "enrichments" / f"{stem}.insight_sentiment.json").write_text(
        json.dumps(
            {"data": {"insights": [{"insight_id": insight_id, "compound": 0.7, "label": label}]}}
        ),
        encoding="utf-8",
    )


def test_topic_conversation_arc_route(tmp_path: Path) -> None:
    _write_arc_bundle(tmp_path, "0001-a", "2024-01-15", "i1", "positive")
    _write_arc_bundle(tmp_path, "0002-b", "2024-01-16", "i2", "negative")
    resp = _client(tmp_path).get("/api/app/topics/topic:ai/conversation-arc")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["topic_id"] == "topic:ai"
    # Both dates are ISO week 2024-W03 → one bucket, volume 2, 1 pos + 1 neg.
    assert len(body["weeks"]) == 1
    wk = body["weeks"][0]
    assert wk["week"] == "2024-W03"
    assert wk["volume"] == 2 and wk["positive"] == 1 and wk["negative"] == 1


def test_topic_card_carries_conversation_arc_week_count(tmp_path: Path) -> None:
    """The card says up front how many weeks its arc has (#2202), matching the arc route."""
    _two_episode_corpus(tmp_path)
    _write_arc_bundle(tmp_path, "0101-a", "2024-01-15", "i1", "positive")
    _write_arc_bundle(tmp_path, "0102-b", "2024-03-04", "i2", "negative")
    client = _client(tmp_path)
    card = client.get("/api/app/topics/topic:ai").json()
    arc = client.get("/api/app/topics/topic:ai/conversation-arc").json()
    assert len(arc["weeks"]) == 2
    assert card["conversation_arc_weeks"] == 2


def test_topic_card_arc_weeks_zero_when_no_insight_is_about_the_topic(tmp_path: Path) -> None:
    """KG-mentioned but no insight ABOUT it — the common case (#2202): the card says 0."""
    _two_episode_corpus(tmp_path)
    card = _client(tmp_path).get("/api/app/topics/topic:ai").json()
    assert card["episode_count"] == 2
    assert card["conversation_arc_weeks"] == 0


def test_topic_card_mine_scope_reports_no_arc(tmp_path: Path) -> None:
    """The arc is corpus-wide with no per-user cut, so "My corpus" never offers it."""
    _two_episode_corpus(tmp_path)
    _write_arc_bundle(tmp_path, "0101-a", "2024-01-15", "i1", "positive")
    body = _client(tmp_path).get("/api/app/topics/topic:ai", params={"scope": "mine"}).json()
    assert body["conversation_arc_weeks"] == 0


def test_topic_card_and_arc_route_share_one_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The arc is an uncached corpus-wide walk; the card's count must not double it."""
    from podcast_scraper.server import cil_queries

    _two_episode_corpus(tmp_path)
    _write_arc_bundle(tmp_path, "0101-a", "2024-01-15", "i1", "positive")
    real = cil_queries.topic_conversation_arc
    calls: list[str] = []

    def counting(*args: object, **kwargs: object) -> list[dict]:
        calls.append(str(args[2]))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cil_queries, "topic_conversation_arc", counting)
    client = _client(tmp_path)
    client.get("/api/app/topics/topic:ai")
    client.get("/api/app/topics/topic:ai/conversation-arc")
    assert calls == ["topic:ai"]


def _fixture_client(tmp_path: Path) -> TestClient:
    """A client over the COMMITTED app-validation corpus, writing its app-data to ``tmp_path``.

    NOT ``_client(fixture_root)``: that derives ``data_dir`` as ``<root>/_appdata``, so pointing it
    at the tracked corpus creates ``tests/fixtures/app-validation-corpus/v3/_appdata/`` and leaves
    the fixture dirty after every run. `.gitignore` force-includes that tree, so it shows up as an
    untracked directory in every later `git status` — which is how a committed fixture quietly grows
    a users/ directory nobody meant to add.
    """
    fixture = Path("tests/fixtures/app-validation-corpus/v3")
    if not fixture.is_dir():
        pytest.skip(f"fixture corpus missing: {fixture}")
    data_dir = tmp_path / "_appdata"
    app = create_app(fixture, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_grouping_perspectives_routes_serve_the_union(tmp_path: Path) -> None:
    """GET /themes/{id}/perspectives and /storylines/{id}/perspectives (2026-10-01).

    Both were added with the "What's said across this theme/storyline" section and had no route
    test — the builder was covered, the HTTP surface was not, so a wiring mistake (wrong kind, wrong
    404, auth missing) would have shipped green.

    Uses the committed app-validation corpus rather than a synthetic one: these routes resolve a
    cluster id through the real cluster maps, which a hand-built two-topic fixture cannot exercise.
    """
    client = _fixture_client(tmp_path)

    theme = client.get("/api/app/themes/tc:safety-practices/perspectives")
    assert theme.status_code == 200, theme.text
    body = theme.json()
    assert body["perspective_count"] == len(body["perspectives"]) > 0
    assert all(p["person_name"].strip() and p["insights"] for p in body["perspectives"])

    # The storyline route takes its ANCHOR TOPIC id, the same second form the card route accepts.
    story = client.get("/api/app/storylines/topic:risk-management/perspectives")
    assert story.status_code == 200, story.text
    assert story.json()["topic_label"] == "Managing risk across domains"


def test_perspectives_are_paged_on_the_server(tmp_path: Path) -> None:
    """2026-10-08: a storyline's perspectives were 160 KB / 1.1 s on prod — every speaker with every
    take — for a section showing three speakers with two takes. Paged; no params = the full list."""
    client = _fixture_client(tmp_path)
    url = "/api/app/storylines/topic:risk-management/perspectives"
    full = client.get(url).json()
    capped = client.get(f"{url}?insights_per_speaker=1").json()
    assert capped["perspective_count"] == full["perspective_count"]
    assert len(capped["perspectives"]) == len(full["perspectives"])
    assert all(len(p["insights"]) <= 1 for p in capped["perspectives"])
    # insight_count stays each speaker's TOTAL, so "all N from this speaker" is still right.
    assert [p["insight_count"] for p in capped["perspectives"]] == [
        p["insight_count"] for p in full["perspectives"]
    ]
    one = client.get(f"{url}?speakers_offset=1&speakers_limit=1").json()
    assert [p["person_id"] for p in one["perspectives"]] == [full["perspectives"][1]["person_id"]]
    assert one["perspectives"][0]["insights"] == full["perspectives"][1]["insights"]


def test_grouping_perspectives_404_on_an_absence_not_a_fault(tmp_path: Path) -> None:
    """A grouping nobody speaks to is a 404, which the client renders as nothing at all.

    `tc:lifelong-learning`'s members are abstract labels ("lifelong learning", "expert interviews")
    that nobody utters aloud, so no member has a speaker-attributable insight. That is an ABSENCE,
    and the section renders nothing rather than an error.
    """
    client = _fixture_client(tmp_path)

    assert client.get("/api/app/themes/tc:lifelong-learning/perspectives").status_code == 404
    assert client.get("/api/app/themes/tc:nope/perspectives").status_code == 404
    # A THEME id is not a storyline — the kind is explicit, never sniffed from the prefix.
    assert client.get("/api/app/storylines/tc:safety-practices/perspectives").status_code == 404


def _search(client: TestClient, kind: str, q: str) -> list[dict]:
    resp = client.get("/api/app/interests/search", params={"kind": kind, "q": q})
    assert resp.status_code == 200, resp.text
    items: list[dict] = resp.json()["items"]
    return items


def test_interest_search_finds_people_by_any_part_of_the_name(tmp_path: Path) -> None:
    """Substring, not exact: "doe" finds Jane Doe, which /entities/search never would."""
    _two_episode_corpus(tmp_path)
    client = _client(tmp_path)
    items = _search(client, "person", "doe")
    assert [(i["id"], i["kind"], i["label"]) for i in items] == [
        ("person:jane-doe", "person", "Jane Doe")
    ]
    # Only the asked-for kind: "a" matches the AI topic, but this is the person section.
    assert all(i["kind"] == "person" for i in _search(client, "person", "a"))


def test_interest_search_ranks_a_prefix_first_then_by_coverage(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        stem="0001-a",
        episode_id="ep1",
        persons=[],
        topics=[("topic:open-ai", "Open AI"), ("topic:ai-safety", "AI safety")],
    )
    _write_episode(
        tmp_path,
        stem="0002-b",
        episode_id="ep2",
        persons=[],
        topics=[("topic:ai-safety", "AI safety"), ("topic:ai", "AI")],
    )
    ids = [i["id"] for i in _search(_client(tmp_path), "topic", "ai")]
    # Both prefix matches come before the later-word match; among them, the topic in TWO episodes
    # ("AI safety") beats the one in one ("AI").
    assert ids == ["topic:ai-safety", "topic:ai", "topic:open-ai"]


def test_interest_search_lists_a_person_once_under_their_best_matching_name(tmp_path: Path) -> None:
    """One person spelled three ways is ONE result, labelled by the spelling that matches best.

    Every spelling is registered for search, so "jane" reaches the person through all three: a
    later word ("Dr Jane Doe"), the start ("Jane Doe"), mid-word ("Majane Doe"). The start wins
    whichever order the episodes load in, so both a better and a worse later match are exercised.
    """
    for n, label in enumerate(["Dr Jane Doe", "Jane Doe", "Majane Doe"], start=1):
        _write_episode(
            tmp_path,
            stem=f"000{n}-x",
            episode_id=f"ep{n}",
            persons=[("person:jane-doe", label)],
            topics=[],
            published=f"2024-0{n}-01T00:00:00",
        )
    items = _search(_client(tmp_path), "person", "jane")
    assert [(i["id"], i["label"]) for i in items] == [("person:jane-doe", "Jane Doe")]


def test_interest_search_skips_groupings_that_do_not_match_or_have_no_label(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    (tmp_path / "search").mkdir(parents=True, exist_ok=True)
    clusters = [
        ("tc:ai", "Artificial Intelligence", "topic:ai"),
        ("tc:food", "Cooking", "topic:ml"),
        ("tc:blank", "", "topic:ai"),
    ]
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": cid,
                "canonical_label": label,
                "members": [{"topic_id": tid, "label": "x"}],
            }
            for cid, label, tid in clusters
        ]
    }
    (tmp_path / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")
    items = _search(_client(tmp_path), "theme", "intelligence")
    assert [i["id"] for i in items] == ["tc:ai"]


def test_interest_search_themes_return_their_tc_token(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    items = _search(_client(tmp_path), "theme", "intelligence")
    assert [(i["id"], i["label"]) for i in items] == [("tc:ai", "Artificial Intelligence")]


def test_interest_search_storylines_follow_by_thc_and_open_by_anchor(tmp_path: Path) -> None:
    """The id is what gets FOLLOWED (thc:), the anchor is what gets OPENED — never swapped."""
    _two_episode_corpus(tmp_path)
    (tmp_path / "enrichments").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "thc:shadow-fleet",
                "canonical_label": "Shadow fleet",
                "members": [
                    {"topic_id": "topic:sanctions", "lift_to_cluster": 3.0},
                    {"topic_id": "topic:tankers", "lift_to_cluster": 1.0},
                ],
            }
        ]
    }
    (tmp_path / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    items = _search(_client(tmp_path), "storyline", "fleet")
    assert items == [
        {
            "id": "thc:shadow-fleet",
            "kind": "storyline",
            "label": "Shadow fleet",
            "anchor_topic_id": "topic:sanctions",
        }
    ]


def test_interest_search_rejects_an_unknown_kind_and_ignores_punctuation_only(
    tmp_path: Path,
) -> None:
    _two_episode_corpus(tmp_path)
    client = _client(tmp_path)
    bad = client.get("/api/app/interests/search", params={"kind": "show", "q": "x"})
    assert bad.status_code == 422
    assert _search(client, "topic", "--") == []
