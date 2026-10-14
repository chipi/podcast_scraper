"""#2345: a listener's stored references follow the corpus — re-keyed or flagged, never deleted."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_collections_store as boards, app_user_state as state
from podcast_scraper.server.reference_integrity import gone_references, repair_user, Resolver, undo

pytestmark = pytest.mark.unit

U = "u1"


def _resolver() -> Resolver:
    return Resolver(
        themes={"tc:reliability"},
        storylines={"thc:managing-risk"},
        successors={"thc:risk-old": "thc:managing-risk", "tc:rel-old": "tc:reliability"},
        gone_clusters={"thc:dissolved"},
        topics={"topic:cargill"},
        persons={"person:tracey-alloway"},
        canonical={"topic:cargil": "topic:cargill", "person:tracy": "person:tracey-alloway"},
        episodes={"p01-aaa"},
    )


def _seed(d: Path) -> None:
    state._write(
        d, U, "interests", ["thc:risk-old", "thc:managing-risk", "topic:cargil", "thc:dissolved"]
    )
    state._write(d, U, "suggest_less", [{"token": "tc:rel-old", "label": "x", "added_at": 1}])
    state._write(
        d,
        U,
        "favorites",
        [
            {"kind": "person", "ref": "person:tracy", "added_at": 1},
            {"kind": "episode", "ref": "p01-gone", "added_at": 2},
            {"kind": "show", "ref": "feed-x", "added_at": 3},
        ],
    )
    state._write(
        d,
        U,
        "notes",
        [{"id": "n1", "target": "storyline", "target_id": "thc:risk-old", "text": "t"}],
    )
    state._write(
        d, U, "learning_paths", [{"subject_id": "thc:risk-old", "subject_kind": "storyline"}]
    )
    state._write(d, U, "reel_progress", {"subject_id": "thc:dissolved", "index": 2})
    state._write(d, U, "highlights", [{"id": "h1", "episode_slug": "p01-gone", "kind": "span"}])
    boards._write(
        d,
        U,
        {
            "collections": [{"id": "col_a", "name": "A", "created_at": 1}],
            "items": {
                "col_a": [
                    {"kind": "storyline", "ref": "thc:risk-old"},
                    {"kind": "episode", "ref": "p01-aaa"},
                ]
            },
        },
    )


def test_renames_are_rekeyed_silently_and_the_gone_are_flagged_not_deleted(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = repair_user(tmp_path, U, _resolver())

    # The renamed storyline is followed once, under its current id; the gone one stays.
    assert state._read(tmp_path, U, "interests", []) == [
        "thc:managing-risk",
        "topic:cargill",
        "thc:dissolved",
    ]
    assert state._read(tmp_path, U, "suggest_less", [])[0]["token"] == "tc:reliability"
    favs = state._read(tmp_path, U, "favorites", [])
    assert favs[0]["ref"] == "person:tracey-alloway"
    assert favs[1]["ref"] == "p01-gone", "an episode is never rewritten, only flagged"
    assert favs[2]["ref"] == "feed-x", "a show is not this pass's to judge"
    assert state._read(tmp_path, U, "notes", [])[0]["target_id"] == "thc:managing-risk"
    assert state._read(tmp_path, U, "learning_paths", [])[0]["subject_id"] == "thc:managing-risk"
    assert state._read(tmp_path, U, "reel_progress", None)["subject_id"] == "thc:dissolved"
    items = boards._read(tmp_path, U)["items"]["col_a"]
    assert items[0]["ref"] == "thc:managing-risk" and items[1]["ref"] == "p01-aaa"

    assert gone_references(tmp_path, U) == [
        {"ref": "p01-gone", "kind": "episode"},
        {"ref": "thc:dissolved", "kind": "storyline"},
    ]
    assert {r["store"] for r in result.receipts} == {
        "interests",
        "suggest_less",
        "favorites",
        "notes",
        "learning_paths",
        "collections:col_a",
    }


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    _seed(tmp_path)
    result = repair_user(tmp_path, U, _resolver(), apply=False)
    assert result.receipts, "it reports what it would change"
    assert state._read(tmp_path, U, "interests", [])[0] == "thc:risk-old"
    assert gone_references(tmp_path, U) == []


def test_undo_puts_back_what_the_pass_changed(tmp_path: Path) -> None:
    _seed(tmp_path)
    before = {
        s: state._read(tmp_path, U, s, [])
        for s in ("suggest_less", "favorites", "notes", "learning_paths")
    }
    result = repair_user(tmp_path, U, _resolver())
    restored = undo(tmp_path, result.receipts)
    assert restored == len(result.receipts)
    for store, rows in before.items():
        assert state._read(tmp_path, U, store, []) == rows
    assert boards._read(tmp_path, U)["items"]["col_a"][0]["ref"] == "thc:risk-old"


def test_a_second_pass_changes_nothing(tmp_path: Path) -> None:
    _seed(tmp_path)
    repair_user(tmp_path, U, _resolver())
    again = repair_user(tmp_path, U, _resolver())
    assert again.receipts == []


def test_resolver_follows_a_chain_and_judges_only_what_it_knows() -> None:
    r = _resolver()
    r.successors["thc:older"] = "thc:risk-old"
    assert r.resolve("thc:older") == ("rekey", "thc:managing-risk")
    assert r.resolve("thc:never-heard-of") == ("gone", None)
    assert r.resolve("topic:cargill") == ("ok", "topic:cargill")
    assert r.resolve("topic:nothing") == ("gone", None)
    assert r.resolve("feed-x") == ("unknown", "feed-x")


def test_a_storyline_saved_by_its_anchor_topic_becomes_its_storyline_id(tmp_path: Path) -> None:
    """Saves made before they carried the storyline id stored the page's anchor topic, which a
    re-clustering can move into another storyline. The pass rewrites it to the storyline the topic
    belongs to now, and flags one whose topic is in no storyline."""
    r = _resolver()
    r.storyline_of_topic = {"topic:cargill": "thc:managing-risk"}
    state._write(
        tmp_path,
        U,
        "favorites",
        [
            {"kind": "storyline", "ref": "topic:cargil", "added_at": 1},  # a spelling of cargill
            {"kind": "storyline", "ref": "topic:orphaned", "added_at": 2},
        ],
    )
    repair_user(tmp_path, U, r)
    refs = [f["ref"] for f in state._read(tmp_path, U, "favorites", [])]
    assert refs == ["thc:managing-risk", "topic:orphaned"]
    assert {"ref": "topic:orphaned", "kind": "storyline"} in gone_references(tmp_path, U)


def test_a_storyline_the_registry_absorbed_takes_its_followers_along(tmp_path: Path) -> None:
    """RFC-128, end to end: the registry folds a storyline into another (its topics faded into
    it), records the successor, and the reference pass moves the follow and the board item —
    from the artifact the enricher writes, not a hand-built resolver."""
    import json

    from podcast_scraper.search.storyline_registry import reconcile
    from podcast_scraper.server.reference_integrity import build_resolver

    def prop(label: str, topics: list[str], eps: list[str]) -> dict:
        return {
            "canonical_label": label,
            "graph_compound_parent_id": "thc:" + label,
            "members": [{"topic_id": t, "episode_ids": eps} for t in topics],
        }

    labels = {t: t for t in ("topic:a", "topic:b", "topic:x", "topic:y")}
    eps = {t: ["e1", "e2"] for t in labels}
    first = [
        prop("big", ["topic:a", "topic:b"], ["e1"]),
        prop("small", ["topic:x", "topic:y"], ["e2"]),
    ]
    stories, registry, history = reconcile(first, None, topic_label=labels, topic_eps=eps, now=0.0)
    registry["entries"]["thc:small"]["topics"] = {"topic:x": 0.1, "topic:y": 0.1}
    prior = {"clusters": stories, "registry": registry, "id_history": history}
    merged = [prop("big", ["topic:a", "topic:b", "topic:x", "topic:y"], ["e1", "e2"])]
    stories, registry, history = reconcile(
        merged, prior, topic_label=labels, topic_eps=eps, now=86400.0
    )
    assert history["successors"] == {"thc:small": "thc:big"}

    root = tmp_path / "corpus"
    (root / "enrichments").mkdir(parents=True)
    (root / "enrichments" / "topic_theme_clusters.json").write_text(
        json.dumps({"data": {"clusters": stories, "id_history": history, "registry": registry}})
    )
    data = tmp_path / "appdata"
    state._write(data, U, "interests", ["thc:small"])
    boards._write(
        data,
        U,
        {
            "collections": [{"id": "col_a", "name": "A", "created_at": 1}],
            "items": {"col_a": [{"kind": "storyline", "ref": "thc:small"}]},
        },
    )
    repair_user(data, U, build_resolver(root))
    assert state._read(data, U, "interests", []) == ["thc:big"]
    assert boards._read(data, U)["items"]["col_a"] == [{"kind": "storyline", "ref": "thc:big"}]
    assert gone_references(data, U) == []
