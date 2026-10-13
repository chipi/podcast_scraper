"""RFC-128 (#2349): the clustering proposes, the storyline registry decides."""

from __future__ import annotations

import copy

import pytest

from podcast_scraper.search.storyline_registry import reconcile

pytestmark = pytest.mark.unit

DAY = 86400.0


def _prop(label: str, topics: list[str], eps: list[str]) -> dict:
    return {
        "canonical_label": label,
        "graph_compound_parent_id": "thc:" + label.replace(" ", "-"),
        "members": [{"topic_id": t, "episode_ids": eps} for t in topics],
    }


def _corpus(*props: dict) -> tuple[dict, dict]:
    labels: dict[str, str] = {}
    eps: dict[str, list[str]] = {}
    for p in props:
        for m in p["members"]:
            labels[m["topic_id"]] = m["topic_id"].split(":")[1]
            eps.setdefault(m["topic_id"], [])
            eps[m["topic_id"]] = sorted(set(eps[m["topic_id"]]) | set(m["episode_ids"]))
    return labels, eps


def _run(props: list[dict], prior: dict | None, now: float, extra_topics: tuple = ()):
    labels, eps = _corpus(*props)
    for t in extra_topics:
        labels.setdefault(t, t)
        eps.setdefault(t, ["e-old"])
    stories, registry, history = reconcile(props, prior, topic_label=labels, topic_eps=eps, now=now)
    return stories, {"clusters": stories, "registry": registry, "id_history": history}


def _ids(stories: list[dict]) -> set[str]:
    return {s["graph_compound_parent_id"] for s in stories}


def test_migration_keeps_every_existing_storyline_with_its_id_and_name() -> None:
    old = {"clusters": [_prop("prompt engineering", ["topic:p", "topic:q"], ["e1", "e2"])]}
    stories, _ = _run([], old, 0.0, extra_topics=("topic:p", "topic:q"))
    assert [(s["graph_compound_parent_id"], s["canonical_label"]) for s in stories] == [
        ("thc:prompt-engineering", "prompt engineering")
    ]


def test_a_proposal_attaches_by_overlap_and_new_episodes_keep_it_active() -> None:
    first = _prop("risk", ["topic:a", "topic:b"], ["e1", "e2"])
    _, state = _run([first], None, 0.0)
    grown = _prop("risk and systems", ["topic:a", "topic:b", "topic:c"], ["e1", "e2", "e3"])
    stories, state = _run([grown], state, 10 * DAY)
    (s,) = stories
    assert s["graph_compound_parent_id"] == "thc:risk" and s["canonical_label"] == "risk"
    assert {m["topic_id"] for m in s["members"]} == {"topic:a", "topic:b", "topic:c"}
    assert s["state"] == "active" and s["last_active_at"] == int(10 * DAY)


def test_a_new_thread_needs_two_topics_five_episodes_and_28_days() -> None:
    _, state = _run([_prop("base", ["topic:a", "topic:b"], ["e1", "e2"])], None, 0.0)
    new = _prop("new thread", ["topic:x", "topic:y"], ["n1", "n2", "n3", "n4", "n5"])
    base = _prop("base", ["topic:a", "topic:b"], ["e1", "e2"])
    stories, state = _run([base, new], state, 1 * DAY)
    assert "thc:new-thread" not in _ids(stories), "proposed once: a candidate"
    stories, state = _run([base, new], state, 20 * DAY)
    assert "thc:new-thread" not in _ids(stories), "19 days: still a candidate"
    stories, state = _run([base, new], state, 30 * DAY)
    assert "thc:new-thread" in _ids(stories), "29 days, 5 episodes, 2 topics: a storyline"

    small = _prop("small", ["topic:s", "topic:t"], ["s1", "s2"])
    for day in (31, 70):
        stories, state = _run([base, small], state, day * DAY)
    assert "thc:small" not in _ids(stories), "two episodes never make a storyline"


def test_a_storyline_with_no_new_episode_goes_quiet_and_is_never_deleted() -> None:
    _, state = _run([_prop("hormuz", ["topic:h", "topic:o"], ["e1", "e2"])], None, 0.0)
    stories, state = _run([], state, 100 * DAY, extra_topics=("topic:h", "topic:o"))
    assert [s["state"] for s in stories] == ["quiet"]
    stories, state = _run(
        [_prop("hormuz", ["topic:h", "topic:o"], ["e1", "e2", "e9"])], state, 101 * DAY
    )
    assert [s["state"] for s in stories] == ["active"], "the next episode wakes it"


def test_a_storyline_whose_topics_all_left_the_corpus_is_gone_for_the_reference_pass() -> None:
    _, state = _run([_prop("gone soon", ["topic:g", "topic:h"], ["e1", "e2"])], None, 0.0)
    stories, state = _run([], state, 1 * DAY)  # neither topic in the corpus any more
    assert stories == [] and state["id_history"]["gone"] == ["thc:gone-soon"]


def test_the_same_input_gives_the_same_registry() -> None:
    props = [
        _prop("one", ["topic:a", "topic:b"], ["e1", "e2"]),
        _prop("two", ["topic:c", "topic:d"], ["e3", "e4"]),
    ]
    a = _run(copy.deepcopy(props), None, 0.0)
    b = _run(list(reversed(copy.deepcopy(props))), None, 0.0)
    assert a[0] == b[0]


def _anchor(story: dict) -> str | None:
    from podcast_scraper.search.storylines import _anchor_topic_id

    return _anchor_topic_id(story["members"])


def test_migration_keeps_the_anchor_a_storyline_page_is_routed_by() -> None:
    """Storyline pages are routed by the anchor; the first registry run must not move it (the
    prod copy moved 22 of 99 when every seeded topic started at the same strength)."""
    old = _prop("risk", ["topic:a", "topic:z"], ["e1", "e2"])
    for m in old["members"]:
        m["lift_to_cluster"] = 9.0 if m["topic_id"] == "topic:z" else 1.0
    grown = _prop("risk", ["topic:a", "topic:b"], ["e1", "e2", "e3"])
    stories, state = _run([grown], {"clusters": [old]}, 0.0, extra_topics=("topic:z",))
    assert _anchor(stories[0]) == "topic:z"
    stories, state = _run([grown], state, 40 * DAY, extra_topics=("topic:z",))
    assert _anchor(stories[0]) == "topic:z", "a later run keeps it too"


def test_the_anchor_moves_only_when_its_topic_leaves_the_corpus() -> None:
    old = _prop("risk", ["topic:a", "topic:z"], ["e1", "e2"])
    old["members"][1]["lift_to_cluster"] = 9.0
    _, state = _run([], {"clusters": [old]}, 0.0, extra_topics=("topic:a", "topic:z"))
    stories, _ = _run([], state, 1 * DAY, extra_topics=("topic:a",))
    assert _anchor(stories[0]) == "topic:a"


def test_a_topic_is_published_in_one_storyline_only() -> None:
    """Today's clustering is a partition; attaching a proposal must not copy a topic another
    storyline holds (the replay over production reached 25 shared topics without this)."""
    _, state = _run(
        [
            _prop("risk", ["topic:a", "topic:b"], ["e1", "e2"]),
            _prop("trade", ["topic:c", "topic:d"], ["e3", "e4"]),
        ],
        None,
        0.0,
    )
    spill = _prop("risk", ["topic:a", "topic:b", "topic:c"], ["e1", "e2", "e3"])
    trade = _prop("trade", ["topic:c", "topic:d"], ["e3", "e4"])
    stories, _ = _run([spill, trade], state, 1 * DAY)
    held = [m["topic_id"] for s in stories for m in s["members"]]
    assert len(held) == len(set(held)), held
    by = {s["graph_compound_parent_id"]: {m["topic_id"] for m in s["members"]} for s in stories}
    assert by["thc:trade"] == {"topic:c", "topic:d"}, "the anchor's storyline keeps it"


def test_a_storyline_whose_topics_all_went_to_others_has_a_successor() -> None:
    _, state = _run(
        [
            _prop("big", ["topic:a", "topic:b", "topic:c"], ["e1", "e2"]),
            _prop("small", ["topic:x", "topic:y"], ["e3", "e4"]),
        ],
        None,
        0.0,
    )
    # "small"'s topics have faded below held, and big is now proposed with them.
    state["registry"]["entries"]["thc:small"]["topics"] = {"topic:x": 0.1, "topic:y": 0.1}
    merged = _prop("big", ["topic:a", "topic:b", "topic:c", "topic:x", "topic:y"], ["e1", "e3"])
    stories, state = _run([merged], state, 1 * DAY)
    assert "thc:small" not in _ids(stories)
    assert state["id_history"]["successors"] == {"thc:small": "thc:big"}
