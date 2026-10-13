"""Theme and storyline ids survive re-clustering (#2345)."""

from __future__ import annotations

import copy

from podcast_scraper.search.stable_cluster_ids import carry_cluster_ids, resolve


def _c(cid: str, *topics: str) -> dict:
    return {"graph_compound_parent_id": cid, "members": [{"topic_id": t} for t in topics]}


def test_a_relabelled_cluster_keeps_its_id() -> None:
    prior = [_c("thc:risk", "a", "b", "c", "d")]
    new = [_c("thc:managing-risk", "a", "b", "c", "e")]  # Jaccard 3/5 = 0.6
    history = carry_cluster_ids(new, prior)
    assert new[0]["graph_compound_parent_id"] == "thc:risk"
    assert history == {"successors": {}, "gone": []}


def test_below_the_overlap_a_new_id_and_the_old_one_has_a_successor() -> None:
    prior = [_c("thc:risk", "a", "b")]
    new = [
        _c("thc:big", "a", "b", "c", "d", "e")
    ]  # Jaccard 0.4: no inheritance, but holds all of a,b
    history = carry_cluster_ids(new, prior)
    assert new[0]["graph_compound_parent_id"] == "thc:big"
    assert history["successors"] == {"thc:risk": "thc:big"}


def test_two_clusters_absorbed_into_one_keep_the_closer_id_the_other_points_to_it() -> None:
    prior = [_c("thc:x", "a", "b", "c"), _c("thc:y", "d", "e")]
    new = [_c("thc:z", "a", "b", "c", "d", "e")]
    history = carry_cluster_ids(new, prior)
    assert new[0]["graph_compound_parent_id"] == "thc:x"  # 3/5 beats 2/5
    assert history["successors"] == {"thc:y": "thc:x"}


def test_a_dissolved_cluster_is_gone() -> None:
    prior = [_c("thc:pair", "a", "b")]
    new = [_c("thc:other", "c", "d")]
    history = carry_cluster_ids(new, prior)
    assert history == {"successors": {}, "gone": ["thc:pair"]}
    assert resolve("thc:pair", history) is None
    assert resolve("thc:other", history) == "thc:other"


def test_a_new_cluster_never_takes_an_inherited_id_by_its_label() -> None:
    prior = [_c("thc:risk", "a", "b")]
    new = [_c("thc:risk", "c", "d"), _c("thc:other", "a", "b")]
    carry_cluster_ids(new, prior)
    ids = [c["graph_compound_parent_id"] for c in new]
    assert ids == ["thc:risk-1", "thc:risk"]


def test_older_history_is_carried_through_this_run() -> None:
    prior = [_c("thc:b", "a", "b")]
    prior_history = {"successors": {"thc:a": "thc:b"}, "gone": ["thc:old"]}
    new = [_c("thc:c", "a", "b", "c", "d", "e")]
    history = carry_cluster_ids(new, prior, prior_history=prior_history)
    assert history["successors"] == {"thc:a": "thc:c", "thc:b": "thc:c"}
    assert history["gone"] == ["thc:old"]


def test_the_result_does_not_depend_on_input_order() -> None:
    prior = [_c("thc:x", "a", "b", "c"), _c("thc:y", "c", "d", "e")]
    new = [_c("n1", "a", "b", "c", "d"), _c("n2", "c", "d", "e", "f")]
    one, two = copy.deepcopy(new), list(reversed(copy.deepcopy(new)))
    h1 = carry_cluster_ids(one, prior)
    h2 = carry_cluster_ids(two, list(reversed(prior)))
    assert {c["graph_compound_parent_id"] for c in one} == {
        c["graph_compound_parent_id"] for c in two
    }
    assert h1 == h2


def test_a_theme_keeps_its_tc_id_when_its_label_changes() -> None:
    """The theme clustering, end to end: the same two topics under new labels keep the id."""
    import numpy as np

    from podcast_scraper.search.topic_clusters import build_topic_clusters_payload, TopicVectorRow

    v = np.array([1.0, 0.0], dtype=np.float32)
    first = build_topic_clusters_payload(
        [
            TopicVectorRow("topic:a", "alpha", ["e1"], v),
            TopicVectorRow("topic:b", "beta", ["e2"], v),
        ],
        threshold=0.9,
        embedding_model="m",
    )
    (tc,) = [c["graph_compound_parent_id"] for c in first["clusters"]]
    second = build_topic_clusters_payload(
        [
            TopicVectorRow("topic:a", "gamma", ["e1"], v),
            TopicVectorRow("topic:b", "delta", ["e2"], v),
        ],
        threshold=0.9,
        embedding_model="m",
        prior_payload=first,
    )
    assert [c["graph_compound_parent_id"] for c in second["clusters"]] == [tc]
    assert second["clusters"][0]["canonical_label"] in {"gamma", "delta"}
    assert second["id_history"] == {"successors": {}, "gone": []}
