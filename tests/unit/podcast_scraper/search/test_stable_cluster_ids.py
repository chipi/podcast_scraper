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


def test_an_old_artifact_maps_to_the_current_ids_without_changing_them() -> None:
    """The backfill: renames from before ids were kept, read from a saved snapshot."""
    from podcast_scraper.search.stable_cluster_ids import map_prior_ids

    old = [_c("thc:risk", "a", "b", "c"), _c("thc:pair", "x", "y"), _c("thc:same", "m", "n")]
    current = [_c("thc:managing-risk", "a", "b", "c", "d"), _c("thc:same", "m", "n")]
    snapshot = copy.deepcopy(current)
    history = map_prior_ids(old, current)
    assert current == snapshot, "the current artifact is read, never changed"
    assert history == {"successors": {"thc:risk": "thc:managing-risk"}, "gone": ["thc:pair"]}


def _unit(*xs: float) -> list[float]:
    n = sum(x * x for x in xs) ** 0.5
    return [x / n for x in xs]


# Topics in a 2-d "meaning" space: a* and b* point one way, z* far off.
_VEC = {
    "a1": _unit(1.0, 0.0),
    "a2": _unit(0.99, 0.1),
    "b1": _unit(0.98, 0.15),
    "b2": _unit(1.0, 0.05),
    "m1": _unit(0.85, 0.53),  # ~0.85 from the a/b direction: successor, not the same theme
    "z1": _unit(0.0, 1.0),
    "z2": _unit(0.1, 0.99),
}


def test_a_theme_that_swapped_its_topics_but_means_the_same_keeps_its_id() -> None:
    """#2349: no member overlap at all, but the centroids agree (>= 0.9): the id is kept."""
    prior = [_c("tc:ai-safety", "a1", "a2")]
    new = [_c("tc:alignment", "b1", "b2"), _c("tc:poetry", "z1", "z2")]
    history = carry_cluster_ids(new, prior, vectors=_VEC)
    assert [c["graph_compound_parent_id"] for c in new] == ["tc:ai-safety", "tc:poetry"]
    assert history == {"successors": {}, "gone": []}


def test_a_theme_near_in_meaning_is_the_successor_and_a_far_one_is_gone() -> None:
    """Between 0.8 and 0.9 the nearest theme is the successor (the id is not kept); below 0.8,
    and with no member in common, the id is gone."""
    centroid = _unit(*(x + y for x, y in zip(_VEC["a1"], _VEC["a2"])))
    sim = sum(x * y for x, y in zip(_VEC["m1"], centroid))
    assert 0.8 <= sim < 0.9, f"fixture: m1 must sit between the two bars, got {sim:.3f}"
    prior = [_c("tc:ai-safety", "a1", "a2"), _c("tc:sonnets", "z1", "z2")]
    new = [_c("tc:governance", "m1")]
    history = carry_cluster_ids(new, prior, vectors=_VEC)
    assert new[0]["graph_compound_parent_id"] == "tc:governance"
    assert history == {"successors": {"tc:ai-safety": "tc:governance"}, "gone": ["tc:sonnets"]}


def test_without_vectors_meaning_plays_no_part() -> None:
    prior = [_c("tc:ai-safety", "a1", "a2")]
    new = [_c("tc:alignment", "b1", "b2")]
    assert carry_cluster_ids(new, prior)["gone"] == ["tc:ai-safety"]
    assert new[0]["graph_compound_parent_id"] == "tc:alignment"
