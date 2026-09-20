"""A person MENTIONED in the episode they also SPOKE in is one Person node, not two.

MEASURED on 400 sampled prod episodes: 20 distinct pairs where one spelling is a speaker and a
near-identical one is a `mentioned` Person in the SAME episode — `Joe Weisenthal` (host) beside
`Joe Wiesenthal` (x5, Odd Lots), `Peter Attia, MD` (host) beside `Peter Attia` (x4),
`Tracy Alloway` (host) beside `Tracy Allaway` (x3), `Mark Galeotti` (host) beside
`Mark Galliotti` (x2), and 15 more across ChinaTalk, MLST, The Journal., Trivium China, Unhedged,
Talk Eastern Europe and Analyse.

Neither existing mechanism closed it: the roster's `_one_name_per_person` operates on diarized
VOICES and never sees a mention, and `_dedupe_nodes_by_id` merges only nodes already sharing a
slug — which `joe-weisenthal` and `joe-wiesenthal` do not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.kg.pipeline import _collapse_mentions_into_speakers

pytestmark = [pytest.mark.unit]


def _person(name: str, role: str, node_id: Optional[str] = None) -> Dict[str, Any]:
    slug = name.lower().replace(" ", "-").replace(",", "")
    return {
        "id": node_id or f"person:{slug}",
        "type": "Person",
        "properties": {"name": name, "role": role},
    }


def _names(nodes: List[Dict[str, Any]]) -> set:
    return {n["properties"]["name"] for n in nodes if n.get("type") == "Person"}


def test_a_misspelled_mention_collapses_onto_the_speaker() -> None:
    """The speaker's spelling wins: a voice is established evidence, a mention is a name the ASR
    heard in prose."""
    nodes, _edges = _collapse_mentions_into_speakers(
        [_person("Joe Weisenthal", "host"), _person("Joe Wiesenthal", "mentioned")], []
    )
    assert _names(nodes) == {"Joe Weisenthal"}


def test_the_edges_of_a_collapsed_mention_are_remapped_not_dropped() -> None:
    """A dangling edge is worse than the duplicate — it references a node that no longer exists."""
    edges = [{"from": "person:joe-wiesenthal", "to": "topic:rates", "type": "MENTIONS"}]
    _nodes, out = _collapse_mentions_into_speakers(
        [_person("Joe Weisenthal", "host"), _person("Joe Wiesenthal", "mentioned")], edges
    )
    assert out == [{"from": "person:joe-weisenthal", "to": "topic:rates", "type": "MENTIONS"}]


def test_an_edge_between_the_two_halves_is_dropped() -> None:
    """Once they are one person, an edge from the mention to the speaker says nothing."""
    edges = [{"from": "person:joe-wiesenthal", "to": "person:joe-weisenthal", "type": "MENTIONS"}]
    _nodes, out = _collapse_mentions_into_speakers(
        [_person("Joe Weisenthal", "host"), _person("Joe Wiesenthal", "mentioned")], edges
    )
    assert out == []


def test_mentions_of_one_family_are_never_collapsed_into_each_other() -> None:
    """THE MEASURED DANGER. In `ChinaTalk — "North Korea's Messiah"`, `Kim Jong-il`, `Kim Jong-un`
    and `Kim Il-sung` are three different people, all mentions, and the KG matcher scores the
    first two at 0.857. Comparing only against SPEAKERS keeps them apart — that episode's only
    speaker is Jordan Schneider."""
    nodes, _edges = _collapse_mentions_into_speakers(
        [
            _person("Jordan Schneider", "host"),
            _person("Kim Jong-il", "mentioned"),
            _person("Kim Jong-un", "mentioned"),
            _person("Kim Il-sung", "mentioned"),
        ],
        [],
    )
    assert _names(nodes) == {"Jordan Schneider", "Kim Jong-il", "Kim Jong-un", "Kim Il-sung"}


def test_a_generational_suffix_keeps_a_father_from_his_son() -> None:
    """`same_person`'s subset rule reads `Sam Lee` as `Sam Lee Jr`; that suffix is the entire
    distinction between two people."""
    nodes, _edges = _collapse_mentions_into_speakers(
        [_person("Sam Lee Jr", "host"), _person("Sam Lee", "mentioned")], []
    )
    assert _names(nodes) == {"Sam Lee Jr", "Sam Lee"}


def test_an_ambiguous_mention_is_left_alone() -> None:
    """Two speakers match, so nothing says which it is — and guessing is the #876 failure."""
    nodes, _edges = _collapse_mentions_into_speakers(
        [
            _person("Jon Smith", "host"),
            _person("Jon Smyth", "guest"),
            _person("Jon Smithe", "mentioned"),
        ],
        [],
    )
    assert _names(nodes) == {"Jon Smith", "Jon Smyth", "Jon Smithe"}


def test_an_unrelated_mention_is_untouched() -> None:
    nodes, _edges = _collapse_mentions_into_speakers(
        [_person("Tracy Alloway", "host"), _person("Greg Brockman", "mentioned")], []
    )
    assert _names(nodes) == {"Tracy Alloway", "Greg Brockman"}


def test_an_episode_with_no_speakers_is_untouched() -> None:
    """A never-diarized episode casts nobody (#2075); there is no authoritative spelling to
    collapse onto."""
    people = [_person("Kim Jong-un", "mentioned"), _person("Kim Jong-il", "mentioned")]
    nodes, _edges = _collapse_mentions_into_speakers(list(people), [])
    assert _names(nodes) == {"Kim Jong-un", "Kim Jong-il"}
