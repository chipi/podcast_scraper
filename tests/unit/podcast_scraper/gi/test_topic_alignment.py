"""``align_gi_topics_with_kg`` — the single channel by which GI receives topics.

THIS FUNCTION HAD NO TESTS. It was found unguarded while inverting its empty-KG branch for
ADR-156 / #2164, and it is not a minor helper: after that ADR it is the ONLY way a Topic node
reaches gi.json, because GI's own bullets fallback is gone. Everything downstream of GI topics —
ABOUT edges, the CIL bridge, insight→topic navigation — is whatever this function leaves behind.

The branch that changed: it used to return 0 early when the KG declared no topics, reasoning it was
"better to keep GI's own topics than to strip an episode's topic vocabulary down to nothing". That
held only while GI had a topic source of its own — the summary-bullets fallback — so the branch's
real effect was to preserve fabrications. Now an empty KG topic set means the episode has no topics
and GI is stripped to match.

No network, no LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from podcast_scraper.gi.topic_alignment import align_gi_topics_with_kg

pytestmark = [pytest.mark.unit]


def _kg(labels: List[str]) -> Dict[str, Any]:
    return {
        "nodes": [{"id": "episode:e1", "type": "Episode", "properties": {}}]
        + [
            {"id": f"topic:{lab.replace(' ', '-')}", "type": "Topic", "properties": {"label": lab}}
            for lab in labels
        ],
        "edges": [],
    }


def _gi(topic_labels: List[str], *, insights: int = 2) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = [{"id": "episode:e1", "type": "Episode", "properties": {}}]
    for i in range(insights):
        nodes.append({"id": f"insight:i{i}", "type": "Insight", "properties": {"claim": f"c{i}"}})
    nodes.append({"id": "person:alice", "type": "Person", "properties": {"name": "Alice"}})
    edges: List[Dict[str, Any]] = [
        {"type": "SPOKEN_BY", "from": "insight:i0", "to": "person:alice"}
    ]
    for lab in topic_labels:
        tid = f"topic:{lab.replace(' ', '-')}"
        nodes.append({"id": tid, "type": "Topic", "properties": {"label": lab}})
        edges.append({"type": "ABOUT", "from": "insight:i0", "to": tid})
    return {"nodes": nodes, "edges": edges}


def _types(art: Dict[str, Any]) -> List[str]:
    return [n["type"] for n in art["nodes"]]


def _topic_labels(art: Dict[str, Any]) -> List[str]:
    return sorted(n["properties"]["label"] for n in art["nodes"] if n["type"] == "Topic")


def _about(art: Dict[str, Any]) -> List[tuple]:
    return sorted((e["from"], e["to"]) for e in art["edges"] if e["type"] == "ABOUT")


class TestTheKGTopicsBecomeGIsTopics:
    def test_kg_labels_replace_gi_labels(self):
        gi = _gi(["stale bullet phrase"])
        applied = align_gi_topics_with_kg(gi, _kg(["monetary policy", "bond yields"]))
        assert applied == 2
        assert _topic_labels(gi) == ["bond yields", "monetary policy"]

    def test_every_insight_gets_an_about_edge_to_every_topic(self):
        gi = _gi([], insights=2)
        align_gi_topics_with_kg(gi, _kg(["a topic", "b topic"]))
        assert len(_about(gi)) == 4, "2 insights x 2 topics"

    def test_non_topic_nodes_and_non_about_edges_survive(self):
        """The function rebuilds two collections; it must not take anything else with them."""
        gi = _gi(["old"], insights=1)
        align_gi_topics_with_kg(gi, _kg(["new topic"]))
        assert "Insight" in _types(gi) and "Person" in _types(gi) and "Episode" in _types(gi)
        assert any(
            e["type"] == "SPOKEN_BY" for e in gi["edges"]
        ), "SPOKEN_BY carries insight attribution — dropping it would unattribute every quote"

    def test_running_twice_is_idempotent(self):
        gi = _gi(["old"])
        kg = _kg(["x topic", "y topic"])
        align_gi_topics_with_kg(gi, kg)
        first = (_topic_labels(gi), _about(gi))
        align_gi_topics_with_kg(gi, kg)
        assert (_topic_labels(gi), _about(gi)) == first


class TestAnEmptyKGStripsGIsTopics:
    """THE INVERSION (ADR-156 / #2164).

    The KG is the only topic source, so "the KG has no topics" means "this episode has no topics".
    GI keeping its own was only ever keeping bullet-derived fabrications alive.
    """

    def test_gi_topics_are_removed_when_the_kg_has_none(self):
        gi = _gi(["Product development in frontier AI requires", "Empirical iteration replaces"])
        applied = align_gi_topics_with_kg(gi, _kg([]))
        assert applied == 0, "no topics were APPLIED"
        assert _topic_labels(gi) == [], (
            "GI kept its own topics against an empty KG. Those can only be bullet-derived now, "
            "which is the fabrication ADR-156 removed."
        )

    def test_the_about_edges_go_with_them(self):
        """An ABOUT edge pointing at a removed Topic is a dangling reference."""
        gi = _gi(["stale one"])
        align_gi_topics_with_kg(gi, _kg([]))
        assert _about(gi) == []

    def test_the_insights_themselves_are_untouched(self):
        """Stripping topics must not cost the episode its knowledge."""
        gi = _gi(["stale"], insights=3)
        align_gi_topics_with_kg(gi, _kg([]))
        assert len([n for n in gi["nodes"] if n["type"] == "Insight"]) == 3
        assert any(e["type"] == "SPOKEN_BY" for e in gi["edges"])

    def test_an_already_topicless_gi_is_a_clean_no_op(self):
        gi = _gi([])
        before = (len(gi["nodes"]), len(gi["edges"]))
        assert align_gi_topics_with_kg(gi, _kg([])) == 0
        assert (len(gi["nodes"]), len(gi["edges"])) == before
