"""Unit tests for #654 — bridge produces a non-mechanical {gi, kg, both}
distribution after #653 slug canonicalisation.

Background: pre-#653, GI Topic IDs were long bullet-slugs while KG IDs were
short canonical slugs. Bridge reconciliation fell back to "mechanical match"
behaviour: every KG topic got matched, producing ``both = 10 × episode_count,
gi_only = 0``. This regression guard asserts that when GI and KG topics
overlap-but-do-not-equal, the bridge produces a genuine three-way split.
"""

from __future__ import annotations

from podcast_scraper.builders.bridge_builder import build_bridge


def _artifact(nodes):
    return {"nodes": nodes, "edges": []}


def _topic(node_id: str, label: str) -> dict:
    return {
        "id": node_id,
        "type": "Topic",
        "properties": {"label": label},
    }


class TestBridgeNonMechanical:
    def test_overlapping_topics_produce_three_way_split(self) -> None:
        """GI has 4 topics, KG has 4 topics, 2 overlap on canonical slug.
        Post-#653 expectation: both=2, gi_only=2, kg_only=2 — NOT mechanical."""
        gi = _artifact(
            [
                _topic("topic:prediction-markets", "prediction markets"),
                _topic("topic:ai-agents", "ai agents"),
                _topic("topic:only-in-gi", "only in gi"),
                _topic("topic:also-gi-only", "also gi only"),
            ]
        )
        kg = _artifact(
            [
                _topic("topic:prediction-markets", "prediction markets"),
                _topic("topic:ai-agents", "ai agents"),
                _topic("topic:only-in-kg", "only in kg"),
                _topic("topic:also-kg-only", "also kg only"),
            ]
        )

        out = build_bridge("episode:test", gi, kg, fuzzy_reconcile=False)
        identities = out["identities"]

        both = [i for i in identities if i["sources"]["gi"] and i["sources"]["kg"]]
        gi_only = [i for i in identities if i["sources"]["gi"] and not i["sources"]["kg"]]
        kg_only = [i for i in identities if not i["sources"]["gi"] and i["sources"]["kg"]]

        assert len(both) == 2, f"expected 2 both, got {len(both)}: {both}"
        assert len(gi_only) == 2, f"expected 2 gi_only, got {len(gi_only)}: {gi_only}"
        assert len(kg_only) == 2, f"expected 2 kg_only, got {len(kg_only)}: {kg_only}"
        # Critical non-mechanical assertion: both < N × max(gi, kg) count.
        # Pre-#653 the bridge mechanically matched every KG topic, producing
        # both = len(kg_topics) which is load-bearing to guard against.
        assert len(both) < max(len(gi["nodes"]), len(kg["nodes"]))

    def test_disjoint_topics_produce_zero_both(self) -> None:
        """Sanity: when no IDs overlap, both=0, gi_only=N, kg_only=M."""
        gi = _artifact(
            [
                _topic("topic:alpha", "alpha"),
                _topic("topic:beta", "beta"),
            ]
        )
        kg = _artifact(
            [
                _topic("topic:gamma", "gamma"),
                _topic("topic:delta", "delta"),
            ]
        )
        out = build_bridge("episode:test", gi, kg, fuzzy_reconcile=False)

        both = [i for i in out["identities"] if i["sources"]["gi"] and i["sources"]["kg"]]
        assert len(both) == 0
        gi_only = [i for i in out["identities"] if i["sources"]["gi"] and not i["sources"]["kg"]]
        kg_only = [i for i in out["identities"] if not i["sources"]["gi"] and i["sources"]["kg"]]
        assert len(gi_only) == 2
        assert len(kg_only) == 2

    def test_identical_topics_produce_all_both(self) -> None:
        """Full overlap: both=N, gi_only=0, kg_only=0. This is the only case
        where 'mechanical-looking' output is actually correct."""
        shared = [
            _topic("topic:shared-one", "shared one"),
            _topic("topic:shared-two", "shared two"),
        ]
        gi = _artifact(shared)
        kg = _artifact(shared)
        out = build_bridge("episode:test", gi, kg, fuzzy_reconcile=False)

        both = [i for i in out["identities"] if i["sources"]["gi"] and i["sources"]["kg"]]
        assert len(both) == 2
        assert all(not i["sources"]["gi"] or i["sources"]["kg"] for i in out["identities"])
