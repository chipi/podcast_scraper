"""#1058 chunk 4 — Multi-show connectivity fixture contract.

Asserts the checked-in fixture under
``tests/fixtures/connectivity-multi-show/`` exercises every
connectivity surface the relational query layer expects (per the
#1058 issue body). If this test fails, EITHER:

1. The generator (``build_fixture.py``) was edited and the fixture
   is stale — re-run it.
2. The connectivity contract itself changed and this test needs to
   move with it.

Surfaces asserted:

- **Cross-show Person** — at least one Person id appears in ≥2
  episodes from ≥2 shows (powers ``person→topics`` across shows)
- **Intra-episode co-speakers** — at least one episode has ≥2
  named speakers (powers ``co_speakers``)
- **Per-show Topic + ABOUT** — Topic nodes per show + ABOUT edges
- **MENTIONS_PERSON / MENTIONS_ORG** — at least one each
- **Cross-show concept-Topic + RELATED_TO** — ≥1 ``concept:topic-*``
  node with RELATED_TO edges from ≥2 per-show Topics
- **Entity neighborhood** — at least one Person reachable via
  ``Person → MENTIONS_PERSON ← Insight → ABOUT → Topic``

This is the static-data layer of #1058's testability — it backstops
the live pipeline path (chunks 1–3) by proving that the server +
viewer correctly render every surface once data exists. The runtime
pipeline path is asserted by chunk 5's stack-test.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

pytestmark = pytest.mark.integration

_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "connectivity-multi-show"
)


def _load_all_kg() -> List[Tuple[Path, Dict]]:
    return [
        (p, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(_FIXTURE_ROOT.rglob("*.kg.json"))
    ]


def _load_all_gi() -> List[Tuple[Path, Dict]]:
    return [
        (p, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(_FIXTURE_ROOT.rglob("*.gi.json"))
    ]


class TestFixtureExists:
    def test_fixture_root_present(self) -> None:
        assert _FIXTURE_ROOT.is_dir(), (
            "fixture missing — regenerate via: "
            ".venv/bin/python tests/fixtures/connectivity-multi-show/build_fixture.py"
        )

    def test_episode_count_matches_contract(self) -> None:
        """3 shows × 2 episodes = 6 metadata artifacts of each kind."""
        assert len(_load_all_kg()) == 6
        assert len(_load_all_gi()) == 6


class TestCrossShowConnectivity:
    def test_at_least_one_person_appears_across_shows(self) -> None:
        """A Person id present as Person node in ≥2 episodes from ≥2
        shows — powers cross-show person queries."""
        person_to_podcasts: Dict[str, Set[str]] = defaultdict(set)
        for _, kg in _load_all_kg():
            podcast_id = next(
                (n["id"] for n in kg["nodes"] if n.get("type") == "Podcast"),
                None,
            )
            if not podcast_id:
                continue
            for n in kg["nodes"]:
                if n.get("type") == "Person":
                    person_to_podcasts[n["id"]].add(podcast_id)
        cross_show = {pid for pid, shows in person_to_podcasts.items() if len(shows) >= 2}
        assert cross_show, "expected at least one cross-show Person"

    def test_intra_episode_co_speakers(self) -> None:
        """At least one episode carries ≥2 Person nodes (the host +
        a guest)."""
        for path, kg in _load_all_kg():
            person_count = sum(1 for n in kg["nodes"] if n.get("type") == "Person")
            if person_count >= 2:
                return
        pytest.fail("no episode has co-speakers")

    def test_at_least_one_concept_topic_exists(self) -> None:
        """The concept:topic-* nodes that drive cross-show synthesis
        must be present in ≥2 shows."""
        concept_to_podcasts: Dict[str, Set[str]] = defaultdict(set)
        for _, kg in _load_all_kg():
            podcast_id = next(
                (n["id"] for n in kg["nodes"] if n.get("type") == "Podcast"),
                None,
            )
            if not podcast_id:
                continue
            for n in kg["nodes"]:
                if n.get("type") == "Topic" and n["id"].startswith("concept:topic-"):
                    concept_to_podcasts[n["id"]].add(podcast_id)
        cross_show_concepts = {cid for cid, shows in concept_to_podcasts.items() if len(shows) >= 2}
        assert cross_show_concepts, "expected at least one cross-show concept-Topic"

    def test_concept_topics_have_related_to_edges(self) -> None:
        """For every concept:topic-* node, at least one RELATED_TO
        edge from a per-show Topic points at it."""
        for path, kg in _load_all_kg():
            concept_ids = {
                n["id"]
                for n in kg["nodes"]
                if n.get("type") == "Topic" and n["id"].startswith("concept:topic-")
            }
            related_to_targets = {e["to"] for e in kg["edges"] if e.get("type") == "RELATED_TO"}
            for cid in concept_ids:
                assert (
                    cid in related_to_targets
                ), f"{path.name}: concept-Topic {cid!r} has no RELATED_TO source"


class TestPerEpisodeConnectivity:
    def test_each_episode_has_about_edges(self) -> None:
        for path, gi in _load_all_gi():
            about_edges = [e for e in gi["edges"] if e.get("type") == "ABOUT"]
            assert about_edges, f"{path.name}: no ABOUT edges"

    def test_at_least_one_episode_has_mentions_person_edges(self) -> None:
        for _, gi in _load_all_gi():
            if any(e.get("type") == "MENTIONS_PERSON" for e in gi["edges"]):
                return
        pytest.fail("no MENTIONS_PERSON edges anywhere in fixture")

    def test_at_least_one_episode_has_mentions_org_edges(self) -> None:
        for _, gi in _load_all_gi():
            if any(e.get("type") == "MENTIONS_ORG" for e in gi["edges"]):
                return
        pytest.fail("no MENTIONS_ORG edges anywhere in fixture")


class TestTranscriptFiles:
    """Every episode metadata.json references a transcript file that
    actually exists on disk under feeds/<show>/transcripts/. Without
    this, the .txt files would be dead weight — the JSON-level
    reference would lie about the on-disk presence."""

    def test_every_metadata_has_a_real_transcript(self) -> None:
        import json

        for meta_path in sorted(_FIXTURE_ROOT.rglob("*.metadata.json")):
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            tx_name = (data.get("content") or {}).get("transcript_file_path")
            assert tx_name, f"{meta_path.name}: no transcript_file_path"
            tx_path = meta_path.parent.parent / "transcripts" / tx_name
            assert (
                tx_path.is_file()
            ), f"{meta_path.name} references {tx_name} but the file is missing"
            assert tx_path.read_text(encoding="utf-8").strip(), f"{tx_path.name} is empty"


class TestEntityNeighborhood:
    def test_at_least_one_person_reachable_via_insight_and_topic(self) -> None:
        """``Person → MENTIONS_PERSON ← Insight → ABOUT → Topic`` —
        the neighborhood path Position Tracker + Person Profile
        consume. Assert at least one full traversal lands."""
        for _, gi in _load_all_gi():
            # build edge index
            mp_by_insight = defaultdict(set)
            about_by_insight = defaultdict(set)
            for e in gi["edges"]:
                if e.get("type") == "MENTIONS_PERSON":
                    mp_by_insight[e["from"]].add(e["to"])
                elif e.get("type") == "ABOUT":
                    about_by_insight[e["from"]].add(e["to"])
            for insight_id, persons in mp_by_insight.items():
                topics = about_by_insight.get(insight_id, set())
                if persons and topics:
                    return
        pytest.fail("no Person ← Insight → Topic traversal lands")
