"""Fixture-invariants for ``tests/fixtures/app-validation-corpus/v3``.

The v3 corpus is the committed, deterministic fixture the consumer app + (soon, #1168) the operator
viewer drive their tier-3 / e2e tests against. These invariants guard it from silently regressing:
every enricher surface must have real fixture data to render, and the key cross-links must exist.

Reads the committed artifacts directly (no server, no ML) — fast + hermetic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_V3 = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _envelope(name: str) -> dict:
    obj: dict = json.loads((_V3 / "enrichments" / f"{name}.json").read_text(encoding="utf-8"))
    return obj


def _data(name: str) -> dict:
    return _envelope(name).get("data") or {}


def test_all_corpus_level_enricher_artifacts_present() -> None:
    """Every corpus-scope enricher a surface reads must have a committed artifact — no mock-only
    surfaces. (topic_similarity is the ADR-108-round fixture gap this PR closes.)"""
    required = {
        "temporal_velocity",
        "topic_theme_clusters",
        "topic_consensus",
        "topic_similarity",
        "grounding_rate",
        "guest_coappearance",
        "topic_cooccurrence_corpus",
    }
    present = {p.stem for p in (_V3 / "enrichments").glob("*.json")}
    missing = required - present
    assert not missing, f"v3 fixture missing corpus enricher artifacts: {sorted(missing)}"


def test_episode_scope_sidecars_present() -> None:
    """Every episode carries insight_density + insight_sentiment sidecars."""
    n_ep = len(list(_V3.glob("feeds/*/run_*/metadata/*.bridge.json")))
    assert n_ep > 0
    for kind in ("insight_density", "insight_sentiment"):
        n = len(list(_V3.glob(f"feeds/*/run_*/metadata/enrichments/*.{kind}.json")))
        assert n == n_ep, f"{kind}: {n} sidecars for {n_ep} episodes"


def test_topic_similarity_has_neighbours() -> None:
    """The EnrichmentEdges "similar" + consumer "Similar topics" surfaces need ≥1 topic with a few
    neighbours to render + pivot."""
    topics = _data("topic_similarity").get("topics", [])
    with_neighbours = [t for t in topics if len(t.get("top_k", [])) >= 3]
    assert len(with_neighbours) >= 1, "no topic has ≥3 similarity neighbours"
    # scores well-formed + in a plausible band
    for t in with_neighbours:
        for n in t["top_k"]:
            assert 0.0 <= n["similarity"] <= 1.0


def test_topic_consensus_has_cross_person_pairs() -> None:
    """The Consensus surface (topic + person) needs ≥1 cross-person corroboration pair."""
    pairs = _data("topic_consensus").get("consensus", [])
    assert len(pairs) >= 1
    p = pairs[0]
    assert p["person_a_id"] != p["person_b_id"]
    assert p["topic_id"].startswith("topic:")


def test_topic_theme_clusters_form_and_membership_sane() -> None:
    """The theme-cluster surface needs ≥1 real cluster (≥2 topics) whose members
    reference known topics and whose declared count matches the member list."""
    d = _data("topic_theme_clusters")
    multi = [c for c in d.get("clusters", []) if c.get("member_count", 0) >= 2]
    assert multi, "no theme cluster with ≥2 members in v3"
    c = multi[0]
    assert len(c["members"]) == c["member_count"]
    assert all(m["topic_id"].startswith("topic:") for m in c["members"])


def test_insight_sentiment_has_label_spread() -> None:
    """Sentiment tint needs more than one label across the corpus (not degenerate all-neutral)."""
    labels: Counter[str] = Counter()
    for f in _V3.glob("feeds/*/run_*/metadata/enrichments/*.insight_sentiment.json"):
        d = json.loads(f.read_text(encoding="utf-8")).get("data") or {}
        labels.update(i.get("label") for i in d.get("insights", []))
    assert len(labels) >= 2, f"sentiment labels not diverse: {dict(labels)}"


def test_cross_show_topic_exists() -> None:
    """≥1 topic must span ≥3 distinct shows (the cross-show timeline / cluster flow)."""
    topic_shows: dict[str, set[str]] = {}
    for gi in _V3.glob("feeds/*/run_*/metadata/*.gi.json"):
        feed = gi.relative_to(_V3 / "feeds").parts[0]
        d = json.loads(gi.read_text(encoding="utf-8"))
        for n in d.get("nodes", []):
            nid = str(n.get("id", ""))
            if nid.startswith("topic:"):
                topic_shows.setdefault(nid, set()).add(feed)
    spanning = {t: s for t, s in topic_shows.items() if len(s) >= 3}
    assert spanning, "no topic spans ≥3 shows"
