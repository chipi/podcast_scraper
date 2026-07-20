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


def test_topic_theme_clusters_super_theme_rollup_present() -> None:
    """graph-v3 tier 7-1a — every theme cluster must carry ``super_theme_id`` +
    ``super_theme_label`` (enricher v1.1.0+). The viewer's hierarchical legend
    (tier 7-1) + top-down mount (tier 8-1) both read these fields; running
    the tier-3 fixture through an older enricher would silently drop them
    and the viewer surfaces would degrade to a flat list. Also check the
    top-level summary fields land."""
    d = _data("topic_theme_clusters")
    clusters = d.get("clusters", [])
    assert clusters, "no theme clusters in v3"
    for c in clusters:
        sid = c.get("super_theme_id", "")
        assert isinstance(sid, str) and sid.startswith(
            "sth:"
        ), f"cluster {c.get('graph_compound_parent_id')} missing super_theme_id"
        assert c.get(
            "super_theme_label"
        ), f"cluster {c.get('graph_compound_parent_id')} missing super_theme_label"
    assert d.get("super_theme_method") == "cross_cluster_lift_avg_linkage"
    n_supers = len({c["super_theme_id"] for c in clusters if c.get("super_theme_id")})
    assert d.get("super_theme_count") == n_supers
    # Between 1 and _SUPER_THEME_MAX (8) super-themes — never more than clusters.
    assert 1 <= n_supers <= 8
    assert n_supers <= len(clusters)


def test_temporal_velocity_has_mention_signal() -> None:
    """Velocity/momentum surfaces need ≥1 topic carrying real mention counts (not an empty series)."""
    topics = _data("temporal_velocity").get("topics", [])
    assert any(
        (t.get("total") or 0) >= 1 for t in topics
    ), "no temporal_velocity topic has mentions"


def test_topic_cooccurrence_corpus_has_pairs_with_lift() -> None:
    """The "co-occurs with" / "discussed alongside" surfaces need ≥1 pair carrying lift + pmi."""
    pairs = _data("topic_cooccurrence_corpus").get("pairs", [])
    assert pairs, "no topic co-occurrence pairs"
    p = pairs[0]
    assert "lift" in p and "pmi" in p
    assert p["topic_a_id"] != p["topic_b_id"]


def test_grounding_rate_discriminates() -> None:
    """The grounding-rate row needs real, discriminating rates (not all-0 / all-1), each in [0, 1]."""
    persons = _data("grounding_rate").get("persons", [])
    assert len(persons) >= 2, persons
    rates = [p["rate"] for p in persons]
    assert all(0.0 <= r <= 1.0 for r in rates)
    assert min(rates) < max(
        rates
    ), f"grounding_rate does not discriminate across the corpus: {rates}"


def test_guest_coappearance_has_pairs() -> None:
    """The "co-appears with" row needs ≥1 real cross-person pair."""
    pairs = _data("guest_coappearance").get("pairs", [])
    assert pairs, "no guest co-appearance pairs"
    p = pairs[0]
    assert p["person_a_id"] != p["person_b_id"]
    assert (p.get("episode_count") or 0) >= 1


def test_guest_coappearance_person_communities_present() -> None:
    """graph-v3 tier 7-4 — the ``communities[]`` rollup (enricher v1.1.0+)
    feeds the viewer's Person community underlay lens. Each community must
    carry ``community_id`` / ``community_label`` / ``member_ids`` and
    ``member_count`` must equal ``len(member_ids)``. Between 1 and N pairs
    worth of communities."""
    d = _data("guest_coappearance")
    communities = d.get("communities", [])
    assert communities, "no guest_coappearance communities in v3"
    assert d.get("community_method") == "connected_components_threshold"
    assert (d.get("community_min_pair") or 0) >= 1
    assert d.get("community_count") == len(communities)
    for c in communities:
        cid = c.get("community_id", "")
        assert isinstance(cid, str) and cid.startswith(
            "pco:"
        ), f"community {cid!r} missing pco: prefix"
        assert c.get("community_label"), f"community {cid} missing label"
        members = c.get("member_ids") or []
        assert len(members) >= 2, f"community {cid} has <2 members"
        assert c.get("member_count") == len(members)
        assert all(isinstance(m, str) and m.startswith("person:") for m in members)


def test_insight_density_bins_insights() -> None:
    """The insight-density strip needs ≥1 episode whose insights bin into segments that sum back to
    the episode total (not an empty/degenerate strip)."""
    seen_nonzero = False
    for f in _V3.glob("feeds/*/run_*/metadata/enrichments/*.insight_density.json"):
        d = json.loads(f.read_text(encoding="utf-8")).get("data") or {}
        counts = d.get("counts") or {}
        total = d.get("total_insights") or 0
        assert sum(counts.values()) == total, f"{f.name}: segment counts != total_insights"
        if total >= 1:
            seen_nonzero = True
    assert seen_nonzero, "no episode carries an insight-density signal"


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
