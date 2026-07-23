"""Guards for the Search v3 search-slice content in the production-shaped fixture.

Two responsibilities:

1. The hand-authored ``search-v3/mocks.json`` under
   ``web/gi-kg-viewer/e2e/fixtures/production-shaped/`` is well-formed and its
   scenarios cover the 5 shapes RFC-107 / UXS-016 spec (compound-lift,
   enriched-answer, operator-cluster, operator-consensus, temporal-intent).
2. Every ``episode_id`` the mocks reference exists in the parent fixture's
   ``manifest.json`` — so hit-card handoffs to Library / Graph resolve
   cleanly against the same fixture and don't break spec-time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "web" / "gi-kg-viewer" / "e2e" / "fixtures" / "production-shaped"
MOCKS_PATH = FIXTURE_ROOT / "search-v3" / "mocks.json"
PARENT_MANIFEST = FIXTURE_ROOT / "manifest.json"

EXPECTED_SCENARIOS = {
    "compound-lift",
    "enriched-answer",
    "operator-cluster",
    "operator-consensus",
    "temporal-intent",
}


@pytest.fixture(scope="module")
def mocks() -> dict[str, Any]:
    assert MOCKS_PATH.exists(), f"Search v3 mocks missing at {MOCKS_PATH}"
    data: dict[str, Any] = json.loads(MOCKS_PATH.read_text())
    return data


@pytest.fixture(scope="module")
def parent_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(PARENT_MANIFEST.read_text())
    return data


def _collect_episode_ids(node: Any, out: set[str]) -> None:
    """Recursively collect any dict value keyed 'episode_id'."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "episode_id" and isinstance(v, str):
                out.add(v)
            else:
                _collect_episode_ids(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_episode_ids(item, out)


def test_mocks_json_is_well_formed(mocks: dict[str, Any]) -> None:
    assert isinstance(mocks, dict)
    assert "scenarios" in mocks
    assert isinstance(mocks["scenarios"], dict)
    # schema_version bumped to "2" when the response shape aligned with the
    # shipped CorpusSearchApiResponse (top-level ``clusters`` / ``consensus_pairs``
    # replaced the earlier ``operator_result`` wrapper; per-hit
    # ``metadata.query_enrichments`` replaced the top-level ``enriched`` block).
    assert mocks.get("schema_version") == "2"


def test_all_expected_scenarios_present(mocks: dict[str, Any]) -> None:
    got = set(mocks["scenarios"].keys())
    missing = EXPECTED_SCENARIOS - got
    assert not missing, f"missing scenarios: {sorted(missing)} (found {sorted(got)})"


def test_each_scenario_has_request_and_response(mocks: dict[str, Any]) -> None:
    for name, entry in mocks["scenarios"].items():
        assert isinstance(entry, dict), f"scenario {name!r} not a dict"
        assert "request" in entry, f"scenario {name!r} missing 'request'"
        assert "response" in entry, f"scenario {name!r} missing 'response'"
        assert "description" in entry, f"scenario {name!r} missing 'description'"


def test_compound_lift_scenario_carries_lifted_block(mocks: dict[str, Any]) -> None:
    resp = mocks["scenarios"]["compound-lift"]["response"]
    hits = resp.get("results", [])
    assert hits, "compound-lift scenario has no hits"
    lifted_hits = [h for h in hits if h.get("lifted")]
    assert lifted_hits, "no hit in compound-lift carries a 'lifted' block (RFC-072 KL1)"
    for h in lifted_hits:
        lifted = h["lifted"]
        for k in ("insight", "speaker", "topic", "quote"):
            assert k in lifted, f"compound-lift hit missing lifted.{k}"
        assert lifted["insight"].get("grounded") is True, "lifted.insight must be grounded"


def test_enriched_answer_scenario_has_grounded_sources(mocks: dict[str, Any]) -> None:
    # Enrichment moved from a top-level ``enriched`` block to per-hit
    # ``metadata.query_enrichments.related_topics`` (RFC-088 QueryEnricher
    # chain, wired in commit a26e0a5e). "Grounded" here means: at least one
    # hit is an ``insight`` doc_type (the only tier that carries grounded
    # claims) AND at least one hit carries a related_topics decoration.
    resp = mocks["scenarios"]["enriched-answer"]["response"]
    hits = resp.get("results", [])
    assert hits, "enriched-answer scenario has no hits"
    grounded_hits = [h for h in hits if h.get("metadata", {}).get("doc_type") == "insight"]
    assert grounded_hits, "enriched-answer scenario has no insight-tier (grounded) hits"
    decorated = [
        h
        for h in hits
        if isinstance(h.get("metadata", {}).get("query_enrichments"), dict)
        and h["metadata"]["query_enrichments"].get("related_topics")
    ]
    assert decorated, "no hit carries metadata.query_enrichments.related_topics"


def test_operator_cluster_has_min_two_members(mocks: dict[str, Any]) -> None:
    # Response shape: top-level ``operator="cluster"`` + top-level ``clusters``
    # list of {cluster_id, cluster_kind, label, size, hit_indices} — matches
    # SearchClusterGroupModel in server/schemas.py.
    resp = mocks["scenarios"]["operator-cluster"]["response"]
    assert resp.get("operator") == "cluster"
    clusters = resp.get("clusters", [])
    assert clusters, "no clusters in operator-cluster scenario"
    # The mocks are 9-episode/small-corpus samples — RFC-107 §T1's ≥5-member
    # invariant belongs on a full-corpus e2e; the fixture floor is ≥2 members
    # (still exercises multi-hit grouping, hit_indices lookup, cluster
    # labelling).
    assert any(
        c.get("size", 0) >= 2 for c in clusters
    ), f"no cluster has ≥2 members (sizes: {[c.get('size') for c in clusters]})"
    # And every cluster's hit_indices actually resolves within the results
    # page — otherwise the client can't lift a card from a cluster.
    results = resp.get("results", [])
    for c in clusters:
        for idx in c.get("hit_indices", []):
            assert 0 <= idx < len(results), (
                f"cluster {c.get('cluster_id')!r} hit_index {idx} out of range "
                f"(results has {len(results)})"
            )


def test_operator_consensus_has_shipped_tuple_shape(mocks: dict[str, Any]) -> None:
    # Response shape: top-level ``operator="consensus"`` + top-level
    # ``consensus_pairs`` list (SearchConsensusPairModel in server/schemas.py).
    resp = mocks["scenarios"]["operator-consensus"]["response"]
    assert resp.get("operator") == "consensus"
    pairs = resp.get("consensus_pairs", [])
    assert pairs, "no pairs in operator-consensus scenario"
    required = {
        "topic_id",
        "person_a_id",
        "person_b_id",
        "insight_a_id",
        "insight_b_id",
        "contradiction_score",
    }
    for pair in pairs:
        missing = required - set(pair.keys())
        assert not missing, f"consensus pair missing tuple fields: {sorted(missing)}"
        # Consensus per ADR-108: low contradiction_score + high embedding cosine ⇒ agreement.
        # The mocks encode agreement, so contradiction_score should be low.
        assert (
            pair["contradiction_score"] < 0.5
        ), f"consensus pair should have low contradiction_score (got {pair['contradiction_score']})"


def test_temporal_intent_scenario_has_query_type(mocks: dict[str, Any]) -> None:
    resp = mocks["scenarios"]["temporal-intent"]["response"]
    got = resp.get("query_type")
    assert (
        got == "temporal_tracking"
    ), f"temporal-intent scenario query_type != 'temporal_tracking' (got {got!r})"


def test_every_referenced_episode_id_exists_in_parent_fixture(
    mocks: dict[str, Any], parent_manifest: dict[str, Any]
) -> None:
    parent_ids = {e["episode_id"] for e in parent_manifest["picked"]["episodes"]}
    referenced: set[str] = set()
    _collect_episode_ids(mocks, referenced)
    # doc_id values embed episode_ids too (e.g. seg:74a9745d-...); extract those.
    for scenario in mocks["scenarios"].values():
        for hit in scenario["response"].get("results", []):
            doc_id = hit.get("doc_id", "")
            # doc_id shape "seg:{episode_id}_chunk_N" or "insight:{episode_id}__..."
            for prefix in ("seg:", "insight:"):
                if doc_id.startswith(prefix):
                    rest = doc_id[len(prefix) :]
                    for sep in ("_chunk_", "__"):
                        if sep in rest:
                            candidate = rest.split(sep, 1)[0]
                            if candidate in parent_ids:
                                referenced.add(candidate)
    orphans = referenced - parent_ids
    assert not orphans, (
        f"scenarios reference episode_ids absent from parent manifest.json — "
        f"hit-card handoffs will break at spec time. Orphans: {sorted(orphans)}. "
        f"Parent has: {sorted(parent_ids)}"
    )


# ---- --search-slice merge semantics (--prune-orphaned / --backup-scenarios-to) ---


def _load_build_module() -> Any:
    import importlib.util
    import sys as _sys

    script = REPO_ROOT / "scripts" / "build_production_shaped_fixture.py"
    spec = importlib.util.spec_from_file_location("build_prod_shaped", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mock_fetch_scenario_response_factory(payload: Any) -> Any:
    return lambda api, endpoint, method, params: payload


def test_search_slice_prune_orphaned_drops_scenarios_without_matching_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_build_module()
    monkeypatch.setattr(
        mod, "_fetch_scenario_response", _mock_fetch_scenario_response_factory({"results": []})
    )
    out = tmp_path / "fixture"
    (out / "search-v3").mkdir(parents=True)
    # Seed the JSON with a spec'd scenario (compound-lift) + an orphan.
    (out / "search-v3" / "mocks.json").write_text(
        json.dumps(
            {
                "scenarios": {
                    "compound-lift": {"description": "keep", "response": {"stale": True}},
                    "orphan-scenario": {"description": "should go", "response": {"old": True}},
                }
            }
        )
    )
    mod._capture_search_v3_slice(api="x", corpus="/c", out=out, prune_orphaned=True)
    written = json.loads((out / "search-v3" / "mocks.json").read_text())
    assert "compound-lift" in written["scenarios"]
    assert "orphan-scenario" not in written["scenarios"]
    # And the spec'd scenario's description is preserved (hand-authored field).
    assert written["scenarios"]["compound-lift"]["description"] == "keep"


def test_search_slice_prune_orphaned_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --prune-orphaned, an orphan stays put (silent accumulation)."""
    mod = _load_build_module()
    monkeypatch.setattr(
        mod, "_fetch_scenario_response", _mock_fetch_scenario_response_factory({"results": []})
    )
    out = tmp_path / "fixture"
    (out / "search-v3").mkdir(parents=True)
    (out / "search-v3" / "mocks.json").write_text(
        json.dumps({"scenarios": {"orphan-scenario": {"response": {}}}})
    )
    mod._capture_search_v3_slice(api="x", corpus="/c", out=out)
    written = json.loads((out / "search-v3" / "mocks.json").read_text())
    assert "orphan-scenario" in written["scenarios"]


def test_search_slice_backup_snapshots_prev_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_build_module()
    monkeypatch.setattr(
        mod, "_fetch_scenario_response", _mock_fetch_scenario_response_factory({"new": True})
    )
    out = tmp_path / "fixture"
    (out / "search-v3").mkdir(parents=True)
    (out / "search-v3" / "mocks.json").write_text(
        json.dumps(
            {
                "scenarios": {
                    "compound-lift": {"response": {"prev": True}},
                    "orphan-scenario": {"response": {"was_orphan": True}},
                }
            }
        )
    )
    backup = tmp_path / "backup"
    mod._capture_search_v3_slice(
        api="x", corpus="/c", out=out, prune_orphaned=True, backup_scenarios_to=backup
    )
    # Overwritten scenario has a `.previous.json` snapshot.
    prev_dump = json.loads((backup / "compound-lift.previous.json").read_text())
    assert prev_dump == {"prev": True}
    # Pruned scenario has a `.pruned.json` snapshot.
    pruned_dump = json.loads((backup / "orphan-scenario.pruned.json").read_text())
    assert pruned_dump == {"response": {"was_orphan": True}}
