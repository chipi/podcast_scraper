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
    assert mocks.get("schema_version") == "1"


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
    resp = mocks["scenarios"]["enriched-answer"]["response"]
    enriched = resp.get("enriched")
    assert enriched, "enriched-answer scenario missing 'enriched' block"
    assert enriched.get("grounded") is True, "enriched.grounded must be True"
    sources = enriched.get("sources", [])
    assert sources, "enriched.sources must not be empty"
    for src in sources:
        assert (
            src.get("grounded") is True
        ), f"every enriched source must be grounded=True (found {src})"


def test_operator_cluster_has_min_five_members(mocks: dict[str, Any]) -> None:
    op = mocks["scenarios"]["operator-cluster"]["response"]["operator_result"]
    assert op.get("kind") == "cluster"
    clusters = op.get("clusters", [])
    assert clusters, "no clusters in operator-cluster scenario"
    # RFC-107 §T1 / S0 acceptance: at least one cluster with ≥5 members.
    assert any(
        c.get("count", 0) >= 5 for c in clusters
    ), f"no cluster has ≥5 members (counts: {[c.get('count') for c in clusters]})"


def test_operator_consensus_has_shipped_tuple_shape(mocks: dict[str, Any]) -> None:
    op = mocks["scenarios"]["operator-consensus"]["response"]["operator_result"]
    assert op.get("kind") == "consensus"
    pairs = op.get("pairs", [])
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
