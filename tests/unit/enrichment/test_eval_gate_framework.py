"""Enricher accuracy-eval + gating framework (RFC-088 gate amendment).

Covers the grading side (scorer protocol/registry/reference scorers), the
generic gold accessors, the gate (mirror of the provider RegressionRule gate),
and the admission cascade that makes ``data/eval`` drive registry/profile
membership — including that ``topic_consensus``'s exclusion is now a
data-driven gate decision, not a hardcoded list edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.eval import (
    admitted_enricher_ids,
    evaluate_gate,
    EXPECTED_ENRICHMENT_KEY,
    gold_for,
    known_enricher_manifests,
    load_latest_eval_metrics,
    metrics_by_enricher,
    register_builtin_scorers,
    run_scorers,
    ScorerRegistry,
)
from podcast_scraper.enrichment.eval.admission import (
    admit_enrichers,
    gate_specs_from_manifests,
    write_gate_metrics,
)
from podcast_scraper.enrichment.eval.gold import all_gold_enricher_ids, collect_episode_gold
from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile
from podcast_scraper.enrichment.protocol import AccuracyGateRule, AccuracyGateSpec

# --------------------------------------------------------------------------- #
# Scorer registry + reference scorers
# --------------------------------------------------------------------------- #


def _registry() -> ScorerRegistry:
    reg = ScorerRegistry()
    register_builtin_scorers(reg)
    return reg


def test_scorer_registry_register_get_and_dup_raises() -> None:
    reg = _registry()
    assert set(reg.all_enricher_ids()) == {
        "grounding_rate",
        "topic_similarity",
        "guest_coappearance",
    }
    assert reg.has("grounding_rate")
    with pytest.raises(ValueError, match="already registered"):
        register_builtin_scorers(reg)  # second registration collides


def test_reference_scorers_produce_expected_metrics() -> None:
    reg = _registry()
    outputs: dict[str, dict[str, Any]] = {
        "grounding_rate": {"persons": [{"grounded_insights": 8, "total_insights": 10}]},
        "guest_coappearance": {
            "pairs": [
                {"person_a_id": "p:a", "person_b_id": "p:b"},
                {"person_a_id": "p:a", "person_b_id": "p:c"},
            ]
        },
        "topic_similarity": {
            "topics": [
                {"topic_id": "t:ml", "top_k": [{"topic_id": "t:llm"}, {"topic_id": "t:nlp"}]}
            ]
        },
    }
    gold: dict[str, dict[str, Any]] = {
        "grounding_rate": {"expected_rate": 0.8, "tolerance": 0.1},
        "guest_coappearance": {"expected_pairs": [["p:a", "p:b"]]},
        "topic_similarity": {"expected_neighbours": {"t:ml": ["t:llm", "t:xyz"]}},
    }
    metrics = metrics_by_enricher(run_scorers(reg, outputs, gold))
    assert metrics["grounding_rate"]["within_tolerance"] == 1.0
    assert metrics["grounding_rate"]["abs_error"] == 0.0
    # pair {a,b} matched of {a,b},{a,c} predicted → precision .5, recall 1.
    assert metrics["guest_coappearance"]["precision"] == 0.5
    assert metrics["guest_coappearance"]["recall"] == 1.0
    # 1 of 2 emitted neighbours in expected → precision .5; 1 of 2 expected → recall .5
    assert metrics["topic_similarity"]["precision_at_k"] == 0.5
    assert metrics["topic_similarity"]["recall_at_k"] == 0.5


def test_run_scorers_skips_when_output_or_gold_absent() -> None:
    reg = _registry()
    results = {r.enricher_id: r for r in run_scorers(reg, {}, {})}
    assert all(r.skipped for r in results.values())
    # skipped results carry no metrics → excluded from the gate's input
    assert metrics_by_enricher(list(results.values())) == {}


def test_topic_similarity_scorer_respects_config_top_k() -> None:
    reg = _registry()
    output = {
        "topics": [
            {
                "topic_id": "t:ml",
                "top_k": [{"topic_id": "t:a"}, {"topic_id": "t:b"}, {"topic_id": "t:c"}],
            }
        ]
    }
    gold = {"expected_neighbours": {"t:ml": ["t:a"]}}
    # top_k=1 → only t:a graded → precision 1.0 (vs 1/3 without the cap)
    res = reg.get("topic_similarity").score(output=output, gold=gold, config={"top_k": 1})
    assert res.metrics["precision_at_k"] == 1.0


# --------------------------------------------------------------------------- #
# Generic gold — keyed by enricher id, no per-enricher field names
# --------------------------------------------------------------------------- #


def test_gold_for_wrapped_and_unwrapped() -> None:
    wrapped = {EXPECTED_ENRICHMENT_KEY: {"grounding_rate": {"expected_rate": 0.7}}}
    assert gold_for(wrapped, "grounding_rate") == {"expected_rate": 0.7}
    # already-unwrapped expected_enrichment mapping
    assert gold_for({"grounding_rate": {"expected_rate": 0.7}}, "grounding_rate") == {
        "expected_rate": 0.7
    }
    assert gold_for(wrapped, "absent") is None


def test_collect_episode_gold_and_id_listing() -> None:
    gts = [
        {"episode_id": "e1", EXPECTED_ENRICHMENT_KEY: {"grounding_rate": {"expected_rate": 0.5}}},
        {"episode_id": "e2", EXPECTED_ENRICHMENT_KEY: {}},
    ]
    got = collect_episode_gold(gts, "grounding_rate")
    assert got == {"e1": {"expected_rate": 0.5}}
    assert all_gold_enricher_ids(gts[0]) == ["grounding_rate"]


# --------------------------------------------------------------------------- #
# Gate — absolute-floor acceptance (mirrors provider RegressionRule)
# --------------------------------------------------------------------------- #


def test_gate_no_spec_promotes() -> None:
    d = evaluate_gate("x", None, None)
    assert d.promoted and "no accuracy gate" in d.reason


def test_gate_missing_data_policy() -> None:
    reject = AccuracyGateSpec(rules=(AccuracyGateRule("precision", 0.5),), on_missing_data="reject")
    admit = AccuracyGateSpec(rules=(AccuracyGateRule("precision", 0.5),), on_missing_data="admit")
    assert evaluate_gate("x", reject, None).promoted is False
    assert evaluate_gate("x", reject, {}).promoted is False
    assert evaluate_gate("x", admit, None).promoted is True


def test_gate_pass_fail_and_reason() -> None:
    spec = AccuracyGateSpec(rules=(AccuracyGateRule("precision", 0.5),))
    passed = evaluate_gate("x", spec, {"precision": 0.9})
    assert passed.promoted and passed.had_metrics
    failed = evaluate_gate("x", spec, {"precision": 0.2})
    assert failed.promoted is False
    assert "precision 0.20 < 0.50" in failed.reason


def test_gate_warning_severity_is_advisory_not_blocking() -> None:
    spec = AccuracyGateSpec(
        rules=(
            AccuracyGateRule("precision", 0.5, severity="error"),
            AccuracyGateRule("recall", 0.9, severity="warning"),
        )
    )
    d = evaluate_gate("x", spec, {"precision": 0.6, "recall": 0.1})
    assert d.promoted is True  # recall miss is a warning → does not gate
    assert any(v.metric_name == "recall" for v in d.violations)


# --------------------------------------------------------------------------- #
# Admission cascade
# --------------------------------------------------------------------------- #


def test_admission_pure_no_gate_admits_gated_drops() -> None:
    specs = {
        "a": None,  # no gate → admitted
        "b": AccuracyGateSpec(rules=(AccuracyGateRule("p", 0.5),), on_missing_data="reject"),
    }
    res = admitted_enricher_ids(["a", "b"], specs, {})
    assert res.admitted == ["a"]
    assert res.is_admitted("a") and not res.is_admitted("b")
    # passing metric for b promotes it
    res2 = admitted_enricher_ids(["a", "b"], specs, {"b": {"p": 0.9}})
    assert res2.admitted == ["a", "b"]


def test_known_manifests_cover_all_nine_and_gated_ml_declare_gates() -> None:
    mans = known_enricher_manifests()
    # 7 deterministic (incl. insight_sentiment) + topic_similarity + topic_consensus.
    assert len(mans) == 9
    gate = mans["topic_consensus"].accuracy_gate  # the one gated ML enricher
    assert gate is not None
    assert gate.on_missing_data == "reject"
    # deterministic + topic_similarity declare no gate → always admitted
    assert mans["grounding_rate"].accuracy_gate is None
    assert mans["topic_similarity"].accuracy_gate is None


def test_load_latest_eval_metrics_absent_and_present(tmp_path: Path) -> None:
    assert load_latest_eval_metrics(tmp_path) == {}  # nothing recorded
    d = tmp_path / "enrichment" / "topic_similarity"
    d.mkdir(parents=True)
    (d / "gate_metrics.json").write_text(
        json.dumps({"enricher_id": "topic_similarity", "metrics": {"precision_at_k": 0.8}}),
        encoding="utf-8",
    )
    got = load_latest_eval_metrics(tmp_path)
    assert got == {"topic_similarity": {"precision_at_k": 0.8}}


def test_admit_enrichers_gates_nli_by_default(tmp_path: Path) -> None:
    # Hermetic: no gate_metrics under this eval_root → nli rejected via on_missing=reject.
    # (Uses an injected root, not the real data/eval, which now carries a measured 0%.)
    res = admit_enrichers(
        ["grounding_rate", "topic_similarity", "topic_consensus"], eval_root=tmp_path
    )
    assert "topic_consensus" not in res.admitted
    assert "grounding_rate" in res.admitted and "topic_similarity" in res.admitted


def test_gate_specs_from_manifests_projects_gate_only() -> None:
    specs = gate_specs_from_manifests(known_enricher_manifests())
    assert specs["topic_consensus"] is not None
    assert specs["grounding_rate"] is None


# --------------------------------------------------------------------------- #
# Integration: profile_sets membership is now gate-driven
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile", ["cloud_thin", "cloud_quality", "prod_dgx_balanced"])
def test_profile_sets_gate_is_data_driven(profile: str) -> None:
    # topic_consensus cleared its eval (precision 0.91 on prod-v2, ADR-108 composite) → admitted.
    # A gated candidate with no passing eval would be excluded — membership is data-driven.
    enabled = enricher_set_for_profile(profile).enabled_enrichers
    assert "topic_consensus" in enabled  # gate cleared → admitted
    assert "topic_similarity" in enabled  # no gate → still shipped
    assert len(enabled) == 9  # 7 deterministic + topic_similarity + topic_consensus


# --------------------------------------------------------------------------- #
# The closed loop: scorers → write_gate_metrics → load → gate
# --------------------------------------------------------------------------- #


def test_write_gate_metrics_round_trips(tmp_path: Path) -> None:
    paths = write_gate_metrics(
        {"topic_similarity": {"precision_at_k": 0.8}}, eval_root=tmp_path, run_id="r1"
    )
    assert len(paths) == 1 and paths[0].name == "gate_metrics.json"
    assert load_latest_eval_metrics(tmp_path) == {"topic_similarity": {"precision_at_k": 0.8}}


def test_closed_loop_scorers_to_gate(tmp_path: Path) -> None:
    reg = _registry()
    outputs: dict[str, dict[str, Any]] = {
        "topic_similarity": {"topics": [{"topic_id": "t:ml", "top_k": [{"topic_id": "t:llm"}]}]}
    }
    gold: dict[str, dict[str, Any]] = {
        "topic_similarity": {"expected_neighbours": {"t:ml": ["t:llm"]}}
    }
    # run scorers → persist metrics → read them back → gate promotes
    write_gate_metrics(metrics_by_enricher(run_scorers(reg, outputs, gold)), eval_root=tmp_path)
    metrics = load_latest_eval_metrics(tmp_path)
    spec = AccuracyGateSpec(rules=(AccuracyGateRule("precision_at_k", 0.5),))
    assert evaluate_gate("topic_similarity", spec, metrics["topic_similarity"]).promoted is True


def test_nli_auto_promotes_when_passing_eval_recorded(tmp_path: Path) -> None:
    # The payoff: a passing precision written to data/eval promotes nli with NO code edit.
    write_gate_metrics({"topic_consensus": {"precision": 0.9}}, eval_root=tmp_path)
    res = admit_enrichers(
        ["grounding_rate", "topic_similarity", "topic_consensus"], eval_root=tmp_path
    )
    assert "topic_consensus" in res.admitted  # auto-promoted by the recorded eval
