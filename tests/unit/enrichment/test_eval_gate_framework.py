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

from podcast_scraper.enrichment.eval import (
    admitted_enricher_ids,
    evaluate_gate,
    EXPECTED_ENRICHMENT_KEY,
    gold_for,
    load_latest_eval_metrics,
    metrics_by_enricher,
    register_builtin_scorers,
    run_scorers,
    ScorerRegistry,
)
from podcast_scraper.enrichment.eval.admission import (
    admit_enrichers,
    write_gate_metrics,
)
from podcast_scraper.enrichment.eval.gold import all_gold_enricher_ids, collect_episode_gold
from podcast_scraper.enrichment.protocol import AccuracyGateRule, AccuracyGateSpec

# --------------------------------------------------------------------------- #
# Scorer registry + reference scorers
# --------------------------------------------------------------------------- #


def _registry() -> ScorerRegistry:
    reg = ScorerRegistry()
    register_builtin_scorers(reg)
    return reg


def test_run_scorers_skips_when_output_or_gold_absent() -> None:
    reg = _registry()
    results = {r.enricher_id: r for r in run_scorers(reg, {}, {})}
    assert all(r.skipped for r in results.values())
    # skipped results carry no metrics → excluded from the gate's input
    assert metrics_by_enricher(list(results.values())) == {}


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


# --------------------------------------------------------------------------- #
# Integration: profile_sets membership is now gate-driven
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The closed loop: scorers → write_gate_metrics → load → gate
# --------------------------------------------------------------------------- #


def test_write_gate_metrics_round_trips(tmp_path: Path) -> None:
    paths = write_gate_metrics(
        {"topic_similarity": {"precision_at_k": 0.8}}, eval_root=tmp_path, run_id="r1"
    )
    assert len(paths) == 1 and paths[0].name == "gate_metrics.json"
    assert load_latest_eval_metrics(tmp_path) == {"topic_similarity": {"precision_at_k": 0.8}}


def test_nli_auto_promotes_when_passing_eval_recorded(tmp_path: Path) -> None:
    # The payoff: a passing precision written to data/eval promotes nli with NO code edit.
    write_gate_metrics({"topic_consensus": {"precision": 0.9}}, eval_root=tmp_path)
    res = admit_enrichers(
        ["grounding_rate", "topic_similarity", "topic_consensus"], eval_root=tmp_path
    )
    assert "topic_consensus" in res.admitted  # auto-promoted by the recorded eval


# --------------------------------------------------------------------------- #
# E2: shipped package gate metrics are found by default resolution (no eval_root)
# --------------------------------------------------------------------------- #


def test_default_resolution_finds_topic_consensus_gate_metrics() -> None:
    # Regression: _default_eval_root() previously resolved to repo-root data/eval via
    # parents[4], which is absent in the installed image (data/ is excluded by .dockerignore
    # and Dockerfile COPY). The fix packages the committed gate_metrics under
    # podcast_scraper.enrichment.eval.gate_metrics and resolves via importlib.resources —
    # deploy-independent, always present in site-packages.
    #
    # This test exercises the full default-resolution path (no eval_root override) and
    # asserts that topic_consensus metrics are present with at least the precision key
    # that its gate requires (on_missing_data="reject", metric_name="precision").
    #
    # Full in-image verification requires a docker build; this test proves the
    # resolution logic + that the metrics load from the shipped package location.
    metrics = load_latest_eval_metrics()  # uses _default_eval_root(), no override
    assert "topic_consensus" in metrics, (
        "_default_eval_root() did not find topic_consensus gate metrics — "
        "the shipped package data may be missing or importlib.resources resolution failed"
    )
    assert "precision" in metrics["topic_consensus"], (
        "topic_consensus gate_metrics.json is missing the 'precision' key "
        "required by its AccuracyGateSpec rule"
    )
    assert (
        metrics["topic_consensus"]["precision"] > 0
    ), "topic_consensus precision must be > 0 (shipped value is 0.9091)"


def test_topic_consensus_admitted_via_default_resolution() -> None:
    # Verifies that topic_consensus is ADMITTED (not rejected) when admit_enrichers
    # uses default resolution — the path that runs in production.
    res = admit_enrichers(["topic_consensus"])  # no eval_root override
    assert res.is_admitted("topic_consensus"), (
        "topic_consensus should be admitted via default package-data resolution; "
        "check that gate_metrics.json is included in pyproject package-data globs"
    )
