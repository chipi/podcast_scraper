"""The chaos acceptance checker must actually detect the incident it was written for.

A checker that passes everything is worse than none: it converts "untested" into "verified" in the
operator's head. So this pins BOTH directions against the real signatures.

The incident: an LLM outage that completed successfully, wrote a full corpus, and recorded
``kg_failures=0`` — while every Topic node was a summary bullet truncated into a sentence
fragment. The exit code was green. That is what the checker exists to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.check_chaos_llm_artifacts import main

pytestmark = pytest.mark.unit

_BULLET = (
    "Product development in frontier AI requires building for model capabilities two to three "
    "months ahead"
)


def _corpus(root: Path, *, metrics: dict, topics: list[str], provenance: str) -> None:
    run = root / "run_1"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run / "metadata" / "ep.metadata.json").write_text(
        json.dumps({"summary": {"bullets": [_BULLET]}}), encoding="utf-8"
    )
    (run / "metadata" / "ep.kg.json").write_text(
        json.dumps(
            {
                "extraction": {"model_version": provenance},
                "nodes": [
                    {"type": "Topic", "id": f"topic:{i}", "properties": {"label": lab}}
                    for i, lab in enumerate(topics)
                ],
            }
        ),
        encoding="utf-8",
    )


_HEALTHY_METRICS = {
    "llm_kg_calls": 4,
    "kg_failures": 0,
    "gi_insights_total": 7,
    "gi_failures": 0,
}
_SILENT_FAILURE_METRICS = {
    "llm_kg_calls": 0,
    "kg_failures": 0,
    "gi_insights_total": 0,
    "gi_failures": 0,
}


def test_a_healthy_run_passes(tmp_path: Path) -> None:
    _corpus(
        tmp_path,
        metrics=_HEALTHY_METRICS,
        topics=["ai regulation", "global oil supply chain"],
        provenance="provider:qwen3",
    )
    assert main([str(tmp_path)]) == 0


def test_the_silent_failure_signature_is_caught(tmp_path: Path) -> None:
    """Both counters zero — neither ran nor failed. The exact incident metric shape."""
    _corpus(
        tmp_path,
        metrics=_SILENT_FAILURE_METRICS,
        topics=["ai regulation"],
        provenance="provider:qwen3",
    )
    assert main([str(tmp_path)]) == 1


def test_a_truncated_bullet_as_a_topic_is_caught(tmp_path: Path) -> None:
    """The label is a strict prefix of a summary bullet — bullets-as-topics."""
    _corpus(
        tmp_path,
        metrics=_HEALTHY_METRICS,
        topics=["Product development in frontier AI requires"],
        provenance="provider:qwen3",
    )
    assert main([str(tmp_path)]) == 1


def test_the_substitution_provenance_is_caught(tmp_path: Path) -> None:
    """``topic_labels`` is minted only on the bullet-substitution path."""
    _corpus(
        tmp_path,
        metrics=_HEALTHY_METRICS,
        topics=["ai regulation"],
        provenance="topic_labels",
    )
    assert main([str(tmp_path)]) == 1


def test_a_proposition_length_topic_is_caught(tmp_path: Path) -> None:
    """Even with clean provenance, a sentence-shaped label is not a topic."""
    _corpus(
        tmp_path,
        metrics=_HEALTHY_METRICS,
        topics=["this label has far too many words to be a noun phrase at all"],
        provenance="provider:qwen3",
    )
    assert main([str(tmp_path)]) == 1


def test_a_missing_directory_is_a_usage_error_not_a_pass(tmp_path: Path) -> None:
    """Exit 2, never 0 — a typo in the path must not read as a clean chaos run."""
    assert main([str(tmp_path / "nope")]) == 2
    assert main([]) == 2
