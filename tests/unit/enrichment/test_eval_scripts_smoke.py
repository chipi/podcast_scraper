"""Smoke tests for the chunk-2/3/4 eval scoring scripts.

These run --help and the no-gold path so the scaffolding is exercised
in CI even before gold fixtures are populated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO_ROOT / "scripts" / "eval" / "score"

_SCRIPTS_TO_CHECK = [
    "enrichment_deterministic.py",
    "enrichment_topic_similarity.py",
    "enrichment_nli_contradiction.py",
]


@pytest.mark.parametrize("script_name", _SCRIPTS_TO_CHECK)
def test_eval_script_help_exits_zero(script_name: str) -> None:
    script = _SCRIPTS / script_name
    assert script.is_file(), f"missing eval scaffolding at {script}"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_enrichment_deterministic_no_gold_exits_zero(tmp_path: Path) -> None:
    """Empty gold dir → status=no_gold + exit 0 (scaffolding mode)."""
    script = _SCRIPTS / "enrichment_deterministic.py"
    result = subprocess.run(
        [sys.executable, str(script), "--gold", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "no_gold" in result.stdout


def test_enrichment_deterministic_match_against_gold(tmp_path: Path) -> None:
    """End-to-end: gold envelope present + matching corpus envelope → exit 0."""
    import json

    gold = tmp_path / "gold"
    gold.mkdir()
    payload = {"foo": {"k": 1}, "rows": [{"id": "a"}, {"id": "b"}]}
    (gold / "x.gold.json").write_text(json.dumps({"data": payload}), encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "enrichments").mkdir(parents=True)
    # Same data shape, list order shuffled — canonicalisation should match.
    (corpus / "enrichments" / "x.json").write_text(
        json.dumps({"data": {"foo": {"k": 1}, "rows": [{"id": "b"}, {"id": "a"}]}}),
        encoding="utf-8",
    )
    script = _SCRIPTS / "enrichment_deterministic.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--gold",
            str(gold),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"status": "match"' in result.stdout


def test_enrichment_deterministic_mismatch_exits_one(tmp_path: Path) -> None:
    """Gold envelope present but corpus envelope differs → exit 1."""
    import json

    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "x.gold.json").write_text(json.dumps({"data": {"k": 1}}), encoding="utf-8")
    corpus = tmp_path / "corpus"
    (corpus / "enrichments").mkdir(parents=True)
    (corpus / "enrichments" / "x.json").write_text(json.dumps({"data": {"k": 2}}), encoding="utf-8")
    script = _SCRIPTS / "enrichment_deterministic.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--gold",
            str(gold),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert '"status": "mismatch"' in result.stdout


def test_enrichment_topic_similarity_scores_recall_at_k(tmp_path: Path) -> None:
    """Wires the recall@K loop end-to-end."""
    import json

    corpus = tmp_path / "corpus"
    out = corpus / "enrichments"
    out.mkdir(parents=True)
    (out / "topic_similarity.json").write_text(
        json.dumps(
            {
                "data": {
                    "topics": [
                        {
                            "topic_id": "topic:ai",
                            "top_k": [
                                {"topic_id": "topic:ml"},
                                {"topic_id": "topic:safety"},
                                {"topic_id": "topic:noise"},
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "g.jsonl").write_text(
        json.dumps(
            {
                "topic_id": "topic:ai",
                "expected_neighbours": ["topic:ml", "topic:safety", "topic:rlhf"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = _SCRIPTS / "enrichment_topic_similarity.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--gold",
            str(gold),
            "--top-k",
            "3",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["status"] == "scored"
    # 2 of 3 expected neighbours present in the top-3 → recall = 2/3.
    assert abs(body["macro_recall@3"] - 0.6667) < 0.01


def test_enrichment_nli_contradiction_scores_precision_recall(tmp_path: Path) -> None:
    """Wires the P/R/F1 loop end-to-end."""
    import json

    corpus = tmp_path / "corpus"
    out = corpus / "enrichments"
    out.mkdir(parents=True)
    # Model detected 2 pairs as contradictions.
    (out / "nli_contradiction.json").write_text(
        json.dumps(
            {
                "data": {
                    "contradictions": [
                        {
                            "insight_a_id": "i1",
                            "insight_b_id": "i2",
                            "contradiction_score": 0.92,
                        },
                        {
                            "insight_a_id": "i3",
                            "insight_b_id": "i4",
                            "contradiction_score": 0.85,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    gold = tmp_path / "gold"
    gold.mkdir()
    # Gold labels: 2 contradictions, 1 neutral, 1 entailment.
    # Model finds i1+i2 (TP) and i3+i4 (FP — labeled neutral).
    # Missing: i5+i6 contradiction (FN).
    (gold / "g.jsonl").write_text(
        "\n".join(
            [
                '{"insight_a_id": "i1", "insight_b_id": "i2", "label": "contradiction"}',
                '{"insight_a_id": "i3", "insight_b_id": "i4", "label": "neutral"}',
                '{"insight_a_id": "i5", "insight_b_id": "i6", "label": "contradiction"}',
                '{"insight_a_id": "i7", "insight_b_id": "i8", "label": "entailment"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script = _SCRIPTS / "enrichment_nli_contradiction.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--gold",
            str(gold),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["true_positives"] == 1
    assert body["false_positives"] == 1
    assert body["false_negatives"] == 1
    assert body["precision"] == 0.5
    assert body["recall"] == 0.5
    assert body["f1"] == 0.5


def test_enrichment_topic_similarity_missing_corpus_output_exits_one(
    tmp_path: Path,
) -> None:
    """topic_similarity scorer requires the corpus to have produced
    enrichments/topic_similarity.json — exits 1 when missing."""
    script = _SCRIPTS / "enrichment_topic_similarity.py"
    result = subprocess.run(
        [sys.executable, str(script), "--corpus", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "no_corpus_output" in result.stdout


def test_enrichment_nli_contradiction_missing_corpus_output_exits_one(
    tmp_path: Path,
) -> None:
    """nli_contradiction scorer requires enrichments/nli_contradiction.json
    on disk — exits 1 when missing."""
    script = _SCRIPTS / "enrichment_nli_contradiction.py"
    result = subprocess.run(
        [sys.executable, str(script), "--corpus", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "no_corpus_output" in result.stdout


def test_enrichment_deterministic_gold_present_without_corpus_exits_two(
    tmp_path: Path,
) -> None:
    """Gold files exist but --corpus omitted/missing → exit 2 (invocation)."""
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "example.gold.json").write_text('{"data": {}}', encoding="utf-8")
    script = _SCRIPTS / "enrichment_deterministic.py"
    result = subprocess.run(
        [sys.executable, str(script), "--gold", str(gold)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "no_corpus" in result.stdout
