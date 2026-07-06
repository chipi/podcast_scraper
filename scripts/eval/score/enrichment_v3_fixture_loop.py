#!/usr/bin/env python3
"""Close the enricher eval loop on the v3 fixture corpus (#1148).

Reads the deterministic enricher output envelopes baked into the built app
validation corpus (``tests/fixtures/app-validation-corpus/v3/enrichments/*.json``,
each wrapping its payload under ``data``) + the authored corpus gold
(``tests/fixtures/ground-truth/v3/expected_enrichment.gold.json``), runs the
registered accuracy scorers, and writes real ``gate_metrics.json`` under
``data/eval/enrichment/<id>/`` so the accuracy gate reads measured numbers.

This is the offline (no-ML) half of the loop: deterministic enrichers only
(guest_coappearance / grounding_rate / …). The embedding/NLI enrichers
(topic_similarity / nli) run separately with ``--with-ml``.

Run::

    python scripts/eval/score/enrichment_v3_fixture_loop.py
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.enrichment.eval import (
    metrics_by_enricher,
    register_builtin_scorers,
    run_scorers,
    ScorerRegistry,
    write_gate_metrics,
)
from podcast_scraper.enrichment.eval.gold import EXPECTED_ENRICHMENT_KEY

_ROOT = Path(__file__).resolve().parents[3]
_CORPUS = _ROOT / "tests" / "fixtures" / "app-validation-corpus" / "v3"
_GOLD = _ROOT / "tests" / "fixtures" / "ground-truth" / "v3" / "expected_enrichment.gold.json"


def _load_output(enricher_id: str) -> dict | None:
    """Load one enricher's emitted payload (the envelope's ``data`` block)."""
    path = _CORPUS / "enrichments" / f"{enricher_id}.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    data = doc.get("data")
    return data if isinstance(data, dict) else None


def main() -> int:
    gold_doc = json.loads(_GOLD.read_text(encoding="utf-8"))
    gold = gold_doc.get(EXPECTED_ENRICHMENT_KEY, {})

    registry = ScorerRegistry()
    register_builtin_scorers(registry)

    outputs: dict[str, dict] = {}
    for eid in registry.all_enricher_ids():
        out = _load_output(eid)
        if out is not None:
            outputs[eid] = out

    results = run_scorers(registry, outputs, gold)
    print("=== v3 fixture enricher scoring ===")
    for r in results:
        if r.skipped:
            print(f"  {r.enricher_id:22} skipped — {r.notes}")
        else:
            print(f"  {r.enricher_id:22} {r.metrics}  (n={r.sample_count})")

    metrics = metrics_by_enricher(results)
    written = write_gate_metrics(metrics, run_id="v3_fixture_loop")
    print(f"\nwrote {len(written)} gate_metrics.json → data/eval/enrichment/<id>/")
    for p in written:
        print(f"  {p.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
