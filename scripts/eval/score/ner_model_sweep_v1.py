"""NER model sweep for #906 Phase A.

Compares spaCy `en_core_web_sm` (current default) vs `en_core_web_trf`
(transformer, RoBERTa-based, stronger) on PERSON entity detection over:

- v2 source transcripts (5 smoke episodes)
- A sample of real-prod transcripts (manual-run-10, episodes with known
  silver-labelled near-duplicate person pairs from #853)

For each model: PERSON entity count, distinct entity count, runtime per
episode. Optionally compares against the v2 spec's known guest/host list to
compute precision/recall (canonical names known per episode metadata).

Usage:
    python scripts/eval/score/ner_model_sweep_v1.py \\
        --v2-sources data/eval/sources/curated_5feeds_raw_v2 \\
        --prod-transcripts-dir \\
            .test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts \\
        --output data/eval/runs/baseline_ner_model_sweep_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS = ["en_core_web_sm", "en_core_web_trf"]


def extract_persons(text: str, nlp: Any) -> list[str]:
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]


# v2 spec — primary guest per episode (from the generator's spec dict).
# Used as a precision/recall ground truth: every primary_guest MUST be
# detected, and the host should always be detected. Other PERSON mentions
# are valid signal too but not asserted.
V2_SPEC_PERSONS: dict[str, set[str]] = {
    "p01_e01": {"Maya", "Liam"},
    "p01_e02": {"Maya", "Sophie"},
    "p01_e03": {"Maya", "Noah"},
    "p02_e01": {"Ethan", "Priya"},
    "p02_e02": {"Ethan", "Jonas"},
    "p02_e03": {"Ethan", "Camila"},
    "p03_e01": {"Rina", "Marco"},
    "p03_e02": {"Rina", "Hanna"},
    "p03_e03": {"Rina", "Owen"},
    "p04_e01": {"Leo", "Ava"},
    "p04_e02": {"Leo", "Tariq"},
    "p04_e03": {"Leo", "Elise"},
    "p05_e01": {"Nora", "Daniel"},
    "p05_e02": {"Nora", "Isabel"},
    "p05_e03": {"Nora", "Kasper"},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-sources", type=Path, required=True)
    p.add_argument("--prod-transcripts-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prod-sample", type=int, default=3, help="Number of prod episodes to sample")
    args = p.parse_args()

    # Load models lazily
    loaded: dict[str, Any] = {}
    for model_name in MODELS:
        try:
            print(f"loading {model_name}...", file=sys.stderr)
            loaded[model_name] = spacy.load(model_name)
        except Exception as exc:
            print(f"  FAILED {model_name}: {exc}", file=sys.stderr)

    if not loaded:
        print("No spaCy models could be loaded", file=sys.stderr)
        return 1

    # v2 transcripts (smoke)
    v2_txts: list[tuple[str, str]] = []
    for ep in sorted(V2_SPEC_PERSONS):
        srcs = list(args.v2_sources.rglob(f"{ep}.txt"))
        if srcs:
            v2_txts.append((ep, srcs[0].read_text(encoding="utf-8")))

    # Prod sample
    prod_txts: list[tuple[str, str]] = []
    if args.prod_transcripts_dir and args.prod_transcripts_dir.is_dir():
        all_prod = sorted(args.prod_transcripts_dir.glob("*.txt"))[: args.prod_sample]
        for path in all_prod:
            ep_short = path.stem.split(" - ")[0].strip()
            prod_txts.append((ep_short, path.read_text(encoding="utf-8")))

    rows: list[dict[str, Any]] = []
    for model_name, nlp in loaded.items():
        for corpus_name, txts, spec in (
            ("v2_smoke", v2_txts, V2_SPEC_PERSONS),
            ("prod_sample", prod_txts, {}),
        ):
            for ep_id, text in txts:
                t0 = time.time()
                persons = extract_persons(text, nlp)
                elapsed = time.time() - t0
                # First-token comparison: spaCy detects "Liam Verbeek" or "Liam"
                # — count an expected guest as detected if any extracted PERSON
                # starts with the same first token.
                expected = spec.get(ep_id, set())
                detected_firsts = {p.split()[0] for p in persons if p}
                hits = expected & detected_firsts
                misses = expected - detected_firsts
                rows.append(
                    {
                        "model": model_name,
                        "corpus": corpus_name,
                        "episode_id": ep_id,
                        "n_person_mentions": len(persons),
                        "n_distinct_persons_first_token": len(detected_firsts),
                        "expected_persons": sorted(expected),
                        "expected_hits": sorted(hits),
                        "expected_misses": sorted(misses),
                        "all_distinct_first_tokens": sorted(detected_firsts),
                        "elapsed_s": round(elapsed, 2),
                    }
                )

    # Aggregate
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = []
    for model_name in MODELS:
        if model_name not in loaded:
            continue
        rs = by_model[model_name]
        v2_rows = [r for r in rs if r["corpus"] == "v2_smoke"]
        prod_rows = [r for r in rs if r["corpus"] == "prod_sample"]

        # v2 spec recall: how many expected persons did we detect across all v2 eps?
        total_expected = sum(len(r["expected_persons"]) for r in v2_rows)
        total_hits = sum(len(r["expected_hits"]) for r in v2_rows)
        v2_recall = round(100 * total_hits / max(total_expected, 1), 1)

        summary.append(
            {
                "model": model_name,
                "v2_episodes": len(v2_rows),
                "v2_expected_total": total_expected,
                "v2_hits_total": total_hits,
                "v2_spec_recall_pct": v2_recall,
                "v2_mean_person_mentions": round(
                    sum(r["n_person_mentions"] for r in v2_rows) / max(len(v2_rows), 1), 1
                ),
                "v2_mean_distinct_first_tokens": round(
                    sum(r["n_distinct_persons_first_token"] for r in v2_rows)
                    / max(len(v2_rows), 1),
                    1,
                ),
                "v2_mean_elapsed_s": round(
                    sum(r["elapsed_s"] for r in v2_rows) / max(len(v2_rows), 1), 2
                ),
                "prod_episodes": len(prod_rows),
                "prod_mean_person_mentions": round(
                    sum(r["n_person_mentions"] for r in prod_rows) / max(len(prod_rows), 1), 1
                ),
                "prod_mean_distinct_first_tokens": round(
                    sum(r["n_distinct_persons_first_token"] for r in prod_rows)
                    / max(len(prod_rows), 1),
                    1,
                ),
                "prod_mean_elapsed_s": round(
                    sum(r["elapsed_s"] for r in prod_rows) / max(len(prod_rows), 1), 2
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(
            {"schema": "metrics_ner_model_sweep_v1", "summary": summary, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'model':<22} v2_recall  v2_mentions  v2_distinct  v2_lat   "
        f"prod_mentions  prod_distinct  prod_lat"
    )
    for s in summary:
        print(
            f"{s['model']:<22} {s['v2_spec_recall_pct']:>8.1f}% "
            f"{s['v2_mean_person_mentions']:>11.1f}  {s['v2_mean_distinct_first_tokens']:>11.1f}  "
            f"{s['v2_mean_elapsed_s']:>5.2f}s   {s['prod_mean_person_mentions']:>12.1f}  "
            f"{s['prod_mean_distinct_first_tokens']:>13.1f}  {s['prod_mean_elapsed_s']:>5.2f}s"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
