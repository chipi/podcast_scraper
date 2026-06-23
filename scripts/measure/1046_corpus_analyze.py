"""#1046 measurement pass 2 — laptop-side analyzer.

Consumes the DGX measurement output (transcripts_small/, transcripts_large/,
timings.csv) plus the source fixtures (tests/fixtures/transcripts/v1/) and
computes:

- Per-episode PERSON + ORG entity counts on (source / small / large)
- Gate decision at the configured threshold for each of (source / small / large)
- Confusion matrix: small.en gate decision vs large-v3 gate decision
  (the operational question — would the cheap gate make the same call
  as a deep-transcript-derived gate?)
- Corpus-level r (fraction of episodes the gate would FIRE on, by each
  signal source)
- Latency stats already in the DGX summary.json, surfaced here for one
  unified report

Writes:
  data/eval/runs/1046-measurement-pass-2/analysis.csv  — per-episode
  data/eval/runs/1046-measurement-pass-2/analysis.json — corpus summary

This is a measurement artifact — write once, never mutate. Future re-runs
should land in a new ``1046-measurement-pass-N`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import spacy

PERSON_ORG = ("PERSON", "ORG")
THRESHOLD = 5  # cfg.dgx_whisper_sniff_gate_min_entities default

REPO_ROOT = Path(__file__).resolve().parent if False else Path.cwd()


def count_entities(nlp, text: str) -> int:
    return sum(1 for ent in nlp(text).ents if ent.label_ in PERSON_ORG)


def main(args) -> None:
    nlp = spacy.load("en_core_web_sm")
    src_dir = Path(args.source_transcripts)
    sniff_dir = Path(args.sniff_transcripts)
    deep_dir = Path(args.deep_transcripts)
    timings_csv = Path(args.timings_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = {}
    with timings_csv.open() as f:
        for r in csv.DictReader(f):
            timings[r["episode"]] = {
                "small_s": float(r["small_seconds"] or 0.0),
                "large_s": float(r["large_seconds"] or 0.0),
                "ratio": (
                    float(r["ratio_large_over_small"] or 0.0)
                    if r["ratio_large_over_small"] not in ("", "None")
                    else None
                ),
            }

    rows = []
    for sniff_path in sorted(sniff_dir.glob("*.txt")):
        ep = sniff_path.stem
        deep_path = deep_dir / f"{ep}.txt"
        source_path = src_dir / f"{ep}.txt"
        if not (deep_path.exists() and source_path.exists()):
            continue
        sniff_text = sniff_path.read_text()
        deep_text = deep_path.read_text()
        source_text = source_path.read_text()

        small_ents = count_entities(nlp, sniff_text)
        large_ents = count_entities(nlp, deep_text)
        source_ents = count_entities(nlp, source_text)

        small_decision = small_ents >= THRESHOLD
        large_decision = large_ents >= THRESHOLD
        source_decision = source_ents >= THRESHOLD

        t = timings.get(ep, {})
        rows.append(
            {
                "episode": ep,
                "source_ents": source_ents,
                "small_ents": small_ents,
                "large_ents": large_ents,
                "source_decision": source_decision,
                "small_decision": small_decision,
                "large_decision": large_decision,
                "small_vs_large_agree": small_decision == large_decision,
                "small_vs_source_agree": small_decision == source_decision,
                "large_vs_source_agree": large_decision == source_decision,
                "small_s": t.get("small_s"),
                "large_s": t.get("large_s"),
                "ratio": t.get("ratio"),
            }
        )

    if not rows:
        raise SystemExit("No rows produced — check that transcripts dir has results.")

    # write per-episode csv
    csv_out = out_dir / "analysis.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # corpus summary
    n = len(rows)

    def _frac(predicate):
        return round(sum(1 for r in rows if predicate(r)) / n, 4)

    # gate-vs-gate confusion (the operational question)
    def _cls(r):
        sd, ld = r["small_decision"], r["large_decision"]
        if sd and ld:
            return "TP"  # small says deep, large agrees → correct deep
        if not sd and not ld:
            return "TN"  # both say skip → correct skip
        if sd and not ld:
            return "FP"  # small said deep, but large would skip → wasted deep
        return "FN"  # small said skip, but large would deep → MISSED a deep-worthy episode

    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in rows:
        confusion[_cls(r)] += 1

    # latency stats (filter rows where small_s is 0 — resumed)
    import math

    ratios = [r["ratio"] for r in rows if r["ratio"] and r["ratio"] > 0 and r["ratio"] < 50]
    geomean = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else None

    summary = {
        "n_episodes": n,
        "threshold": THRESHOLD,
        "r_by_small": _frac(lambda r: r["small_decision"]),
        "r_by_large": _frac(lambda r: r["large_decision"]),
        "r_by_source": _frac(lambda r: r["source_decision"]),
        "gate_agreement_small_vs_large": _frac(lambda r: r["small_vs_large_agree"]),
        "gate_agreement_small_vs_source": _frac(lambda r: r["small_vs_source_agree"]),
        "gate_agreement_large_vs_source": _frac(lambda r: r["large_vs_source_agree"]),
        "confusion_small_vs_large": confusion,
        "false_negative_rate": round(
            confusion["FN"] / max(confusion["FN"] + confusion["TP"], 1), 4
        ),
        "false_positive_rate": round(
            confusion["FP"] / max(confusion["FP"] + confusion["TN"], 1), 4
        ),
        "latency_geomean_ratio_large_over_small": round(geomean, 3) if geomean else None,
        "latency_min_ratio": round(min(ratios), 3) if ratios else None,
        "latency_max_ratio": round(max(ratios), 3) if ratios else None,
        "break_even_r": round(1 - 1 / geomean, 3) if geomean else None,
    }

    json_out = out_dir / "analysis.json"
    json_out.write_text(json.dumps(summary, indent=2))

    print("\n=== #1046 measurement pass 2 — corpus summary ===")
    print(f"n_episodes={summary['n_episodes']}  threshold={THRESHOLD} (PERSON+ORG)")
    print(f"\n[gate fire rate r at threshold {THRESHOLD}]")
    print(f"  by sniff (small.en):  {summary['r_by_small']:.3f}")
    print(f"  by deep  (large-v3):  {summary['r_by_large']:.3f}")
    print(f"  by source (ground):   {summary['r_by_source']:.3f}")
    print("\n[gate decision agreement]")
    print(f"  small vs large:  {summary['gate_agreement_small_vs_large']:.3f}")
    print(f"  small vs source: {summary['gate_agreement_small_vs_source']:.3f}")
    print(f"  large vs source: {summary['gate_agreement_large_vs_source']:.3f}")
    print("\n[confusion small.en gate vs large-v3 gate]")
    print(f"  TP (both fire):           {confusion['TP']}")
    print(f"  TN (both skip):           {confusion['TN']}")
    print(f"  FP (small fires, large skips): {confusion['FP']}  → wasted deep")
    print(f"  FN (small skips, large fires): {confusion['FN']}  → MISSED deep-worthy")
    print(f"  false-negative rate: {summary['false_negative_rate']:.3f}")
    print(f"  false-positive rate: {summary['false_positive_rate']:.3f}")
    print("\n[latency]")
    print(f"  ratio large/small (geomean): {summary['latency_geomean_ratio_large_over_small']}")
    print(f"  ratio min:                   {summary['latency_min_ratio']}")
    print(f"  ratio max:                   {summary['latency_max_ratio']}")
    print(f"  break-even r*:               {summary['break_even_r']}")
    print(f"\nArtifacts:\n  {csv_out}\n  {json_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-transcripts", default="tests/fixtures/transcripts/v1")
    ap.add_argument(
        "--sniff-transcripts", default="data/eval/runs/1046-measurement-pass-2/transcripts_small"
    )
    ap.add_argument(
        "--deep-transcripts", default="data/eval/runs/1046-measurement-pass-2/transcripts_large"
    )
    ap.add_argument("--timings-csv", default="data/eval/runs/1046-measurement-pass-2/timings.csv")
    ap.add_argument("--out-dir", default="data/eval/runs/1046-measurement-pass-2")
    main(ap.parse_args())
