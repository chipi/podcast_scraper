"""v2 deliberate-ambiguity validation for tuned entity-canon thresholds (#853).

Per #904 spec (and the #903 audit), v2 fixtures encode three CIL-relevant
canonicalization tests. Validate the recommended thresholds against each:

1. **Two distinct people, same first name** — `person:marco` (p03 dive
   Marco) vs `person:marco-bianchi` (p05 tax-loss researcher) MUST stay
   distinct.
2. **Title canonicalization** — synthetic surface "Dr. Elena Fischer" vs
   "Elena Fischer" MUST merge (single token-count diff means current
   predicate can't fix it — surfaced as a known gap in the eval report).
3. **First-name alias** — "Liam Verbeek" (Whisper-invented surname) vs
   "Liam" — same token-count-mismatch class as above.

Cases 2 and 3 fail under any threshold setting in the current predicate
(it rejects token-count mismatches by design). The point of validating
is to surface this honestly — the recommended thresholds don't fix those
cases, and a predicate redesign is in scope for #904, not #853.

Usage:
    python scripts/eval/score/entity_canon_v2_validate.py \
        --output data/eval/runs/baseline_entity_canon_v2_validate/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_PATH = PROJECT_ROOT / "scripts" / "eval" / "score" / "entity_canon_sweep_v1.py"
_spec = importlib.util.spec_from_file_location("entity_canon_sweep_v1", _PATH)
assert _spec and _spec.loader
sweep = importlib.util.module_from_spec(_spec)
sys.modules["entity_canon_sweep_v1"] = sweep
_spec.loader.exec_module(sweep)


CASES = [
    # (case_id, label_a, label_b, expected_label, class_note)
    (
        "two-marcos-distinct",
        "Marco",
        "Marco Bianchi",
        "DIFFERENT",
        "two-people-same-first-name (token-count mismatch)",
    ),
    (
        "fischer-merge",
        "Dr. Elena Fischer",
        "Elena Fischer",
        "SAME",
        "title-prefix canonicalization (token-count mismatch)",
    ),
    (
        "liam-alias",
        "Liam Verbeek",
        "Liam",
        "SAME",
        "first-name-only alias (token-count mismatch)",
    ),
    # Real ASR garbles confirmed in the manual-run-10 silver — same threshold class
    (
        "bessent-garble",
        "Scott Bessent",
        "Scott Bessett",
        "SAME",
        "ASR surname single-letter swap",
    ),
    (
        "alloway-garble",
        "Tracy Alloway",
        "Tracy Allaway",
        "SAME",
        "ASR surname single-letter swap",
    ),
    (
        "weisenthal-quartet",
        "Joe Weisenthal",
        "Joe Wassenthal",
        "SAME",
        "ASR first-name garble",
    ),
    (
        "geithner-garble",
        "Tim Geithner",
        "Tim Geidner",
        "SAME",
        "ASR surname swap",
    ),
]

# Settings to validate against
SETTINGS = [
    ("baseline_defaults", 0.78, 0.85),
    ("recommended", 0.65, 0.70),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    results = []
    for setting_name, tr, ov in SETTINGS:
        for case_id, a, b, expected, note in CASES:
            predicted_merge = sweep.are_variants(a, b, token_ratio=tr, overall_ratio=ov)
            predicted_label = "SAME" if predicted_merge else "DIFFERENT"
            passed = predicted_label == expected
            results.append(
                {
                    "setting": setting_name,
                    "token_ratio": tr,
                    "overall_ratio": ov,
                    "case_id": case_id,
                    "label_a": a,
                    "label_b": b,
                    "expected": expected,
                    "predicted": predicted_label,
                    "pass": passed,
                    "class": note,
                }
            )

    by_setting: dict[str, dict[str, int]] = {}
    for r in results:
        s = r["setting"]
        by_setting.setdefault(s, {"pass": 0, "fail": 0})
        by_setting[s]["pass" if r["pass"] else "fail"] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "metrics_entity_canon_v2_validate_v1",
        "cases": results,
        "summary_by_setting": by_setting,
    }
    (args.output / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = [
        "# Entity-canon v2 deliberate-ambiguity validation — #853",
        "",
        f"Cases: {len(CASES)} per setting.",
        "",
        "## Summary",
        "",
        "| Setting | Pass | Fail |",
        "| --- | ---: | ---: |",
    ]
    for s, counts in by_setting.items():
        report.append(f"| {s} | {counts['pass']} | {counts['fail']} |")
    report += ["", "## Per-case", ""]
    for setting_name, _, _ in SETTINGS:
        report += [
            f"### {setting_name}",
            "",
            "| case | expected | predicted | pass | class |",
            "| --- | --- | --- | :---: | --- |",
        ]
        for r in results:
            if r["setting"] != setting_name:
                continue
            tick = "✓" if r["pass"] else "✗"
            report.append(
                f"| `{r['case_id']}` ({r['label_a']!r} vs {r['label_b']!r}) | "
                f"{r['expected']} | {r['predicted']} | {tick} | {r['class']} |"
            )
        report.append("")

    (args.output / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"summary: {by_setting}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
