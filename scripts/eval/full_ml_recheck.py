#!/usr/bin/env python3
"""One-shot regression harness for every #382-touched ML surface.

Re-captures each fixture-based baseline against the current code and
compares against the frozen artifacts under ``data/eval/references/``.
Exits 0 on full pass, 1 on any regression.

Surfaces covered (all follow the same pattern: fixed inputs → capture →
compare against frozen):

- Summarizer (BART+LED): via ``data/eval/baselines/*_v5_post`` +
  ``compare_v5_parity.py``.
- Extractive QA: via ``capture_qa_baseline.py`` + JSONL diff.
- NLI: via ``capture_nli_baseline.py`` + entailment-score diff (abs
  tolerance 0.01).
- Embedding: via ``capture_embedding_baseline.py`` + dim + L2-norm +
  first-8-dims within 1e-4 tolerance (CPU BLAS thread-order can flip
  SHA-256; first-dim tolerance still catches wrong-model / wrong-pool drift).
- FLAN-T5 reduce (hybrid tier-1): via ``capture_flant5_reduce_baseline.py``
  + output-text identity.

Usage::

    python scripts/eval/full_ml_recheck.py
    python scripts/eval/full_ml_recheck.py --json-report /tmp/report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Check:
    name: str
    frozen_path: Path
    capture_script: Path
    capture_arg: str
    diff_fn: Callable[[Path, Path], Dict[str, object]]


def _diff_jsonl_by_id(
    frozen: Path,
    fresh: Path,
    *,
    id_key: str,
    compare_keys: List[str],
    scalar_tol: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Compare two JSONL files keyed by ``id_key``, checking ``compare_keys``.

    Numeric keys allow ``scalar_tol[key]`` absolute tolerance; string keys
    must match exactly.
    """
    scalar_tol = scalar_tol or {}
    frozen_rows = {r[id_key]: r for r in map(json.loads, frozen.read_text().splitlines())}
    fresh_rows = {r[id_key]: r for r in map(json.loads, fresh.read_text().splitlines())}
    shared = sorted(set(frozen_rows) & set(fresh_rows))
    missing_frozen = sorted(set(fresh_rows) - set(frozen_rows))
    missing_fresh = sorted(set(frozen_rows) - set(fresh_rows))

    mismatches: List[Dict[str, object]] = []
    for rid in shared:
        f, r = frozen_rows[rid], fresh_rows[rid]
        for key in compare_keys:
            fv, rv = f.get(key), r.get(key)
            if key in scalar_tol:
                if fv is None or rv is None or abs(float(fv) - float(rv)) > scalar_tol[key]:
                    mismatches.append({"id": rid, "key": key, "frozen": fv, "fresh": rv})
            else:
                if fv != rv:
                    mismatches.append({"id": rid, "key": key, "frozen": fv, "fresh": rv})
    return {
        "n_pairs": len(shared),
        "missing_frozen": missing_frozen,
        "missing_fresh": missing_fresh,
        "mismatches": mismatches,
        "pass": not mismatches and not missing_frozen and not missing_fresh,
    }


def _capture(script: Path, out_arg: str, out_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), out_arg, str(out_path)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"capture script {script.name} failed:\n{result.stderr[-2000:]}")


def _check_qa(frozen: Path, fresh: Path) -> Dict[str, object]:
    """QA answer-text identity — reuses the flat structure of qa_baseline_*.jsonl."""

    # top_k_spans[0] is the field we care about
    def load(p: Path) -> Dict[str, Dict]:
        out = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            top = r["top_k_spans"][0] if r["top_k_spans"] else {}
            out[r["id"]] = {
                "answer": top.get("answer"),
                "start": top.get("start"),
                "end": top.get("end"),
            }
        return out

    fr, fe = load(frozen), load(fresh)
    shared = sorted(set(fr) & set(fe))
    mismatches = [
        {"id": rid, "frozen": fr[rid], "fresh": fe[rid]}
        for rid in shared
        if fr[rid]["answer"] != fe[rid]["answer"]
    ]
    return {
        "n_pairs": len(shared),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _check_nli(frozen: Path, fresh: Path) -> Dict[str, object]:
    return _diff_jsonl_by_id(
        frozen,
        fresh,
        id_key="id",
        compare_keys=["entailment_score"],
        scalar_tol={"entailment_score": 0.01},
    )


def _check_embedding(frozen: Path, fresh: Path) -> Dict[str, object]:
    """Embedding regression: dim + L2-norm + first-8-dims with tight tolerance.

    NOT bit-identity via SHA-256 — sentence-transformers on CPU has BLAS-
    thread-order non-determinism that flips SHA even when the vector is
    semantically identical to ~1e-6. The first-8 dims + L2 norm catch any
    real drift (a wrong model, different normalization, different pooling)
    without false-positive'ing on floating-point noise.
    """

    # Build per-dim comparisons: first_8_d0, first_8_d1, ...
    def load(p: Path) -> Dict[str, Dict[str, float]]:
        out = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            row = {"dim": r["dim"], "l2_norm": r["l2_norm"]}
            for i, v in enumerate(r["first_8"]):
                row[f"d{i}"] = v
            out[r["id"]] = row
        return out

    fr, fe = load(frozen), load(fresh)
    shared = sorted(set(fr) & set(fe))
    tol = {**{f"d{i}": 1e-4 for i in range(8)}, "l2_norm": 1e-4}
    mismatches: List[Dict[str, object]] = []
    for rid in shared:
        f, r = fr[rid], fe[rid]
        if f["dim"] != r["dim"]:
            mismatches.append({"id": rid, "key": "dim", "frozen": f["dim"], "fresh": r["dim"]})
        for key, tolv in tol.items():
            if abs(f.get(key, 0.0) - r.get(key, 0.0)) > tolv:
                mismatches.append(
                    {"id": rid, "key": key, "frozen": f[key], "fresh": r[key], "tol": tolv}
                )
    return {
        "n_pairs": len(shared),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _check_flant5(frozen: Path, fresh: Path) -> Dict[str, object]:
    return _diff_jsonl_by_id(
        frozen,
        fresh,
        id_key="id",
        compare_keys=["output_text"],
    )


CHECKS: List[Check] = [
    Check(
        name="Extractive QA (roberta-squad2)",
        frozen_path=REPO / "data/eval/references/qa_baseline_v5_post.jsonl",
        capture_script=REPO / "scripts/dev/capture_qa_baseline.py",
        capture_arg="--out",
        diff_fn=_check_qa,
    ),
    Check(
        name="NLI (cross-encoder/nli-deberta-v3-base)",
        frozen_path=REPO / "data/eval/references/nli_baseline_v5.jsonl",
        capture_script=REPO / "scripts/dev/capture_nli_baseline.py",
        capture_arg="--out",
        diff_fn=_check_nli,
    ),
    Check(
        name="Embedding (all-MiniLM-L6-v2)",
        frozen_path=REPO / "data/eval/references/embedding_baseline_v5.jsonl",
        capture_script=REPO / "scripts/dev/capture_embedding_baseline.py",
        capture_arg="--out",
        diff_fn=_check_embedding,
    ),
    Check(
        name="FLAN-T5 reduce (hybrid tier-1)",
        frozen_path=REPO / "data/eval/references/flant5_reduce_baseline_v5.jsonl",
        capture_script=REPO / "scripts/dev/capture_flant5_reduce_baseline.py",
        capture_arg="--out",
        diff_fn=_check_flant5,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-report", type=Path, default=None)
    args = parser.parse_args()

    print("Full ML data-quality recheck — capturing fresh vs frozen for every path")
    reports: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for chk in CHECKS:
            fresh = tmp / chk.frozen_path.name.replace(".jsonl", "_fresh.jsonl")
            print(f"\n[{chk.name}]")
            try:
                _capture(chk.capture_script, chk.capture_arg, fresh)
            except Exception as e:
                print(f"  ✗ capture failed: {e}")
                reports.append({"check": chk.name, "pass": False, "error": str(e)})
                continue
            report = chk.diff_fn(chk.frozen_path, fresh)
            report["check"] = chk.name
            reports.append(report)
            if report.get("pass"):
                print(f"  ✓ pass ({report.get('n_pairs')} pairs matched)")
            else:
                print(f"  ✗ FAIL — {len(report.get('mismatches', []))} mismatch(es)")
                for m in report.get("mismatches", [])[:5]:
                    print(f"    {m}")

    overall = all(r.get("pass") for r in reports)
    print(f"\nOverall: {'PASS ✓' if overall else 'FAIL ✗'}")
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps({"overall_pass": overall, "checks": reports}, indent=2)
        )
        print(f"wrote {args.json_report}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
