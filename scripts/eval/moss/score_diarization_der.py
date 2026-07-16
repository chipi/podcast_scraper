#!/usr/bin/env python3
"""Score MOSS hypothesis RTTMs vs the v3 fixture ground truth: speaker count + DER (#1174).

Bar (community-1 on full-45): count 40/45 exact, DER 7.1%.
"""

import glob
import json
import os
import pathlib

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

REPO = str(pathlib.Path(__file__).resolve().parents[3])
REF_DIR = f"{REPO}/tests/fixtures/transcripts/v3"
HYP_DIR = os.environ.get("HYP_DIR", "/tmp/moss_rttm")


def load_rttm(path):
    ann = Annotation()
    for line in open(path):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur, spk = float(f[3]), float(f[4]), f[7]
            if dur > 0:
                ann[Segment(start, start + dur)] = spk
    return ann


def main():
    metric = DiarizationErrorRate()
    rows = []
    count_exact = count_within1 = 0
    ders = []
    for hyp_path in sorted(glob.glob(f"{HYP_DIR}/*.rttm")):
        stem = os.path.basename(hyp_path)[:-5]
        ref_path = f"{REF_DIR}/{stem}.rttm"
        gt_path = f"{REF_DIR}/{stem}.groundtruth.json"
        if not os.path.exists(ref_path):
            continue
        ref, hyp = load_rttm(ref_path), load_rttm(hyp_path)
        der = metric(ref, hyp)
        ders.append(der)
        exp = None
        if os.path.exists(gt_path):
            exp = json.load(open(gt_path)).get("expected_diarized_voices")
        n_hyp = len(hyp.labels())
        n_ref = len(ref.labels())
        exact = exp is not None and n_hyp == exp
        within1 = exp is not None and abs(n_hyp - exp) <= 1
        count_exact += exact
        count_within1 += within1
        rows.append((stem, exp, n_ref, n_hyp, der * 100))

    rows.sort(key=lambda r: -r[4])
    print(f"{'fixture':16s} {'exp':>4} {'ref':>4} {'moss':>4} {'DER%':>7}")
    for stem, exp, n_ref, n_hyp, der in rows:
        flag = "" if (exp and n_hyp == exp) else "  <-- count off"
        print(f"{stem:16s} {str(exp):>4} {n_ref:>4} {n_hyp:>4} {der:7.1f}{flag}")
    n = len(rows)
    print(f"\n=== MOSS diarization on {n} fixtures ===")
    print(f"count exact:   {count_exact}/{n}   (bar: community-1 40/45)")
    print(f"count within1: {count_within1}/{n}")
    print(f"mean DER:      {sum(ders)/len(ders)*100:.1f}%   (bar: community-1 7.1%)")
    print(f"median DER:    {sorted(ders)[len(ders)//2]*100:.1f}%")


if __name__ == "__main__":
    main()
