#!/usr/bin/env python3
"""Part 2 reality-check scorer: MOSS/whisper/pyannote vs Deepgram nova-3 silver on prod audio.

Transcription WER (MOSS vs DG, whisper vs DG) + diarization DER (MOSS vs DG, pyannote vs DG),
plus raw and dominant (>=5% time) speaker counts. No ground truth — Deepgram is the shared silver.
All inputs live under the gitignored .test_outputs/moss-eval/prod/.
"""

import collections
import glob
import os
import re

import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

BASE = ".test_outputs/moss-eval/prod"
DG = f"{BASE}/deepgram_ref"
MOSS = f"{BASE}/moss"
WHISPER = f"{BASE}/whisper"
PYANNOTE = f"{BASE}/pyannote"

_PUNCT = re.compile(r"[^0-9a-z\s]")
_WS = re.compile(r"\s+")


def norm(t):
    return _WS.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return jiwer.wer(r, h) if (r and h) else 1.0


def load_rttm(path):
    ann = Annotation()
    for line in open(path):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            s, d = float(f[3]), float(f[4])
            if d > 0:
                ann[Segment(s, s + d)] = f[7]
    return ann


def counts(path):
    by = collections.defaultdict(float)
    tot = 0.0
    for line in open(path):
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            by[f[7]] += float(f[4])
            tot += float(f[4])
    raw = len(by)
    dom = sum(1 for d in by.values() if tot and d / tot >= 0.05)
    return raw, dom


def read(p):
    return open(p).read() if os.path.exists(p) else ""


def main():
    eps = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{DG}/*.txt"))
    print(
        f"{'ep':6} {'MOSS_wer':>9} {'whis_wer':>9} | {'MOSS_der':>9} {'pyan_der':>9} "
        f"| {'DGdom':>5} {'MOSSdom':>7} {'pyandom':>7}"
    )
    agg = collections.defaultdict(list)
    der = DiarizationErrorRate()
    for e in eps:
        dg_txt = read(f"{DG}/{e}.txt")
        mw = wer(dg_txt, read(f"{MOSS}/{e}.txt")) if os.path.exists(f"{MOSS}/{e}.txt") else None
        ww = (
            wer(dg_txt, read(f"{WHISPER}/{e}.txt"))
            if os.path.exists(f"{WHISPER}/{e}.txt")
            else None
        )
        md = pd = None
        dg_dom = moss_dom = pyan_dom = None
        if os.path.exists(f"{DG}/{e}.rttm"):
            ref = load_rttm(f"{DG}/{e}.rttm")
            _, dg_dom = counts(f"{DG}/{e}.rttm")
            if os.path.exists(f"{MOSS}/{e}.rttm"):
                md = der(ref, load_rttm(f"{MOSS}/{e}.rttm"))
                _, moss_dom = counts(f"{MOSS}/{e}.rttm")
            if os.path.exists(f"{PYANNOTE}/{e}.rttm"):
                pd = der(ref, load_rttm(f"{PYANNOTE}/{e}.rttm"))
                _, pyan_dom = counts(f"{PYANNOTE}/{e}.rttm")
        for k, v in [("mw", mw), ("ww", ww), ("md", md), ("pd", pd)]:
            if v is not None:
                agg[k].append(v)
        fmt = lambda x, s=1: (f"{x*s:.1f}" if x is not None else "-")  # noqa: E731
        print(
            f"{e:6} {fmt(mw,100):>9} {fmt(ww,100):>9} | {fmt(md,100):>9} {fmt(pd,100):>9} "
            f"| {str(dg_dom):>5} {str(moss_dom):>7} {str(pyan_dom):>7}"
        )

    def m(k):
        return f"{sum(agg[k])/len(agg[k])*100:.1f}%" if agg[k] else "n/a"

    print(f"\n=== Part 2 reality check ({len(eps)} prod eps, ref = Deepgram nova-3 silver) ===")
    print(f"Transcription WER: MOSS {m('mw')}  |  whisper {m('ww')}")
    print(f"Diarization  DER : MOSS {m('md')}  |  pyannote {m('pd')}")


if __name__ == "__main__":
    main()
