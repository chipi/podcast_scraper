# MOSS bake-off v1 — fixture run artifacts (#1174 / #1177)

**Date:** 2026-07-16 · **Hardware:** DGX GB10 (sm_121) · **Runtime:** `nvcr.io/nvidia/vllm:26.05-py3`
(torch 2.12/CUDA13, transformers 5.6), MOSS via the upstream `moss_transcribe_diarize` package.

Head-to-head of **MOSS-Transcribe-Diarize** vs the incumbent audio stack on the **45 v3 RTTM
fixtures** (the `say`-rendered set with exact per-turn ground truth). This directory holds the
**fixture-derived** artifacts only — reproducible, no prod data. Prod-run artifacts are **never
committed** (they live in the gitignored `.test_outputs/moss-eval/`).

## What's here

- `moss/` — MOSS output per fixture: `<stem>.rttm` (speaker turns) + `<stem>.txt` (transcript).
- `whisper/` — faster-whisper large-v3 (`:8000`) transcript per fixture (`<stem>.txt`).
- `pyannote/` — pyannote community-1 (`:8001`, **no count hint**) RTTM per fixture.
- `*_summary.txt` / `scores.json` — the scored numbers.

Harness: [`scripts/eval/moss/`](../../../../scripts/eval/moss/) —
`validate_moss.py` (single-clip smoke), `moss_transcribe_diarize_batch.py` (DGX batch → rttm+txt),
`score_diarization_der.py` (DER + count vs RTTM truth), `score_transcription_wer.py` (WER vs the
source `.txt`).

## Results — the head-to-head on 45 fixtures

**Diarization** (DER + speaker count vs the RTTM ground truth; **neither** model given a count hint):

| Metric | MOSS | pyannote community-1 (same session) |
|---|---|---|
| mean DER | **3.3%** | 7.9% |
| median DER | **1.8%** | 6.9% |
| count exact | 38/45 | **40/45** |
| count within-1 | 45/45 | 45/45 |

MOSS **beats pyannote community-1 on DER by >2×**; pyannote edges exact count. (The fresh 7.9% /
40-count reproduces the #1170 community-1 baseline of 7.1% / 40, validating the harness.)

**Transcription** (WER vs the source text; MOSS vs faster-whisper large-v3):

| Metric | MOSS | whisper large-v3 |
|---|---|---|
| mean WER | 9.0% | **8.4%** |
| median WER | **3.1%** | 4.3% |
| head-to-head | **wins 25/45** | wins 20/45 |

Essentially a **tie** — MOSS better median + wins more fixtures, whisper better mean (a few
multi-accent/garble fixtures skew MOSS's mean; on one, p04_e02, MOSS 13.6% vs whisper 43.6%).

**Speed:** MOSS 6.0× realtime (short) / 2.1× (44-min episode) in **bare transformers**; large-v3 is
7.8×. MOSS is slower on long audio — the 50–100× claim needs the vLLM backend (untested sm_121).

## The load-bearing caveat — fixtures ≠ real long audio

On a **real 44-min prod episode** (The Daily, run saved to the gitignored `.test_outputs/moss-eval/`)
MOSS emitted **28 speaker labels** — long-audio label drift — vs the sane 2–3 it gives on every
fixture. **The fixture numbers overstate real-world diarization.** Transcription there was still
excellent; speed was **2.1× realtime** (bare transformers), i.e. slower than large-v3's 7.8× — the
headline speed rationale needs the (untested-on-sm_121) vLLM backend.

## Part 2 — the prod bake-off (next)

Real-audio subset (prod-v2 / `diar-prod90`) has **no ground-truth labels**, so score against:
- **Silver = Deepgram nova-3** across the whole subset (precedent: #952 `EVAL_WHISPER_ENGINE_DRIFT`).
- **Golden = a hand-labeled stick** (#1189 / the deferred #1170 real-10 RTTM labeling) on a few eps.

MOSS / whisper / pyannote all scored vs the same silver (and golden where it exists): WER + DER +
speaker-count sanity on long audio. **Prod artifacts stay gitignored — never committed.**
