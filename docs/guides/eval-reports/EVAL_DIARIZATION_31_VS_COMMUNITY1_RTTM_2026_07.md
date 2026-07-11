# EVAL — Diarization: pyannote 3.1 vs community-1, count + DER on v3 fixtures

2026-07-11 — full v3 fixture set (45 episodes) scored against **exact per-turn RTTM
ground truth**, both the brittle speaker-**count** metric and time-weighted **DER**.

**Issue:** #1170
**Branch:** `feat/enrichment-surfaces`
**Dataset:** v3 audio fixtures, 45 episodes (macOS `say` generation; `_fast` excluded — no RTTM)
**Models:** `pyannote/speaker-diarization-3.1` (pyannote.audio 3.4) vs
`pyannote/speaker-diarization-community-1` (v4, pyannote.audio 4.0.7)
**Status:** **Measured.** Cross-validated against the locked curated-8 baseline
(3.1 = 8/8, community-1 = 7/8) — reproduced exactly.

---

## TL;DR

**On the full fixture set, community-1 (v4) is the better model on both metrics** —
count 40/45 vs 32/45, DER 7.1% vs 10.8%. The earlier "3.1 is better" conclusion was
an artifact of the **curated-8 subset + count-only metric**: that subset over-weights
the one brief-cameo case where 3.1 happens to win. community-1 substantially
outperforms 3.1 on **multi-speaker (3-way) panel separation** — precisely 3.1's known
same-gender-merge / over-segmentation weakness — at the cost of being slightly worse on
two brief-voice fixtures.

The entire DER gap is **speaker confusion**: missed-detection and false-alarm rates are
identical across models (both diarizers detect the same speech regions vs the RTTM);
they differ only in *who* they attribute it to.

## Results

### Full set (45 episodes)

| Model | Count match | DER | Confusion | Missed | False-alarm |
| --- | --- | --- | --- | --- | --- |
| pyannote 3.1 | 32/45 (71.1%) | 10.84% | 9.77% | 0.97% | 0.11% |
| **community-1 (v4)** | **40/45 (88.9%)** | **7.13%** | **6.05%** | 0.97% | 0.11% |

### Curated-8 subset (baseline cross-check)

| Model | Count match | DER | Confusion | Missed | False-alarm |
| --- | --- | --- | --- | --- | --- |
| pyannote 3.1 | **8/8** | 7.26% | 6.28% | 0.94% | 0.04% |
| community-1 (v4) | 7/8 | **6.25%** | 5.27% | 0.94% | 0.04% |

The curated-8 count exactly reproduces the locked baseline (3.1 = 8/8, community-1 = 7/8),
validating the RTTM ground truth + scorer. Note that even where 3.1 wins on **count**,
community-1 has **lower DER** — it misses only the ~1.4 s cameo (a tiny time penalty)
while labelling everything else more tightly.

## Per-episode breakdown (count, full set)

Both correct: **30** · community-1-only: **10** · 3.1-only: **2** · neither: **3**

| Bucket | Episodes | Reading |
| --- | --- | --- |
| community-1 only | p03_e01–e04, p04_e04, p05_e01, p05_e05, p06_e02–e04 | Mostly **3-speaker panels** where 3.1 under-counts to 2 (merges co-speakers) or over-segments to 4; community-1 gets 3. |
| 3.1 only | p01_multi_e03, p01_multi_e05 | Brief 2nd voice that community-1 merges away. |
| neither | p01_multi_e01, p01_multi_e02, p02_e05 | Micro-greetings + the designed ~1.4 s cameo — **model-independent brief-voice floors**. |

This matches the prior diagnosis that 3.1's real weakness is same-gender / co-speaker
merges on panels. community-1 fixes most of them; the residual misses for *both* models
are brief cameos, a shared floor no clustering threshold reaches.

## Method

1. **Ground truth** — each fixture carries an exact per-turn RTTM
   (`tests/fixtures/transcripts/v3/<ep>.rttm`) emitted by
   `transcripts_to_mp3.py --rttm-only` from the deterministic `say` aiff timeline.
   No transcription, no alignment — free and exact (contrast the v2 Deepgram-aligned
   approach in `EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md`).
2. **Hypothesis** — `scripts/eval/score/diarization_dump_segments_v1.py`, run on the
   DGX (mounted at `/work/dump_segments.py`), writes one `segments_<ep>.json` per
   fixture per model (raw `itertracks` output).
3. **Score** — `scripts/eval/score/diarization_der_rttm_v1.py` scores count +
   DER (pyannote.metrics, which solves the optimal speaker mapping) locally.

## Reproduce

DGX segment dumps run **fully offline** from local model caches (no download). The
non-obvious part: **pyannote 3.x caches to `/root/.cache/torch/pyannote`; pyannote 4.x
uses the standard HF hub cache** — mount each model where its pyannote version looks.

```bash
# --- 3.1 (pyannote 3.4, base image), offline from torch-cache ---
docker run --rm --gpus all \
  -v ~/diar-sweep:/work \
  -v ~/diar-sweep/torch-cache:/root/.cache/torch/pyannote \
  -e HF_HUB_OFFLINE=1 -e DIAR_MODEL=pyannote/speaker-diarization-3.1 -e OUT_DIR=segments_31 \
  podcast-pyannote:0.1.0 python /work/dump_segments.py

# --- community-1 (v4), offline from the HF hub cache ---
docker run --rm --gpus all \
  -v ~/diar-sweep:/work \
  -v ~/diar-sweep/hf-cache:/root/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 -e DIAR_MODEL=pyannote/speaker-diarization-community-1 -e OUT_DIR=segments_v4 \
  podcast-pyannote:0.1.0 bash -lc \
    'pip install --no-deps "pyannote.audio>=4,<5" && \
     pip install opentelemetry-exporter-otlp-proto-http opentelemetry-exporter-otlp-proto-common rich && \
     python /work/dump_segments.py'

# --- score locally against the RTTM ground truth ---
.venv/bin/python scripts/eval/score/diarization_der_rttm_v1.py \
  --rttm-dir tests/fixtures/transcripts/v3 \
  --segments-run 3.1:runs/pyannote-3.1 \
  --segments-run community-1:runs/community-1 \
  --output runs/der_out
```

`~/diar-sweep/hf-cache` (unified, 74 MB, all four pyannote repos) is the one-time
populated offline cache; `torch-cache` is 3.x's cache. `dump_segments.py` and the
segment outputs are **local-only** (not versioned).

## Caveats

- **Synthetic fixtures.** These are `say`-generated; a real 10-episode set (1/feed)
  is the planned next step and may surface different failure modes.
- **DER excludes `_fast`** (an ffmpeg truncation with no RTTM) — 45 of 46 fixtures.
- **Count and DER are complementary**, reported side-by-side, never swapped. Count
  punishes a 1 s cameo miss as a full −1; DER weights by mis-attributed time.
