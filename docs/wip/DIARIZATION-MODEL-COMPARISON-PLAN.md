# Diarization model / embedding comparison — plan

**Question:** are the persistent diarization failures a **model limit** (fixable by a better
embedding / a different engine) or **bad ground truth** (fixtures whose expected count is
acoustically unreachable)? Two pyannote versions (3.1, community-1) both cap at ~4–5/8 on the
8-fixture small set and fail the **same** fixtures — but they are **not independent**: they share
the same `segmentation-3.0` + `wespeaker` embedding backbone, so their agreement only shows the
**shared backbone's** limit, not that the ground truth is wrong. To answer the question we must
run **genuinely different** models — different embedding, and different engines — plus audit the
fixtures. Related issues: #1170 (tuning), #1171 (v4).

## Fixed method (identical across every experiment)

- **Dataset:** the 8-fixture small set (`data/eval` small set — monologue/interview/panel × ad ×
  cameo × multi_accent; the guardrails: panel `p05_e04`, cameo `p01_multi_e05`, over-count
  `p08_e04`/`p09_e01`, under-count `p09_e02`/`p09_e03`). Ground truth = the v3 sidecars'
  `expected_diarized_voices`.
- **Scoring:** exact count-match + over/under split + **gates** (panel==3, cameo not dropped, no
  voice merged away). Each experiment writes a `results.json` in the same shape; a final table
  compares all.
- **Sweep:** each model's own relevant knob(s) — don't force one model's params onto another.
- **Compute:** DGX GPU via ephemeral containers off `podcast-pyannote:0.1.0` (or a NeMo image);
  never touch the live pyannote-server or autoresearch. Waveforms preloaded in-memory.

## Experiments (run in parallel — not in a rush; thoroughness over speed)

### A. Embedding swap inside the pyannote pipeline  ·  the surgical test
Keep `segmentation-3.0` + agglomerative clustering; swap the **embedding** model (the component
that decides whether two similar voices separate):

| embedding | id | notes |
|---|---|---|
| wespeaker (baseline) | `pyannote/wespeaker-voxceleb-resnet34-LM` | already tested (3.1/v4) |
| **SpeechBrain ECAPA** | `speechbrain/spkrec-ecapa-voxceleb` | strong on similar-voice separation |
| NeMo TitaNet | `nvidia/speakerverification_en_titanet_large` | different training data |
| pyannote embedding | `pyannote/embedding` | pyannote's own |

Build `SpeakerDiarization(segmentation=..., embedding=<X>, clustering="AgglomerativeClustering")`,
sweep `clustering.threshold` per embedding, score. **If ECAPA/TitaNet separates `p09_e02/e03`,
the failure was the wespeaker embedding → swap it.** If all embeddings still merge them, the
fixtures move to the audit's suspect list.

### B. Deepgram (different engine entirely)  ·  independent read
The repo already has a Deepgram diarization provider. Deepgram Nova diarization is a fully
separate stack (no pyannote). Needs a `DEEPGRAM_API_KEY` (cloud; local/manual only — never CI).
Run the 8 fixtures, score. A true independent data point.

### C. NeMo (NVIDIA, different engine)  ·  independent read on-GPU
NeMo diarization (MSDD or the newer **Sortformer** end-to-end). Runs on the DGX GPU via an
`nvcr.io/nvidia/nemo` container. Different segmentation + embedding (TitaNet) + a neural
diarizer. Score. On-box, no API key.

### D. Audit (parallel, cheap)  ·  is the ground truth reachable?
For each persistent-failure fixture, extract **per-speaker talk-time** + **which labels merged**:
- `p09_e02/e03` (exp 3 → det 2 everywhere): did the two accented humans merge, or a human + the
  robotic ad? Listen / inspect. If two synthetic voices are acoustically identical, `expected=3`
  is unreachable → fix the fixture or the count.
- `p08_e04`/`p09_e01` (exp 2 → det 3): is there a genuine 3rd voice (intro sting / ad bleed) →
  `expected` is wrong, or a real spurious cluster → a fragment.
Outcome: corrected sidecars for any fixture whose ground truth was wrong, **before** re-scoring.

## Phase 2 — 2nd dataset: real episodes (local, unversioned) → the FULL matrix

After the synthetic-fixture matrix is solid, build a **2nd dataset from real corpus episodes** and
**re-run every model/embedding/engine** on it. Then the comparison is a **full matrix**:

```
                 synthetic fixtures     real episodes
  pyannote+wespeaker      X                  X
  pyannote+ECAPA          X                  X
  pyannote+TitaNet        X                  X
  Deepgram                X                  X
  NeMo                    X                  X
```

**This is what localizes the bottleneck:**
- a model passes **synthetic but fails real** → the bottleneck is the **real audio** (noise, music
  beds, dynamic-ad-insertion, montage clips) — not the model.
- a model **fails both** → the **model** is the limit.
- models **pass real but fail synthetic** → the **synthetic fixtures** are the problem (TTS voices
  too similar / bad ground truth) — the audit's suspicion, confirmed.

### Real-episode dataset (Phase 2 specifics)
- **Local only, NOT versioned** — real audio is bridge-only / never rehosted (it lives under
  `.test_outputs/manual/prod-v2/corpus/**/media/*.mp3`, gitignored). The dataset manifest +
  hand-established ground truth stay local (e.g. `data/eval/datasets/diarization_real_local/`,
  gitignored), never committed.
- **Curated, varied:** include the known over-segmented ones (the prediction-markets episode:
  15 labels for ~2 speakers), a clean 2-host interview, a panel/roundtable, a news/montage
  episode, and one with real ads — so it spans the same axes as the synthetic set.
- **Ground truth is the hard part:** real episodes have no clean speaker-count label. Establish
  it by hand for a *small* curated set (listen / read the transcript's named speakers /
  cross-check `content.speakers`). Keep it small (≈6–10 episodes) because ground-truthing is
  manual. This is the honest bottleneck of Phase 2 — do it carefully, few episodes, high
  confidence.

## Decision

Comparison table across **both datasets** (rows = model/embedding/engine × dataset), with the
audit's corrected ground truth applied. Pick the model+config that reaches **100% + gates** on the
fixtures AND does best on the real set. The full matrix tells us where the bottleneck is (model vs
synthetic fixtures vs real audio), so we fix the *right* thing rather than overfitting one cell.

Only **after** the full matrix is understood do we adopt the winner as the `config` defaults and
run the production re-diarization.

## Status / results (fill in as experiments land)

> **Superseded for pyannote 3.1 vs community-1 (2026-07-11, #1170).** The rows below
> were an EARLY sweep, before the fixture fixes (voice-collision + period-regex) and
> the segment-length squelch. The locked, cross-validated result is in
> [`EVAL_DIARIZATION_31_VS_COMMUNITY1_RTTM_2026_07`](../guides/eval-reports/EVAL_DIARIZATION_31_VS_COMMUNITY1_RTTM_2026_07.md):
> on the curated-8 **3.1 = 8/8, community-1 = 7/8**; on the full 45-fixture set
> **community-1 wins** (count 40/45 vs 32/45, DER 7.1% vs 10.8%). community-1 is the
> promoted default. The Exp A/B/C embedding rows below remain open.

| model | embedding | best exact | over | under | gate | config |
|---|---|---|---|---|---|---|
| pyannote 3.1 | wespeaker | 8/8 curated / 32/45 full | | | see report | one-voice fixtures + squelch |
| pyannote community-1 (v4) | wespeaker | 7/8 curated / 40/45 full | | | **default** | one-voice fixtures + squelch |
| pyannote + ECAPA | speechbrain | — | | | | (Exp A) |
| pyannote + TitaNet | nvidia | — | | | | (Exp A) |
| Deepgram | — | — | | | | (Exp B) |
| NeMo | TitaNet/Sortformer | — | | | | (Exp C) |
