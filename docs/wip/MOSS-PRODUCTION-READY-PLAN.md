# MOSS production-ready plan (#1177 impl / #1174 eval)

**Date:** 2026-07-16 · in-domain follow-on to #1173, before the #1191 epic.

MOSS-Transcribe-Diarize: one 0.9B Apache-2.0 model emits transcript + speaker labels + timestamps
in a single pass. #1177 = implementation + DGX service; #1174 = adoption eval (gated on English
quality, which has **zero published English numbers** — every MOSS benchmark is Mandarin CER).

## Current state (assessed 2026-07-16)

**Done:**
- `providers/moss/moss_provider.py` (transcription) + `providers/ml/diarization/moss_provider.py`
  (diarization) — both wired into their factories (transcription factory line 279; diarization
  factory line 84). Deepgram two-call shape; DGX service caches by audio digest so the 2nd stage
  is free.
- `providers/moss/sats.py` (SATS stream parser) + tests. Config fields (`moss_port`, native-
  screenplay set, provider Literals). **15 unit tests green** (fixtures, no live service).
- `infra/dgx/moss-server/app.py` (FastAPI service) + `experiment_dgx_moss.yaml`.

**Gaps to #1177 acceptance:**

| # | Gap | Where | Gated |
|---|---|---|---|
| G1 | Service never run vs the real model — `_processor(audio=path)` API + `max_new_tokens=8192` (won't cover a 45-min episode) unvalidated | `moss-server/app.py` | **DGX** |
| G2 | No deploy artifacts — Dockerfile, requirements, `sats.py` vendoring, compose entry, converge, gpu-mode-swap `:8004` | `moss-server/` | local build, DGX deploy |
| G3 | No `prod_dgx_moss.yaml` + no-cloud-leak assertion | `config/profiles/` | local |
| G4 | 10-episode comparison run vs prod-v3 (`compare_corpora_v1.py`) — also #1174's adoption gate | eval | **DGX** |
| G5 | Possible `capabilities.py` entry; ModelRegistry entry (#1174 deliverable) | providers | local |

## Plan

### Local (crush now — no GPU)
1. **`prod_dgx_moss.yaml`** — mirror `experiment_dgx_moss` minus the experiment gate; MOSS for
   transcription+diarization, `qwen3.5:35b` summary/GI/KG, fully local. + a profile-resolves /
   **no-cloud-leak** assertion test.
2. **Deploy artifacts** for `moss-server/`: `Dockerfile` (transformers, `trust_remote_code`),
   `requirements.txt`, a build step that vendors `sats.py` next to `app.py`, a compose service on
   `:8004`, and the gpu-mode-swap `prod` health entry. Reviewable; deployed in the DGX session.
3. **Harden `app.py` for long audio** — 8192 tokens can't cover 45-min episodes in one pass; add
   chunked inference (or a dynamic token budget) with segment-time stitching. This is the biggest
   correctness risk after the processor API.
4. **capabilities/ModelRegistry** entry if missing; docs (recipe + the eval-report skeleton).

### DGX-gated (needs a GPU session + explicit gpu-mode-swap approval)
5. Deploy the service, download weights, **validate the processor API + long-audio path** against
   the real model (G1) — expect to fix `_infer` once the real API is known.
6. Health-check + survive `gpu-mode-swap.sh prod`.
7. The **10-episode eval** (G4/#1174) vs prod-v3 — the adoption verdict. English quality is the
   load-bearing unknown; MOSS may not clear the sync/WER bars, in which case it stays a wired-but-
   unadopted option (profile present, not a prod default).

## DGX validation results (2026-07-16) — MOSS runs on GB10

Ran in `nvcr.io/nvidia/vllm:26.05-py3` (torch 2.12/CUDA13, transformers 5.6) via the upstream
`moss_transcribe_diarize` package. cuFFT is **stubbed** in that image → whisper's GPU mel-STFT
fails ("cuFFT error 50"); dodged by forcing feature extraction to **CPU** (`prepare_inputs
device=None`), which is cheap.

| Audio | Dur | Infer | Realtime | Diar | Transcription |
|---|---|---|---|---|---|
| `p01_e04` fixture (clean) | 82 s | 13.6 s | **6.0×** | 2 spk (correct) | accurate, clean |
| `e070` prod (The Daily) | 2636 s | 1254 s | **2.1×** | **28 spk** | **accurate on real audio** |

**Verdict:**
- **English transcription: excellent** on both fixture and real prod audio — the load-bearing
  unknown (#1174) is resolved positively.
- **Speed: 2.1× realtime** on the long episode in **bare transformers**. This is the catch: the
  adoption rationale was "50–100× vs large-v3's 7.8×." Bare transformers is **~4× SLOWER than the
  incumbent large-v3**. The speed win requires the **vLLM backend** (the package ships
  `app/vllm_runner.py`) — which is exactly what #1177 avoided as untested on sm_121.
- **Diarization: 28 raw speaker labels** on a ~few-speaker news episode (has archival clips, so
  not necessarily wrong) — must be judged by **DER + count vs the RTTM fixtures** (#1174 bar: beat
  community-1's 40/45 count, 7.1% DER) before trusting. Not yet scored.

**Implication:** MOSS is a strong **transcription** provider (great English, single model) but its
headline **speed advantage does not hold in bare transformers**, and diarization quality is
unproven. Adoption now hinges on (a) does vLLM work on GB10 for the speed, and (b) DER on fixtures.

## Honest framing
Everything code-side can be made production-grade locally, but the **production-ready verdict is
DGX + English-eval gated**. Deliver the local artifacts so one DGX session closes it; do not flip
any prod default to MOSS until #1174's English numbers clear the bars.
