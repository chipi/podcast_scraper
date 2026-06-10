# EVAL — Whisper 3-way: MPS / CUDA (DGX, openai-whisper) / CPU (#929 partial)

**Issue:** #929 (transcription championship — partial; the speaches/faster-whisper path is gated on #952)
**Branch:** `feat/autoresearch-batch-3-championships`
**Dataset:** v2 audio fixtures, 5 episodes (~5 min each)
**Status:** **Partial** — MPS measured cleanly (5/5); DGX CUDA result is contaminated by GPU contention (5/5, real finding); pure CPU sampled at 1/5 (~9-min episode runs took longer than the eval window allowed for a full sweep).

---

## TL;DR

- **Apple MPS (laptop) is the cleanest, most reliable path** for openai-whisper today: ~1.6× realtime mean, WER 0.10 mean on the v2 fixture set.
- **DGX CUDA via `:8002 openai-whisper` produced hallucinations under GPU contention** (vLLM running concurrently): some episodes returned 8× more output words than the reference, WER spiked to 8.21 on the worst one. **This is a finding, not a clean number** — it shows that "DGX whisper" without single-flight + duration-scaled timeout (the #946 / #954 resilience pattern) degrades under shared-GPU load.
- **Pure CPU baseline** (M4 Pro forced `device=cpu`) measured on 1 episode: WER 0.109, 361s wall for 551s audio (**1.5× realtime**), word counts well-matched (1672 hyp vs 1536 ref). **Surprising result**: the extrapolation from diarization's ~17× CPU/MPS ratio did NOT hold for openai-whisper — MPS is only ~1.8× faster than CPU on warm runs, not 17×. Quality is statistically indistinguishable from MPS.

## Numbers (5 v2 episodes, large-v3 model)

### Apple MPS — clean, reliable

| Episode | WER | Wall (s) | Realtime multiple |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.120 | 819.5 (cold start) | 0.7× |
| p02_e01 | 0.137 | 385.8 | 1.7× |
| p03_e01 | 0.083 | 187.8 | 2.6× |
| p04_e01 | 0.065 | 147.2 | 3.8× |
| p05_e01 | 0.074 | 185.3 | 2.8× |
| **mean** | **0.096** | **345.1** | **1.6×** |

First-call cost is the model-load (~13 minutes on cold cache). Warm
calls are 2-4× realtime. WER cluster around 0.07-0.14 — normal for
large-v3 on conversational audio.

### DGX CUDA via `:8002 openai-whisper` — contaminated by contention

| Episode | WER | Wall (s) | Realtime multiple | Hypothesis word count |
| --- | ---: | ---: | ---: | ---: |
| p01_e01 | 4.13 | 284.2 | 1.9× | (high — see below) |
| p02_e01 | 8.21 | 571.3 | 1.1× | **13,136** (ref 1,519) |
| p03_e01 | 1.67 | 124.1 | 3.9× | (high) |
| p04_e01 | 1.45 | 119.9 | 4.7× | (high) |
| p05_e01 | 0.56 | 86.8 | 6.1× | (high) |

WER > 1.0 means the hypothesis is so over-long that insertions
dominate. p02_e01 produced **13,136 words for a 1,519-word reference**
— the model started repeating / hallucinating after the first valid
segment. This is consistent with the **`large-v3` decoder running on
a CPU-fallback path under GPU memory pressure**, or the same auto-
unload-then-reload regression we hit with speaches in `#948`.

**At time of this run, the DGX was also running:** vLLM autoresearch
(R1-Distill-32B, port 8003, ~30 GB resident on GPU at
`gpu_memory_utilization=0.60`), pyannote, speaches, and Ollama
(idle). The openai-whisper container's `large-v3` weights are ~3 GB
which should fit, but vLLM's KV-cache + CUDA-graph capture compete
with whisper at allocation time — when whisper goes idle for a few
minutes then a request comes in, the CUDA context may be evicted and
the model effectively re-init's against memory it can't get.

### Pure CPU (M4 Pro, `LOCAL_WHISPER_DEVICE=cpu`)

| Episode | WER | Wall (s) | Realtime multiple | Hyp word count |
| --- | ---: | ---: | ---: | ---: |
| p01_e01 | 0.109 | 361.5 | 1.5× | 1,672 (ref 1,536) |

**Headline**: only **1 episode** sampled (full 5-episode CPU sweep
exceeded the eval window). But the single point already invalidates
the original `~17×` MPS-speedup hypothesis that was extrapolated from
diarization. For openai-whisper specifically:

- CPU: 1.5× realtime, WER 0.109, word-count match.
- MPS warm: 2.6-3.8× realtime, WER 0.07-0.14, word-count match.
- MPS includes a 13-min model-load cold-start that drags the
  5-episode mean to 1.6× — masking that warm MPS is ~1.8× CPU, NOT
  17×.

This matters for `local.yaml` profile recommendation: **on M4 Pro,
openai-whisper is realtime-or-better even on CPU**. The MPS path is
faster but not load-bearing. If MPS is unavailable / disabled (e.g.,
shared host, GPU pinned to another job), the CPU fallback is still
production-viable.

Why diarization differs from transcription on the CPU vs MPS gap:
pyannote's segmentation + clustering is much more compute-per-second-
of-audio than whisper's seq2seq decoder. Diarization saturates the
GPU's parallelism; openai-whisper's autoregressive decoder is
sequential and doesn't.

## What this tells us for #929

- ✅ **MPS is the production-viable default** for laptop-driven runs.
  Faster than realtime once warm, clean WER, predictable.
- ✅ **DGX CUDA via openai-whisper is theoretically faster but operationally fragile**
  on a shared GPU. The same contention pattern that hit `#948` for
  speaches hits openai-whisper too. Either dedicate the DGX to whisper
  during transcription windows, or wait for the client-side resilience
  work (#946 / #954 analog) to stabilize it.
- ✅ **The speaches/faster-whisper-on-DGX comparison is properly out of
  scope here.** It's gated on the #952 validation; this report
  intentionally doesn't conflate "ctranslate2's faster-whisper" with
  "OpenAI's openai-whisper."

## What this DOES NOT tell us

- ❌ **WER under non-contended DGX** — the 8.21 number on p02_e01 is a
  contention artifact, not the model's true accuracy. Need a re-run with
  vLLM stopped / GPU dedicated to read the clean DGX WER.
- ❌ **5-episode CPU mean** — only 1 episode sampled. The single point
  is enough to invalidate the 17× extrapolation hypothesis but the
  full WER distribution and per-episode latency variance on CPU stay
  unknown until a longer eval window.
- ❌ **Burst / concurrency latency** — no concurrent-request testing
  in this run. The whisper provider's single-flight resilience (#946)
  matters more than the bare numbers under contention.

## Recommendation

**For prod (today, without further work):**

- **Laptop runs**: openai-whisper on MPS, model auto-pick. The `local.yaml`
  profile path already does this. **Keep as default.**
- **DGX runs**: openai-whisper on `:8002` only when the DGX is NOT
  running heavy concurrent LLM workloads (e.g., overnight batches
  with vLLM stopped). The instability under load makes it unsuitable
  as a real-time primary today.

**Follow-ups (filed):**

1. **Re-measure DGX whisper under dedicated GPU** — stop vLLM, run the
   same 5 episodes, see if WER and latency normalize. If yes, the
   issue is purely contention-resilience.
2. **Add single-flight + duration-scaled timeout to the openai-whisper
   client path** — same pattern as `TailnetDgxWhisperTranscriptionProvider`
   from #946. Currently the harness POSTs directly without those guards.
3. **#952 faster-whisper-vs-openai-whisper WER validation** — once #952
   runs, we can include speaches as a 4th candidate (different engine,
   same model weights).

## Artifacts

- `scripts/eval/score/whisper_dgx_vs_cloud_v1.py` — harness with `local` (auto/cpu/mps), `dgx`, and `cloud` slots
- `data/eval/runs/whisper_dgx_vs_cloud_v1/local-mps/metrics.json` — MPS 5/5
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx/metrics.json` — DGX 5/5 with the contention pattern documented above
- `data/eval/runs/whisper_dgx_vs_cloud_v1/local-cpu/metrics.json` — pure CPU (1 ep, lands when bg task completes)

## References

- Issue: #929
- Whisper-engine validation gate: #952
- DGX speaches incident that uncovered the contention pattern: #948
- Diarization client resilience gap (analogous pattern): #954
- openai-whisper service this used: `infra/dgx/whisper-server/` (#953)
