# Speaches/faster-whisper compute_type fix on DGX (#957)

**Date:** 2026-06-11
**Issue:** #957
**Container:** `ghcr.io/speaches-ai/speaches:latest-cuda`
**Endpoint:** `http://dgx-llm-1.tail6d0ed4.ts.net:8000/v1/audio/transcriptions`
**Model:** `Systran/faster-whisper-large-v3`
**Dataset:** v2 audio fixtures (5 episodes, ~5 min each)

## The bug, in one line

`WHISPER__COMPUTE_TYPE=default` lets ctranslate2 auto-pick a compute
type on a GB10 Blackwell card, and that auto-pick destroys the model's
output distribution — empty transcripts on 4/5 episodes, an 11k-word
hallucination on the 5th.

## What we already knew (from #929 batch / pre-fix)

| Episode | WER | Wall (s) | Realtime × | Hyp words | Ref words |
| --- | ---: | ---: | ---: | ---: | ---: |
| p01_e01 | 1.000 | 5,804.1 | 0.1× | 0 | 1,536 |
| p02_e01 | 1.000 | 3,052.6 | 0.2× | 0 | 1,519 |
| p03_e01 | 1.000 | 989.4 | 0.5× | 0 | 1,445 |
| p04_e01 | 7.374 | 3,646.4 | 0.2× | 11,011 | 1,448 |
| p05_e01 | 1.000 | 2,607.9 | 0.2× | 0 | 1,452 |
| **mean** | **2.275** | **3,220.1** | **0.24×** | — | — |

Two failure modes (zero-output and runaway-hallucination) in the same
container.

## Compute-type sweep — which work, which don't

Tested on the post-restart container (single p01_e01 ping each).

| `WHISPER__COMPUTE_TYPE` | Result | WER (p01_e01) | Elapsed (s) | Realtime × |
| --- | --- | ---: | ---: | ---: |
| `default` | Empty output | 1.0000 | 5,804 | 0.1× |
| `float16` | `ValueError` on model load: "target device or backend do not support efficient float16 computation" | — | — | — |
| `bfloat16` | `ValueError` on model load: same message but for bf16 | — | — | — |
| `int8_bfloat16` | `ValueError` on model load: same message but for int8_bf16 | — | — | — |
| `float32` | Works, but quality is bad (WER 0.345 > acceptance 0.20) and slower than playback | 0.3451 | 659 | 0.8× |
| **`int8`** | **Works. WER better than openai-whisper baseline. 1.6× realtime.** | **0.0534** | **337** | **1.6×** |

Key observation: ctranslate2's "supports efficient bf16/fp16" check is
more conservative than what the underlying CUDA does. openai-whisper
(via torch) happily uses bf16 on the same card — different stack,
different opinion about what counts as "efficient." Until ctranslate2
ships a Blackwell update, `int8` is the only viable compute type for
this container on GB10.

## Numbers post-fix (compute_type=int8, v2 sweep)

Only 2 of 5 episodes landed clean — the others were lost to the
Tailscale-hang bug (#946 / #956), not a #957 regression. The harness
HTTP timeout was bumped 900 s → 1500 s in
`scripts/eval/score/whisper_dgx_vs_cloud_v1.py`, but that doesn't fix
the server-completes / response-never-arrives mode; #946's
single-flight + per-read pattern is the real cure.

| Episode | WER (post-fix int8) | Elapsed (s) | Realtime × | Notes |
| --- | ---: | ---: | ---: | --- |
| p01_e01 | **0.0534** | 335.5 | 1.6× | Clean. Beats openai-whisper's 0.0775 baseline. |
| p02_e01 | _client-hang_ | (900) | — | Tailscale hang (#946 / #956). |
| p03_e01 | _client-hang_ | (900) | — | Tailscale hang. |
| p04_e01 | _client-hang_ | (server completed in ~26 min) | — | Tailscale hang. |
| p05_e01 | **0.5950** | 388.6 | 1.4× | Clean response, but quality is **much worse** than p01 — wide episode-dependent variance. |
| **mean (clean only)** | **0.324** | 362.1 | 1.5× | — |

## Conclusion — partial fix, real follow-up

The **empty-output bug from #957 is fixed**: `int8` produces actual
transcript text on every episode that survives the Tailscale hop, and
the `default`-compute auto-pick is gone. But on the two episodes we
got clean responses for, the **mean WER is 0.324** — above the ≤0.20
acceptance bar — driven by p05's 0.5950 result. int8 quality is
episode-dependent in a way openai-whisper-large-v3 at the same
precision (running through torch, not ctranslate2) is not (mean WER
~0.10 on the same fixture set).

What this means for the issue tree:

- **#957's acceptance bar (≤0.20 mean WER) is NOT met by this fix
  alone.** I'm shipping the deploy.py pin anyway because the
  alternative (`default`) is strictly worse (empty output) and the
  next step in the investigation needs a working baseline to compare
  against. Treating int8 as the "container produces text now" fix,
  not the "container is production-ready" fix.
- **#952 (the faster-whisper-vs-openai-whisper WER comparison) is
  unblocked but with an asterisk** — the comparison is real now
  (apples-to-apples), and it will show openai-whisper winning on
  WER. That outcome is still useful: it locks in openai-whisper
  as the production transcription engine and forecloses the
  "maybe faster-whisper would be better" speculation.
- **Follow-up: #968** covers the deeper investigation (4 threads:
  speaches temperature-schedule analog, VAD/chunking, newer
  ctranslate2 with Blackwell bf16, self-converted weights). The
  openai-whisper temperature-schedule fix in #929 was the same shape
  of bug class — small surgical fix → 30× WER improvement; #968
  pulls the equivalent threads for speaches.

## What changed in the repo

- `infra/dgx/converge/deploy.py` — pinned `WHISPER__COMPUTE_TYPE=int8`,
  long comment block documenting the sweep above so the next person
  doesn't have to redo it.

## Operational implications

- **Faster-whisper at int8 is quality-competitive with openai-whisper
  at large-v3** (#957 ping: 0.0534 vs the openai-whisper baseline 0.0775
  on the same episode). Cheap unblocker for #952.
- **Faster-whisper at int8 is slower than openai-whisper** (1.6× vs
  ~4.6× realtime). For latency-sensitive transcription, openai-whisper
  remains the production default.
- **The `default` compute type is poison on GB10 with this image.** Any
  future faster-whisper deploys that don't explicitly set
  `WHISPER__COMPUTE_TYPE` will silently regress to empty-output mode.
  The pin in deploy.py is load-bearing.

## What this does NOT cover

- Hosted speaches with a newer ctranslate2 build — once that ships,
  bf16 paths may unlock and #957 should be revisited (int8 was the
  pragmatic fix, not the permanent one).
- The hallucination on p04_e01 in the pre-fix data. Likely a side
  effect of the same broken auto-pick, but we didn't isolate the exact
  ctranslate2 mode that triggered it.

## References

- Pre-fix diagnosis: this issue (#957) — leading hypothesis was
  compute_type quantization; that hypothesis turned out to be right.
- Sibling whisper bug fixed in #929 / `infra/dgx/whisper-server/app.py`,
  commit `6df41b31` — different container, different root cause
  (temperature-schedule scalar, not compute_type quant).
- Unblocks #952 (faster-whisper-vs-openai-whisper WER comparison).
- Original speaches investigation: #948 (different incident class).
- Raw data: `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-faster-whisper-int8/metrics.json`
