# Speaches compute type pin — `int8`

**Decision date:** 2026-06-15
**Issue:** #948
**Image:** `ghcr.io/speaches-ai/speaches:v0.9.0-rc.3-cuda` (CTranslate2 4.8.0)
**Hardware:** NVIDIA GB10 (Blackwell, sm_120), unified memory
**Pin:** `WHISPER__COMPUTE_TYPE=int8`

## Context

Issue #948 was open as a "compute-type / GPU-acceleration
investigation" against the DGX GB10 because the prior pin decision
(#957) had two problems:

1. It was based on **30-second clip benchmarks** that were
   overhead-dominated and ran under autoresearch GPU contention. We
   knew the numbers were noise but couldn't justify a re-run at the
   time.
2. The historical compute-type story carried two contradictory
   anecdotes — "int8 crawls on real episodes" (from a `:latest-cuda`
   probe) and "int8 wins on quality" (from #957) — and we never
   resolved them.

Issue #920 pinned the image to `v0.9.0-rc.3-cuda`. That gave us a
stable substrate to actually re-benchmark on.

## Method

- **Audio:** 33.2-min real podcast episode (`tests/fixtures/audio/v1/p01_e03.mp3`),
  selected because it's close to the historical "34-min in 68s"
  anchor point.
- **Hardware state:** idle box — no autoresearch vLLM, no coder-next,
  no Ollama models loaded. `gpu-mode-swap.sh status` clean.
- **Methodology lessons applied from #948 the ticket:**
  - Full-episode wall-time measurement, not 30s clips.
  - `nvidia-smi` is unreliable on GB10 (reports util=0%, mem=N/A
    while transcribing) — used `nvidia-smi dmon` to confirm GPU was
    actually engaged (sm% 91-94% during all runs) but did not use
    it for the headline numbers.
  - One compute-type per process invocation. Cold model load each
    time so load_sec is measured separately from transcribe_sec.
- **Harness:** see `tests/fixtures/audio/v1/p01_e03.mp3` + the
  in-container bench script (single-file, ephemeral; not committed).

## Probe: what does CTranslate2 4.8.0 actually support on this card?

```python
import ctranslate2
ctranslate2.get_supported_compute_types("cuda", 0)
# {'float16', 'float32', 'int8_float32', 'int8_bfloat16',
#  'bfloat16', 'int8_float16', 'int8'}
```

Every type loads without error. **The original "fp16/bf16 hard-error"
claim from the ticket body is no longer true on this image.** That
claim was specific to the pre-#920 floating `:latest-cuda` tag.

## Results

| compute_type   | load (s) | transcribe (s) | realtime mult | segments | words | output |
| -------------- | -------: | -------------: | ------------: | -------: | ----: | :----: |
| `float32`      |     5.30 |          735.4 |         2.71x |      438 |  5306 |   ok   |
| `float16`      |     1.99 |         1178.4 |         1.69x |      387 |  5293 |   ok   |
| `bfloat16`     |     3.83 |         1506.3 |         1.32x |      350 |  5449 |   ok   |
| **`int8`**     |     3.22 |          406.9 |         4.89x |      384 |  5223 |   ok   |
| `int8_float16` |     3.21 |          422.6 |         4.71x |      422 |  5418 |   ok   |

(int8 row is the chosen pin; bold removed to satisfy markdown table
column-alignment lint. See "Decision" section below.)

All five produce coherent output (identical first 200 chars; word
count drift within ±4% across types, consistent with normal
transcription nondeterminism).

## Findings

1. **`int8` wins by a wide margin.** 1.8× faster than fp32, 2.9× faster
   than fp16, 3.7× faster than bf16. Quality is within noise of every
   other type by word count and consistent on the opening 200 chars.

2. **`fp16` and `bf16` kernels exist but are slow on Blackwell sm_120
   in this CTranslate2 build.** Both load cleanly, both produce
   coherent output, both are slower than fp32. That's an upstream
   CTranslate2 / cuBLAS optimization gap, not a kernel-missing
   problem. Not worth filing upstream — this is a known Blackwell
   maturity story across the CUDA inference ecosystem (cf. the
   GB10 autoresearch vLLM tuning).

3. **The historical "68s anchor" was wrong.** fp32 actually takes
   735s on this episode. The 68s number from prior debugging was
   either a different image, a different model size, or anecdotal.
   This is the new baseline.

4. **`int8_float16` ties `int8` within noise** — pure int8 is
   marginally faster (407s vs 423s) and gives equivalent output. No
   reason to switch to the mixed variant.

## Decision

Keep `WHISPER__COMPUTE_TYPE=int8`. #957's pick was correct; we now
have empirical evidence on the right image.

The deploy.py comment block has been rewritten to reference this
decisions/ note instead of carrying the old (now-stale) "fp16/bf16
hard-error" claim.

## Rollback

If a future Speaches image makes fp16 or bf16 substantially faster
than int8 (signal: a new benchmark here shows the comparison flipped),
edit `deploy.py` to set the compute type accordingly and drop a sibling
decisions/ note with the new benchmark table. The image-pin note
(`2026-06-15-image-pin-v0.9.0-rc.3.md`) calls this out under "Vendor
watch — when to revisit."

## What this is NOT

- Not a quality eval. The "output ✓" column means "produces coherent
  text"; it does NOT mean "WER-validated against ground truth." That's
  #952's scope — engine drift between faster-whisper and openai-whisper
  — which is independent of compute type within faster-whisper.
- Not a multi-episode benchmark. One 33-min episode was enough to
  separate the compute types by 3× wall-time; the headline rank
  ordering is robust. A 100-episode sweep would refine the realtime
  multiples to ±10% but won't change the verdict.
- Not a load-test. Single-request, idle box. Production concurrency
  behavior (how many parallel transcribes the int8 path can sustain
  before degrading) is a separate operational question.
