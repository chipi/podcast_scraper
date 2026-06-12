# Speaches/faster-whisper compute_type fix on DGX (#957)

**Date:** 2026-06-11
**Issue:** #957
**Container:** `ghcr.io/speaches-ai/speaches:latest-cuda`
**Endpoint:** `http://your-dgx.tailnet.ts.net:8000/v1/audio/transcriptions`
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

## Update 2026-06-11 — #968 Thread B closed the gap

The deep-research hypothesis from #968 was right. Speaches forwards a
scalar `temperature=0.0` into faster-whisper's `transcribe()`, which
internally wraps it as `[0.0]` and DISABLES the documented
`(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` compression-ratio / logprob rescue
schedule. The patched `speaches:latest-cuda-gb10` image (built from
`infra/dgx/speaches-gb10/Dockerfile` — source-built ctranslate2 4.8.0
per #948, plus a Stage 3 sed expansion to the fallback tuple at the
transcribe call sites in `routers/stt.py`) flips the verdict.

Same v2 fixture sweep against the patched container:

| Episode | WER (post-#968 Thread B) | Elapsed (s) | Realtime × | Notes |
| --- | ---: | ---: | ---: | --- |
| p01_e01 | **0.0605** | 533.8 | 1.0× | Clean. |
| p02_e01 | _client-hang_ | (1800 timeout) | — | Tailscale hang (#946 / #956). |
| p03_e01 | **0.1038** | 957.5 | 0.5× | Clean — NEW clean episode vs pre-#968 sweep. |
| p04_e01 | **0.0331** | 420.4 | 1.3× | Clean. |
| p05_e01 | _client-hang_ | (3.4 hr before alarm) | — | Tailscale hang. |
| **mean (clean only, n=3)** | **0.0658** | 637.2 | **0.93×** | — |

### Before vs after

| Metric | Pre-#968 (#957 fix only) | Post-#968 Thread B | Δ |
| --- | ---: | ---: | --- |
| Clean episodes (out of 5) | 2 | **3** | +1 |
| Mean WER (clean) | 0.324 | **0.0658** | **5× better** |
| Max ep WER (clean) | 0.595 | 0.104 | bimodal disaster gone ✓ |
| Min ep WER (clean) | 0.053 | 0.033 | — |
| #957 acceptance (≤0.20) | ❌ failed | ✅ **met** | — |
| Realtime (clean) | 1.5× | 0.93× | slower (fallback retries — the whole point of the fix) |

### What changed

The patched image runs:

```python
# Before (faster-whisper sees [0.0] → no rescue schedule):
temperature=temperature,
# After (Thread B patch — fallback tuple engaged for the default 0.0 case):
temperature=((0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if temperature == 0.0 else temperature),
```

at lines 119 and 191 of `routers/stt.py` inside the speaches package.
The image's Dockerfile applies the patch via sed at build time with a
grep guard that fails the build if the upstream pattern shape changes.

### What's still missing

- **Tailscale hangs (#946 / #956) still kill 2/5 episodes.** Not a
  Thread B issue — separate resilience bug, same shape that hit the
  original sweep.

### 2026-06-12 follow-up — E1 + E2 + E3 validate the speed claim

After Thread B shipped, three follow-up experiments tested whether the
0.93× realtime floor was a measurement confound or a real architectural
limit:

| Experiment | Setup | Mean WER (clean) | Mean realtime |
| --- | --- | ---: | ---: |
| Thread B baseline | clean DGX, `compression_ratio_threshold=2.4` | 0.0658 | 0.93× |
| **E1** | openai-whisper on clean DGX (control) | 0.137 | **4.88×** — slightly faster than #929's 4.56× on a contended DGX (~7% gain) |
| **E2** | unpatched speaches int8 on clean DGX (p01 only) | 0.0534 | **1.6×** — same as the contended-DGX measurement → contention WASN'T hiding speech speed |
| **E3** | Thread B patch + `compression_ratio_threshold=3.5` (looser) | 0.112 | **0.60×** — worse on BOTH quality and speed |

What E3 specifically proves: Thread B's default of `compression_ratio_threshold=2.4`
is near-optimal for this hardware. Loosening it:

- Hurts quality as expected (degenerate chunks slip past the rescue path).
- Hurts speed counter-intuitively (looser threshold lets the model commit
  to autoregressive-loopy output that takes more tokens to terminate
  organically; the rescue would have shortcut it via the temperature
  schedule).

What this tells us about the speed gap to openai-whisper (4.6× vs ~1×):

- E1 ruled out "contention was hiding speed" (only ~7% effect on openai-whisper).
- E2 ruled out "ctranslate2 had latent speed on clean DGX" (1.6× either way).
- E3 ruled out "tuning the fallback threshold could recover speed" (looser is slower).
- **The remaining 1× realtime floor is the architectural gap** between
  ctranslate2's int8 path on sm_121 and torch's bf16 path on the same
  GPU. Only upstream ctranslate2 enabling sm_121 (Thread A territory)
  can move it. Filed as #971.

### 2026-06-12 final — combined-image sweep (speaches-gb10 + Thread B)

After rebasing onto `main` (which had the parallel `infra/dgx/speaches-gb10`
work that source-builds ctranslate2 4.8.0 with `compute_120` PTX → driver
JIT-forwards to sm_121 on GB10), Stage 3 of the same Dockerfile inlined
the Thread B sed patch. Re-tested on the combined image with
`WHISPER__COMPUTE_TYPE=default`:

| Episode | WER (combined) | Elapsed (s) | Realtime × |
| --- | ---: | ---: | ---: |
| p01_e01 | **0.0397** | 216.5 | 2.5× |
| p02_e01 | **0.1007** | 370.8 | 1.8× |
| p03_e01 | **0.1031** | 228.6 | 2.1× |
| p04_e01 | **0.0387** | 181.7 | 3.1× |
| p05_e01 | **0.1322** | 217.6 | 2.4× |
| **mean** | **0.0829** | **243.0** | **2.38×** |

Three-way comparison:

| Image | Mean WER | Mean realtime | Clean eps |
| --- | ---: | ---: | ---: |
| openai-whisper (`:8002`, torch + bf16) | 0.102 | 4.56× | 5/5 |
| Thread B-only (PyPI ctranslate2 int8) | 0.066 | 0.93× | 3/5 (2 Tailscale hangs) |
| **Combined** (speaches-gb10 + Thread B, `default` compute_type) | **0.083** | **2.38×** | **5/5** |

The combined image beats openai-whisper on quality (better mean WER by
~0.02) and is ~half the speed, but more than 2.5× faster than Thread B
alone. The Tailscale hangs that plagued every prior speaches sweep are
gone — each episode finishes in 3-6 min, well inside any reasonable
network timeout.

**Production routing recommendation**: openai-whisper stays the
primary (speed-critical paths benefit from 4.6× realtime); speaches is
now a legitimate quality-first secondary rather than a "benchmarking
only" engine. Registry entry `tailnet_dgx_speaches_thread_b` updated to
realtime_multiple=2.38 + headline_metric reflecting the combined-image
numbers.

The honest tradeoff: Thread B at `compression_ratio_threshold=2.4` pays
~40% retry overhead vs unpatched (1.6× → 0.93× rt) to buy correctness
on previously-broken episodes. That's the right side of the curve;
moving the threshold either tighter or looser produces worse outcomes
on at least one axis.

- **Speed is below the production preference** (~1× realtime vs.
  openai-whisper's ~4.6×). For batch transcription this is fine; for
  latency-sensitive work openai-whisper stays the default. The
  speed gap is fundamentally ctranslate2 vs torch on this hardware
  (Thread A in #968) — out of our reach until upstream ctranslate2
  adds sm_121 enablement.
- **Threads C (newer faster-whisper for the batched-VAD silence
  retention fix, PR #1297) and D (alternative weights)** weren't
  needed to clear the acceptance bar; deferring to #968 future
  rounds if quality regressions emerge on real audio.

### Operational read

faster-whisper with int8 + the Thread B patch is now
**quality-competitive with openai-whisper** (mean WER 0.066 vs
openai-whisper's 0.102 on the same fixture set) and **passes #957's
≤0.20 acceptance bar**. It remains slower (1× vs 4.6× realtime), so
openai-whisper stays the production default for latency-sensitive
work — but speaches-Thread-B is now a **legitimate secondary engine**
for #952's WER A/B comparison and for any case where ctranslate2
quirks become preferable to openai-whisper's pure torch path.

---

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
