# EVAL — Whisper 4-way: MPS / CPU / DGX openai-whisper / DGX faster-whisper (#929)

**Issue:** #929 (transcription championship)
**Branch:** `feat/autoresearch-batch-3-championships`
**Dataset:** v2 audio fixtures, 5 episodes (~5 min each), large-v3 model on every backend that supports it
**Status:** **DONE — 4-way comparison complete.**

> **Note**: filename retains the original "3-way" tag for git history
> continuity, but this report now covers the 4-way result after Track
> 3 (DGX faster-whisper) and the post-fix DGX openai-whisper re-run
> landed during the same eval window.

---

## TL;DR

- **Apple MPS (laptop) is the clean local default**: WER 0.096 mean,
  1.6× realtime (warm ~2.6–3.8×).
- **Pure CPU on M4 Pro is a viable fallback**: WER 0.137 mean, 2.34×
  realtime — quality 30% behind MPS but production-tolerable. The
  surprise finding from earlier (MPS only ~1.4× faster than CPU on
  this workload, not 17×) holds at scale.
- **DGX openai-whisper, after the temperature-schedule fix
  (`infra/dgx/whisper-server/app.py`): the new speed leader.** WER
  0.102 mean (matches MPS within 6%), **4.56× realtime** (~3× faster
  than MPS, ~2× faster than CPU).
- **DGX faster-whisper (speaches image, port 8000) is still broken**
  — returns empty output on 4/5 episodes, hallucinates on the 5th.
  Different bug from the openai-whisper one (lives in the speaches
  container's config, not our app.py). Filed as a follow-up.

**Production recommendation**: route DGX-equipped profiles to the
**fixed openai-whisper on `:8002`** (after this PR lands). Best
combination of quality + latency. MPS stays the laptop default.
Cloud Whisper API stays the choice for `cloud_*` profiles.

## The temperature-schedule bug (root-caused this PR)

### Symptom (the partial report's "DGX is broken" story)

Before the fix, the DGX `whisper-openai` container produced WER 3.20
mean — 2-9× more output words than the reference, repeating phrases
at the end of each transcript. Documented as "container is broken"
because:

- Behavior was **byte-identical** whether vLLM was contending for
  GPU memory or not (5/5 episodes had identical WER and hyp word
  counts in both the contended and non-contended runs).
- Pattern was consistent with autoregressive runaway: the decoder
  loops past the natural EOS and keeps generating until the
  max-tokens cap.

The original synthesis recommended NOT routing transcription to the
DGX because of this. That recommendation is **withdrawn after the
fix landed**.

### Root cause

`infra/dgx/whisper-server/app.py` (line 138, pre-fix):

```python
temperature: float = Form(0.0, ge=0.0, le=1.0),
...
transcribe_kwargs = {
    "temperature": temperature,
    "fp16": _DEVICE == "cuda",
}
result = _MODEL.transcribe(tmp_path, **transcribe_kwargs)
```

`openai-whisper`'s `transcribe()` default `temperature` is a
**schedule** `(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` — it starts deterministic
at 0.0 but falls back to higher temperatures when the decoder
triggers a hallucination check (`compression_ratio_threshold`,
`logprob_threshold`). That fallback is what rescues long-audio
transcription from autoregressive loops.

Passing a **scalar** `0.0` disables the fallback. The decoder gets
no recovery path; once it enters a loop, it keeps generating.

The laptop's MPS call goes through `model.transcribe(audio_path,
verbose=False, fp16=True)` with **no `temperature` kwarg**, so it
uses the default schedule and recovers gracefully. That's why MPS
worked and DGX didn't.

### Verification (in-container A/B test)

Ran `whisper.transcribe()` directly inside the `whisper-openai`
container on `p01_e01.mp3` with both modes:

| Test | Kwargs | Hyp words | Elapsed | Notes |
| --- | --- | ---: | ---: | --- |
| A: scalar 0.0 (pre-fix) | `temperature=0.0` | **7,311** | 328.9s | Last 200c repeating: `"...land stewardship change the calculus here? Where do most teams get it wrong? How does land stewardship change the calculus here?"` |
| B: default schedule (fix) | (no `temperature`) | **1,446** | 175.3s | Natural ending: `"That's it for today's episode of Single Track Sessions. See you next time."` |

Same container, same model, same audio. Only the kwarg differs.
**Confirms scalar temperature kwarg disables the schedule, schedule
fallback rescues long audio.** Also: schedule mode is FASTER (no
broken-decode → fallback re-run loop eats wall time).

### Fix

`infra/dgx/whisper-server/app.py` (post-fix):

- `temperature: Optional[float] = Form(None, ...)` — default None.
- Only pass `transcribe_kwargs["temperature"] = temperature` when
  the caller explicitly set a value. Otherwise let openai-whisper
  use its default schedule.

Deployed via `make dgx-deploy` (rebuilds the whisper-server image
and restarts the container). Verified by re-running the 5-episode
sweep — see Numbers section below.

## Numbers — 5 v2 episodes, large-v3 weights

### Apple MPS (laptop) — clean local default

| Episode | WER | Wall (s) | Realtime multiple |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.120 | 819.5 (cold start) | 0.7× |
| p02_e01 | 0.137 | 385.8 | 1.7× |
| p03_e01 | 0.083 | 187.8 | 2.6× |
| p04_e01 | 0.065 | 147.2 | 3.8× |
| p05_e01 | 0.074 | 185.3 | 2.8× |
| **mean** | **0.096** | **345.1** | **1.6×** |

First-call cost is the model-load (~13 min on cold cache). Warm runs
are 2-4× realtime.

### Pure CPU (M4 Pro, `LOCAL_WHISPER_DEVICE=cpu`) — viable fallback

| Episode | WER | Wall (s) | Realtime multiple |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.109 | 361.5 | 1.5× |
| p02_e01 | 0.115 | 293.9 | 2.2× |
| p03_e01 | 0.107 | 297.8 | 1.6× |
| p04_e01 | 0.072 | 152.2 | 3.7× |
| p05_e01 | 0.281 | 197.8 | 2.7× |
| **mean** | **0.137** | **260.6** | **2.34×** |

4 of 5 episodes match MPS quality. One outlier (p05_e01 at 0.281)
drags the mean. **MPS is only ~1.4× faster than CPU** on this
fixture set — the previously-extrapolated 17× CPU/MPS ratio from
diarization didn't hold for openai-whisper. CPU fallback for
`local.yaml` is real.

### DGX openai-whisper on `:8002` — POST-FIX (production winner)

After applying the temperature-schedule fix and re-running:

| Episode | WER | Wall (s) | Realtime multiple | Hyp words | Ref words |
| --- | ---: | ---: | ---: | ---: | ---: |
| p01_e01 | 0.077 | 132.1 | 4.2× | 1,608 | 1,536 |
| p02_e01 | 0.161 | 165.9 | 4.0× | 1,650 | 1,519 |
| p03_e01 | 0.093 | 96.2 | 5.1× | 1,529 | 1,445 |
| p04_e01 | 0.065 | 113.4 | 5.0× | 1,512 | 1,448 |
| p05_e01 | 0.115 | 116.3 | 4.5× | 1,534 | 1,452 |
| **mean** | **0.102** | **124.8** | **4.56×** | — | — |

Hyp word counts now match reference within ±10% on every episode
(no more 7,000-word runaways). Quality is within scoring noise of
MPS (0.102 vs 0.096 = 6% gap, smaller than the inter-episode
variance on either backend). Realtime multiple ~3× MPS, ~2× CPU —
the GPU's actual win.

### DGX openai-whisper on `:8002` — PRE-FIX (kept for the audit trail)

| Episode | WER | Hyp words | Ref words |
| --- | ---: | ---: | ---: |
| p01_e01 | 4.131 | 7,368 | 1,536 |
| p02_e01 | 8.211 | 13,136 | 1,519 |
| p03_e01 | 1.671 | 3,371 | 1,445 |
| p04_e01 | 1.448 | 2,821 | 1,448 |
| p05_e01 | 0.558 | 1,943 | 1,452 |
| **mean** | **3.204** | — | — |

These numbers are **the result of the temperature-scalar bug**, not
a property of the DGX. Documented for the audit trail because the
synthesis report's earlier "DGX whisper unfit for production"
recommendation was based on them.

### DGX faster-whisper on `:8000` (speaches image) — PRE-FIX (kept for audit trail)

| Episode | WER | Wall (s) | Realtime multiple | Hyp words | Ref words |
| --- | ---: | ---: | ---: | ---: | ---: |
| p01_e01 | 1.000 | 5,804.1 (97 min) | 0.1× | **0** | 1,536 |
| p02_e01 | 1.000 | 3,052.6 | 0.2× | **0** | 1,519 |
| p03_e01 | 1.000 | 989.4 | 0.5× | **0** | 1,445 |
| p04_e01 | 7.374 | 3,646.4 | 0.2× | 11,011 | 1,448 |
| p05_e01 | 1.000 | 2,607.9 | 0.2× | **0** | 1,452 |
| **mean** | **2.275** | **3,220.1** | **0.24×** | — | — |

Two failure modes in one container:

1. **Zero-output (4/5)**: 200 OK with empty transcript after tens
   of minutes wall time.
2. **Hallucination (1/5)**: 11,011 words for 1,448 reference.

Root cause (diagnosed in #957): `WHISPER__COMPUTE_TYPE=default` was
letting ctranslate2 auto-pick a broken quantization on the GB10
Blackwell card. CTranslate2 also rejects `float16`, `bfloat16`, and
`int8_bfloat16` on this hardware with "target device or backend do not
support efficient X computation" — so the only viable compute type is
**`int8`** (pure int8 weights + int8 compute). See
[EVAL_SPEACHES_COMPUTE_TYPE_2026_06.md](./EVAL_SPEACHES_COMPUTE_TYPE_2026_06.md)
for the full sweep.

### DGX faster-whisper on `:8000` (speaches image) — POST-FIX (compute_type=int8)

Tested on the same v2 fixture set, post-`int8` pin in `deploy.py`. The
full 5-episode sweep was destroyed by the **Tailscale-hang bug
(#946 / #956)** — speaches completed all five server-side, but the
response bodies for episodes 2-5 never reached the laptop (the harness
hit its HTTP read timeout). That's a separate issue from #957; it
would have hit any DGX-served stack with similarly long responses.

Two episodes landed cleanly (re-shot p05 with the bumped HTTP timeout
after the original sweep wedged on episodes 2-4):

| Episode | WER (post-fix int8) | Wall (s) | Realtime × | Notes |
| --- | ---: | ---: | ---: | --- |
| p01_e01 | **0.0534** | 335.5 | 1.6× | Beats openai-whisper's 0.0775 baseline. |
| p05_e01 | **0.5950** | 388.6 | 1.4× | Much worse than p01 — wide episode-dependent variance. |
| p02 / p03 / p04 | — | — | — | Tailscale-hang (#946 / #956). Server-side processing completed; client never got the response. Not a #957 regression. |
| **mean (clean)** | **0.324** | 362.1 | 1.5× | — |

Operational read: **the compute-type fix unblocks the empty-output
bug but does NOT make faster-whisper production-ready.** Mean WER
0.324 across the two clean episodes is above #957's ≤0.20 acceptance
bar — int8 has episode-dependent quality variance that openai-whisper
at the same precision (via torch, not ctranslate2) doesn't exhibit.
For now: openai-whisper stays the production default for everything;
faster-whisper-int8 is usable only for #952's apples-to-apples WER
comparison (which will show openai-whisper winning). A follow-up to
investigate the int8 variance (newer ctranslate2 build, alternative
weights, VAD/chunking interaction, speaches temperature-schedule
analog) is filed as **#968**; see
[EVAL_SPEACHES_COMPUTE_TYPE_2026_06.md](./EVAL_SPEACHES_COMPUTE_TYPE_2026_06.md)
for the threads.

## 4-way summary

| Backend | WER mean | Realtime × | Status | Per-PR change |
| --- | ---: | ---: | --- | --- |
| **DGX openai-whisper FIXED (`:8002`)** | **0.102** | **4.56×** | ✅ new production winner | Fix in this PR |
| **MPS (laptop, openai-whisper)** | 0.096 | 1.6× | ✅ laptop production default | No change |
| CPU (laptop, openai-whisper) | 0.137 | 2.34× | ✅ viable fallback | New finding documented |
| **DGX faster-whisper post-#968 Thread B (`:8000`, int8 + temperature-fallback patch)** | **0.066** (3 clean eps, max 0.104) | 0.93× | ✅ quality-competitive with openai-whisper; speed gap from ctranslate2 vs torch (Thread A still open) | Patched image `podcast-speaches:0.1.0` (FROM speaches:latest-cuda + sed expansion of scalar temp to fallback tuple) |
| DGX faster-whisper post-#957 only (`:8000`, int8 + default scalar temperature) | 0.324 (2 clean eps; bimodal 0.05/0.60) | 1.5× | ⚠️ superseded by Thread B above — audit trail only | — |
| DGX faster-whisper PRE-#957 (`:8000`, default) | 2.275–7.374 | 0.24× | ❌ (audit trail only — broken compute_type auto-pick) | Fixed by #957 |
| DGX openai-whisper PRE-FIX (`:8002`) | 3.204 | 2.30× | ❌ (audit trail only) | Fixed this PR |

## What this tells us for #929

- ✅ **DGX openai-whisper is the new production transcription default
  for `cloud_with_dgx_*` profiles** — quality within scoring noise of
  MPS, ~3× faster on realtime, free per token. The temperature-
  schedule bug was the only thing blocking it.
- ✅ **MPS stays the `local.yaml` default**. No reason to change —
  it's clean and the operator already has it. CPU fallback is also
  viable if MPS is unavailable.
- ❌ **DGX faster-whisper (speaches) is not the production path** —
  separate bug, separate fix. Filed.
- ⏭️ **Cloud Whisper API stays the choice for `cloud_*` profiles** —
  not retested this batch.

## What this DOES NOT tell us

- ❌ **The speaches/faster-whisper container's root cause.** We
  documented the empirical failure but haven't yet bisected (the
  `compute_type=default` hypothesis is the leading suspect but not
  verified). Filed as follow-up. The container should not be the
  production path until this is resolved.
- ❌ **WER on real podcast audio (90 min episodes).** All 5 episodes
  are v2 synthesized fixtures (~5 min each). Real podcasts have
  different challenges (noise, accents, music interludes). Synthesis
  recommendation reasons from the v2 numbers, which the operator has
  consistently used as the proxy for production audio.
- ❌ **Burst / concurrency latency.** No concurrent-request testing
  in this run. The whisper provider's single-flight pattern from
  #946 and the broader DGX-over-Tailscale resilience patterns from
  #956 are still the right wrap for any prod DGX whisper consumer.

## Tailscale client-side caveat

The first two attempts at the post-fix 5-episode sweep hung after
episode 1/2 due to a **DGX-over-Tailscale HTTP response stuck
mid-transit** — server returned 200 OK but the response body
didn't reach the laptop. Reproducible. Worked around with a
per-episode loop driven by `perl -e 'alarm 1200'` (each episode is
an independent harness invocation; a hang only kills that one
episode).

**This is NOT a property of DGX whisper.** It's a generic
blocking-HTTP-over-Tailscale failure mode that bites every long-
running client. Filed as **#956** (DGX-over-Tailscale client
resilience). Until #956 lands the shared resilience layer, the
operator should expect this pattern on any long DGX call from the
laptop and use the per-call timeout workaround.

## Recommendation

**For prod (after this PR lands):**

- **Laptop runs (`local.yaml`)**: openai-whisper on MPS, model
  auto-pick. CPU fallback is viable. **No change.**
- **DGX-equipped runs (`cloud_with_dgx_*`)**: route transcription
  to **`whisper-openai` on `:8002`** (the fixed container).
  Quality matches MPS within noise, ~3× faster realtime.
- **Cloud-only runs (`cloud_*`)**: cloud Whisper API. Not retested
  this batch.

**Follow-ups (filed):**

1. **#956** — DGX-over-Tailscale client resilience (shared
   timeout/retry/keepalive layer across all DGX clients). Covers
   the hang pattern we hit during this eval.
2. **Speaches/faster-whisper root cause** (separate ticket — file
   if not already; container produces empty output on 4/5
   episodes, likely `compute_type=default` → bad quantization).
3. **#946 / #954** — whisper / diarization client resilience.
   Should expand to use the #956 shared layer instead of bespoke
   patterns per backend.

## Artifacts

- `infra/dgx/whisper-server/app.py` — the fix (temperature now
  Optional[float], only passed to transcribe() when explicit)
- `scripts/eval/score/whisper_dgx_vs_cloud_v1.py` — harness with
  `local` (auto/cpu/mps), `dgx`, and `cloud` slots
- `data/eval/runs/whisper_dgx_vs_cloud_v1/local-mps/metrics.json` — MPS 5/5
- `data/eval/runs/whisper_dgx_vs_cloud_v1/local-cpu/metrics.json` — pure CPU episode 1
- `data/eval/runs/whisper_dgx_vs_cloud_v1/local-cpu-rest/metrics.json` — pure CPU episodes 2-5
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx/metrics.json` — DGX openai-whisper, PRE-fix, contended (audit trail)
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-noncontended/metrics.json` — DGX openai-whisper, PRE-fix, non-contended (audit trail; identical to contended)
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-fixed/metrics.json` — DGX openai-whisper, POST-fix (the production-recommended path)
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-fixed-perep/<ep>/metrics.json` — per-episode raw metrics for the post-fix sweep (the Tailscale-hang workaround pattern)
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-faster-whisper/metrics.json` — DGX faster-whisper (still broken; separate bug)

## References

- Issue: #929
- This PR fixes: `infra/dgx/whisper-server/app.py` — the temperature-schedule disable
- DGX-over-Tailscale client resilience (filed this batch): #956
- Whisper client resilience (existing): #946
- Diarization client resilience (analogous pattern): #954
- Whisper-engine validation gate: #952 (blocked on speaches root-cause — see follow-ups)
- DGX speaches background context: #948 (originally framed as ctranslate2 CPU-only build issue; current container boots with CUDA but produces broken output — different bug class)
- openai-whisper service deploy: `infra/dgx/whisper-server/` (#953) — the deploy was correct; the temperature contract was the bug
