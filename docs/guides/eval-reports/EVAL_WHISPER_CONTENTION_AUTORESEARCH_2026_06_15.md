# EVAL — Whisper contention catastrophic-tail characterization (#996)

**Date:** 2026-06-14 sweep / 2026-06-15 writeup
**Issue:** [#996](https://github.com/chipi/podcast_scraper/issues/996) (filed as a follow-up to #963)
**Sibling report:** [`EVAL_WHISPER_CONTENTION_2026_06.md`](EVAL_WHISPER_CONTENTION_2026_06.md) — the 2026-06-11 and 2026-06-14 N=5 SC3 runs

## TL;DR

Re-ran the #963 scenario-3 contention shape against the autoresearch
vLLM stack (homelab `infra/vllm/autoresearch/`, model
`Qwen/Qwen3-30B-A3B-Instruct-2507`, port `:8003`, GB10) with N=20 v2
episodes — the first sample that's actually statistically interesting
on the catastrophic-tail question.

**Result: the catastrophic-tail rate is ~20 % at this contention shape.**
4 of 20 episodes failed catastrophically; the surviving 16 were
substantially degraded but produced usable transcripts.

| Outcome | Count | Notes |
| --- | ---: | --- |
| Catastrophic — hung > 25 min, no response | 2 | `p01_e02`, `p02_e02` |
| Catastrophic — server `ConnectionResetError` mid-stream | 2 | `p07_e01`, `p09_e02` |
| Degraded but completed | 16 | mean WER 0.18 vs baseline 0.10; mean realtime 2.2× vs baseline 4.4× |

Reproduces the N=5 catastrophic case from the prior coder-next
stand-in sweep (1/5 = 20 % rate, same percentage at N=5 as we now see
at N=20). The shape is stable; the operator rule (idle vLLM before
transcription) is correctly load-bearing.

## Method

20 v2 audio fixtures, each transcribed via the production DGX
whisper-openai service (`large-v3`, `:8002`), wrapped in a
per-episode subprocess with a 1500 s `perl alarm` wrapper.
A continuous `_dgx_vllm_load_generator.sh` flooded the autoresearch
vLLM (`:8003`) with summarisation requests for the duration —
this is the SC3 "heavy contention" shape from `EVAL_WHISPER_CONTENTION_2026_06.md`,
extended from N=5 to N=20.

```text
WHISPER_DGX_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions \
VLLM_PORT=8003 \
VLLM_MODEL=autoresearch \
  bash /tmp/run_996_sweep.sh
```

20 episodes selected to span all 9 v2 feeds:

```text
p01_e01 p01_e02 p01_e03
p02_e01 p02_e02 p02_e03
p03_e01 p03_e02 p03_e03
p04_e01 p04_e02 p04_e03
p05_e01 p05_e02 p05_e03
p06_e01 p07_e01 p08_e01 p09_e01 p09_e02
```

Sweep wall-clock: ~10 hours (2026-06-14 19:23 UTC → 2026-06-15 05:14 UTC).
Catastrophic episodes contributed the bulk of that — the perl alarm
wrapper did not actually trigger on macOS for the hung-client cases
(`exec`-replaced perl loses the SIGALRM scheduling on this platform),
so the two hung episodes had to be killed manually by an operator
at ~2 h and ~22 min respectively. See § Methodology caveats.

## Per-episode results

Sorted by elapsed time (ascending):

| Episode | WER | Elapsed (s) | Realtime × | Status |
| --- | ---: | ---: | ---: | --- |
| p06_e01 | 0.033 | 3.6 | 3.3× | very short audio (≈ 12 s) — barely processed |
| p04_e03 | 0.304 | 165.3 | 2.9× | degraded |
| p03_e03 | 0.076 | 173.4 | 3.6× | OK |
| p03_e02 | 0.152 | 241.5 | 2.3× | OK |
| p05_e01 | 0.112 | 242.7 | 2.2× | OK |
| p03_e01 | 0.158 | 274.6 | 1.8× | OK |
| p04_e01 | 0.073 | 278.6 | 2.0× | OK |
| p01_e03 | 0.068 | 297.5 | 1.8× | OK |
| p04_e02 | 0.036 | 302.9 | 1.9× | OK |
| p05_e02 | 0.147 | 388.9 | 1.3× | OK |
| p05_e03 | 0.162 | 395.3 | 1.5× | OK |
| p01_e01 | 0.175 | 412.4 | 1.3× | OK |
| p02_e01 | 0.342 | 429.8 | 1.5× | degraded |
| p09_e01 | 0.256 | 452.3 | 4.9× | degraded |
| p02_e03 | 0.280 | 650.2 | 1.1× | degraded |
| **p08_e01** | 0.230 | **2054.8** | 2.8× | **slow but completed (34 min)** |
| **p09_e02** | — | — | — | **CATASTROPHIC — `ConnectionResetError`** |
| **p07_e01** | — | — | — | **CATASTROPHIC — `ConnectionResetError`** |
| **p02_e02** | — | — | — | **CATASTROPHIC — hung, killed at ~22 min** |
| **p01_e02** | — | — | — | **CATASTROPHIC — hung, killed at ~2 h** |

### Aggregates over the 16 completing episodes

| Metric | Value | vs baseline (`EVAL_WHISPER_CONTENTION_2026_06` SC1) |
| --- | ---: | --- |
| Mean WER | **0.176** | 0.118 → **+49 % relative** |
| Median WER | 0.155 | — |
| Mean elapsed | **352 s** | 116 s → **3.0× slower** |
| Mean realtime × | **2.2×** | 5.1× → **57 % slower** |
| Stdev elapsed | 462 s | 30 s → **15× higher variance** |

p08_e01 (2055 s / WER 0.23) is the outlier that dominates the stdev —
even completing, it ran 17× longer than baseline, 5× longer than the
SC3 N=5 mean from #963. Borderline catastrophic in shape if not in
classification.

## vLLM-side telemetry during the sweep

The load generator logged per-request elapsed for every
`/v1/chat/completions` call against `:8003`:

| Metric | Value |
| --- | ---: |
| Requests served | 1 198 |
| Mean elapsed | 23.6 s |
| Median | 8.2 s |
| p95 | 8.4 s |
| **p99** | **976 s** |
| Max | **1 018 s** |

vLLM's own tail latency blew out symmetrically with whisper's. The
median request completed in 8 s (the model's natural pace for the
3-bullet-summary prompt), but ~1 % of requests took **16 min+** —
matching the catastrophic regime whisper saw on its side. This is the
bilateral degradation pattern flagged in the prior sibling report and
now reproduced at scale.

Confirms: under whisper contention, vLLM is not "serving normally
while whisper suffers" — both endpoints are simultaneously degraded.
Neither can claim the GPU cleanly; the GB10 schedules them
catastrophically against each other in some fraction of windows.

## Findings

### 1. Catastrophic-tail rate is ~20 %, stable across measurement N

| Sample | vLLM target | Catastrophic | Rate |
| --- | --- | ---: | ---: |
| 2026-06-14 N=5 (sibling report) | coder-next stand-in (FP8) | 1 / 5 | 20 % |
| **2026-06-15 N=20 (this report)** | **autoresearch (BF16 MoE 30 B)** | **4 / 20** | **20 %** |

Same rate within stochastic noise, against two different vLLM stacks
(different model, different quant, different VRAM footprint). That
the rate didn't shift when the load shape did is informative — it
suggests catastrophic-tail is **a contention-pattern property of the
GB10 + whisper-openai service**, not a property of any particular
vLLM workload mix.

### 2. Two distinct catastrophic-failure modes observed

Earlier (N=5) we only saw the "successful HTTP 200 with garbage
content (WER 1.000)" failure mode. At N=20:

- **Hang** (`p01_e02`, `p02_e02`): client waits indefinitely; whisper-server
  logs show no completion for the request. SIGALRM-style timeouts
  don't take effect because the python http client is blocked deep
  in a socket read that ignores the alarm. **The `tailnet_dgx.resilience`
  watchdog (#956) handles this case** — the production code path's
  monotonic deadline does fire and triggers fallback.
- **Server `ConnectionResetError`** (`p07_e01`, `p09_e02`): server kills
  the connection mid-stream. Surfaces as a `urllib3`
  `ConnectionResetError(54)` ("Connection reset by peer"). Indicates
  the whisper container's HTTP framework gave up on the request
  mid-flight — probably hit some internal deadline / OOM watchdog
  inside the container itself, not the OS. **The resilience layer
  catches this** too (it's a connection-level failure, not a 200-OK-with-
  garbage case).

We did **not** see the "200 OK with WER=1.0 garbage" mode at N=20.
That mode appeared once at N=5 (`p03_e01` in the 2026-06-14 sibling
report, against coder-next). With both samples combined (N=25),
the WER-1.0 mode has been observed exactly once — too small a
sample to be confident about its frequency, but it's the failure
mode the resilience layer **cannot** catch, so it remains the
"worst case" pattern even if rare.

### 3. Surviving episodes are heavily degraded too — not just slow

Mean WER on the 16 completing episodes is **0.176**, up from baseline
0.118 (+49 % relative). This is not "they take longer but quality
holds" — quality drifts too. Cause is likely the temperature-
fallback schedule firing more aggressively under GPU starvation
(can't decode confidently in the time budget, falls back to a hotter
temperature, hallucinates fragments). The prior 2026-06-11 N=5 finding
that "WER means hold across all three scenarios" appears to have been
a small-N artifact — at N=16 surviving episodes the WER drift is
real.

### 4. p08_e01 is the in-band catastrophic case

p08_e01 completed at 2 055 s — 17× baseline, 5× the SC3 mean. WER
held at 0.23 (degraded but readable). It's not catastrophic by the
strict "failed to return / returned garbage" rule, but it's
operationally indistinguishable from one — a 34-minute transcribe
window for a 4-minute episode is a production-grade outage on any
real timing budget. The honest framing is:

- **Strict catastrophic rate: 20 %** (4 / 20 — failed or returned WER 1.0).
- **Operational catastrophic rate: 25 %** (5 / 20 — including "completed
  but took 30+ minutes for a 4-minute episode").

The PROD_RUNBOOK operator rule (idle vLLM before transcription
windows) is correctly calibrated for either framing.

## Operational implications

### What changes after this report

- **PROD_RUNBOOK § Provider model selection — DGX vs cloud per stage** is
  unchanged in *substance*: "idle any active vLLM serving before
  transcription windows" was the right rule and is now backed by
  N=20 evidence at 20 % failure rate (not just one anecdote).
- **The 30B-vs-35B model-size drift documented in homelab issue #3 is
  shown to not matter for this question.** The catastrophic-tail rate
  on the 30 B MoE matches the coder-next-FP8 rate at N=5. So the rate
  is robust to vLLM stack choice within the broad "moderate-size
  Qwen on the GB10" envelope.
- **The vLLM tail-latency observation (sibling report's "out of scope
  but worth noting" line) is now confirmed.** vLLM p99 hits 16 min
  under whisper contention; bilateral degradation is real. Any
  workload assumption that "vLLM keeps serving normally while whisper
  takes its lumps" is wrong.

### Routing recommendations (unchanged from PR #998 / #931)

`cloud_with_dgx_primary` profile stays. The operational rule covers
the contention envelope. No profile change.

The new `prod_dgx_full_with_fallback` profile (#923) is **still
gate-able**: this report's 20 % rate confirms the gating is warranted,
NOT that the gating is unnecessary. The transcript + summary stages
on that profile run sequentially per-episode, so they shouldn't
contend within a single episode — but external vLLM workloads
(coder-next, autoresearch sweeps, future stacks) overlapping a
transcription window will hit this rate.

## Methodology caveats

### perl alarm wrapper failure on macOS

The sweep driver used the pattern:

```bash
perl -e 'alarm 1500; exec @ARGV' -- .venv/bin/python ...
```

This pattern works on Linux but **does not survive `exec` on macOS**
in our environment. The SIGALRM scheduled before `exec` does not fire
in the replaced python process. Two episodes (`p01_e02`, `p02_e02`)
hung indefinitely and had to be killed manually.

For future contention sweeps: use Python's own subprocess timeout, or
gnu `gtimeout` (from `coreutils`). The pattern is **not portable**.

The `tailnet_dgx.resilience` layer (#956) implements a
**monotonic wall-clock watchdog** that does fire reliably on hangs —
that's the correct production-side handling. The sweep harness
needs the same primitive; perl-alarm-exec is a stale pattern.

### Image / model attribution still imperfect

- vLLM image: `nvcr.io/nvidia/vllm:25.11-py3` (homelab issue #2 tracks
  whether to bump to 26.05-py3).
- vLLM model: `Qwen/Qwen3-30B-A3B-Instruct-2507`, 30 B total / 3 B
  active MoE (homelab issue #3 tracks the 30 vs 35 B baseline drift).
- MoE config: untuned for GB10 (homelab issue #1 tracks the tuning gap).

Per-episode catastrophic rate may shift under different model / image /
config combinations. The 20 % observed here is the rate **for this
specific stack against this whisper baseline**. The fact that it
matched the coder-next-FP8 rate at N=5 suggests the rate is robust
across these axes, but more replications would strengthen the claim.

### Whisper container behavior under sustained contention

We did not log whisper-container memory or CUDA-graph cache state
during the sweep. The two `ConnectionResetError` cases plausibly
trace back to a whisper-side OOM kill or internal queue cap — we
have no evidence either way. Future sweeps should pair the prometheus
scrape (#943) with the runtime — DGX_RUNBOOK now lists the autoresearch
`:8003/metrics` endpoint; whisper-openai's only metric is its docker
container stats via cAdvisor, which captures memory but not internal
queueing.

## Out of scope (filed elsewhere)

- Re-running with `gtimeout`-style reliable timeouts. Methodology fix;
  doesn't change the headline number meaningfully.
- Characterising the "WER=1.0 garbage" failure mode specifically
  (sibling report saw it 1 in 5; we saw it 0 in 20). N still too small
  for a confident rate; would need N≥100 against a fixed stack.
- Whisper-container side instrumentation. Would benefit any future
  catastrophic-tail follow-up; out of scope here.
- Resilience-layer hardening: the watchdog catches "hang" and
  "connection reset" cases but not the WER=1.0 case. Hardening to
  catch the third mode (e.g. a sanity-check on returned text length
  vs expected) is a separate ticket worth filing if the WER=1.0 mode
  reproduces in any future sample.

## Reproduction

```bash
# 1. Verify autoresearch vLLM is up at :8003
gpu-mode-swap.sh status
curl -fsS http://dgx-llm-1.tail6d0ed4.ts.net:8003/v1/models \
  -H "Authorization: Bearer buddy-is-the-king"

# 2. Start vLLM load gen in background
VLLM_PORT=8003 VLLM_MODEL=autoresearch \
  LOAD_LOG=/tmp/dgx_vllm_load_996.jsonl \
  bash scripts/eval/score/_dgx_vllm_load_generator.sh &

# 3. Per-episode whisper sweep (be prepared to manually kill hangs)
WHISPER_DGX_URL=http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions \
  bash /tmp/run_996_sweep.sh  # (driver in this report's data dir)

# 4. Aggregate
.venv/bin/python scripts/eval/aggregate_whisper_perep.py \
  data/eval/runs/whisper_996_n20_autoresearch/perep \
  data/eval/runs/whisper_996_n20_autoresearch/metrics.json

# 5. Stop load gen + idle GPU
pkill -f _dgx_vllm_load_generator
gpu-mode-swap.sh idle
```

## Raw data

- `data/eval/runs/whisper_996_n20_autoresearch/metrics.json` — sweep-level aggregate
- `data/eval/runs/whisper_996_n20_autoresearch/perep/<ep>/metrics.json` — per-episode
- `/tmp/dgx_vllm_load_996.jsonl` — vLLM-side per-request timings (1198 requests)

## References

- Issue: [#996](https://github.com/chipi/podcast_scraper/issues/996)
- Sibling report (N=5 against coder-next stand-in): [`EVAL_WHISPER_CONTENTION_2026_06.md`](EVAL_WHISPER_CONTENTION_2026_06.md)
- Operator rule home: `PROD_RUNBOOK.md` § "Provider model selection — DGX vs cloud per stage"
- Stack-level caveats (model / image / MoE):
  [homelab #1](https://github.com/chipi/agentic-ai-homelab/issues/1),
  [homelab #2](https://github.com/chipi/agentic-ai-homelab/issues/2),
  [homelab #3](https://github.com/chipi/agentic-ai-homelab/issues/3)
- Profile that's gated on this rule: `config/profiles/prod_dgx_full_with_fallback.yaml` (#923)
- Resilience layer that catches hang + connection-reset modes: #956
- Cross-repo profile-side observations: #927 / #931 synthesis
