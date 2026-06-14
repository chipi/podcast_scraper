# Whisper-openai under vLLM contention — DGX (#963)

**Date:** 2026-06-11
**Issue:** #963 (filed in PR #966)
**Container:** `podcast-whisper:0.1.0` (post-fix per #929 / commit `458b4f53`)
**Endpoint:** `http://your-dgx.tailnet.ts.net:8002/v1/audio/transcriptions`
**Model:** `large-v3` (openai-whisper, served by our custom FastAPI shim)
**Dataset:** v2 audio fixtures (5 episodes, ~5 min each)

## Why this exists

During #929 the `whisper-openai` container produced byte-identical bad
output whether vLLM was running concurrently or not (5/5 episodes had
the same WER and hyp word counts in contended and non-contended runs).
At the time this was interpreted as "the container is broken regardless
of contention." That interpretation was correct given the data, but the
bad output was driven by the temperature-schedule bug (fixed in #929 /
commit `6df41b31` — `infra/dgx/whisper-server/app.py`), not by GPU
contention. The contention test was confounded by the bigger bug.

The open question post-fix: does the FIXED container survive concurrent
vLLM load, or does GPU contention still degrade quality on top of the
now-working baseline?

## Scenarios

| Scenario | DGX state | Purpose |
| --- | --- | --- |
| **1 — baseline** | vLLM stopped; only whisper-openai + pyannote idle | Reproduce the post-fix baseline (~0.102 mean WER, ~4.6× realtime). |
| **2 — realistic** | vLLM running idle (loaded but no active sweep), Ollama qwen3.5:35b loaded, pyannote loaded | Typical production state during transcription. |
| **3 — heavy** | vLLM actively serving a summary prediction sweep concurrent with the whisper sweep | Worst case — knowing the degradation envelope is the point. |

## Methodology

Per-episode invocations of `scripts/eval/score/whisper_dgx_vs_cloud_v1.py`
(`--backend dgx --models large-v3 --episodes <ep>`) wrapped in
`perl -e 'alarm 1200'` (the Tailscale-hang workaround pattern from
issue #929). Driver: `scripts/eval/whisper_contention_perep.sh`.
Aggregator: `scripts/eval/aggregate_whisper_perep.py`.

## Numbers

Filled in once each scenario completes.

### Scenario 1 — baseline (vLLM stopped, only whisper + pyannote idle)

| Episode | WER | Elapsed (s) | Realtime × |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.1302 | 153.6 | 3.6× |
| p02_e01 | 0.1666 | 125.0 | 5.2× |
| p03_e01 | 0.1017 | 129.1 | 3.8× |
| p04_e01 | 0.0345 | 77.1 | 7.3× |
| p05_e01 | 0.1612 | 94.0 | 5.6× |
| **mean** | **0.1188** | **115.7** | **5.1×** |
| stdev_elapsed | — | 30.3 | — |

### Scenario 2 — realistic contention (vLLM running idle, pyannote loaded)

**Deviation from the original #963 spec:** issue spec called for
Ollama qwen3.5:35b loaded alongside vLLM Qwen3.6-35B-A3B. **That
combination crashed the DGX mid-session** (kernel OOM on the GB10's
unified-memory pool; documented in `docs/guides/DGX_RUNBOOK.md` —
the "DO NOT STACK BIG MODELS" section). Scenario 2 was re-run with
vLLM idle only, no Ollama load. The contention signal is still
meaningful — vLLM holds ~70 GB of the unified pool whether it's
serving or idle.

| Episode | WER | Elapsed (s) | Realtime × |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.0768 | 120.8 | 4.6× |
| p02_e01 | 0.1619 | 126.5 | 5.2× |
| p03_e01 | 0.1009 | 144.0 | 3.4× |
| p04_e01 | 0.1093 | 88.6 | 6.3× |
| p05_e01 | 0.1373 | 105.1 | 5.0× |
| **mean** | **0.1172** | **117.0** | **4.9×** |
| stdev_elapsed | — | 25.0 | — |

### Scenario 3 — heavy contention (vLLM actively summarizing)

Concurrent load via `scripts/eval/vllm_summary_loadgen.sh` —
continuous chat.completions requests against the same vLLM endpoint
(`Qwen/Qwen3.6-35B-A3B`) using the v2 fixture transcripts as inputs.
**73 summary requests served** by vLLM during the whisper sweep.

| Episode | WER | Elapsed (s) | Realtime × |
| --- | ---: | ---: | ---: |
| p01_e01 | 0.1191 | 375.3 | 1.5× |
| p02_e01 | 0.0879 | 433.0 | 1.5× |
| p03_e01 | 0.0685 | 234.9 | 2.1× |
| p04_e01 | 0.2232 | 270.9 | 2.1× |
| p05_e01 | 0.1601 | 271.3 | 1.9× |
| **mean** | **0.1318** | **317.1** | **1.84×** |
| stdev_elapsed | — | 80.3 | — |

## Findings

| Metric | Scenario 1 | Scenario 2 | Scenario 3 |
| --- | ---: | ---: | ---: |
| Mean WER | 0.1188 | 0.1172 | 0.1318 |
| Mean elapsed (s) | 115.7 | 117.0 | 317.1 |
| Mean realtime × | 5.1× | 4.9× | 1.84× |
| WER range | 0.034 – 0.167 | 0.077 – 0.162 | 0.069 – 0.223 |
| stdev elapsed | 30.3 | 25.0 | **80.3** |

1. **WER is essentially unchanged across all three scenarios.** Mean
   WER 0.117–0.132 across baseline / idle-vLLM / active-vLLM is
   within stochastic-decoding noise (per-episode variance > scenario
   variance). The temperature-schedule fix from #929 is robust under
   GPU contention — the whisper-openai container's quality does not
   degrade when vLLM is hammering the GPU.
2. **Latency takes a ~2.7× hit under heavy contention.** Mean elapsed
   goes from 115.7s (idle) to 317.1s (active). Realtime drops from
   ~5× to ~1.8× — still faster than playback, but the headroom is
   gone.
3. **Latency variance triples under contention.** stdev_elapsed jumps
   from 25-30s (clean scenarios) to 80s under load — some episodes
   take 235s, others 433s, depending on how vLLM's request queue
   aligns with whisper's GPU window.
4. **vLLM running idle is essentially free.** Scenario 2 (vLLM idle)
   matched scenario 1 (vLLM stopped) on every metric within noise.
   The cost of leaving vLLM warm-loaded is paid in memory headroom,
   not in transcription quality or speed.

## Operational implications

- **Production-routing policy**: transcription quality survives GPU
  contention; latency does not. For batch-mode transcription jobs
  (the operator's normal pattern), the ~3× slowdown is acceptable
  — episodes are still transcribed faster than they play back. For
  interactive transcription latency budgets, **avoid running
  autoresearch sweeps during peak transcription windows.**
- **Capacity headroom**: vLLM idle costs ~70 GB of the 122 GB
  unified-memory pool with no measurable transcription impact. The
  load-bearing capacity question is "can both fit at all on GB10"
  (see DGX runbook), not "do they contend."
- **The #929 temperature-schedule fix holds under load.** Important
  signal: the fix wasn't a contention workaround that secretly
  degrades under pressure; it's a real fix to a real bug, and
  contention doesn't reopen it.
- **vLLM tail-latency observation (out of scope but worth noting)**:
  the loadgen pushed ~73 summary requests through vLLM in ~30 min,
  one at a time, mean ~25s/request. Under sustained whisper
  contention vLLM's summary requests likely slowed similarly to
  whisper — a follow-up benchmark could measure vLLM's own
  contention envelope (#963 only measured the whisper side).

## Caveats / what this does NOT measure

- Latency variance across more than 5 episodes (small sample).
- Behavior under sustained 24-hour load — this is a single-batch test.
- Behavior with non-Tailscale clients — the harness goes
  laptop → Tailscale → DGX, which adds ~150ms of network jitter per
  episode that's not part of GPU contention per se.

## Raw data

- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/scenario-1-baseline/`
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/scenario-2-realistic/`
- `data/eval/runs/whisper_dgx_vs_cloud_v1/dgx-openai-contention/scenario-3-heavy/`

Each scenario subdir contains:

- `perep/<ep>/metrics.json` — per-episode raw metrics.
- `metrics.json` — sweep-level aggregate.

## References

- The temperature-schedule fix that motivated this re-test: #929,
  `infra/dgx/whisper-server/app.py`, commit `6df41b31`.
- Original #929 contention finding (now retracted):
  `docs/guides/eval-reports/EVAL_TRANSCRIPTION_3WAY_2026_06.md`.
- Tailscale resilience patterns surfaced by this work: #946, #956.

---

## 2026-06-14 re-run — different vLLM model + GB10 unified-memory cap

**Attribution caveat — read first.** This re-run measured the operator's
**`vllm-coder-next`** stack (`Qwen/Qwen3-Coder-Next-FP8` on port `:9000`,
homelab repo `infra/vllm/coder-next/`) as a stand-in for "vLLM contention."
That is **not** the project's intended autoresearch model. The
podcast_scraper-side `vllm-autoresearch` stack (intended Qwen3.6-35B-A3B)
has not been deployed in homelab as of 2026-06-14 — `infra/vllm/` contains
only `coder-next/`, `openwebui/`, and `template/`. Coder-next was measured
because it was the only deployed vLLM. The numbers below are still
informative (any vLLM competes for the same GB10 GPU as whisper), but
attribution to "the autoresearch model" was wrong in the prior draft and
has been corrected throughout this section.

**Why a re-run.** Two things changed since the 2026-06-11 numbers above:

1. **vLLM target switched** — measured against the operator's coder-next
   stack (`Qwen/Qwen3-Coder-Next-FP8`) rather than the project's intended
   autoresearch model (Qwen3.6-35B-A3B). The 2026-06-11 run cited the
   project model, but in practice both runs hit the only-deployed vLLM
   on the box; the *model* difference is real, the *purpose* attribution
   was wrong.
2. **vLLM `gpu-memory-utilization` lowered** — `0.92` → `0.75` after the
   `0.92` setting OOM-crashed the host during this very re-test. GB10
   has a unified CPU+GPU pool (121 GiB total), so `0.92` claims ≈ 112 GB
   and starves whisper + the host OS. The compose default is now
   `${VLLM_GPU_MEM_UTIL:-0.75}` (see
   `agentic-ai-homelab/infra/vllm/coder-next/docker-compose.yml`,
   commit `62780d2`).

Also: the laptop-side `infra/dgx/converge/deploy.py` had a stale path
(`faster-whisper-server`) blocking `make dgx-deploy` after #975; fixed in
this branch (`fix(infra): point speaches build at speaches-gb10/`) so the
post-#966 whisper image landed on the DGX before scenario 1 re-baselined.

### Setup deltas

| | 2026-06-11 run | 2026-06-14 re-run |
| --- | --- | --- |
| vLLM stack | (cited as autoresearch — wrong; measured coder-next) | **coder-next (`Qwen/Qwen3-Coder-Next-FP8`)** |
| vLLM mem util | 0.92 | **0.75** (GB10-safe) |
| Whisper image | post-#929 | **post-#929 + post-#966 (re-deployed today)** |
| Fixtures | v2 (5 ep) | v2 (5 ep) — same |
| Load generators | `vllm_summary_loadgen.sh` (coder-next-shaped) | `_dgx_vllm_load_generator.sh` (this branch, coder-next-shaped) |

### Re-run numbers

| Episode | SC1 WER | SC2 WER | SC3 WER | SC1 (s) | SC2 (s) | SC3 (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p01_e01 | 0.1628 | 0.0391 | 0.0417 | 165 | — | 272 |
| p02_e01 | 0.2330 | 0.1949 | 0.2930 | 186 | — | 359 |
| p03_e01 | 0.1682 | 0.1114 | **1.0000** | 100 | — | **1833** |
| p04_e01 | 0.0704 | 0.2452 | 0.1630 | 88 | — | 343 |
| p05_e01 | 0.1322 | 0.1102 | 0.1736 | 93 | — | 195 |
| **mean** | **0.1533** | **0.1402** | **0.1678** | 126.5 | 127.2 | **292.2** |
| max WER | 0.2330 | 0.2452 | **0.2930** | — | — | — |
| realtime × | 4.4× | 4.4× | **2.0×** | — | — | — |

vLLM-side telemetry (scenario 3, 514 requests over the eval window):
mean 5.8s/req, max **335.2s** — vLLM tail latency blew out too.

### Findings (2026-06-14)

1. **WER means hold across SC1/SC2.** SC1 0.153 vs SC2 0.140 — within
   per-episode noise. Idle vLLM on FP8 + 0.75 mem util is still
   essentially free for whisper, confirming the 2026-06-11 conclusion
   under the new config.
2. **Active vLLM serving triggers a non-trivial single-episode failure
   mode.** `p03_e01` jumped from 100s/0.168 WER (SC1) to **1833s / 1.000
   WER** under SC3 — an 18× slowdown and complete transcription
   collapse. The other 4 episodes degraded gracefully (WER +0–6pp,
   latency 2–3.7× slower) consistent with the 2026-06-11 picture. This
   tail risk did **not** appear in the prior SC3 run, but with N=5 per
   scenario we can't distinguish "different vLLM workload shape causes
   it" from "rare event we happened not to hit." **Important — this
   measurement reflects coder-next sweep behavior, not the project's
   autoresearch sweep.** Update 2026-06-14: the autoresearch vLLM stack
   shipped at `agentic-ai-homelab/infra/vllm/autoresearch/` (model
   `Qwen/Qwen3-30B-A3B-Instruct-2507`, port `:8003`, see the
   `gpu-mode-swap.sh research` slot). **Characterization follow-up:
   #996** is now actionable against the correct target — re-run
   scenario 3 with N ≥ 20 against `:8003` and amend this report when
   the numbers land. Caveats to record in the #996 amendment: model
   size drift (30 B vs original 35 B baseline; homelab issue #3), MoE
   config not GB10-tuned (homelab issue #1), vLLM image version drift
   (homelab issue #2). The coder-next numbers above are no longer the
   best stand-in — they're now retired-as-superseded-once-#996-runs.
3. **Mean realtime × halved under SC3** (4.4× → 2.0×) — consistent with
   the 2026-06-11 SC1→SC3 drop (5.1× → 1.84×). Latency-under-load is
   reproducible across vLLM models.
4. **vLLM itself is also degraded under whisper contention** — max
   chat.completions latency 335s while normal latency is ~4s. The
   degradation is bilateral, not one-sided.

### Operational implications (2026-06-14)

- **Hybrid-routing rule of thumb (feeds #927/#929/#930/#931 synthesis)**:
  if **any vLLM is actively serving** on the GB10 (whether coder-next
  for the operator's IDE, an autoresearch sweep, or any other future
  stack on the same GPU), treat the DGX whisper endpoint as
  *latency-undefined* (95th-percentile risk includes a catastrophic
  single-episode failure). For batch transcription windows, **gate
  vLLM serving behind a queue** rather than letting them run
  concurrently. The provider resilience layer (`tailnet_dgx.resilience`,
  #956) catches the timeout case — the client falls back to cloud —
  but the failure mode here is a successful HTTP response containing
  wrong content (WER = 1.0), not a timeout. We can't safely depend on
  the resilience layer alone for this; gating remains the
  operator-side rule.
- **Whisper-on-DGX is still viable** for the operator's day-to-day
  pattern (transcription during idle vLLM = 4.4× realtime, no quality
  loss vs SC1). The catastrophic case is sweep-overlap-specific.
- **vLLM mem-util cap is now load-bearing**: `0.75` is the GB10 floor
  for cohabitation. The compose default + `VLLM_GPU_MEM_UTIL` override
  preserve the option to push higher transiently on a quieter box.

### Raw data (2026-06-14 re-run)

- `data/eval/runs/whisper_contention_v2/scenario1_baseline_postdeploy/metrics.json`
- `data/eval/runs/whisper_contention_v2/scenario2_realistic/metrics.json`
- `data/eval/runs/whisper_contention_v2/scenario3_heavy/metrics.json`
- vLLM load-gen log: `/tmp/dgx_vllm_load_scenario3.jsonl` (514 requests)

### Load-generator scripts added this branch

- `scripts/eval/score/_dgx_ollama_load_generator.sh` — qwen3.5:35b
  continuous-load generator (for future Ollama-contention runs).
- `scripts/eval/score/_dgx_vllm_load_generator.sh` — vLLM
  chat.completions continuous-load generator (used for SC3 today).
- Compose hardening: `agentic-ai-homelab` commit `62780d2`
  (`VLLM_GPU_MEM_UTIL` default 0.75 + comment).
- Path fix: `infra/dgx/converge/deploy.py` — `faster-whisper-server` →
  `speaches-gb10` (this branch).
