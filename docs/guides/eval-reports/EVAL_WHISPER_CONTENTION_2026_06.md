# Whisper-openai under vLLM contention — DGX (#963)

**Date:** 2026-06-11
**Issue:** #963 (filed in PR #966)
**Container:** `podcast-whisper:0.1.0` (post-fix per #929 / commit `458b4f53`)
**Endpoint:** `http://dgx-llm-1.tail6d0ed4.ts.net:8002/v1/audio/transcriptions`
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
