# DGX inference services — isolated load/perf baseline plan

Follow-up to **[INCIDENT-2026-08-04](../incidents/INCIDENT-2026-08-04-dgx-diarization-oom-lock.md)** (#1397).
The OOM happened because we ran every model service *together* under a batch burst
with no idea of each one's memory-vs-load curve. This plan characterises each DGX
service **in isolation** so we know its concurrency ceiling, memory-per-request, and
`/dev/shm` behaviour before co-residing them again.

**Blocked on:** DGX power-cycle (box currently OOM-locked). Prepare now, run when it's back.

## The box (constraint)

DGX Spark GB10, **130.7 GB unified** CPU+GPU memory. "GPU memory" *is* system RAM —
a model's reservation and a `/dev/shm` burst draw from the same pool. That's why
per-service memory ceilings matter more here than on a discrete-GPU box.

## Services under test (isolate one at a time)

| Service | Server (`infra/dgx/`) | Endpoint | Workload |
| --- | --- | --- | --- |
| Diarization | `pyannote-server` | `:8001 POST /v1/diarize` (multipart audio) | real episode wav (short + long) |
| Whisper Turbo | `whisper-server` / `speaches` | `:8000 POST /v1/audio/transcriptions` (multipart) | real episode wav (short + long) |
| vLLM (Qwen) | vLLM | `:8003 POST /v1/completions` (OpenAI-style) | representative prompt + `max_tokens` |
| (secondary) MOSS | `moss-server` | `:? POST` (http) | TBD |
| (secondary) ollama | ollama | `:11434` | TBD |

**Isolation setup:** load only the target service (stop the others so the unified pool
isn't pre-reserved) — otherwise the baseline is contaminated by the ~100 GB of other
reservations. Record what *else* is resident when you run.

## Method

For each service, **ramp concurrency** (1 → 2 → 4 → 8 …) with a fixed per-request
workload, sustaining each level for ~60 s, and stop when the box approaches the OOM
cliff. At every level capture:

**Client-side** (from the harness): p50/p95 latency, throughput (req/s; for
audio: audio-minutes processed/s; for vLLM: tokens/s), error rate, first-failure
concurrency.

**Server-side** (from homelab VictoriaMetrics, correlated to the run window):
`node_memory_MemAvailable_bytes` (minimum), `node_memory_Shmem_bytes` (peak + delta —
the `/dev/shm` signal that caused the incident), GPU util/mem where exposed.

The harness — `scripts/perf/dgx_service_loadtest.py` — does all of this and prints a
per-level table, plus a **ceiling guard** that halts the ramp when `MemAvailable`
dips below 10 GB (so a baseline run can't repeat the incident). Example:

```
python scripts/perf/dgx_service_loadtest.py \
    --service diarize --port 8001 --audio episode.wav \
    --concurrency 1,2,4,8 --duration 60 \
    --vm-ssh homelab-claude --out reports/dgx-diarize-baseline.json
```

`--dry-run` validates VM connectivity without sending load (works even while the DGX is
down — verified 2026-08-04).

## Baseline template (fill one per service)

| Concurrency | p50 | p95 | throughput | MemAvailable min | Shmem peak | Shmem Δ | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | | | | |
| 2 | | | | | | | |
| 4 | | | | | | | |

Then conclude, per service:
- **Idle/resident footprint** (GB reserved at load) and **memory-per-concurrent-request** (Δ per +1).
- **Concurrency ceiling** — where latency knees, errors start, or `MemAvailable` nears the cliff.
- **`/dev/shm` behaviour** — does `Shmem` grow with load? Did the incident-fix
  (`set_sharing_strategy('file_system')` in `pyannote-server`) flatten it?
- **Improvement opportunities** — GPU-util headroom (are we GPU- or memory-bound?),
  batching, `max_tokens`/audio-chunking, the right per-service concurrency cap + `MemoryMax`.

## Tracking

One GH issue per service (label `incident` / `perf`), each capturing its baseline table
+ opportunities. Umbrella: #1397.
