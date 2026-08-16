# DGX faster-whisper stall investigation (2026-07-28 reprocess run)

- **Status**: Investigation paused — handed to a separate **service-stability** work stream.
- **Date**: 2026-07-29
- **Context**: During the v2.3.5 reprocess run (profile `reprocess_v23_turbo`, DGX turbo ASR),
  the pipeline stalled repeatedly on the DGX faster-whisper server. This doc records what the
  observability plane could and could not tell us, so the stability stream starts from evidence,
  not from scratch.

## TL;DR

The DGX whisper server **dropped its in-flight transcription mid-job** (GPU compute fell to 0)
while **keeping its TCP port open**, twice-confirmed, for ~22 min each. The client behaved
correctly: its TCP health check stayed green, the transcription *request* hung, the
duration-scaled request timeout fired, and ADR-122 hold-mode paused-and-probed (≤900 s) rather
than failing over. **Root cause of the server-side job drop is unknown** — it lives in the whisper
server's own stdout, which is **not shipped anywhere** (see "Logs gap"). This is a DGX service
stability problem, not a client/resilience bug.

## What the 900 s actually is (mechanism, not a guess)

Verified against `src/podcast_scraper/config.py:1442-1482` + profile
`config/profiles/reprocess_v23_turbo.yaml` (sets `resilience_failure_strategy: hold`,
`resilience_run_context: reprocess`; does **not** override the numeric knobs → defaults apply):

| Stage | Knob | Value | Behaviour |
|---|---|---|---|
| 1. Retry backoffs (these accumulate) | `resilience_backoff_schedule_sec` × `resilience_retries_before_trip` | `[30,60,120]` × 3 | retry same endpoint at +30/+60/+120 s ≈ **210 s**, then the fuse trips |
| 2. Hold-and-probe (single window) | `resilience_on_open_max_wait_sec` | **900** | after trip: pause + re-probe every `resilience_probe_interval_sec`=30 s, up to **900 s continuous**, then give up + alert. **Never fails over** (hold strategy, ADR-122) |

So the 900 s is **one continuous ceiling**, *not* five backoffs summed. Each stall ≈
~210 s accumulating backoff → fuse trips → up to ~900 s single hold-probe window → recover.

## Evidence (all from the homelab observability plane)

Observability endpoints: VictoriaMetrics `http://<HOMELAB_IP>:8428`,
VictoriaLogs `http://<HOMELAB_IP>:9428`. DGX telemetry reaches VM via the mini's `dgx-scrape`
launchd pull (`~/agentic-ai-homelab/infra/dgx-scrape/`) — GPU (DCGM) + cadvisor + TCP health only.

### 1. GPU metrics — server stopped computing, port stayed up

Two clean stalls in the 20:45–22:35 UTC slice (2026-07-28):

- `DCGM_FI_DEV_GPU_UTIL`: pegged **81–96 %** up to the onset, then **→ 0** and flat for ~22 min,
  then resumes. Stall 1: `21:02:30=87 → 21:03:00=0 … 21:25:30=74`. Stall 2:
  `21:59=96 → 22:00:00=0 … 22:22:30=72`.
- `DCGM_FI_DEV_POWER_USAGE`: 50 W → **~10 W** (idle) during the stalls.
- `dgx_service_up{service="whisper"}` (a **TCP-open** check): **=1 the entire time** — the
  listening socket never dropped.
- `DCGM_FI_DEV_XID_ERRORS`: **0** — no GPU hardware/driver fault, no OOM XID.

Interpretation: the server went **idle** (not pegged) with its **port open** → it **stopped
processing the in-flight job**, it did not crash and did not stay busy.

### 2. Client-side — genuinely blocked on the DGX (not busy elsewhere)

VictoriaLogs `instance=podcast_scraper-ai-ml-improvements`, `pipeline_progress`/`pipeline_stage`
events: two progress **gaps of 34.1 min (20:54:44→21:28:48)** and **32.7 min
(21:51:43→22:24:26)**, each bracketing a GPU-idle window. So the pipeline emitted no progress
across the stalls — it was waiting on whisper, confirming client-stall ↔ GPU-idle alignment.

### 3. Health check is TCP liveness (already hardened against "busy = down", #956)

`src/podcast_scraper/providers/tailnet_dgx/health.py`: `check_faster_whisper_health` →
`_check_dgx_http_health` uses `tcp_endpoint_listening` (a raw `socket.create_connection`) for the
proceed/fallback decision. A GPU-busy server (TCP accepts, HTTP `/v1/models` queued behind the
job) classifies as **BUSY = UP** and still gets the request; only a **refused/unreachable TCP
connect** is `DOWN`. So a pegged GPU **cannot** produce a false-negative here — the "flaky health
under load" hypothesis is defended against by design. The mini's `dgx-scrape` uses the same
TCP-open rationale (documented: inference servers "stop answering HTTP `/health` while still
serving").

## Ruled OUT (with evidence)

- **Health false-negative under load** — health is TCP, stayed green (`dgx_service_up=1`); client
  never saw DOWN.
- **GPU hardware/driver fault / OOM** — `XID_ERRORS=0` throughout.
- **Server crash** — TCP port stayed open the whole stall (`dgx_service_up=1`).
- **"Pegged and can't answer"** — GPU went to **0**, not stayed at 96 %.
- **ADR-123 coverage/quality failover ("turbo long-episode cliff")** — that's a quality path and
  hold-mode does not fail over anyway; this was an infra stall, not a coverage decision.
- **900 s = accumulated backoffs** — it's a single hold-probe ceiling (see mechanism table).

## NOT determined / open (needs the stability stream)

- **Why the server dropped the in-flight job** — the decisive signal (whisper server stdout:
  decode watchdog abort? worker restart? VRAM pressure? input-specific hang?) is **not
  observable** from the current plane.
- **Hang vs crash-and-reload** — could not distinguish: DCGM `FB_USED`/`FB_FREE` (VRAM residency)
  are **not exported** on this box (only `DCGM_FI_DEV_MEMORY_TEMP`, `DCGM_FI_DEV_MEM_COPY_UTIL`).
  `MEM_COPY_UTIL` was flat 0 in the sampled window (no obvious reload burst), but that's weak.
- **Full stall count** — 2 stalls metric-confirmed in the 20:45–22:35 UTC slice; the run log
  referenced ~5–6. The earlier ones were not re-verified in metrics.
- **Which episodes were in flight** — not pinned; the run's transcription stdout was not located
  (per-feed `run.jsonl` carries only enrichment events; the transcription hold-probe logging goes
  to pipeline stdout, which was not captured to a findable file this session).

## Logs gap (the reason we can't root-cause from here)

The DGX ships **metrics only**. `~/agentic-ai-homelab/infra/dgx-scrape/README.md`: the mini pulls
DCGM + cadvisor + TCP health-checks; it ships **no logs** (grep for `logsql|9428|VictoriaLogs` in
`dgx-scrape/` = empty). The designed log collector (node-exporter + Alloy **on the DGX**) is
"blocked on the DGX's login being wedged (no shell)." A full 7-day VictoriaLogs stream inventory
shows **zero DGX streams**. So the whisper server's own stdout exists only on the DGX box, which
has no working SSH. **Closing the root cause requires getting logs off the DGX** (fix the wedged
login / stand up the on-box collector) — tracked in the stability stream + the homelab repo.

## Suggested next steps for the stability stream

1. Restore log egress from the DGX (unblock shell / stand up the on-box Alloy collector) so the
   whisper server stdout reaches VictoriaLogs.
2. With logs, correlate the 21:03 / 22:00 UTC drops to server-side events (decode timeout, worker
   OOM/restart, specific audio input).
3. Consider a server-side request/decode watchdog + a client `max_wait` alert so a 900 s hold is
   surfaced live, not found the next morning.
