# INCIDENT-2026-08-04 — DGX OOM-locked by concurrent diarization (`/dev/shm` exhaustion)

| Field | Value |
| --- | --- |
| Date | 2026-08-04 (onset 2026-08-03 23:35 UTC) |
| Duration | 2026-08-03 23:35 UTC (OOM onset) → **ongoing** (box hung, awaiting physical power-cycle) |
| Severity | SEV-2 — DGX inference host down (vLLM/Qwen, Whisper, diarization unavailable); no customer-facing prod impact |
| Affected services | DGX `dgx-llm-1` / `spark-2c14`: `pyannote-app` (diarization), Whisper, vLLM, ollama, and tailnet access to the box |
| Author(s) | operator: Marko Dragoljevic. agent: Claude Code (Claude Opus 4.8). |
| Status | **open** — root cause identified from observability; box awaiting hard reboot; fixes in progress |
| Last updated | 2026-08-04 |

## Summary

During a batch reprocessing run (10 episodes re-diarized, then a second batch), the DGX Spark (GB10, 130.7 GB **unified** CPU+GPU memory) ran out of memory. The batch drove **concurrent diarization requests** to the DGX `pyannote-app` service; each request uploads a whole-episode audio file and runs PyTorch speaker-embedding extraction, which allocates large **`/dev/shm` shared-memory** segments. The DGX service has no concurrency or memory limit, and the box was already ~110/130 GB committed to resident model reservations (vLLM/Qwen et al.), leaving only ~20 GB headroom. At **23:35 UTC** ~17 GB was consumed in ~2 minutes, the OOM-killer began reaping, the batch/driver re-issued the failed work (**crash-loop**), and by ~23:50–00:20 UTC the box's userspace was hung — including `tailscaled` and `sshd`. The box is still up at the kernel level (answers ICMP + TCP SYN) but cannot complete an SSH handshake, so it needs an out-of-band power-cycle.

## Impact

- **Customer-facing**: none. The DGX is an internal inference host (invoked from the operator's laptop pipeline via the tailnet DGX provider); prod podcast/player were unaffected.
- **Service impact**: all DGX-hosted inference (vLLM/Qwen, Whisper Turbo, pyannote diarization, ollama) unavailable from ~00:20 UTC. Any pipeline stage depending on the DGX fails over or blocks.
- **Data lost or corrupted**: none identified. The in-flight batch's diarization outputs for the final episodes were not produced; those episodes need re-running once the box is back.
- **Time to detect (TTD)**: ~lagging. The box went offline on the tailnet at 00:20 UTC; noticed the following morning (~07:00 UTC) as "DGX not on the tailnet." No alert fired on the memory cliff.
- **Time to resolve (TTR)**: **ongoing.** Recovery is blocked on a physical power-cycle (no remote power controller / BMC reachable). Root cause was fully diagnosed from the homelab observability stack while the box remained hung.

---

## Phase 1: Facts (timeline)

UTC throughout. All figures from the homelab VictoriaMetrics stack (DGX `node_exporter` + app HTTP metrics) unless noted.

| Time (UTC) | Event | Source |
| --- | --- | --- |
| ~17:00–23:21 | DGX steady state: `MemAvailable` flat at ~18–20 GB of 130.7 GB (≈85% used by resident model reservations). `Shmem` flat at 0.5 GB. Diarization service serving intermittent requests. | VictoriaMetrics `node_memory_*` |
| ~23:28–23:32 | Second batch's diarization requests arrive; `pyannote-app` `http_requests_total` climbs to 1100. | `http_requests_total{job="pyannote-app"}` |
| **23:35 → 23:37** | **Memory cliff**: `MemFree` 18.9 → 1.4 GB in ~2 min. `Shmem` 0.5 → 3.7 GB, `Cached` +1.7 GB; `AnonPages` 5.5 → 1.3 GB (OOM-killer already reaping). `pyannote-app` request counter freezes at 1100 (in-flight requests never complete). | `node_memory_*`, `http_requests_total` |
| 23:39 | `Shmem` still climbing to 8.1 GB; `MemFree` 1.1 GB. | `node_memory_Shmem_bytes` |
| 23:45 | OOM-killer reaps a large process → `MemFree` bounces to 24 GB, `Shmem` back to 0.6 GB (brief recovery). | `node_memory_*` |
| 23:47 | Workload re-issues the failed work → `Shmem` back to 7.8 GB, `MemFree` → 1.0 GB (**crash-loop**). | `node_memory_*` |
| ~23:50 | `node_exporter` + services stop reporting (OOM-killed / starved). | last-sample time per job |
| **00:20 (08-04)** | `tailscaled` dies → DGX drops off the tailnet (`LastSeen 2026-08-04T00:20Z`). This is the "6h ago, lost access" symptom. Key still valid (`KeyExpiry 2026-12-02`) — NOT a credential issue. | `tailscale status --json` |
| ~07:05 | `alloy` stops pushing; box goes fully silent to observability. | VictoriaMetrics staleness |
| ~07:00–09:30 | Investigation (this session). From homelab LAN: box answers ICMP (0% loss) and TCP SYN on :22, but `sshd` never returns its banner → **userspace hung / OOM-locked**. No remote power controller (Kasa/HA/MQTT/IPMI) found. | agent probes via `homelab-claude` jump |
| — | **Awaiting** physical power-cycle. | — |

---

## Phase 2: Analysis

### Root cause

**Unbounded concurrent diarization on a memory-saturated unified-memory box.** The batch reprocessing drove multiple concurrent `POST /v1/diarize` requests to the DGX `pyannote-app` service. Each request:

1. uploads a whole-episode audio file (held in RAM by the service), and
2. runs PyTorch speaker-embedding extraction, which allocates large **`/dev/shm` shared-memory** tensors (the memory-composition signature at the cliff was `Shmem` + `Cached` growth, **not** process RSS — RSS actually fell as the OOM-killer reaped).

The DGX service has **no concurrency limit and no memory cap**, and the box was already ~110/130 GB committed to resident model reservations (vLLM/Qwen etc. pre-reserve upfront), leaving ~20 GB headroom. A few concurrent diarizations exceeded that headroom in ~2 minutes → OOM. The OOM-kill then failed the in-flight requests, which were **re-issued** (crash-loop), grinding the box — including `tailscaled` and `sshd` — to a hung state.

### Contributing factors

1. **Client single-flight lock is per-process only.** `providers/tailnet_dgx/diarization_provider.py` guards concurrency with a module-level `threading.Lock()` (`_dgx_diarize_single_flight`). It serializes diarization *within one Python process*, but a parallel batch (episodes across multiple processes, or a second concurrent run) **bypasses** it, so several requests reach the single-GPU DGX at once.
2. **No server-side concurrency/memory guardrail.** The DGX `pyannote-app` accepts unlimited concurrent requests, buffering each full-audio upload and spawning PyTorch `/dev/shm` embeddings. There is no semaphore, no `429/503` backpressure, no cgroup `MemoryMax`, and `/dev/shm` defaults to 50% of RAM (~65 GB) — far too permissive.
3. **PyTorch default shared-memory strategy.** Embedding extraction uses the `file_descriptor`/`/dev/shm` sharing strategy; `set_sharing_strategy('file_system')` (or `num_workers=0`) would avoid the shmem blow-up.
4. **Thin headroom under full load (capacity-assessment gap).** The GPU-mode design assumed one heavy service at a time; the "full test of everything together" ran all resident services *plus* a batch diarization burst on ~20 GB of spare, unified memory.
5. **Retry/re-issue fed the crash-loop.** The 23:45 → 23:47 re-consumption shows failed work being re-driven onto an OOM-ing box, escalating a contained OOM into total hang.
6. **Critical processes not OOM-protected.** `tailscaled` and `sshd` had no `OOMScoreAdjust`, so the OOM-killer took our remote access — which is why the box is unrecoverable without physical intervention.
7. **Observability blind to per-process/`/dev/shm` memory.** The DGX exports node-level memory + app HTTP counters only — no process-exporter, no `/dev/shm` gauge, no GPU `FB_USED`. Attribution required reverse-engineering from memory *composition* + request timing rather than a direct per-PID metric.

### Why detection took as long as it did

No alert fired on the memory cliff (no threshold on `MemAvailable` / `Shmem`). The first signal was the tailnet drop, noticed the next morning. A simple `MemAvailable < 15 GB` (or `Shmem` rate) alert would have surfaced it at 23:35.

### Why recovery is slow

The OOM-killer took `sshd`'s ability to respond (banner never completes) and there is **no out-of-band power control** for the Spark reachable from the network. Recovery is therefore gated on a physical power-cycle. This is the single biggest mitigation gap: an OOM on this box is currently *unsurvivable remotely*.

### Counterfactuals (what didn't break that could have)

- **The homelab observability stack kept running** (it's a separate Mac mini) — without the DGX's shipped `node_exporter`/app metrics, the cliff would have been invisible and the root cause unknowable.
- **vLLM/Qwen was innocent** — its ~103 GB reservation never moved; the failure was the diarization batch's `/dev/shm`, not model serving.
- **No prod impact** — the DGX is internal; a customer-facing dependency on it would have made this a SEV-1.

---

## Phase 3: Improvement plan

### Prevention (would have stopped this happening)

| Item | Where | Owner | Target |
| --- | --- | --- | --- |
| DGX `pyannote-app`: hard **concurrency=1 semaphore** + `429/503` backpressure (process one diarization at a time; don't buffer extra uploads) | DGX service | operator/agent | post-reboot |
| DGX `pyannote-app`: `torch.multiprocessing.set_sharing_strategy('file_system')` (or `num_workers=0`) to kill `/dev/shm` growth at source | DGX service | agent | post-reboot |
| DGX service: cgroup `MemoryMax` + bound `/dev/shm` (e.g. `--shm-size`/tmpfs cap) so it fails its own alloc, not the box | DGX host | operator | post-reboot |
| Client: make diarization single-flight **cross-process** (file lock) or hold DGX diarization concurrency at 1 | `tailnet_dgx/diarization_provider.py` | agent | this session (client-side) |
| Capacity rule: don't co-resident all model services *and* run a batch burst; reserve ≥25–30 GB headroom | GPU-mode design / runbook | operator | v2.8 |

### Detection (would have surfaced the problem sooner)

| Item | Where | Owner | Target |
| --- | --- | --- | --- |
| Alert on DGX `node_memory_MemAvailable_bytes < 15 GB` and on fast `Shmem` growth | Grafana/alerting | operator | v2.8 |
| Add per-process memory (process-exporter) + `/dev/shm` gauge + DCGM `FB_USED` on the DGX | DGX observability | operator | v2.8 |

### Mitigation (would have reduced impact / recovery time)

| Item | Where | Owner | Target |
| --- | --- | --- | --- |
| `OOMScoreAdjust=-1000` on `tailscaled` + `sshd` (survive-access — the OOM-killer takes a model worker, never our access) | DGX systemd drop-ins | agent | post-reboot |
| Client: never re-issue diarization on an OOM/5xx-overload signal (distinguish from a blip) — stop feeding the crash-loop | `tailnet_dgx/diarization_provider.py` | agent | this session |
| Remote power control for the Spark (smart plug / BMC) so a future OOM is recoverable without a house visit | homelab | operator | v2.8 |

### Process

| Item | Where | Owner | Target |
| --- | --- | --- | --- |
| Runbook: "DGX OOM / hung box" recovery (LAN-jump diagnosis via homelab, power-cycle, re-`tailscale up`, re-run affected episodes) | homelab/DGX runbook | agent | post-reboot |

---

## Recovery runbook (how to bring a hung box back)

When a box is OOM-livelocked (kernel answers ping, but `sshd` never completes its banner
and all metrics are stale) it will **not** self-heal — the OOM-killer is reactive and can't
restart the daemons it already killed (see "Why the OOM-killer won't recover it" below).

**DGX Spark (`dgx-llm-1` / `192.168.0.59`):**

1. Cut power at the smart plug (remote), wait ~15 s, restore power.
2. The Spark has a physical micro power-switch and (currently) does **not** auto-boot on
   AC restore, so someone must press the button. WoL does **not** help after a full power
   cut (the NIC loses standby power).
3. Once up: `ssh dgx-llm-1 'sudo tailscale up'`; confirm it rejoins the tailnet; verify GPU
   clean (`nvidia-smi`); restart the model services; re-run the episodes the batch didn't finish.

**Make future power-cycles fully remote (do these to remove the house-visit):**

- **DGX firmware:** set **"Restore on AC Power Loss = Power On"** (a.k.a. AC Recovery) in
  BIOS/firmware. Then a smart-plug off→on boots it — no button, no WoL.
- **Mac mini (`homelab`, Macmini8,1):** same risk — `pmset -g` shows `autorestart 0`
  (verified 2026-08-04). Enable auto-boot-on-power-restore with `sudo pmset -a autorestart 1`
  (and keep `sleep 0` so a headless server never sleeps).
- **Both:** `OOMScoreAdjust=-1000` on `tailscaled` + `sshd` (systemd drop-ins) so the OOM
  killer takes a workload process, not our remote access — the highest-leverage fix.

### Why the OOM-killer won't recover it (it fired once — why not again?)

At 23:45 the OOM-killer *did* fire and freed ~23 GB (it killed the largest process); the
workload immediately re-consumed it. It won't "come again and fix it" now because:

1. **The killer is reactive, not a sweeper.** It fires only when a process requests memory
   that reclaim can't satisfy. In a livelock the remaining processes are blocked/waiting,
   not making fresh large requests, so nothing triggers it — the kernel just pins on page
   reclaim.
2. **The memory it would need is likely unreclaimable.** A wedged GPU/CUDA context leaves
   its holder in uninterruptible sleep (D-state); the OOM-killer **cannot** kill a D-state
   process, and GPU/driver memory isn't reclaimable page cache.
3. **Freeing RAM ≠ recovery.** `sshd` and `tailscaled` were killed hours ago. Freeing
   memory doesn't respawn them — a wedged userspace (init not making progress) can't launch
   new processes. Recovery needs *starting* daemons, which the killer never does.

So the 23:45 kill "worked" only in the narrow sense of satisfying one re-request; by then
the cascade had already taken the control-plane daemons, and the box is now in a state the
killer can neither act on nor repair.

## What went well

- **Full root-cause diagnosis with the box hung**, purely from the homelab observability stack: pinned the 2-minute cliff, the `/dev/shm` signature, the crash-loop, and correlated it to the diarization service by request timing (moss-app idle, vLLM steady).
- **homelab as a diagnostic bastion** — LAN reachability let us confirm "kernel alive, userspace hung" instead of guessing.

## What went wrong

- **A single batch could take down the whole box** — no server-side concurrency/memory guardrail on a shared inference host.
- **The OOM-killer took our remote access** — no `OOMScoreAdjust` protection means an OOM is unrecoverable remotely.
- **No alert on the memory cliff** — the first signal was hours later (tailnet drop).
- **No per-process/`/dev/shm` telemetry** — attribution had to be reverse-engineered.

## Lessons learned

- **On unified-memory boxes, "GPU memory" IS system memory.** vLLM's upfront reservation left thin headroom; a CPU-side `/dev/shm` burst (not GPU) is what tipped it. Budget the whole unified pool, including batch/`/dev/shm`.
- **Client-side concurrency guards are per-process; servers must self-protect.** A `threading.Lock()` can't stop a parallel batch. The heavy inference service needs its own semaphore + memory cap — it can't trust callers.
- **Protect the control plane from the OOM-killer.** `tailscaled`/`sshd` with `OOMScoreAdjust=-1000` is the difference between "kill a model, stay online" and "drive to the house."
- **Diagnose the dead box from its neighbours.** When the box is hung, its shipped telemetry on a *separate* host is the whole story — which is why fixing the observability gap (per-process + `/dev/shm`) pays for itself on the next incident.

---

## References

### Code

- `src/podcast_scraper/providers/tailnet_dgx/diarization_provider.py` — DGX diarization client (per-process `threading.Lock` single-flight, retries, breaker)
- `src/podcast_scraper/providers/guardrails/diarization.py` — existing output-validity guardrail (empty-segments; NOT a memory guardrail)
- `src/podcast_scraper/config.py` — `transcription_parallelism` (default 1; Whisper >1 flagged "may cause memory/GPU contention"), `processing_parallelism` (default 2)

### Observability queries (homelab VictoriaMetrics `:8428`)

- `node_memory_{MemAvailable,MemFree,Shmem,Cached,AnonPages}_bytes{instance="dgx-llm-1"}` — the cliff + composition
- `http_requests_total{job="pyannote-app"}` vs `{job="moss-app"}` — service attribution
- `tailscale status --json` (DGX `LastSeen 2026-08-04T00:20Z`, `KeyExpiry 2026-12-02`)

### GH issues

- [#1397](https://github.com/chipi/podcast_scraper/issues/1397) — DGX diarization OOM guardrail (open; `incident` label)

### Prior PIRs

- [INCIDENT-2026-05-29 — prod VPS destroyed by unintended `tofu apply` cascade](INCIDENT-2026-05-29-prod-rebuild-cascade.md)
