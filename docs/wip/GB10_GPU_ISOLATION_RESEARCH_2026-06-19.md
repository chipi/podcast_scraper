# GB10 GPU isolation primitives — research findings for #1000

**Date:** 2026-06-19
**Branch:** `feat/autoresearch-followups-2026-06-18`
**Issue:** [#1000](https://github.com/chipi/podcast_scraper/issues/1000)
**Status:** Research only — no infra changes, no code changes, no recommendation
to ship anything beyond what Section 5 concludes.

## TL;DR

On GB10 (DGX Spark / Grace Blackwell consumer-workstation tier, sm_121,
unified-memory 128 GB), the GPU-isolation hardware menu is **empty**:

- **MIG**: not exposed on GB10. (MIG ships on data-center Blackwell B200 /
  GB200, professional RTX PRO 6000 Blackwell, and the unrelated `GB10B`
  Thor iGPU — none of which is what we have.)
- **NVIDIA MPS**: technically usable on Blackwell compute capability, but
  has **no fault isolation** (a fault in one client puts the whole MPS
  server in `FAULT` state and terminates every client sharing it). The
  20 % failure modes we measured in #996 (whisper hang, ConnectionReset,
  bilateral p99 spike) are exactly the class of fault MPS does not isolate
  — and adding MPS would *correlate* failures across whisper + vLLM
  instead of just letting them contend.
- **CUDA stream priority**: real API, supported on Blackwell, but a *hint*
  to the scheduler — not isolation. Inapplicable here because both
  consumers are externally-maintained closed-loop services (vLLM, whisper)
  whose inference loops we don't own.
- **cgroup-based time-slicing via nvidia-container-toolkit**: pure
  software round-robin scheduler. Documentation explicitly says no
  fault or memory isolation between replicas. On GB10 this is the
  *only* sharing primitive the official sources suggest, but it
  fundamentally cannot help with our failure shape — it'd reshuffle
  whose kernels run when, not prevent the catastrophic hang / reset
  modes.
- **`CUDA_VISIBLE_DEVICES`**: confirmed N/A (single GPU, can't split).
- **Application-layer queueing**: the only primitive that actually
  matches our contention shape, but it's not a "GPU isolation" primitive
  in any meaningful sense — it's serialization. Whether to build it is a
  separate architectural question.

**Categorical conclusion (Section 5):** Not worth pursuing GPU-level
isolation on GB10. Operator-rule architecture stands. Close #1000 as
"GPU isolation on GB10 won't move the needle for this workload."

---

## Section 1 — What NVIDIA exposes on GB10 today

### 1.1 MIG (Multi-Instance GPU) — NOT supported on GB10

NVIDIA's official MIG User Guide [^mig-profiles] enumerates the SKUs that
support MIG:

- **Data center**: A100, A30, H100 (80 / 94 / 96 GB), H200, B200
- **Workstation (professional)**: RTX PRO 6000 Blackwell (Workstation,
  Max-Q, Server), RTX PRO 5000 Blackwell, RTX PRO 4500 Blackwell
- **Integrated**: Thor iGPU (`GB10B`) — note the `B` suffix; this is the
  Jetson AGX Thor SoC, not the DGX Spark GB10

The DGX Spark / GB10 (no `B` suffix, 48 SMs, 128 GB unified LPDDR5X) is
**not** on the MIG-supported list. The Collabnix DGX Spark Kubernetes
guide [^collabnix-spark] states this directly: "GB10 doesn't expose MIG,
so time-slicing is your GPU-sharing mechanism on a single Spark."

The issue's claim ("MIG not supported on GB10") is **confirmed**.

[CONJECTURE] The architectural reason is plausibly the unified-memory
substrate — MIG's value proposition is hardware-partitioning the L2 +
HBM framebuffer, and GB10 has no discrete framebuffer (`nvmlDeviceGetMemoryInfo`
returns `NVML_ERROR_NOT_SUPPORTED` on GB10 per the NVIDIA forums [^mps-gb10]).
A MIG implementation would need a different partitioning model for
LPDDR5X unified memory; NVIDIA hasn't shipped one.

**Operator verification commands** (do not run from this research session;
listed for future on-DGX confirmation):

```bash
# Confirm MIG mode is disabled (and unsupported)
nvidia-smi --query-gpu=mig.mode.current --format=csv
nvidia-smi -L
# Attempt to enable MIG — should fail
sudo nvidia-smi -mig 1
```

### 1.2 CUDA streams + priority — supported, but a hint not isolation

`cudaStreamCreateWithPriority` is available on all GPUs of compute
capability ≥ 3.5 [^cuda-stream-priority]. GB10 is compute capability
10.0 / 12.1 [^gb10-cc], so the API is available.

Critical caveats from the CUDA Programming Guide [^cuda-prog-guide]:

- Stream priorities are *scheduler hints*. The runtime tries to launch
  pending work from higher-priority streams first, but does not
  preempt or guarantee. Quote: *"At runtime, the GPU scheduler
  utilizes stream priorities to determine task execution order, but
  these priorities serve as hints rather than guarantees."*
- Only **compute kernels** are affected — H2D / D2H memory copies are
  not prioritized.
- Range is queryable via `cudaDeviceGetStreamPriorityRange()`; lower
  numeric value = higher priority. The actual numeric range on GB10
  is not confirmed here ([UNVERIFIED] — common modern values are
  `[0, -1]` or `[0, -5]`, but the operator can run a one-liner to
  confirm).

**Operator verification command**:

```bash
# Print [leastPriority, greatestPriority] for GB10
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 \
  bash -c "cat > /tmp/p.cu <<'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
  int hi, lo; cudaDeviceGetStreamPriorityRange(&lo, &hi);
  printf(\"leastPriority=%d greatestPriority=%d\\n\", lo, hi);
}
EOF
nvcc -arch=sm_121 /tmp/p.cu -o /tmp/p && /tmp/p"
```

### 1.3 NVIDIA MPS — usable but inappropriate for our shape

The NVIDIA MPS architecture doc [^mps-arch] establishes:

- **Volta+ MPS** ships with hardware support for separate per-client
  address spaces and resource provisioning. GB10 is post-Volta (compute
  capability 10.0), so this generation of MPS applies. NVIDIA has not
  *publicly* confirmed MPS support on GB10 specifically [^mps-gb10]
  (forum thread, no definitive answer), but the compute-capability
  threshold is met.
- **`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`** lets you cap SM utilization
  per client [^mps-thread-pct]. E.g. set whisper to 60 %, vLLM to 40 %.
  Default is `100 / MaxSharedClientsPerGPU`.
- **Fault isolation: NONE.** This is the deal-breaker. NVIDIA's own
  docs say [^mps-arch]: *"a fatal fault from one client may bring down
  a different user's client that shares any GPU with the faulting
  client."* Community write-ups [^mps-fault-cascade] elaborate: when
  the MPS server hits a fatal exception from any client, it transitions
  to `FAULT` state and forcibly terminates every other client sharing
  the daemon. The MPS control daemon must restart before normal
  operation resumes.

This second point is fatal for #1000's purpose. The whole motivation for
isolation is to keep whisper failures from being a symptom of vLLM
contention. Adding MPS *correlates* failures: a vLLM OOM or kernel
exception now nukes the whisper container too.

**Operator verification commands**:

```bash
# Confirm MPS works on GB10 (compute-capability check + start)
nvidia-smi -q | grep "CUDA Capability"
sudo nvidia-cuda-mps-control -d
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
```

### 1.4 cgroup time-slicing via nvidia-container-toolkit — software-only

NVIDIA's time-slicing documentation [^nvidia-time-slicing] is explicit
about what this primitive is:

- **Software-only**, implemented in the NVIDIA Kubernetes Device Plugin
  and surfaced via nvidia-container-toolkit ConfigMaps.
- **No fault or memory isolation**: quote — *"Unlike Multi-Instance GPU
  (MIG), there is no memory or fault-isolation between replicas."*
- The mechanism is *time-multiplexing*: a single physical GPU is
  exposed as N "replicas" that round-robin through compute time. Each
  replica gets *"an equal share of time to all GPU processes."*

The Collabnix DGX Spark guide [^collabnix-spark] confirms this is the
nvidia-recommended sharing primitive on GB10 specifically. It is
available, GB10-compatible, and config-only at the compose layer.

However: time-slicing **does not change kernel-level contention**. It
shapes *which container gets a kernel-launch window next*. Both vLLM
and whisper would still each get full GPU access for their slot;
neither sees memory isolation; OOM in one container affects the
whole unified-memory pool. For our 20 % catastrophic-tail rate, this
is the wrong shape — the hangs are *inside a single kernel-launch
window* where vLLM's batched decode and whisper's encode/decode
collide on the same SMs.

**Operator verification commands**:

```bash
# Confirm nvidia-container-toolkit version and time-slicing support
nvidia-ctk --version
# Inspect device plugin config (if k8s-style time-slicing is set up)
cat /etc/nvidia-container-runtime/config.toml
```

### 1.5 `CUDA_VISIBLE_DEVICES` partitioning — N/A

GB10 = single GPU. `CUDA_VISIBLE_DEVICES=0` and `=` (empty) are the only
two states. The issue is correct to mark this N/A.

### 1.6 Application-layer queueing — not a primitive, not in scope here

Not a GPU isolation primitive. It's a serialization mechanism that
runs entirely *outside* the GPU stack. The issue's framing is correct:
this is trivial to build and would fully eliminate contention by
turning concurrency into sequencing — but it would also cap throughput
at the slower of the two consumers (whisper finishes, *then* vLLM
runs, repeat).

Worth noting separately: today's "operator rule" architecture is
*exactly* application-layer queueing implemented in human wetware.
The operator is the queue. Building a software queue is the same
architecture, automated.

---

## Section 2 — Plausibility for our workload (per primitive)

The 5 things we know about the workload (from #996 / #963):

- 2 cohabiting consumers, single GB10, unified memory
- Both have large kernels (vLLM batched MoE decode, whisper large-v3
  encode + greedy/beam decode)
- Failure manifests as *catastrophic tail* (hang, connection reset,
  WER=1.0), not graceful throughput degradation
- Mixed precision (NVFP4 vLLM Qwen3-30B-A3B, fp16/bf16 faster-whisper)
- Container deployment (docker-compose on DGX Spark)

### 2.1 MIG

Filtered at Section 1. Cannot pursue.

### 2.2 CUDA stream priority

- **What it would change**: whisper kernels marked high-priority would
  get launch-order preference over vLLM kernels in the GPU scheduler.
- **Expected win on whisper failure rate**: [UNVERIFIED — speculative]
  modest at best, ~5–15 % reduction per the issue's own estimate.
  Stream priority is a hint; it doesn't preempt running kernels, and
  the hang failure mode looks like a runtime-level deadlock or driver
  state corruption, not a fair-launch problem.
- **Implementation complexity**: **disqualifying.** Both vLLM and
  faster-whisper / whisper-openai create their own CUDA streams
  internally. To use priority streams we would need to fork vLLM
  (modify `vllm/worker/worker.py` and the model-runner stream
  creation) AND fork faster-whisper / whisper-openai (modify the
  CTranslate2 stream init or the openai-whisper inference loop).
  Both are externally-maintained projects on active release cadence.
  The issue's "downstream-possibilities" section explicitly rules
  out forking vLLM or whisper.
- **Failure modes introduced**: maintenance burden of the forks;
  divergence from upstream; bug reports nobody can reproduce.
- **Disqualifying**: yes — requires forks the issue explicitly rules out.

### 2.3 NVIDIA MPS

- **What it would change**: both processes share one CUDA context via
  the MPS daemon, reducing context-switch overhead. Per-client SM
  caps can be set via `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.
- **Expected win on whisper failure rate**: [UNVERIFIED]
  - Best case: with SM caps, whisper gets guaranteed 40–50 % of SMs,
    reducing the "vLLM monopolized the GPU during whisper's window"
    contribution to hangs.
  - Realistic case: **win is likely negative.** The fault correlation
    (any vLLM crash → whisper killed too) is a brand-new failure
    mode we don't have today. Our current 20 % rate is bad but
    *independent*; the resilience layer can fall back per-episode.
    Under MPS a single bad vLLM request takes down the whisper
    container, requiring a daemon restart before either recovers.
- **Implementation complexity**: medium-high — extra daemon, separate
  systemd unit, container `ipc: host` requirement so processes can
  reach the daemon socket, MPS daemon must outlive both containers,
  every restart requires daemon-state coordination.
- **Failure modes introduced**: correlated crashes (above); MPS daemon
  itself is a single point of failure; telemetry hole on GB10 (NVML
  per-process memory queries already broken per [^mps-gb10]).
- **Disqualifying**: not strictly, but the cost-benefit is upside-down
  for our failure shape.

### 2.4 cgroup time-slicing via nvidia-container-runtime[^dgx-spark-runtime]

- **What it would change**: the container runtime alternates which
  container gets kernel-launch access in fixed time intervals.
  Smoothes mean latency; doesn't address per-window contention.
- **Expected win on whisper failure rate**: [UNVERIFIED — likely
  near zero] The catastrophic failure modes (hang, connection reset)
  appear to happen *during* a granted slot, not because of waiting
  for one. Time-slicing reshuffles arrival order but doesn't reduce
  kernel-level contention or fix the unified-memory pressure that
  plausibly triggers whisper's internal watchdog / OOM.
- **Implementation complexity**: low — compose-level config change.
- **Failure modes introduced**: in `--time-slice=N` mode whisper
  could wait N×slot-time for its turn, extending latency. NVIDIA's
  docs warn explicitly: *"no memory or fault-isolation between
  replicas"* — same unified-memory pool, same blast radius.
- **Disqualifying**: not strictly, but the expected win is too small
  to be worth even a small bench. The failure shape is wrong for
  this primitive.

### 2.5 `CUDA_VISIBLE_DEVICES` partitioning

N/A.

### 2.6 Application-layer queueing

- **What it would change**: vLLM and whisper requests serialize through
  a gateway. Never overlap on the GPU. Contention rate → 0 %.
- **Expected win on whisper failure rate**: high — eliminates the
  *contention* failure modes by definition. Does not eliminate
  vLLM- or whisper-specific failures (vLLM still has its own p99,
  whisper still has its own ~0–5 % baseline failure rate).
- **Implementation complexity**: medium. A request-queue gateway is
  a new service. The existing autoresearch eval harness already does
  ad-hoc gating (`gpu-mode-swap.sh research|idle`); productionizing
  that pattern is the gateway.
- **Failure modes introduced**: throughput cap (whisper finishes,
  then vLLM gets the GPU). For a 24×7 pipeline this matters; for
  the current single-operator workload it's the same as today's
  human-operated gate.
- **Disqualifying**: not, but it's **not GPU isolation**. It's a
  procurement-shaped question: "do we want to invest in a queue
  gateway, or do we keep the human-operated gate?" That decision
  belongs in a separate ticket scoped as "automate the operator
  rule," not in #1000.

---

## Section 3 — Bench design (if any primitive were worth one)

Per Section 2, **no primitive is plausibly worth a small bench**:

- MIG, CUDA stream priority: filtered at Section 1–2 (unavailable or
  requires forks).
- MPS: would need a bench, but a positive result is implausible
  ([UNVERIFIED] — and the fault-correlation downside is structural,
  not measurable, so a "fewer hangs but correlated crashes" result
  doesn't even argue for adoption).
- Time-slicing: expected win near zero for our failure shape; not
  worth the bench cost.
- Application-layer queueing: trivial to implement, and the
  "implement it and measure it" path is indistinguishable from
  "implement it and ship it" — there's nothing to bench, just to
  decide and build.

If you nonetheless want a smallest-possible bench for MPS (the
single primitive where the answer isn't categorical), the sketch
would be:

**Bench: MPS on / off, single GB10, N=10 whisper × continuous vLLM load**

- **Setup**: identical to the #996 sweep harness. Two arms:
  - Arm A (control): `gpu-mode-swap.sh research` mode, vLLM and
    whisper run as today (no MPS).
  - Arm B (MPS): start `nvidia-cuda-mps-control -d` on the host,
    add `ipc: host` to both compose services, set
    `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60` for whisper container,
    `40` for vLLM container.
- **Workload**: 10 v2 fixture episodes through whisper-openai
  (`:8002`), concurrent vLLM summary load generator at `:8003`
  (existing `scripts/eval/score/_dgx_vllm_load_generator.sh`).
- **Measurements**:
  - Whisper: catastrophic rate (hang / connection reset / WER=1.0),
    mean WER, mean elapsed, stdev elapsed.
  - vLLM: requests served, mean / p99 / max elapsed.
  - **New metric**: correlated-failure count — episodes where vLLM
    crashed and whisper *simultaneously* errored. Today this is
    structurally impossible; under MPS it becomes possible. This
    is the deciding measurement.
- **Acceptance criteria**:
  - "MPS helps" = whisper catastrophic rate drops from ~20 % to
    <10 %, AND zero correlated-failure events in the run.
  - Either condition failed → MPS doesn't help.
- **Operator GPU-time estimate**: ~3 h per arm (10 episodes × ~10 min
  mean elapsed under load + harness overhead), so ~6 h total + 1 h
  setup / teardown. Single workday.

I do **not** recommend running this bench. The fault-correlation
downside is structural — even a positive whisper-rate result would
need to be weighed against a brand-new failure correlation that
the resilience layer (`tailnet_dgx.resilience` #956) is not
designed for. The cost-benefit doesn't pencil.

---

## Section 4 — Sanity-check the graduated decision path

The issue's path:

1. Verify NVIDIA surface
2. Prototype on small bench
3. Plumb into prod stack
4. Document negative result, close

### Per-primitive walk-through

- **MIG**: filtered at step 1. Confirmed not supported on GB10. Done.
- **CUDA stream priority**: filtered at step 1–2. Available, but
  requires forking vLLM and whisper — explicitly out-of-scope per
  issue #1000's own "What this ticket explicitly does NOT commit to"
  section. Done.
- **MPS**: would pass step 1 (compute-capability ≥ Volta). At step 2
  the bench-cost vs structural-downside calculus argues no. The
  fault-correlation downside isn't measurable by a single bench —
  it's a categorical change in failure mode. Step 2 is wasted work
  here.
- **Time-slicing**: would pass step 1 (available on GB10 per NVIDIA
  docs and Collabnix). At step 2 the expected win is near-zero for
  our failure shape (catastrophic-tail is about kernel-level
  contention inside a slot, not scheduling between slots). Step 2
  is wasted work here.
- **Application-layer queueing**: leapfrogs to step 3. There's
  nothing to bench — the primitive is "build a queue." That's a
  separate ticket about "automate the operator rule," not a GPU
  isolation question.

### Path-level conclusion

The graduated path works *as a triage*: it correctly filters MIG
at step 1, correctly filters CUDA stream priority at step 1–2.
For MPS and time-slicing the path technically continues, but the
filtering criteria at step 2 should be augmented with "and the
expected-win × risk-adjusted is positive" — without that, you'd
burn a workday on MPS or time-slicing without learning anything
that wasn't already predictable from the architecture.

For application-layer queueing the graduated path is the wrong
shape entirely; that primitive is a build-or-don't decision.

---

## Section 5 — Honest categorical conclusion

**Not worth pursuing — operator-rule architecture stands. Close #1000
as "GPU isolation on GB10 won't move the needle for this workload."**

### Specifics

1. **MIG** is the only primitive that would actually solve this class
   of problem (hardware-partitioned isolation, independent failure
   domains, per-partition memory caps). NVIDIA does not ship it on
   GB10. Procurement-side options (H100 / B200 / RTX PRO 6000) are
   out of scope per the issue.

2. **MPS** is the only primitive where a bench could be informative,
   but the structural downside (fault correlation) is a categorical
   downgrade from today's independent failure modes. Even a positive
   whisper-rate result wouldn't justify shipping it — we'd be
   trading 20 % independent hangs for a smaller rate of *correlated*
   crashes. The resilience layer is built for the former, not the
   latter.

3. **CUDA stream priority** requires forking vLLM and whisper, which
   #1000 itself rules out.

4. **Time-slicing** is wrong-shape for the catastrophic-tail failure
   mode. The 20 % failure rate is about kernel-level contention,
   memory-pressure-triggered watchdogs, and OS-level connection
   teardowns — none of which time-slicing addresses.

5. **`CUDA_VISIBLE_DEVICES`** is N/A on a single GPU.

6. **Application-layer queueing** is what we already do, just
   manually. The architectural decision to "automate the operator
   rule" is a separate question and a separate ticket. It is not a
   GPU-isolation primitive.

### What this means for #923 and PROD_RUNBOOK

The operator rule (idle vLLM before transcription windows, per
PROD_RUNBOOK § "Provider model selection — DGX vs cloud per stage")
is the **correct architecture for GB10**. It's not a workaround
pending a hardware fix — it's load-bearing because the hardware
fix is not available at this tier.

For `prod_dgx_full_with_fallback` (#923) the implication is:

- The profile is correct as-is for single-operator workflows.
- For 24×7 multi-stage operation the bottleneck is **not** GPU
  isolation but rather "is there a software gate that enforces the
  operator rule when no operator is at the keyboard." That's a
  different ticket: "automate the operator rule via request-queue
  gateway." Filing that is the operator's call.
- For autoresearch sweep windows the existing `gpu-mode-swap.sh
  research|idle` pattern is the right primitive — it's
  application-layer serialization, just operator-triggered. If we
  later want to run sweeps while transcription is in flight, the
  answer is "rent an H100 or B200 in the cloud for the sweep, leave
  the GB10 to whisper" — which is the out-of-scope "decouple
  autoresearch from prod whisper" item from #1000 itself.

### Suggested issue resolution

Close #1000 with the following summary comment:

> Researched the 6 candidate primitives against the GB10 sm_121
> hardware surface (Section 1) and our specific contention shape
> (Section 2). MIG is the only one that would address the failure
> class and it is not exposed on GB10. MPS is technically usable
> but introduces fault correlation (a categorical downgrade for our
> resilience model). Stream priority requires upstream forks ruled
> out by the issue itself. Time-slicing is the wrong shape for the
> catastrophic-tail failure mode. The operator rule (idle vLLM
> before transcription) remains load-bearing because the hardware
> alternative does not exist at this tier. Future relaxation of
> the rule is a procurement / architecture question (rent H100 for
> sweeps; build a request-queue gateway to automate the rule), not
> a GPU-isolation question.
>
> Closes "operator rule is the right architecture for GB10."

---

## Confidence + open questions

### What I'm high-confidence on

- MIG is not exposed on GB10 (NVIDIA docs; Collabnix GB10
  Kubernetes guide; consistent across sources).
- Time-slicing on GB10 is software-only, no fault isolation
  (NVIDIA's own time-slicing docs say so explicitly).
- MPS has no fault isolation across clients (NVIDIA MPS
  architecture doc says so explicitly).
- CUDA stream priority needs in-process control of stream
  creation, which means forking the consumers.

### What I'm low-confidence on (marked [UNVERIFIED] in body)

- The exact numeric value of GB10's stream priority range —
  not confirmed; needs `cudaDeviceGetStreamPriorityRange()` on
  device.
- Whether MPS is *officially* supported on GB10 — the NVIDIA
  forum thread [^mps-gb10] surfaces it as an open question; the
  compute-capability bar is met but NVIDIA hasn't made a
  positive statement.
- Quantitative win estimates for MPS / time-slicing — these are
  bounded by the issue's own "5–15 %" / "modest" framing; I have
  not run benches and treat them as conjecture.
- Whether the "hang" failure mode is plausibly a UVM page-fault
  cascade triggered by the unified-memory pool being saturated —
  could not verify in this research session, would require
  inspecting `dmesg` and `nvidia-smi` telemetry from the actual
  failure events.

### Operator-only verification (not done here; commands provided
above)

1. `nvidia-smi --query-gpu=mig.mode.current` — confirm MIG is
   structurally not available (expect `[Not Supported]`).
2. `cudaDeviceGetStreamPriorityRange()` one-liner — get the
   actual priority range numeric values (cosmetic; doesn't
   change Section 5).
3. `nvidia-cuda-mps-control -d` startup test — confirm MPS daemon
   starts on GB10 with CUDA 13 / sm_121. (Cosmetic; doesn't
   change Section 5.)
4. Check whether the resilience layer (#956) catches a
   simulated MPS-correlated-failure (kill vLLM mid-stream,
   observe whisper behavior with MPS enabled). Only relevant
   if Section 5 is reversed and MPS is pursued.

### What this research did NOT cover (and why)

- Hardware procurement alternatives (renting H100 / B200, buying
  RTX PRO 6000) — explicitly out of scope per #1000.
- Forking vLLM / whisper for stream-priority support — explicitly
  out of scope per #1000.
- Detailed UVM / unified-memory page-fault analysis — would
  require live DGX access; out of scope for this research session.
- DCGM / NVML telemetry gaps on GB10 — orthogonal to isolation;
  tracked separately (see [^mps-gb10]).

---

## References

[^mig-profiles]: NVIDIA Multi-Instance GPU User Guide — Supported MIG
  Profiles. https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html
[^collabnix-spark]: Collabnix — *NVIDIA DGX Spark + Kubernetes: Run
  GPU Workloads on the GB10 Grace Blackwell Superchip*.
  https://collabnix.com/nvidia-dgx-spark-kubernetes-run-gpu-workloads-on-the-gb10-grace-blackwell-superchip/
[^mps-gb10]: NVIDIA Developer Forums — *MPS Support and Telemetry on
  Grace Blackwell (GB10) with Unified Memory*.
  https://forums.developer.nvidia.com/t/mps-support-and-telemetry-on-grace-blackwell-gb10-with-unified-memory/363137
[^cuda-stream-priority]: NVIDIA CUDA Runtime API — Stream Management
  (`cudaStreamCreateWithPriority`, `cudaDeviceGetStreamPriorityRange`).
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
[^cuda-prog-guide]: NVIDIA CUDA Programming Guide — Asynchronous
  Execution / Stream Priority.
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
[^gb10-cc]: NVIDIA Blackwell Compatibility Guide 13.3 — Blackwell
  compute capability (10.x / 12.x).
  https://docs.nvidia.com/cuda/blackwell-compatibility-guide/
[^mps-arch]: NVIDIA Multi-Process Service — Architecture.
  https://docs.nvidia.com/deploy/mps/architecture.html
[^mps-thread-pct]: ORNL — *Introduction to CUDA's Multi-Process
  Service (MPS)* — covers `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`.
  https://www.olcf.ornl.gov/wp-content/uploads/2021/06/MPS_ORNL_20210817.pdf
[^mps-fault-cascade]: Sagar Parmar — *Demystifying NVIDIA MPS*.
  https://sagar-parmar.medium.com/demystifying-nvidia-mps-how-multi-process-service-improves-gpu-sharing-and-performance-9f633878318a
[^nvidia-time-slicing]: NVIDIA GPU Operator Documentation —
  *Time-Slicing GPUs in Kubernetes*.
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html
[^dgx-spark-runtime]: NVIDIA DGX Spark — *Container Runtime for
  Docker*. https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html

### Internal references

- Originating issue: [#1000](https://github.com/chipi/podcast_scraper/issues/1000)
- Closed parent: [#996](https://github.com/chipi/podcast_scraper/issues/996)
  (catastrophic-tail N=20 sweep)
- Sibling research: [#999](https://github.com/chipi/podcast_scraper/issues/999)
  (response-shape guardrails — ADR-105)
- Profile: `config/profiles/prod_dgx_full_with_fallback.yaml`
  ([#923](https://github.com/chipi/podcast_scraper/issues/923))
- Operator-rule home: `docs/guides/PROD_RUNBOOK.md` §
  "Provider model selection — DGX vs cloud per stage"
- Resilience layer: `tailnet_dgx.resilience`
  ([#956](https://github.com/chipi/podcast_scraper/issues/956))
- Eval evidence (N=20):
  `docs/guides/eval-reports/EVAL_WHISPER_CONTENTION_AUTORESEARCH_2026_06_15.md`
- Eval evidence (N=5):
  `docs/guides/eval-reports/EVAL_WHISPER_CONTENTION_2026_06.md`
