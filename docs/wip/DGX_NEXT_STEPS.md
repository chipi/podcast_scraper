# DGX next steps — what runs where, when, and why

**Status**: Living document. Updated 2026-06-09 (initial).
**Owner**: Marko + Claude. Append rather than overwrite — the history of decisions matters.

This doc captures the multi-month thinking on DGX as a first-class inference target — what's already there, what's moving there, what stays local, and the **operational frame** that decides each.

---

## TL;DR — the future-state architecture we're aiming for

```
┌──────────────────────────────────────────────────────────────┐
│  DGX GB10 — 128 GB unified memory                            │
├──────────────────────────────────────────────────────────────┤
│  Speaches      :8000    Whisper large-v3            3 GB     │ ✓ shipped (#814)
│  pyannote      :8001    speaker-diarization-3.1     2 GB     │ ✓ shipped (#926)
│  vLLM-prod     :8003    champion model @ fp16      70 GB     │ ⏳ planned
│  Ollama        :11434   autoresearch zoo       +6-25 GB      │ ✓ shipped (#811)
│  ─────────────────────────────────────────────               │
│  Steady state                              ~80 GB / 118 GB   │
│  Peak (autoresearch on 32B model)         ~107 GB / 118 GB   │
└──────────────────────────────────────────────────────────────┘

KEPT LOCAL (host running the podcast_scraper pipeline):
  spaCy en_core_web_trf  (NER, ~500ms/ep, 0.5GB)
  sentence-transformers  (MiniLM embeddings, 7ms/chunk, 0.08GB)
```

Cost at this end-state: **~$0/episode** (vs current cloud-Gemini ~$0.05/ep). Major unlock for high-volume corpus rebuilds.

---

## Current state (2026-06-09)

| Service | Where it runs | Status | Notes |
|---|---|---|---|
| Whisper transcription | DGX (Speaches container) | ✓ Prod-ready | [#814](https://github.com/chipi/podcast_scraper/issues/814) shipped; profile: `cloud_with_dgx_primary` |
| pyannote diarization | DGX (pyannote container) | ✓ Prod-ready as of this PR | [#926](https://github.com/chipi/podcast_scraper/issues/926); same profile wires it |
| Summary + GI + KG (LLM) | Cloud Gemini Flash Lite | Prod default | Will move to DGX after autoresearch settles champion ([#923](https://github.com/chipi/podcast_scraper/issues/923)) |
| NER (spaCy trf) | Pipeline host (in-process) | Settled local | See "Per-service offload analysis" |
| Embeddings (MiniLM) | Pipeline host (in-process) | Settled local per [ADR-098](../adr/ADR-098-embedding-provider-profile-axis.md) | A/B showed local beats DGX-nomic |
| Autoresearch matrix | DGX (Ollama daemon) | ✓ Working | qwen3.5:35b current champion; qwen3.6:latest emerging contender per [v2 sweep](../guides/eval-reports/EVAL_SMOKE_V2_DGX_REFRESH_2026_06.md) |
| Operator's coder | DGX (vLLM container) | Operator-managed | Qwen3-Coder-Next; separate from autoresearch |

---

## The decision frame — "should this service go to DGX?"

Single rule that decides every offload question:

> **Offload to DGX makes sense when `inference_time >> network_RTT`** (rule of thumb: ≥50×).
>
> Keep in-process when calls_per_episode × network_RTT exceeds (or even approaches) the inference time saved by GPU acceleration. Also keep in-process when ops complexity outweighs the speedup.

### Applied to every pipeline service

| Service | Inference time | Calls/episode | RTT cost | Verdict |
|---|---|---|---|---|
| Whisper transcribe | 5-30 s | 1 | 10 ms (0.03-0.2%) | ✓ Offload |
| pyannote diarize | 5-15 s | 1 | 10 ms (0.07-0.2%) | ✓ Offload |
| Summary LLM | 5-30 s | 1 | 10 ms (0.03-0.2%) | ✓ Offload (when champion settles) |
| GI extraction (LLM call) | 3-15 s | 1 | 10 ms (0.07-0.3%) | ✓ Offload — reuses summary LLM |
| KG extraction (LLM call) | 3-15 s | 1 | 10 ms (0.07-0.3%) | ✓ Offload — reuses summary LLM |
| NER (spaCy trf) | 500-2000 ms | 1-3 | 10 ms (0.5-2%) | **Stay local** — gain marginal, ops cost high |
| Embeddings (MiniLM) | 7 ms | 50-200 | 10 ms (**~140%**) | **Stay local** — RTT exceeds inference time |

The two "stay local" decisions are not arbitrary — they're driven by the math. Embeddings in particular are a clear loss: doing 50× round-trips at 10ms each (500ms) to save 50× a 7ms compute (350ms) makes the pipeline 40% **slower**, not faster.

### When this could flip

NER — if a future workload upgrades spaCy trf → a heavier model (e.g., a 1-2GB fine-tuned BERT-based recognizer specifically for podcast hosts/guests), THAT becomes 2-5s inference time and offload wins. We're not there yet.

Embeddings — if we switch from MiniLM (CPU-friendly) to a much larger embedding model (e.g., bge-large or e5-mistral at 7B params), RAM pressure on the host forces offload. Per ADR-098, our empirical A/B says MiniLM beats nomic on our corpus, so this isn't pressing.

---

## Three scenarios that determine vLLM vs Ollama for prod summary

The single most consequential question for the future-state is: **will we run multiple pipelines concurrently?**

### Scenario A — Sequential single-pipeline prod (current shape, no changes)

One episode at a time. transcribe → diarize → clean → summary → GI → KG → embed → done. Then next episode.

- **vLLM gives nothing.** Continuous batching needs concurrent requests. Single sequential pipeline = single request at a time.
- **Ollama is fine.** Q4 quantization is a ~2% quality loss vs fp16 — invisible against the noise floor we already established.
- **Recommendation**: stay Ollama. Simpler ops, less memory pressure, easy hot-swap during autoresearch transitions.

### Scenario B — Multi-pipeline parallel prod (planned future per operator)

Pipeline A processing feed 1 / episode 5 (GI step) **at the same time as** Pipeline B processing feed 2 / episode 3 (summary step) **at the same time as** Pipeline C processing feed 3 / episode 1 (diarize step).

- Now Whisper, pyannote, summary LLM, GI LLM, KG LLM are ALL serving simultaneously to different pipelines.
- **vLLM's continuous batching is exactly designed for this. Throughput could be 5-10× Ollama's.**
- Ollama would serialize the LLM calls — bottleneck.
- **Recommendation**: vLLM wins decisively.

### Scenario C — Hybrid (likely actual end-state)

vLLM-prod hosts the champion (qwen3.5:35b or qwen3.6:latest) at fp16 with continuous batching = serves all of summary + GI + KG (they share the same LLM, different prompts).

Ollama keeps the autoresearch zoo — 20+ models hot-swappable for sweep experimentation.

**Memory ceiling at 128GB:**
- Steady state: ~80GB (Speaches + pyannote + vLLM-prod + Ollama idle)
- Peak with autoresearch testing a 32B model: ~107GB → JUST fits
- Constraint: cannot autoresearch llama3.3:70b at Q4 (42GB) concurrent with prod (70GB + 5GB services = 117GB → over budget)
  - Workaround: schedule 70B autoresearch outside prod hours OR temporarily unload vLLM-prod

This is what operator is heading toward.

---

## vLLM vs Ollama — the trade-off in one paragraph

**Ollama** is "many models, one daemon": you can have 100 models on disk, swap the hot one in 5-30s, single endpoint serves all autoresearch. GGUF/Q4 quantization wins on memory (4× smaller per model). Serial inference — no continuous batching.

**vLLM** is "one model, optimized": dedicated container per model, fp16 native, continuous batching = 5-10× throughput under load, PagedAttention scales context cleanly. High memory commitment per container (model size × 1.3-1.5 with KV cache reservation).

**Memory math at 128GB (revised from earlier 64GB assumption)**:

| Model | fp16 (vLLM) | Q4_K_M (Ollama) | Fits at fp16? |
|---|---|---|---|
| llama3.3:70b | ~140 GB | ~42 GB | ❌ needs AWQ/GPTQ |
| qwen3.5:35b | ~70 GB | ~22 GB | ✓ tight |
| qwen3.6:latest (36B MoE) | ~72 GB | ~17 GB | ✓ tight |
| qwen3.5:27b | ~54 GB | ~17 GB | ✓ comfortable |
| phi4:14b | ~28 GB | ~9 GB | ✓ |

At 128GB, every reasonable champion candidate fits at fp16 on vLLM. The choice becomes "what's the workload pattern?" — answered by scenarios A/B/C above.

---

## Recommended sequencing (post-PR #941)

Tied to the autoresearch programme (originally tracked in
`AUTORESEARCH_NEXT_PHASE_AGENT_PLAN.md`, removed 2026-06-24 second-pass
cleanup — #907 / #927 epics + children all closed; sequencing below stands):

1. **Phase 0-2 of agent plan** — autoresearch settles on a champion (qwen3.5:35b vs qwen3.6:latest, gated on #932 G-Eval + #933 prod-curated validation)
2. **Add `summary_provider: vllm` support to the codebase** — new provider in `src/podcast_scraper/providers/vllm/`, OpenAI-compatible client mirroring `ollama_provider.py`. Maybe 1-2 days of work.
3. **Deploy vLLM container for the champion on DGX** — port :8003, model pinned, `gpu_memory_utilization: 0.5` initially (room for Ollama coexistence)
4. **A/B shadow run** — same episodes through both Ollama-prod (current) and vLLM-prod (new) for ~1 week. Verify outputs numerically equivalent within acceptable drift.
5. **Flip `cloud_with_dgx_primary.yaml`** — `summary_provider: vllm` + `vllm_api_base: http://your-dgx.tailnet.ts.net:8003/v1`. Keep Ollama route as documented fallback.
6. **Add `PIPELINE_PARALLELISM` config** — allow N podcast pipelines running concurrently against the same DGX. vLLM's continuous batching makes this efficient.
7. **First parallel-prod stress test** — run 4 feed pipelines concurrently. Measure throughput vs sequential baseline.

Each step is independent and reversible. Steps 1-5 have ZERO impact on prod (shadow only). Step 6 is the unlock that justifies the vLLM investment.

---

## Observability gap (2026-06-09)

Currently we have **zero metrics, zero error tracking** for DGX itself. Only visibility is SSH + `docker logs` + `nvidia-smi`. This needs to land soon — when vLLM-prod ships and we hit cross-service issues, debugging without metrics will be painful.

Two issues filed:

- **[#942 — Sentry on DGX services](https://github.com/chipi/podcast_scraper/issues/942)** (~½ day). Add `sentry_sdk.init(...)` to pyannote-server and future vLLM-prod containers. Separate Sentry project from the pipeline so DGX errors don't pollute pipeline error rates. Would have caught the 7 compat bugs we hit during the pyannote deploy.
- **[#943 — Prometheus + Grafana](https://github.com/chipi/podcast_scraper/issues/943)** (~1-2 days). Deploy `dcgm-exporter` + `node-exporter` + `cadvisor` on DGX, scrape over tailnet from existing Prometheus, add `grafana-dashboard-dgx.json` with per-service GPU memory split + request rates + tailnet RTT.

These slot after PR #941 lands but before the vLLM-prod work (step 3 of the recommended sequencing) — debugging vLLM resource issues without metrics is masochistic.

## Open questions (capture as they come up)

- **Q1**: At what point does NER offload become worth it? Concrete trigger: ?
- **Q2**: Do we want a `vllm_fallback_provider: ollama` chain (so vLLM-down doesn't drop to cloud)? Probably yes — keeps cost at $0 even during vLLM container restart.
- **Q3**: ~~How do we monitor vLLM + Ollama memory pressure live so we don't OOM during autoresearch overlapping prod?~~ → answered: #943 ships the GPU/container exporter + Grafana panel.
- **Q4**: vLLM container restart strategy — does the prod pipeline retry on 502s during a hot-swap? Currently TailnetDgxWhisperProvider has retry+fallback; need same shape for the LLM provider.
- **Q5**: Should we evaluate AWQ/GPTQ quantization for vLLM to make llama3.3:70b viable? Would put us back into Ollama-Q4 quality territory but with vLLM's batching.
- **Q6**: Where does Prometheus actually run today? `compose/grafana-agent.yaml` is in the repo — verify it scrapes from a centralized location accessible over tailnet. Needed before #943 can scrape DGX.

---

## What this doc replaces / supersedes

Nothing — supplemental to:

- `AUTORESEARCH_NEXT_PHASE_AGENT_PLAN.md` — the autoresearch programme's
  2-agent execution plan (removed 2026-06-24; #907 / #927 epics closed)
- `AUTORESEARCH_NEXT_PHASE_DEPENDENCIES.md` — issue dependency map
  (removed 2026-06-24; same reason)
- [ADR-096](../adr/ADR-096-dgx-spark-prod-primary-with-fallback.md) — prod-with-fallback degradation contract
- [ADR-098](../adr/ADR-098-embedding-provider-profile-axis.md) — embedding provider profile axis (the "embeddings stay local" decision)
- [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md) — original DGX bring-up
- [DGX_RUNBOOK.md](../guides/DGX_RUNBOOK.md) — operator-facing operational reference

This doc differs by being the **strategic** view — when to run what where, and the reasoning frame for future decisions. The other docs are tactical / decision-record / runbook.

---

## Changelog

- **2026-06-09** — Initial draft after a conversation surfaced (a) corrected 128GB memory assumption, (b) operator's plan to run parallel pipelines (Scenario B → C), and (c) the NER + embeddings offload question.
- **2026-06-09** — Added "Observability gap" section. Filed #942 (Sentry) + #943 (Prometheus/Grafana). Operator-prompted: "what about adding DGX to Sentry and Grafana?" — both should land before vLLM-prod ships to make resource debugging tractable.
- **2026-06-09** — Phase 0 (#939 Opus silver upgrade) landed locally on `feat/907-autoresearch-batch-2`. Champion ranking flipped dramatically under the new silver: mistral:7b (#1, 0.329), llama3.2:3b (#2, 0.326), llama3.1:8b (#3, 0.307), with qwen3.5:35b dropping to #11 and qwen3.6:latest to #12. ROUGE spread WIDENED (0.024 → 0.086) — refuting the original score-compression worry. This validates the Sonnet-mimicry hypothesis even more strongly than expected, and reinforces #932 + #933 urgency: any prod champion swap must wait for G-Eval (#932) + prod-curated validation (#933). The current finding does NOT pick a new champion — it picks a less-biased metric. Filed [#945](https://github.com/chipi/podcast_scraper/issues/945) to close the tuned-vs-untuned fairness gap (Phase 0.5 candidates are getting hand-tuned prompts; the older top-3 still use Qwen-clone prompts → unfair comparison). Phase 0.5 agent assignments rebalanced (Agent 1: HIGH-impact #935+#938; Agent 2: MEDIUM #937+#936; either picks up #945 as optional sidecar).
