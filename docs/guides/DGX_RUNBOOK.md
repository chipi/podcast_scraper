# DGX Spark runbook (RFC-089)

Operator guide for the NVIDIA DGX Spark on the podcast_scraper tailnet. Hardware
bring-up runs **after** the RFC-089 code/ACL merge unless you are testing incrementally.

Related: [RFC-089](../rfc/RFC-089-dgx-spark-tailnet-integration.md),
[ADR-096](../adr/ADR-096-dgx-spark-prod-primary-with-fallback.md),
[ADR-097](../adr/ADR-097-self-hosted-gha-runner-policy.md),
[PROD_RUNBOOK](PROD_RUNBOOK.md).

## P0 — Bring-up checklist (on DGX)

1. Power on, update OS, verify GPU: `nvidia-smi` (GB10, ~128 GB unified memory).
2. Install [Tailscale](https://tailscale.com); sign in with the **operator account** (not an auth-key).
3. In Tailscale admin, tag the device `tag:dgx-llm-host`.
4. Note MagicDNS name (expected `dgx-llm-1.<tailnet>`). Set repo/operator var `DGX_TAILNET_FQDN`.
5. Install [Ollama](https://ollama.com); enable systemd service.
6. Pull baseline models (overnight is fine):
   - `llama3.3:70b-instruct`
   - `qwen2.5:72b-instruct`
   - `gemma2:27b-instruct`
   - `whisper-large-v3` (or the exact tag used in prod profile `dgx_whisper_model`)
7. Deploy embedding shim: see `infra/dgx/embedding-shim/README.md` on port **8001**.
8. Apply ACL from `tailscale/policy.hujson` via OpenTofu (`make infra-apply` from operator machine).

### Post-hardware smokes

From your laptop (on tailnet):

```bash
export DGX_TAILNET_FQDN=your-dgx.tailnet.ts.net
host=$(bash scripts/ops/resolve_dgx_tailnet_host.sh)
curl -fsS "http://${host}:11434/api/tags"
curl -fsS "http://${host}:8001/health"
```

From prod or drill VPS (after ACL apply): same curls using the resolved host.

## P1 — Laptop / dev env flip

Copy `config/examples/dgx-dev.env.example` and export:

- `OLLAMA_API_BASE=http://<dgx-host>:11434/v1`

Stop local Ollama if it conflicts with port 11434 on the laptop.

Run autoresearch smoke config `autoresearch_prompt_ollama_llama33_70b_dgx_smoke_bullets_v1`.

### Embeddings (ADR-098)

DGX runs **only Ollama** for the pipeline. Embeddings stay in-process via
sentence-transformers on the host running the pipeline (laptop / pre-prod
VPS / prod VPS / CI runner). This is the empirical answer — see ADR-098 for
the A/B that produced it, and `data/eval/embedding_provider_comparison/` for the
numbers.

The architecture supports DGX-served embeddings via
`vector_embedding_provider: ollama` if you want to re-evaluate later, but no
shipped profile enables it. The shim that used to live on `:8001` is gone.

## P2 — Pipeline profiles

- `config/profiles/local_dgx_balanced.yaml` — laptop → DGX Ollama; **no LLM cloud fallback** (operator decision 2026-06-07 — DGX outages should be visible, not silently routed to paid cloud).
- `config/profiles/local_dgx_full.yaml` — measurement only (no cloud fallback in profile).
- `config/profiles/preprod_local_whisper.yaml` — Stage A dress rehearsal for the prod profile, with laptop-local Whisper instead of DGX Whisper (no DGX dependency for this validation pass).
- `config/profiles/cloud_with_dgx_primary.yaml` — prod target; Whisper-on-DGX with cloud fallback at the transcription layer (`transcription_fallback_provider: openai`), LLM is already cloud Gemini.

### P2 — Operator E2E smoke (#811 AC#6)

Single-episode end-to-end run through `local_dgx_balanced` against the fast fixture (60s of audio), validated 2026-06-07:

| Stage | Wall | Provider | Tokens | Cost |
| --- | --- | --- | --- | --- |
| Whisper transcription | 4.2 s | laptop CPU (mps), `small.en` | — | $0 |
| Summarization | 2.7 s | DGX `qwen3.5:9b` | 665 in / 18 out | $0 |
| GI (insights + evidence) | 10.5 s | DGX `qwen3.5:9b` (7 calls) | 967 in / 80 out | $0 |
| KG | 6.3 s | from summary path | — | $0 |
| LanceDB index + topic clusters | 0.2 s | in-process MiniLM (ADR-098) | — | $0 |
| **Total** | **30 s** | | | **$0** |

Quality gates:

- 3 insights generated, 4 quotes; **grounding rate 100%**, **quote verbatim validity 100%**.
- KG: 2 topics, 6 entities (incl. `Liam Verbeek`, `Maya Koster`, `Cascadia Alliance`, `Strava`) — all extracted from the qwen3.5:9b summary path.
- Vector index, topic clusters, bridge.json all written.

Reproducing the smoke (laptop):

```bash
# Terminal 1 — fixture HTTP server (fast variant)
.venv/bin/python scripts/tools/run_e2e_mock_server.py --port 18766 --fast-fixtures

# Terminal 2 — pipeline (one episode, profile-driven, DGX Ollama)
export PYTHONPATH="$(pwd)/src:$(pwd):${PYTHONPATH}"
.venv/bin/python -m podcast_scraper.cli \
  http://127.0.0.1:18766/feeds/podcast1/feed.xml \
  --profile local_dgx_balanced \
  --output-dir /tmp/dgx_e2e_smoke/out \
  --max-episodes 1
```

Caveats / notes for re-runs:

- Default `--rss` flag is for *additional* feeds; for a single feed pass the URL positionally.
- DGX Ollama must already be reachable on the tailnet (`curl http://${DGX_TAILNET_FQDN}:11434/api/tags`); the profile does not bring the daemon up.
- Re-verified 2026-06-07 with diarize ON after the wave-2 audio migration (pyannote 4 / numpy 2 / torch 2.12 venv refresh); `SPEAKER_00` / `SPEAKER_01` correctly labeled in segments.

### P2 — `local_dgx_balanced` fallback semantics (operator decision 2026-06-07)

`local_dgx_balanced` does NOT enable LLM cloud fallback. Rationale: the profile is for laptop-driven runs against DGX. If DGX Ollama goes down mid-run, the affected episode degrades visibly (summary missing) rather than silently routing to a paid cloud provider. The operator fixes the DGX side and re-runs.

The `FallbackAwareSummarizationProvider` wrapper at `src/podcast_scraper/summarization/fallback.py` (RFC-089 #5) stays available — any profile that opts in via `degradation_policy.fallback_provider_on_failure` gets it. Local dev profiles intentionally don't opt in. Cloud fallback IS enabled in prod, but at the Whisper layer; see P4.

### P2 — `qwen3.5:9b` pipeline mode (investigation 2026-06-07)

`local_dgx_balanced` uses `llm_pipeline_mode: staged`, not `bundled`. The investigation: under bundled mode (one fused clean+summarize call), `qwen3.5:9b` emits inconsistently malformed JSON — literal newlines inside string values, mid-bullet truncation, missing fields. Failure rate ~50-67% on small N. Even when it succeeds, output is degraded (1-3 bullets vs target 6-8).

This is **model-specific**, not a code regression:

- Same prompts, same bundled mode, called directly: `qwen3.5:27b`, `qwen2.5:32b`, `llama3.1:8b` all return valid JSON. `qwen3.5:9b` is on the edge of what a 9B can reliably structure in a single fused pass.
- The autoresearch v2 evaluation that crowned `qwen3.5:9b bundled` as a champion scored against RougeL vs silver-Sonnet, not JSON-parse pass/fail — degraded but parseable output still scored.
- Staged mode (two separate Ollama calls — clean, then summarize) sidesteps the issue. Trade: ~1.5× wall-clock. Acceptable since DGX RTT is ~30 ms and free.

If you want to revisit `bundled` later: bump the model to `qwen3.5:27b` (the size where bundled becomes reliable per the model comparison) and re-test before flipping the profile back. Tracking the real fix in #912 (lenient repair pass + autoresearch JSON-parse gate); the profile change here is a workaround, not a fix.

## Pre-prod laptop validation (Stage A → B → C ladder)

Three gates before flipping prod to `cloud_with_dgx_primary`. Each stage runs **from your laptop** — no separate pre-prod VPS needed.

### Stage A — laptop with local Whisper (no DGX Whisper service required)

Profile: `config/profiles/preprod_local_whisper.yaml`. Mirrors the prod profile exactly except `transcription_provider: whisper` (laptop-local openai-whisper on Apple MPS / CUDA) instead of `tailnet_dgx_whisper`. Validates: prompts, pipeline glue, diarization, screenplay, Gemini summary/GI/KG/vector path. Does NOT validate the DGX HTTP transcribe call (that's Stage B).

```bash
# Quick iteration with small.en (~10× real-time)
make preprod-local RSS=https://feeds.example.com/your-show.rss EPISODES=1

# Final dress rehearsal with large-v3 (matches what prod DGX will run)
make preprod-local RSS=https://... EPISODES=3 WHISPER_MODEL=large-v3
```

What to look for in the run output:

- `segments.json` files have `speaker` labels (`SPEAKER_00`, `SPEAKER_01`, ...) — confirms diarize ran
- `gi_grounding_rate_pct` near 100 in `metrics.json`
- `gi_quote_validity_rate_pct` at 100 (verbatim quotes from transcript)
- `kg_topic_nodes_total` and `kg_entity_nodes_total` non-zero
- `search/lance_index/` written
- No `llm_summary_fallback_active_count` (this profile uses Gemini directly; nothing to fall back FROM)

If anything looks wrong, fix it on the prompt / pipeline side before any DGX Whisper work — no point installing a service on DGX while pipeline shape is broken.

### Stage B — laptop → DGX Whisper service (blocked on #814)

Once #814 lands faster-whisper-server on DGX, the same prod profile becomes runnable end-to-end from your laptop:

```bash
podcast-scraper <rss> --profile cloud_with_dgx_primary --output-dir ~/preprod-stage-b
```

Three chaos gates here:

1. **Happy path** — DGX up, profile runs through, output structure matches Stage A's by diff.
2. **DGX denied mid-run** — block the DGX Whisper port via local proxy returning 503; verify the pipeline falls back to OpenAI cloud Whisper (`transcription_fallback_provider: openai`), emits a `dgx_fallback_active` Sentry breadcrumb, episode still completes.
3. **DGX + cloud both denied** — both proxied to 503; verify clean abort with operator-visible error, no half-baked output.

Soak: `make preprod-soak` (added with #814) runs nightly happy-path against the last 24h of operator corpus, appends a row to [DGX_PROD_VALIDATION_LOG](../operations/DGX_PROD_VALIDATION_LOG.md), refuses to allow promotion until the rolling 4-week fallback rate is under 1% per ADR-096.

### Stage C — prod flip

Operator profile YAML on prod VPS → `cloud_with_dgx_primary`. Restart API container. Watch Sentry `dgx.fallback` breadcrumbs for 48 h. Rollback = revert profile + restart (≤ 5 min).

## P3 — GitHub Actions self-hosted runner (ADR-097)

All three layers required before registering the runner:

1. **Ephemeral** registration (`--ephemeral`).
2. Repo setting **Require approval for all outside collaborators** (see [REPO_SETTINGS_AUDIT](../operations/REPO_SETTINGS_AUDIT.md)).
3. Workflow listed in `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md`.

CI enforces the allowlist via `make check-test-policy` (runs
`scripts/tools/check_self_hosted_runner_allowlist.py` on every `python-app` lint job).

Register labels `self-hosted`, `dgx-spark`. Set GitHub repo variables `DGX_TAILNET_FQDN`, `DGX_OLLAMA_API_BASE` for `autoresearch-eval-nightly.yml`.

## P4 — Prod Whisper primary (ADR-096)

Profile: `config/profiles/cloud_with_dgx_primary.yaml` (`screenplay: true`, `diarize: true` — same rules as local Whisper; requires `HF_TOKEN` on the pipeline host for pyannote). See [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md).

Pre-prod validation: log results in [DGX_PROD_VALIDATION_LOG](../operations/DGX_PROD_VALIDATION_LOG.md) (4 weeks, fallback rate under 1%).

Prod rollout: flip operator profile to `cloud_with_dgx_primary`; watch Sentry for `dgx_fallback_active` breadcrumbs. Grafana fallback panel ships with #803.

Fast disable without code: revert profile in `viewer_operator.yaml` and restart API, or remove prod to DGX ACL rule.

## Day-2 operations

- Model updates: `ollama pull` on DGX; document tag changes in this file.
- Logs: `journalctl -u ollama` / embedding shim service.
- GPU: `nvidia-smi` over SSH.
- Embedding determinism: GPU indexes are not byte-identical to CPU; compare top-K overlap only.

## ⚠️ GB10 unified memory — DO NOT STACK BIG MODELS

**The DGX Spark / GB10 has 122 GB UNIFIED memory shared between CPU and
GPU. There is no separate VRAM pool.** This breaks the mental model that
"x86 server + dedicated NVIDIA GPU" gives you. Stacking models that
would fit comfortably on a server with 80 GB VRAM + 256 GB RAM will
OOM the box.

### What happened on 2026-06-11 (incident)

I had vLLM Qwen3.6-35B-A3B bf16 (~70 GB) + 3 small whisper containers
(~10 GB) running. I then warmed Ollama qwen3.5:35b (~23 GB) to set up
a multi-tenant test. Within seconds of the warm-up returning, the
kernel OOM killer fired and killed user-session systemd (PID 75167)
and a 13 GB uvicorn worker. SSH access died; only ping kept working
briefly. Required a hard power-cycle to recover.

Math: 70 + 23 + 10 + (Docker + OS + buffer cache + Tailscale + sshd +
user session services) ≈ 105–115 GB. The system started thrashing,
OOM killer fired indiscriminately, user session died, SSH stopped
authenticating.

### Hard rules going forward

1. **Total resident model size ≤ 80 GB at any moment.** Leave 30%+ of
   the 122 GB pool for the OS, page cache, Docker runtime, networking,
   user session, and short-lived spikes. This is the load-bearing
   number — not 122, not 100.
2. **Never run vLLM Qwen3.6-35B-A3B (~70 GB) alongside any other
   large-context LLM.** No qwen3.5:35b, no deepseek-r1:70b, no
   qwen2.5:72b. Use qwen2.5:7b (4.7 GB) or smaller if you need a
   secondary LLM loaded for contention testing.
3. **Before warming any model**, check `free -h` total used vs.
   available, AND `nvidia-smi --query-gpu=memory.used,memory.total
   --format=csv,noheader`. If `free -h` already shows < 30 GB available
   memory, stop and unload something first.
4. **`nvidia-smi memory.used` LIES on GB10.** It reports GPU-allocated
   memory but tells you nothing about CPU-side pressure on the same
   122 GB pool. The OS will OOM before nvidia-smi shows a problem.
   Always cross-check with `free -h`.
5. **Single-flight scenario tests.** If a test calls for "vLLM,
   Ollama, and pyannote simultaneously," redesign to load one big
   model at a time and document the deviation. Don't try to faithfully
   recreate a multi-model production state that doesn't fit on this
   hardware.

### Pre-flight checklist before loading a model

Run this BEFORE any `docker compose up`, `ollama pull`, or vLLM
warm-up:

```bash
ssh dgx-llm-1 'free -h && echo "---" && sudo nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
```

Decision tree:

- `free -h` "available" column ≥ 50 GB → safe to load up to a 30 GB
  model.
- `free -h` "available" column 20–50 GB → safe only to load up to a
  10 GB model. Anything bigger, unload something first.
- `free -h` "available" column < 20 GB → STOP. Unload an existing
  service before adding anything. You are already 1 OOM-trigger away
  from an incident.

### Loaded-model size reference (for arithmetic)

| Service / model | Approx resident |
| --- | ---: |
| vLLM Qwen3.6-35B-A3B (bf16) | ~70 GB |
| vLLM smaller MoE / 13B dense | ~30 GB |
| Ollama deepseek-r1:70b (Q4) | ~42 GB |
| Ollama qwen3.5:35b (Q4) | ~23 GB |
| Ollama qwen2.5:32b (Q4) | ~19 GB |
| Ollama qwen3.6 (Q4) | ~23 GB |
| Ollama qwen2.5:7b (Q4) | ~4.7 GB |
| whisper-openai large-v3 (bf16 via torch) | ~3 GB |
| faster-whisper large-v3 (int8 via ctranslate2) | ~3 GB |
| pyannote diarization | ~3 GB |
| OS + Docker + Tailscale + user session + buffer cache (steady state) | **~15-20 GB** |

Examples of SAFE stacking (~80 GB total):

- vLLM Qwen3.6-35B-A3B + 3 whisper services + qwen2.5:7b ≈ 70+10+4.7 = 84.7 GB ✗ (slightly over — leave Ollama unloaded)
- vLLM Qwen3.6-35B-A3B + 3 whisper services ≈ 80 GB ✓ (right at the bar)
- Ollama qwen3.5:35b + 3 whisper services ≈ 33 GB ✓ (plenty of headroom)
- Ollama deepseek-r1:70b + 3 whisper services ≈ 52 GB ✓

Examples of UNSAFE stacking:

- vLLM Qwen3.6-35B-A3B + Ollama qwen3.5:35b + 3 whisper services ≈ 103 GB ✗ (incident shape)
- vLLM Qwen3.6-35B-A3B + Ollama deepseek-r1:70b ≈ 112 GB ✗
- Any two of (vLLM 35B, Ollama 35B+, Ollama 70b) simultaneously ✗

### Recovery if it happens again

Symptoms: SSH connections hang at auth, ping eventually fails, no
response to anything. The kernel is alive but user session is dead.

1. Try `ssh root@dgx-llm-1` if you have direct root SSH enabled —
   the root session systemd may still be intact.
2. If that fails: physical access required. Recessed power button is
   behind the next-to-top plug on the back of the unit. Short press
   first; hold 10 sec if no response.
3. After reboot: check `sudo journalctl --boot=-1 -p err --no-pager`
   for OOM kill entries. Look for `Out of memory: Killed process`.
   Pids in the 1000+ UID range = user-session services died.
4. Update the safe-stacking calculation in this section if the
   resident-size estimates above were off.

## Networking — long-blocking HTTP over Tailscale (#956)

The DGX is reached over a Tailscale tunnel. Long-blocking sync HTTP calls
(whisper, pyannote, vLLM) against the DGX over that tunnel will eventually
**hang after the server has already returned 200 OK** — the response body
never reaches the laptop and the TCP connection stays `ESTABLISHED`
indefinitely. Surfaced during #929, codified as the failure mode this
section addresses.

### Why it happens

Three things compound:

1. **Tailscale switches paths mid-connection** (DERP relay → direct UDP →
   re-relay) when keepalives go quiet. The TCP layer doesn't know the
   underlying path changed; packet loss during the switch loses the response.
2. **macOS TCP keepalive default is 2 hours.** The kernel doesn't probe an
   idle socket for 2 hr before declaring it dead — so a lost mid-response
   packet leaves the client waiting "forever" from a user perspective.
   Linux defaults to ~75 s; the laptop side is the harsh one.
3. **`requests`-style `timeout=N` is a wall-clock budget**, not per-read.
   Bytes-arrive-then-stop never trips it.

### What prod consumers do

Every prod DGX consumer (`whisper_provider.py`, `diarization_provider.py`)
uses `src/podcast_scraper/providers/tailnet_dgx/resilience.py`, which
applies the three defences:

| Layer | What it does | Knob |
| --- | --- | --- |
| `dgx_http_client` | httpx client with `(connect, read)` per-read timeout, `Connection: close`, SO_KEEPALIVE + TCP_KEEPALIVE at ~30 s | every per-request call |
| `run_with_watchdog` | hard process-side wall-clock deadline that bypasses httpx's stuck-read state (covers the case where even per-read timeout doesn't fire because bytes are trickling) | every long-blocking call |
| `CircuitBreaker` | once DGX has failed N times in a window, open a cooldown so callers skip the per-request timeout tax on a wedged batch; half-open probe re-tests for self-heal | every consumer's call site |
| `effective_timeout_sec` | duration-scale the request timeout from audio length so 90-min episodes don't false-fail under brief contention | per-request `(duration, base, per_min)` |

Coverage proof: unit tests at
`tests/unit/podcast_scraper/providers/test_tailnet_dgx_resilience.py`
include `test_raises_timeout_when_overrunning` (watchdog fires) and
`test_client_sets_connection_close_and_is_closeable` (httpx config).
`test_tailnet_dgx_diarization.py` adds `test_dgx_raises_then_falls_back`,
`test_timeout_does_not_requeue_and_falls_back`,
`test_open_breaker_skips_dgx_entirely`,
`test_watchdog_hard_deadline_forces_fallback` (#954 — the diarization
analog with mandatory local-pyannote fallback).

### Diarization-specific config knobs (#954)

| Config field | Default | Purpose |
| --- | --- | --- |
| `dgx_diarize_request_timeout_sec` | `180.0` | Base request budget for DGX diarization calls (duration-scaled per audio minute on top of this). |
| `diarization_provider` | `local` | `local` / `tailnet_dgx` / `gemini`. `tailnet_dgx` enables the resilient client below; falls back to in-process pyannote on persistent failure. |

`TailnetDgxDiarizationProvider` (`src/podcast_scraper/providers/tailnet_dgx/diarization_provider.py`)
holds a process-wide `threading.Lock` so concurrent callers serialise on a
single in-flight DGX request — piling on a contended shared GPU just
multiplies the slowdown. On retry-after-transient or hard failure, the
provider falls back to local in-process pyannote (LOCAL not cloud,
because diarization audio shouldn't leave the trust boundary).

### What eval scripts do NOT do

`scripts/eval/score/whisper_dgx_vs_cloud_v1.py` and
`scripts/eval/score/diarization_dgx_vs_cloud_v1.py` use bare `requests.post`
with a flat wall-clock timeout. They were the original repro of the failure
mode. They use a `perl alarm` shell wrapper as a kill switch
(`scripts/eval/whisper_contention_perep.sh`) so a single-episode hang
doesn't kill the whole sweep — fine for eval, NOT a pattern to ship to
prod. If you write a new eval-class script that targets the DGX, prefer
importing `dgx_http_client` from the resilience layer rather than reusing
the eval pattern.

### When to design Tier-1 (async) vs Tier-2 (sync-with-resilience)

Tier-2 (what we have today) covers the failure mode for current load. For
any future DGX service whose typical response time exceeds **30 s**, prefer
a Tier-1 design from the start: `POST /jobs` → `202` → `{jobId, statusUrl}` →
poll or SSE for result. Eliminates the long-blocking socket entirely.
Tradeoff: every server needs a job queue.

## Make helper

```bash
make dgx-smoke
```

Uses `DGX_TAILNET_FQDN` and `resolve_dgx_tailnet_host.sh`; exits 0 with a warning when DGX is offline (for CI-less laptops).

## DGX observability (#943 / #942)

Three exporters on DGX feed the existing Grafana Cloud Prometheus +
the existing pipeline-side `compose/grafana-agent.yaml` scrape config.
**No new Grafana Cloud subscription**: this rides on the free tier the
pipeline already uses.

### Free-tier sizing (do not exceed)

| Cap (Grafana Cloud free) | Headroom we use | Discipline |
| --- | --- | --- |
| 10k active series | ~175 (1.7%) | Strip `id` / `pod` / `namespace` / `container_label_*` from cAdvisor at scrape; keep ~10 metric names per container. |
| 50 GB Prometheus ingest/mo | ~420 MB/mo (<1%) | 60s scrape for node/cAdvisor/pyannote-app; 30s for DCGM only. |
| 14-day retention | n/a | inherited |
| 5k Sentry errors/mo | ~0 expected | `before_send` drops the 503-still-loading boot noise; per-fingerprint rate limit upstream in Sentry project settings. |
| 10k Sentry transactions/mo | ~1.5k @ 0.01 sample × ~150k req/mo | `SENTRY_TRACES_SAMPLE_RATE` env override available; do not raise without re-budgeting. |

If a future panel needs higher-cardinality metrics (per-handler labels,
per-feed labels, etc.), price the additional series first: each new
series × 2880 scrapes/day × 30 bytes ≈ ~85 KB/day of ingest.

### Endpoints (added in #943; vLLM autoresearch added 2026-06-14)

| Port | Exporter | Scrape interval | Why |
| --- | --- | --- | --- |
| `:9400` | DCGM exporter | 30s | GPU state changes fast; util/mem/temp/power/SM ~15 series. |
| `:9100` | node-exporter | 60s | Host CPU/mem/disk/net — low churn. |
| `:8080` | cAdvisor | 60s | Per-container resource use. `id`/`pod`/`namespace` labels dropped at scrape. |
| `:8001/metrics` | pyannote-server | 60s | Request rate / latency histo / status codes via `prometheus-fastapi-instrumentator`. |
| `:8003/metrics` | vllm-autoresearch | 30s | Native vLLM Prometheus exporter — TTFT, queue depth, KV-cache util, GPU mem tracking. Only up while `gpu-mode-swap.sh research` is the active mode; Grafana Agent drops the target cleanly when the port isn't listening. ~30 series. |
| `:8002/metrics` | whisper-server (#996 follow-up) | 60s | Request rate / latency histo / status codes via `prometheus-fastapi-instrumentator`. Mirrors the pyannote pattern. Pair with contention sweeps so whisper-side queue depth + status-code drift land next to vLLM TTFT in the same dashboard window. |

Tailscale ACL (`tailscale/policy.hujson`) opens the three new ports
on `tag:dgx-llm-host` for `autogroup:admin`, `tag:gha-deployer`, and
`tag:prod` (the pipeline VPS). Scrape config lives in
`compose/grafana-agent.yaml` and is shipped by the existing prod
overlay.

### Dashboard

Importable: `config/grafana/grafana-dashboard-dgx.json`. 11 panels in
4 rows (GPU / System / Containers / App). The DGX panels reference the
existing Prometheus datasource — no new datasource setup needed.

### Sentry integration (#942)

`infra/dgx/pyannote-server/app.py` initializes `sentry_sdk` when
`SENTRY_DSN` is set in the operator's `/home/markodragoljevic/.env`.
No-op when unset. Tags applied to every event: `service=pyannote-server`,
`dgx_host=spark-2c14`, `gpu=GB10`, `environment=dgx-prod`. The
`before_send` hook drops the boot-time `pyannote pipeline not yet loaded`
503s — they're health-check noise, not actionable errors.

Future DGX FastAPI services (vLLM-prod, etc.) should mirror this pattern
verbatim — same DSN, same tags, same `before_send` filter.

### First-time operator steps

1. Set `SENTRY_DSN` in `/home/markodragoljevic/.env` on DGX (or leave
   unset to skip Sentry).
2. Push the Tailscale ACL change (Tailscale admin console — pull-request
   the JSON, merge, propagation is ~10s).
3. `make dgx-deploy` from the laptop — this lays down the new
   `/opt/observability/docker-compose.yml` and brings up the three
   exporters. The pyannote-server image is rebuilt with the new
   Sentry / Prometheus deps.
4. Verify scrape from the pipeline VPS:
   `curl http://dgx-llm-1.tail6d0ed4.ts.net:9400/metrics | head` (DCGM),
   same for `:9100`, `:8080`, `:8001/metrics`.
5. Import `config/grafana/grafana-dashboard-dgx.json` into Grafana
   Cloud (Dashboards → New → Import → upload JSON).

### When to break the free-tier budget

If the operator wants higher-fidelity metrics for a one-off
investigation (e.g. characterizing #996 catastrophic-tail in real
time):

- Drop scrape interval to 15s in `compose/grafana-agent.yaml` per-job
  (≈ 4× current ingest, still under 2 GB/mo).
- Add per-handler latency labels to pyannote-server temporarily.
- Revert after the investigation — sustained traffic at higher
  fidelity will eventually exceed the free tier.
