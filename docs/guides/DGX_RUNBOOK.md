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
export DGX_TAILNET_FQDN=dgx-llm-1.tail6d0ed4.ts.net
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
- `config/profiles/cloud_with_dgx_whisper_primary.yaml` — prod target; Whisper-on-DGX with cloud fallback at the transcription layer (`transcription_fallback_provider: openai`), LLM is already cloud Gemini.

### P2 — Operator E2E smoke (#811 AC#6)

Single-episode end-to-end run through `local_dgx_balanced` against the fast fixture (60s of audio), validated 2026-06-07:

| Stage | Wall | Provider | Tokens | Cost |
| --- | --- | --- | --- | --- |
| Whisper transcription | 4.2 s | laptop CPU (mps), `small.en` | — | $0 |
| Summarization | 2.7 s | DGX `qwen3.5:9b` | 665 in / 18 out | $0 |
| GI (insights + evidence) | 10.5 s | DGX `qwen3.5:9b` (7 calls) | 967 in / 80 out | $0 |
| KG | 6.3 s | from summary path | — | $0 |
| FAISS index + topic clusters | 0.2 s | in-process MiniLM (ADR-098) | — | $0 |
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

If you want to revisit `bundled` later: bump the model to `qwen3.5:27b` (the size where bundled becomes reliable per the model comparison) and re-test before flipping the profile back.

## Pre-prod laptop validation (Stage A → B → C ladder)

Three gates before flipping prod to `cloud_with_dgx_whisper_primary`. Each stage runs **from your laptop** — no separate pre-prod VPS needed.

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
- `search/vectors.faiss` written
- No `llm_summary_fallback_active_count` (this profile uses Gemini directly; nothing to fall back FROM)

If anything looks wrong, fix it on the prompt / pipeline side before any DGX Whisper work — no point installing a service on DGX while pipeline shape is broken.

### Stage B — laptop → DGX Whisper service (blocked on #814)

Once #814 lands faster-whisper-server on DGX, the same prod profile becomes runnable end-to-end from your laptop:

```bash
podcast-scraper <rss> --profile cloud_with_dgx_whisper_primary --output-dir ~/preprod-stage-b
```

Three chaos gates here:

1. **Happy path** — DGX up, profile runs through, output structure matches Stage A's by diff.
2. **DGX denied mid-run** — block the DGX Whisper port via local proxy returning 503; verify the pipeline falls back to OpenAI cloud Whisper (`transcription_fallback_provider: openai`), emits a `dgx_fallback_active` Sentry breadcrumb, episode still completes.
3. **DGX + cloud both denied** — both proxied to 503; verify clean abort with operator-visible error, no half-baked output.

Soak: `make preprod-soak` (added with #814) runs nightly happy-path against the last 24h of operator corpus, appends a row to [DGX_PROD_VALIDATION_LOG](../operations/DGX_PROD_VALIDATION_LOG.md), refuses to allow promotion until the rolling 4-week fallback rate is under 1% per ADR-096.

### Stage C — prod flip

Operator profile YAML on prod VPS → `cloud_with_dgx_whisper_primary`. Restart API container. Watch Sentry `dgx.fallback` breadcrumbs for 48 h. Rollback = revert profile + restart (≤ 5 min).

## P3 — GitHub Actions self-hosted runner (ADR-097)

All three layers required before registering the runner:

1. **Ephemeral** registration (`--ephemeral`).
2. Repo setting **Require approval for all outside collaborators** (see [REPO_SETTINGS_AUDIT](../operations/REPO_SETTINGS_AUDIT.md)).
3. Workflow listed in `.github/SELF_HOSTED_RUNNER_ALLOWLIST.md`.

CI enforces the allowlist via `make check-test-policy` (runs
`scripts/tools/check_self_hosted_runner_allowlist.py` on every `python-app` lint job).

Register labels `self-hosted`, `dgx-spark`. Set GitHub repo variables `DGX_TAILNET_FQDN`, `DGX_OLLAMA_API_BASE` for `autoresearch-eval-nightly.yml`.

## P4 — Prod Whisper primary (ADR-096)

Profile: `config/profiles/cloud_with_dgx_whisper_primary.yaml` (`screenplay: true`, `diarize: true` — same rules as local Whisper; requires `HF_TOKEN` on the pipeline host for pyannote). See [Audio Pipeline Guide](AUDIO_PIPELINE_GUIDE.md).

Pre-prod validation: log results in [DGX_PROD_VALIDATION_LOG](../operations/DGX_PROD_VALIDATION_LOG.md) (4 weeks, fallback rate under 1%).

Prod rollout: flip operator profile to `cloud_with_dgx_whisper_primary`; watch Sentry for `dgx_fallback_active` breadcrumbs. Grafana fallback panel ships with #803.

Fast disable without code: revert profile in `viewer_operator.yaml` and restart API, or remove prod to DGX ACL rule.

## Day-2 operations

- Model updates: `ollama pull` on DGX; document tag changes in this file.
- Logs: `journalctl -u ollama` / embedding shim service.
- GPU: `nvidia-smi` over SSH.
- Embedding determinism: GPU indexes are not byte-identical to CPU; compare top-K overlap only.

## Make helper

```bash
make dgx-smoke
```

Uses `DGX_TAILNET_FQDN` and `resolve_dgx_tailnet_host.sh`; exits 0 with a warning when DGX is offline (for CI-less laptops).
