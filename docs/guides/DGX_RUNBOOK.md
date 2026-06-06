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

### Embeddings via Ollama (ADR-098 / #897)

DGX profiles now serve embeddings via Ollama on `:11434` instead of the old
shim on `:8001` (deleted). The `local_dgx_balanced.yaml` profile already wires
`vector_embedding_provider: ollama` + `model: nomic-embed-text`. On first run
under that profile, the FAISS index is rebuilt automatically because the
provider stamp on existing indexes won't match — see
`REASON_EMBEDDING_PROVIDER_MISMATCH` in `server/index_staleness.py`.

One-time pull (overnight is fine, ~280 MB):

```bash
ssh <dgx-host> 'ollama pull nomic-embed-text'
```

Then build / rebuild the corpus index:

```bash
python -m podcast_scraper.cli index --output-dir ./output --rebuild
```

`--rebuild` is optional; the staleness check triggers a rebuild on the next
index run regardless. Use it when you want determinism.

## P2 — Pipeline profiles

- `config/profiles/local_dgx_balanced.yaml` — DGX Ollama + degradation fallback to Gemini.
- `config/profiles/local_dgx_full.yaml` — measurement only (no cloud fallback in profile).

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
