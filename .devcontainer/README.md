# Codespaces devcontainer

This directory configures the GitHub Codespaces environment used by
[RFC-081](../docs/rfc/RFC-081-pre-prod-environment-and-control-plane.md)
Phase 1.

## What gets created when a codespace starts

- 4-core / 16 GB / 32 GB SSD machine (default Codespaces tier).
- Python 3.12 + docker-in-docker + github-cli + Node 22.
- VS Code extensions: Python, Black, Volar (Vue), Prettier, Docker.
- Three privately-forwarded ports (GitHub auth required to visit):
  - `8000` — api (FastAPI direct).
  - `8090` — viewer (Nginx + reverse proxy). **This is the operator's URL.**
    Auto-opens preview on first start.
  - `5173` — viewer dev (Vite). Ignored by default; only relevant for
    operators running `npm run dev` instead of the production build.

## Codespaces Secrets

The `secrets:` block in `devcontainer.json` declares the env vars the
codespace expects. **None are required to boot** — each feature lights
up lazily based on which secrets are present.

To configure them, go to **GitHub repo → Settings → Secrets and
variables → Codespaces**.

| Secret | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | cloud_thin Whisper transcription. |
| `GEMINI_API_KEY` | cloud_thin summarisation (Flash Lite). |
| `ANTHROPIC_API_KEY` | optional, cloud_quality / cloud_balanced profiles. |
| `PODCAST_SENTRY_DSN_API` | Sentry DSN for the api process; unset = skip. |
| `PODCAST_SENTRY_DSN_PIPELINE` | Sentry DSN for pipeline subprocesses; unset = skip. |
| `PODCAST_JOB_WEBHOOK_URL` | Slack incoming-webhook URL (or HA / Shortcuts relay). |
| `GRAFANA_CLOUD_API_KEY` | Grafana Agent's API key. |
| `GRAFANA_CLOUD_PROM_USER` | Grafana Cloud Prometheus user / instance ID. |
| `GRAFANA_CLOUD_LOKI_USER` | Grafana Cloud Loki user / instance ID. |
| `GRAFANA_CLOUD_PROM_URL` | Grafana Cloud Prometheus remote_write URL. |
| `GRAFANA_CLOUD_LOKI_URL` | Grafana Cloud Loki push URL. |

## Bringing up the compose stack

Until the GHCR `publish` workflow lands and pushes the first images,
the stack must be started manually inside the codespace terminal:

```bash
docker compose \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml \
  up -d
```

Then visit the auto-forwarded port `8090` URL.

Once `publish` lands and `:main` images exist on
`ghcr.io/chipi/podcast-scraper-stack-*`, this command will move into
`postStartCommand` so the stack is up automatically on every wake.

## Pulling pinned tags

```bash
PODCAST_IMAGE_TAG=sha-abcdef0 docker compose ... up -d
```

The default `:main` tag tracks the latest passing main; pin to a
specific SHA for incident-response or to `:pr-NNN` for collaborator UAT.

## Notes

- The pipeline (ML) image is **not** pulled into Codespaces. Only
  `api` / `viewer` / `pipeline-llm` are published to GHCR (RFC-081
  §Layer 1; the ML image carries Llama 3.2 license restrictions on
  redistribution).
- `pipeline-llm` is a one-shot service spawned by the api job factory
  via `docker compose run --rm pipeline-llm ...` whenever a job is
  triggered. It does not auto-start with `up -d`.
- If the codespace is deleted (30-day inactivity default), the daily
  backup workflow has snapshotted the corpus to
  `chipi/podcast_scraper-backup` as a release asset; restore via
  `make restore-corpus`.

See [RFC-081](../docs/rfc/RFC-081-pre-prod-environment-and-control-plane.md)
for the full design.
