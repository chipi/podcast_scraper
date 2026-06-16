# Prod operator cheat sheet

Fast companion to the full
[Prod runbook](PROD_RUNBOOK.md). Use this for day-to-day operations.

## Scope

- Prod host: Hetzner VPS
- Access path: tailnet only (`prod-podcast.<tailnet>.ts.net`)
- Deploy model: **`main`** is the source of truth; **Stack test** on `main` gates the Dockerized path; **GHCR** publishes **`api` / `viewer` / `pipeline-llm`** tags including **`sha-<short>`**; **`deploy-prod.yml`** applies images on the VPS (today often **`workflow_dispatch`** until **`workflow_run: Stack test`** is merged — see [ADR-082](../adr/ADR-082-gitops-app-deploy-via-stack-test-and-gha.md)). **Infra** (`infra/**`) applies only via manual **`infra-apply.yml`** after PR plan review.
- Full procedures and edge cases: see [Prod runbook](PROD_RUNBOOK.md)
- Architecture narrative (diagrams): [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md)

## Golden rules

- Never expose prod publicly by opening Hetzner TCP 80/443.
- Prefer tailnet URL or in-container health checks.
- Keep infra changes in `infra/` and app changes in normal PR flow.
- Always keep `/srv/podcast-scraper/.env` names exact (for example
  `OPENAI_API_KEY`, not `OPENAI_KEY`).

## Core endpoints and paths

- Prod URL: `https://prod-podcast.<tailnet>.ts.net`
- Health: `https://prod-podcast.<tailnet>.ts.net/api/health`
- Repo on host: `/srv/podcast-scraper`
- Runtime env: `/srv/podcast-scraper/.env`
- Viewer basic auth file: `/srv/podcast-scraper/.htpasswd`
- **Host corpus directory:** take from **`PODCAST_CORPUS_HOST_PATH`** in
  `/srv/podcast-scraper/.env` (do not assume a path until you check). Typical
  value on a standard deploy: `/srv/podcast-scraper/corpus`.

```bash
grep '^PODCAST_CORPUS_HOST_PATH=' /srv/podcast-scraper/.env
```

- **Same data on the host vs in containers:** SSH and host Python use
  `$PODCAST_CORPUS_HOST_PATH`. The `api` (and pipeline) containers mount that
  tree at **`/app/output`** (see `docker/api/Dockerfile` `serve --output-dir`).

## Topic clusters (manual maintenance) {#topic-clusters-manual-maintenance}

A successful **Configuration → Run pipeline** job runs **`full_incremental_pipeline`**
only. It may rebuild the vector index when `vector_search` is true in the
profile, but it **does not** run **`topic-clusters`**. The viewer/API read
`search/topic_clusters.json` if present; Missing file means clustering was never
built for that corpus. See [RFC-075](../rfc/RFC-075-corpus-topic-clustering.md)
and [Prod runbook — FAQ](PROD_RUNBOOK.md#corpus-directory-host-vs-appoutput).

**Prerequisite:** `search/lance_index/` under the corpus (needs a profile with
`vector_search: true` for indexing, e.g. `cloud_balanced`).

**Option A — host venv (no container for Python):**

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net
cd /srv/podcast-scraper
HOST_CORPUS=$(grep '^PODCAST_CORPUS_HOST_PATH=' .env | cut -d= -f2-)
# stock Debian images often need: sudo apt install python3.12-venv
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[search]"
.venv/bin/python -m podcast_scraper.cli topic-clusters --output-dir "$HOST_CORPUS"
ls -la "$HOST_CORPUS/search/topic_clusters.json"
```

**Option B — use Python inside the already-running `api` container** (no
`compose run`; replace container name if `docker ps` shows a different one):

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net
docker exec compose-api-1 python -m podcast_scraper.cli topic-clusters \
  --output-dir /app/output
```

**Verify API (optional):** `GET /api/corpus/topic-clusters?path=<host corpus path>`

If the JSON file is owned by root after an in-container write, fix ownership so
`deploy` can manage backups: `sudo chown deploy:docker …/search/topic_clusters.json`
(adjust group to match your layout).

## Reprocess from existing transcripts (no re-transcription)

Use this when pipeline logic changed and you want fresh metadata / GI / KG / index
without paying transcription cost again.

**Keep:** corpus `transcripts/` files.

**Rebuild:** derived artifacts (`metadata/`, `search/`, and run outputs that depend
on old prompts/parsers).

**Host path (recommended, no compose run):**

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net
cd /srv/podcast-scraper
HOST_CORPUS=$(grep '^PODCAST_CORPUS_HOST_PATH=' .env | cut -d= -f2-)

# Optional: keep a rollback copy of derived artifacts only.
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p "$HOST_CORPUS/.reprocess-backup-$STAMP"
cp -a "$HOST_CORPUS/search" "$HOST_CORPUS/.reprocess-backup-$STAMP/" 2>/dev/null || true
cp -a "$HOST_CORPUS/run_"* "$HOST_CORPUS/.reprocess-backup-$STAMP/" 2>/dev/null || true

# Remove only derived layers; keep transcripts/.
rm -rf "$HOST_CORPUS/search"
find "$HOST_CORPUS" -type d -name metadata -prune -exec rm -rf {} +

# Re-run from existing transcripts; never transcribe again.
.venv/bin/python -m podcast_scraper.cli \
  --config "$HOST_CORPUS/viewer_operator.yaml" \
  --feeds-spec "$HOST_CORPUS/feeds.spec.yaml" \
  --output-dir "$HOST_CORPUS" \
  --skip-existing \
  --no-transcribe-missing

# Optional post-step when vectors exist
.venv/bin/python -m podcast_scraper.cli topic-clusters --output-dir "$HOST_CORPUS"
```

If this run creates root-owned files (for example after mixed host/container history),
fix ownership before backup jobs:

```bash
sudo chown -R deploy:docker "$HOST_CORPUS"
```

## Daily commands

```bash
# Confirm deploy history
gh run list --workflow deploy-prod.yml --repo chipi/podcast_scraper --limit 5

# Manual deploy (latest main image tags; confirm token required since #796)
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f confirm=PROD_DEPLOY

# Post-deploy smoke over tailnet (same probes as deploy-prod workflow)
export PROD_TAILNET_FQDN=prod-podcast.<tailnet>.ts.net
# path= must be the in-container corpus root (PODCAST_DEFAULT_CORPUS_PATH), not the host bind path
make smoke-prod SMOKE_CORPUS_PATH=/app/output

# Pre-deploy corpus path check (SSH; same as deploy-prod preflight step)
bash scripts/ops/preflight_prod_corpus_path.sh deploy@prod-podcast.<tailnet>.ts.net

# Confirm backup freshness + weekly restore verify (#798)
gh run list --workflow backup-corpus-prod.yml --repo chipi/podcast_scraper --limit 5
gh run list --workflow verify-backup-restore.yml --repo chipi/podcast_scraper --limit 5
gh release list --repo chipi/podcast_scraper-backup --limit 10 | rg snapshot-prod-
```

## Health checks by context

```bash
# Laptop or CI (authoritative external check)
curl -fsS https://prod-podcast.<tailnet>.ts.net/api/health | jq .

# VPS host (run check inside api container)
ssh deploy@prod-podcast.<tailnet>.ts.net \
  'cd /srv/podcast-scraper && docker compose --env-file /srv/podcast-scraper/.env -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml exec -T api curl -fsS http://127.0.0.1:8000/api/health'

# Viewer path on host loopback
ssh deploy@prod-podcast.<tailnet>.ts.net \
  'curl -fsS http://127.0.0.1:${VIEWER_PORT:-8080}/api/health'
```

## Fast incident playbook

### Deploy red

1. Open latest `deploy-prod.yml` run logs.
2. If pull failed, rerun workflow.
3. If container start failed, pin previous `sha-<short>` image tag and `up -d`.

### Viewer up but pipeline failing

1. Check latest job logs under corpus jobs.
2. Verify key env names in `/srv/podcast-scraper/.env`.
3. Verify api container has non-empty provider env vars.

### Host unreachable over tailnet

1. Check `tailscale status` on your laptop.
2. Try hostname with numeric suffix (`prod-podcast-1`, `prod-podcast-2`).
3. Use Hetzner console for recovery if needed.

## Most important secrets

- `HCLOUD_TOKEN`: Hetzner API (infra apply).
- `OPERATOR_SSH_PUBLIC_KEY`: operator laptop pubkey for OpenTofu (repo **secret** so CI logs mask it).
- `TS_AUTHKEY`: tailnet join auth for workflows and machine registration.
- `TS_API_KEY`: tailnet management API for Terraform provider.
- `TFSTATE_AGE_KEY`: decrypts encrypted OpenTofu state.
- Host `.env`: provider API keys, Grafana credentials, Sentry DSNs.

## Rotation rhythm

- Tailscale keys on Free plan: rotate before 90-day expiry.
- Hetzner token: rotate after personnel/device changes or suspicious activity.
- Age key: rotate carefully with state re-encryption and backup first.
- Provider keys and auth passwords: rotate on incident or access changes.

## Rollback and DR shortcuts

```bash
# Roll back app images by known-good short sha
ssh deploy@prod-podcast.<tailnet>.ts.net
cd /srv/podcast-scraper
PODCAST_IMAGE_TAG=sha-<previous-good-short-sha> \
  docker compose -f compose/docker-compose.stack.yml \
                 -f compose/docker-compose.prod.yml \
                 -f compose/docker-compose.vps-prod.yml \
  up -d --remove-orphans

# Restore corpus from backup release (prod layout)
# Preferred: prod-restore-corpus.yml in GitHub Actions (confirm PROD_RESTORE).
# On-host Make only: make restore-corpus-prod, then recycle api + viewer (see PROD_RUNBOOK).
make restore-corpus-prod
```

## When to use full runbook

- First bootstrap of prod host.
- Credential rotation procedures.
- Grafana/Sentry setup and validation.
- Tailscale suffix drift and ACL edits.
- Disaster recovery drill and full re-provision.
