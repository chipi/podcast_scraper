# Prod operator cheat sheet

Fast companion to the full
[Prod runbook](PROD_RUNBOOK.md). Use this for day-to-day operations.

## Scope

- Prod host: Hetzner VPS
- Access path: tailnet only (`prod-podcast.<tailnet>.ts.net`)
- Deploy model: GitOps (`main` + green CI -> `deploy-prod.yml`)
- Full procedures and edge cases: see [Prod runbook](PROD_RUNBOOK.md)

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
- Corpus path: `/srv/podcast-scraper/corpus`

## Daily commands

```bash
# Confirm deploy history
gh run list --workflow deploy-prod.yml --repo chipi/podcast_scraper --limit 5

# Manual deploy (latest main image tags)
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper

# Confirm backup freshness
gh run list --workflow backup-corpus-prod.yml --repo chipi/podcast_scraper --limit 5
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
- `TS_AUTHKEY`: tailnet join auth for workflows and machine registration.
- `TS_API_KEY`: tailnet management API for Terraform provider.
- `TFSTATE_AGE_KEY`: decrypts encrypted OpenTofu state.
- Host `.env`: provider API keys, Grafana credentials, Sentry DSNs.
- `.htpasswd`: viewer basic-auth credentials.

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

# Restore corpus from latest backup release
make restore-corpus
```

## When to use full runbook

- First bootstrap of prod host.
- Credential rotation procedures.
- Grafana/Sentry setup and validation.
- Tailscale suffix drift and ACL edits.
- Disaster recovery drill and full re-provision.
