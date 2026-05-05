# Prod runbook (always-on Hetzner VPS)

Operator-facing runbook for the production deploy defined in
[RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md). The RFC
describes *what we decided*; this runbook describes *what to do today*.

> **For the prerequisites checklist** (Hetzner account + Tailscale credentials —
> auth key + API access token on Free plan, see "Tailscale credentials" below
> for the why — sops/age + GHA secrets), see
> [#714](https://github.com/chipi/podcast_scraper/issues/714).
> All commands below assume those are done.

## Sections

1. [First-time bootstrap](#first-time-bootstrap) — includes [API health checks by context](#api-health-checks-by-context) (GH-745)
2. [Daily operations](#daily-operations)
3. [Basic-auth credentials](#basic-auth-credentials)
4. [Corpus migration from pre-prod (Codespace) to prod (VPS)](#corpus-migration)
5. [Rollback (deploy went red mid-way)](#rollback)
6. [Disaster recovery (VPS gone)](#disaster-recovery)
7. [Credential rotation](#credential-rotation)
8. [Environment variable reference](#environment-variable-reference)
9. [Observability setup walkthrough](#observability-setup-walkthrough)
10. [Tailscale operations](#tailscale-operations)
11. [Hetzner operations](#hetzner-operations)
12. [Operator hot-fix workflow](#operator-hot-fix-workflow)
13. [FAQ / Troubleshooting](#faq-troubleshooting)
14. [Constraints to know](#constraints-to-know)

---

## First-time bootstrap

One-shot setup that takes prod from "nothing" to "viewer reachable on the
tailnet". ~30–45 min wall.

### Pre-bootstrap (one-time, on operator's laptop)

```bash
# 1. Tools
brew install opentofu sops age actionlint shellcheck
gh auth login

# 2. age keypair for sops state encryption
age-keygen -o ~/.config/sops/age/keys.txt
# Copy the public key (the `# public key:` line) into infra/.sops.yaml,
# replacing the `age1PLACEHOLDER…` value. Save the PRIVATE key contents to
# your password manager as `sops/podcast-scraper/tofu-state-age-key`.

# 3. Stage GHA secrets (from #714) — Free-plan auth: see "Tailscale credentials"
#    section below for why TS_AUTHKEY + TS_API_KEY (two separate creds) instead
#    of OAuth client ID/secret (which would be on Premium+ plans).
gh secret set HCLOUD_TOKEN          --repo chipi/podcast_scraper --app actions --body '<token>'
gh secret set TS_AUTHKEY            --repo chipi/podcast_scraper --app actions --body '<tskey-auth-...>'
gh secret set TS_API_KEY            --repo chipi/podcast_scraper --app actions --body '<tskey-api-...>'
gh secret set TFSTATE_AGE_KEY       --repo chipi/podcast_scraper --app actions --body "$(cat ~/.config/sops/age/keys.txt)"
gh secret set BACKUP_REPO_TOKEN     --repo chipi/podcast_scraper --app actions --body '<backup-repo-pat>'

# 4. Stage GHA repo variables
gh variable set OPERATOR_SSH_PUBLIC_KEY --repo chipi/podcast_scraper --body "$(cat ~/.ssh/id_ed25519.pub)"
gh variable set TAILNET_NAME            --repo chipi/podcast_scraper --body 'tail-xxxxx.ts.net'
# PROD_TAILNET_FQDN is set after first apply (it depends on the assigned hostname);
# default value is "prod-podcast.<TAILNET_NAME>".
```

### First `tofu apply` (operator's laptop)

```bash
cd infra
export HCLOUD_TOKEN=$(op read 'op://Personal/Hetzner Cloud/podcast-scraper-prod/api-token')
export TF_VAR_hcloud_token="$HCLOUD_TOKEN"
export TF_VAR_tailscale_api_key=$(op read 'op://Personal/Tailscale/podcast-scraper/api-key')
export TF_VAR_tailscale_tailnet="tail-xxxxx.ts.net"   # your tailnet
export TF_VAR_ssh_public_key="$(cat ~/.ssh/id_ed25519.pub)"

./tofu init
./tofu plan
./tofu apply
# Outputs: server_id, ipv4_address, tailnet_url, ssh_target.
```

**Expected resources (per [#716](https://github.com/chipi/podcast_scraper/issues/716)):**
1 server + 1 firewall + 1 network + 1 SSH key + 1 Tailscale auth key, plus an
optional Volume + attachment if `volume_size_gb > 0`.

After apply, set the GHA variable:

```bash
gh variable set PROD_TAILNET_FQDN --repo chipi/podcast_scraper \
  --body "prod-podcast.tail-xxxxx.ts.net"
```

### When the live hostname has a numeric suffix (`-1`, `-2`, …) {#tailscale-suffix-drift}

After a failed replace or a stale machine record, Tailscale may keep the
MagicDNS name `prod-podcast.<tailnet>` on an **offline** orphan while the
live VPS registers as `prod-podcast-1.<tailnet>` (or `-2`, and so on). That
breaks copy-paste SSH and `curl` until the name lines up again.

**GitHub Actions:** `deploy-prod.yml` and `backup-corpus-prod.yml` join the
tailnet, run `scripts/ops/resolve_prod_tailnet_host.sh`, and use the resolved
FQDN for SSH and the `/api/health` probe. Workflows still require
`vars.PROD_TAILNET_FQDN` as the operator’s **canonical** intent; the resolver
falls back to `prod-podcast-1.<tailnet>`, … when the canonical name is not
online.

**Local laptop (on the tailnet):** print the live host the repo workflows
would pick:

```bash
export PROD_TAILNET_FQDN='prod-podcast.tail-xxxxx.ts.net'
bash scripts/ops/resolve_prod_tailnet_host.sh
```

For tests without `tailscaled`, set `TAILSCALE_STATUS_JSON_PATH` to a saved
`tailscale status --json` file.

When the printed name differs from the repo variable, update the variable so
logs and docs match reality, and remove stale machines in the [Tailscale
admin machines](https://login.tailscale.com/admin/machines) list if you want
the unsuffixed name back. See [GitHub issue
744](https://github.com/chipi/podcast_scraper/issues/744).

### Stage the host-side `.env` (one-time, post-apply)

The `.env` holds runtime secrets (provider API keys, Grafana credentials,
Sentry DSN). Cloud-init drops a sentinel file `/srv/podcast-scraper/.bootstrap-needs-env`;
the systemd unit refuses to start while it exists.

> See **[Environment variable reference](#environment-variable-reference)**
> below for the *complete* list with intent + format for each var. The
> heredoc here is the minimum to boot — anything missing from it must be
> added before the relevant subsystem (Sentry, Grafana, the LLM pipeline,
> etc.) will function. **The exact variable names matter** — typos like
> `OPENAI_KEY` instead of `OPENAI_API_KEY` cost an hour of debugging the
> first time around.

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net   # over Tailscale
# `deploy` user has no sudo (cloud-init: sudo: false) but owns
# /srv/podcast-scraper, so write directly — no `sudo install` needed.
install -m 600 /dev/stdin /srv/podcast-scraper/.env <<'ENV'
# === Required: ingress + paths ===
PODCAST_DOCKER_PROJECT_DIR=/srv/podcast-scraper
PODCAST_CORPUS_HOST_PATH=/srv/podcast-scraper/corpus
PODCAST_HTPASSWD_PATH=/srv/podcast-scraper/.htpasswd
PODCAST_ENV=prod
PODCAST_AVAILABLE_PROFILES=cloud_balanced,cloud_thin
PODCAST_DEFAULT_PROFILE=cloud_balanced

# === LLM provider keys (cloud_balanced uses openai + gemini) ===
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
# Optional providers — set only if you intend to use them
ANTHROPIC_API_KEY=sk-ant-...

# === Grafana Cloud — separate Prom + Loki user IDs ===
# (Single GRAFANA_CLOUD_USER no longer works; see env var reference.)
GRAFANA_CLOUD_PROM_URL=https://prometheus-prod-NN-prod-REGION.grafana.net/api/prom/push
GRAFANA_CLOUD_LOKI_URL=https://logs-prod-NNN.grafana.net/loki/api/v1/push
GRAFANA_CLOUD_PROM_USER=NNNNNNN
GRAFANA_CLOUD_LOKI_USER=NNNNNNN
GRAFANA_CLOUD_API_KEY=glc_eyJ...

# === Sentry DSNs — one per project ===
PODCAST_SENTRY_DSN_API=https://...@o....ingest.de.sentry.io/...
PODCAST_SENTRY_DSN_PIPELINE=https://...@o....ingest.de.sentry.io/...
ENV

# Stage the basic-auth password file too (see #Basic-auth credentials below).
sudo htpasswd -bcB /srv/podcast-scraper/.htpasswd <user> <pass>
sudo chown root:docker /srv/podcast-scraper/.htpasswd
sudo chmod 640         /srv/podcast-scraper/.htpasswd

# Release the systemd gate.
sudo rm /srv/podcast-scraper/.bootstrap-needs-env
sudo systemctl restart podcast-scraper.service
```

### API health checks by context (GH-745) {#api-health-checks-by-context}

Use the right URL for each layer. Mixing them causes false alarms (for
example `curl http://127.0.0.1:8000/api/health` on the **VPS host** can fail
while the API container and the tailnet URL are healthy, because compose
**does not publish** api port `8000` to the host — only `expose` for the
Docker network).

| Context | Authoritative check | Notes |
| --- | --- | --- |
| **Compose / inside `api` container** | `curl -fsS http://127.0.0.1:8000/api/health` | Same as `compose/docker-compose.stack.yml` `healthcheck` and `infra/deploy/deploy.sh` after #745. |
| **On the VPS host over SSH** | `docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml -f compose/docker-compose.vps-prod.yml exec -T api curl -fsS http://127.0.0.1:8000/api/health` | Runs the check **inside** the api netns. From `/srv/podcast-scraper`, a short form is `docker compose exec -T api curl -fsS http://127.0.0.1:8000/api/health` if your shell already exports the same `-f` list as systemd. |
| **Host loopback via viewer (nginx → api)** | `curl -fsS http://127.0.0.1:${VIEWER_PORT:-8080}/api/health` | `VIEWER_PORT` defaults to `8080` (`compose/docker-compose.stack.yml`). Exercises the same path as much of the UI; prod nginx leaves `/api/health` **without** basic auth. |
| **Laptop / CI on the tailnet** | `curl -fsS https://prod-podcast.<tailnet>/api/health` | MagicDNS + `tailscale serve`; this is what `deploy-prod.yml` probes after deploy. |

**Prefer** tailnet or container-local checks when triaging production. Treat
host `:8000` alone as invalid unless you have added an explicit `ports:` map
for `api` (not in the stock compose files).

### Smoke validation

Use the [health-check table](#api-health-checks-by-context) above so
each step hits the intended layer.

```bash
# 1. Tailnet reachability
curl -fsS https://prod-podcast.tail-xxxxx.ts.net/api/health | jq .
# Expected: {"status":"ok","feeds_api":true,...}

# 2. Basic auth (closes #713 verification)
curl -i https://prod-podcast.tail-xxxxx.ts.net/                     # expect 401
curl -i https://prod-podcast.tail-xxxxx.ts.net/welcome              # expect 200 + landing.html
curl -fsS -u user:pass https://prod-podcast.tail-xxxxx.ts.net/      # expect 200 + SPA HTML

# 3. grafana-agent shipping
ssh deploy@prod-podcast.tail-xxxxx.ts.net 'docker logs compose-grafana-agent-1 --tail 20'

# 4. Sentry validation ping
ssh deploy@prod-podcast.tail-xxxxx.ts.net \
  'docker exec compose-api-1 python -c "
from podcast_scraper.utils.sentry_init import init_sentry
import sentry_sdk; init_sentry(\"api\")
sentry_sdk.capture_message(\"prod bootstrap validation ping\", level=\"info\")
"'
# Check sentry.io within ~1 min for the event under environment=prod.

# 5. Grafana Cloud query (~30 s after agent's first scrape)
#    https://<org>.grafana.net → Explore → Prometheus
#    Query:  up{component="api",env="prod"}
#    Expected: 1 series, value=1
```

### Trigger the first deploy

```bash
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper
gh run watch --repo chipi/podcast_scraper
```

Once a few green deploys have passed and you trust the loop, file the
follow-up PR that flips `deploy-prod.yml` to also auto-trigger on
`workflow_run: ["Stack test"]` (RFC-082 Decision 6 GitOps loop).

---

## Daily operations

### Where to look first

| Symptom | Where |
| --- | --- |
| Viewer slow / unreachable | `gh run list --workflow deploy-prod.yml --limit 3` then Sentry → environment=prod |
| Pipeline run failing | Sentry → environment=prod, component=pipeline; viewer Library → Job logs |
| Deploy went red | GHA UI → Deploy to prod VPS → most recent run; api logs are dumped on health-check failure |
| "Did the alert fire because of X?" | Grafana Cloud → podcast-scraper folder → filter `env=prod` |

### Manual deploy

```bash
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f override_image_sha=                       # blank = deploy current main
# or pin to a specific image:
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f override_image_sha=abc1234
```

### Pipeline run via the viewer

Standard flow — open the viewer, hit Configuration → Run pipeline. Same
control plane as pre-prod. Profile dropdown is restricted to
`cloud_balanced,cloud_thin` (no ML profiles in prod, per RFC-082).

### Backup status

```bash
gh run list --workflow backup-corpus-prod.yml --repo chipi/podcast_scraper --limit 5
gh release list --repo chipi/podcast_scraper-backup --limit 10 | grep snapshot-prod-
```

---

## Basic-auth credentials

The viewer is gated by HTTP Basic Auth (#713). Credentials are stored in
`/srv/podcast-scraper/.htpasswd` on the VPS; nginx reads via the bind-mount
in `compose/docker-compose.vps-prod.yml`.

### Initial setup

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
sudo htpasswd -bcB /srv/podcast-scraper/.htpasswd <user> <pass>
sudo chown root:docker /srv/podcast-scraper/.htpasswd
sudo chmod 640         /srv/podcast-scraper/.htpasswd
sudo systemctl restart podcast-scraper.service   # picks up the new file
```

The `-c` flag CREATES the file (overwrites if present). Use `-b` only on
subsequent invocations to add/update users without erasing existing ones.

### Add a collaborator

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
sudo htpasswd -bB /srv/podcast-scraper/.htpasswd <new-user> <new-pass>   # NO -c
sudo systemctl restart podcast-scraper.service
```

### Remove a user

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
sudo htpasswd -D /srv/podcast-scraper/.htpasswd <user>
sudo systemctl restart podcast-scraper.service
```

### Programmatic callers

Home Assistant, Slack→GHA→api, and other headless callers send standard
HTTP Basic Auth headers:

```bash
curl -fsS -u "$HA_USER:$HA_PASS" https://prod-podcast.tail-xxxxx.ts.net/api/jobs
```

`/api/health` is intentionally left unauthed so the deploy workflow's external
probe (#720) can hit it without juggling credentials.

---

## Corpus migration

One-time migration on cutover day. Use the latest snapshot from the backup
repo rather than streaming files between hosts (more reliable; matches RFC-082
Decision 4).

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
cd /srv/podcast-scraper
make restore-corpus               # pulls latest snapshot.tgz from
                                  # chipi/podcast_scraper-backup, untars
                                  # into /srv/podcast-scraper/corpus/

# Verify
ls -la corpus/feeds/ | head
find corpus -name '*.gi.json' | wc -l
docker compose -f compose/docker-compose.stack.yml \
               -f compose/docker-compose.prod.yml \
               -f compose/docker-compose.vps-prod.yml \
  restart api viewer

# In the viewer (over Tailscale): Library tab should now show all
# episodes from the snapshot.
```

After cutover, future backups come from the VPS via
[`backup-corpus-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/backup-corpus-prod.yml).

---

## Rollback

`deploy-prod.yml` runs `infra/deploy/deploy.sh`, which does
`git pull && docker compose pull && docker compose up -d`. Three failure modes:

### Failure 1: image pull failed (network blip / GHCR auth issue)

**Symptom:** workflow's deploy step exits non-zero on `compose pull`.
**Recovery:** re-run the workflow. No state changed; old containers still
running with old images.

### Failure 2: pull succeeded but a container won't start

**Symptom:** workflow exits non-zero on `compose up`. New images on disk; old
container gone (compose recreates by default).
**Recovery:** SSH in, manually pin the previous image:

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
cd /srv/podcast-scraper
PODCAST_IMAGE_TAG=sha-<previous-good-short-sha> \
  docker compose -f compose/docker-compose.stack.yml \
                 -f compose/docker-compose.prod.yml \
                 -f compose/docker-compose.vps-prod.yml \
    up -d --remove-orphans
```

The `:sha-<short>` tags from the publish job are the rollback target — they're
never garbage-collected.

### Failure 3: stack up but functionally broken

**Symptom:** workflow green, but operator notices the bug from the viewer.
**Recovery:** same as Failure 2. Then file a bug + ship a fix-forward via the
hotfix path.

---

## Disaster recovery

If the Hetzner instance is irrecoverable (account issue, hardware failure,
accidental `tofu destroy`):

```bash
# 1. Re-provision (~5–10 min)
cd infra
./tofu apply              # same hostname, same Tailscale registration
                          # (cloud-init re-runs the bootstrap)

# 2. Restore corpus (~3–5 min for ~20 MB snapshot)
ssh deploy@prod-podcast.tail-xxxxx.ts.net 'cd /srv/podcast-scraper && make restore-corpus'

# 3. Re-stage host-side .env  + .htpasswd (operator's responsibility)
#    See "First-time bootstrap → Stage the host-side .env"
#    and "Basic-auth credentials → Initial setup".

# 4. Verify (~5 min)
#    See "Smoke validation".
```

**Total wall time: ~15–20 min** assuming the operator knows the runbook.
Corpus is recoverable to within ~24 h of pre-disaster state (last
`backup-corpus-prod.yml` run).

[#724](https://github.com/chipi/podcast_scraper/issues/724) tracks the
end-to-end DR drill that calibrates these numbers against reality.

---

## Credential rotation

### Hetzner API token

1. Hetzner console → Security → API Tokens → Generate new (Read+Write).
2. `gh secret set HCLOUD_TOKEN --repo chipi/podcast_scraper --app actions --body '<new>'`
3. Update your password manager entry `Hetzner Cloud / podcast-scraper-prod / api-token`.
4. Revoke the old token in the Hetzner console.
5. Trigger one workflow_dispatch run of `infra-apply.yml` to confirm the new
   token works (will be a no-op apply if no infra changes).

### Tailscale credentials (Free-plan workaround)

OAuth clients are gated to Tailscale Premium+ tiers. On Personal Free we use
**two separate credentials** instead, both rotated independently:

| Credential | Purpose | Where it's used | Where to generate |
| --- | --- | --- | --- |
| `TS_AUTHKEY` | Joins the GHA runner / VPS to the tailnet (device-level auth) | `tailscale/github-action@v2` in deploy-prod.yml + backup-corpus-prod.yml; cloud-init's `tailscale up` | [admin/settings/keys](https://login.tailscale.com/admin/settings/keys) → **Auth keys** tab |
| `TS_API_KEY` | Authenticates terraform's `tailscale` provider to the management API (creates per-server auth keys, syncs ACL) | infra-ci.yml + infra-apply.yml's `TF_VAR_tailscale_api_key` | [admin/settings/keys](https://login.tailscale.com/admin/settings/keys) → **API access tokens** tab |

Both expire at most every 90 days on Free plan — Tailscale doesn't allow
non-expiring keys. Set a calendar reminder.

**Rotating `TS_AUTHKEY`:**

1. [admin/settings/keys](https://login.tailscale.com/admin/settings/keys) → Auth keys → Generate new (Reusable, Ephemeral, Pre-approved, Tags: `tag:gha-deployer`).
2. `gh secret set TS_AUTHKEY --repo chipi/podcast_scraper --app actions --body '<tskey-auth-...>'`
3. Update your password manager (entry: `Tailscale / podcast-scraper / authkey`).
4. Revoke the old auth key in the Tailscale admin.
5. Trigger one workflow_dispatch run of `deploy-prod.yml` to confirm.

**Rotating `TS_API_KEY`:**

1. [admin/settings/keys](https://login.tailscale.com/admin/settings/keys) → API access tokens → Generate new.
2. `gh secret set TS_API_KEY --repo chipi/podcast_scraper --app actions --body '<tskey-api-...>'`
3. Update your password manager (entry: `Tailscale / podcast-scraper / api-key`).
4. Revoke the old token in the Tailscale admin.
5. Trigger one workflow_dispatch run of `infra-apply.yml` to confirm (no-op apply if no infra changes).

### age key (sops state)

> **Caution:** the age key encrypts the OpenTofu state. Losing it without a
> backup means the encrypted state in `infra/terraform/terraform.tfstate.enc`
> is unrecoverable; you'd need to import all live Hetzner resources into a
> fresh state. Always back the new key up before retiring the old one.

1. Generate new keypair: `age-keygen -o ~/.config/sops/age/keys.txt.new`
2. Decrypt current state with the OLD key, re-encrypt with the NEW key:

   ```bash
   cd infra
   ./tofu state pull > /tmp/state.json   # via the wrapper, OLD key
   # Update infra/.sops.yaml to the NEW public key
   sops -e /tmp/state.json > infra/terraform/terraform.tfstate.enc
   shred /tmp/state.json
   ```

3. `gh secret set TFSTATE_AGE_KEY --repo chipi/podcast_scraper --app actions --body "$(cat ~/.config/sops/age/keys.txt.new)"`
4. Update your password manager (keep the OLD key for ~30 days as a safety net).
5. After ~30 days of green CI, delete the OLD key.

### Host `.env` secrets (provider API keys, Sentry DSN, etc.)

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
sudo sed -i 's|^OPENAI_API_KEY=.*|OPENAI_API_KEY=<new>|' /srv/podcast-scraper/.env
sudo systemctl restart podcast-scraper.service
```

For multiple keys, edit the `.env` directly: `sudo -e /srv/podcast-scraper/.env`.

### Basic-auth password

See [Basic-auth credentials → Add a collaborator](#basic-auth-credentials):
overwrite the user with `htpasswd -bB`, then restart the service.

---

## Environment variable reference

The host-side `/srv/podcast-scraper/.env` file is the single source of
truth for runtime config on the VPS. Owned by `deploy:deploy`, mode
`600`. Loaded by:

- the systemd unit (`EnvironmentFile=` — runs on boot + on `systemctl restart`)
- the explicit `--env-file` flag we pass when running `docker compose`
  directly (see [Operator hot-fix workflow](#operator-hot-fix-workflow))
- the api code's nested `docker compose run pipeline-llm` (auto-passes
  `--env-file` if the project root has a `.env` — see
  `pipeline_docker_factory.py`)

### Required vars

| Var | Format | Purpose | Notes if missing |
| --- | --- | --- | --- |
| `PODCAST_DOCKER_PROJECT_DIR` | absolute path | Repo path inside the api container (must match host because of bind mount) | api refuses to start; volume interpolation fails |
| `PODCAST_CORPUS_HOST_PATH` | absolute path | Where the corpus lives on the host | api spawn fails: `volumes.corpus_data.driver_opts.device: required variable PODCAST_CORPUS_HOST_PATH is missing a value` |
| `PODCAST_HTPASSWD_PATH` | absolute path | Bind-mount source for nginx `auth_basic` | viewer container fails to start (mount source missing) |
| `PODCAST_ENV` | `prod` | Tags Sentry + Grafana labels with `env=prod` | events tagged `env=preprod` (default) — wrong dashboard filtering |
| `PODCAST_AVAILABLE_PROFILES` | csv | Profile dropdown allowlist | dropdown shows ALL on-disk profiles, including ones whose images aren't published in prod (run will fail mid-job) |
| `PODCAST_DEFAULT_PROFILE` | string | Preselected profile in the viewer | dropdown opens unselected; api 400s if Run hit before Save |

### LLM provider keys (Docker job mode)

These are read by the **pipeline container**, not the api. The api passes
them through via its `environment:` block in
`compose/docker-compose.prod.yml` so the nested `docker compose run`
inherits them.

| Var | Required when | Format |
| --- | --- | --- |
| `OPENAI_API_KEY` | profile uses openai (cloud_balanced default) | `sk-...` |
| `GEMINI_API_KEY` | profile uses gemini (cloud_balanced default for diarization + summary) | `AIza...` |
| `ANTHROPIC_API_KEY` | profile uses anthropic | `sk-ant-...` |
| `MISTRAL_API_KEY` | profile uses mistral | provider-specific |
| `DEEPSEEK_API_KEY` | profile uses deepseek | provider-specific |
| `GROK_API_KEY` | profile uses grok | provider-specific |

**Variable name pitfall:** the code expects `OPENAI_API_KEY` and
`GEMINI_API_KEY` — *with* `_API_` in the middle. Names like `OPENAI_KEY`
or `GEMINI_KEY` silently resolve to empty defaults at compose parse
time, then the pipeline crashes on first provider call with
`OpenAI API key required for OpenAI providers`. Use `grep -E
'^(OPENAI|GEMINI)_API_KEY=' /srv/podcast-scraper/.env` to confirm.

### Observability vars (Grafana Cloud + Sentry)

| Var | Where to get it | Notes |
| --- | --- | --- |
| `GRAFANA_CLOUD_PROM_URL` | Grafana stack page → Prometheus → Send Metrics → "Remote Write Endpoint" | URL ends in `/api/prom/push` |
| `GRAFANA_CLOUD_LOKI_URL` | Grafana stack page → Loki → Send Logs → "Endpoint" | URL ends in `/loki/api/v1/push` |
| `GRAFANA_CLOUD_PROM_USER` | Same Prometheus page → "Username / Instance ID" (numeric) | Distinct from Loki user — Grafana issues a separate instance ID per service |
| `GRAFANA_CLOUD_LOKI_USER` | Same Loki page → "User" (numeric) | Same pattern as Prom user |
| `GRAFANA_CLOUD_API_KEY` | Grafana Cloud → Access Policies → New policy with `metrics:write` + `logs:write` → Add token | Single token authenticates both services; format `glc_eyJ...` |
| `PODCAST_SENTRY_DSN_API` | Sentry → api project → Settings → Client Keys (DSN) | URL like `https://<key>@o<org>.ingest.de.sentry.io/<project>` |
| `PODCAST_SENTRY_DSN_PIPELINE` | Sentry → pipeline project → Settings → Client Keys (DSN) | Different project from api so issues stay separable |

### Optional / advanced vars

| Var | Default | Effect when set |
| --- | --- | --- |
| `PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS` | `llm` (in prod overlay) | Host-wide fallback when operator YAML omits `pipeline_install_extras`. Must be `ml` or `llm` |
| `PODCAST_IMAGE_TAG` | `main` | Pin image to a specific tag (`sha-<short>` for rollback) |
| `PODCAST_RELEASE` | empty | Sentry release tag (defaults to image SHA via build-time injection) |
| `PODCAST_METRICS_ENABLED` | `1` | Set to `0` to disable api `/metrics` exposition |
| `VIEWER_PORT` | `8080` | Host port for the viewer container; only matters before `tailscale serve` is set up |

### Verifying what's actually loaded

```bash
# Names only (no values), sorted, deduped
ssh deploy@prod-podcast.<tailnet> \
  "awk -F= '/^[A-Z_][A-Z0-9_]*=/ {print \$1}' /srv/podcast-scraper/.env | sort -u"

# Confirm a specific var made it into the running api container
ssh deploy@prod-podcast.<tailnet> \
  'docker exec compose-api-1 sh -c "for k in OPENAI_API_KEY GEMINI_API_KEY PODCAST_CORPUS_HOST_PATH; do
     v=\$(printenv \"\$k\"); [ -z \"\$v\" ] && echo \"\$k=<EMPTY>\" || echo \"\$k=<set, length=\${#v}>\"
   done"'
```

---

## Observability setup walkthrough

### Grafana Cloud (one-time, per-stack)

1. Create a free Grafana Cloud account at <https://grafana.com/auth/sign-up>.
2. Default stack is created on signup. Note the region (`us`, `eu-west-2`,
   etc.) — it's baked into every endpoint URL.
3. **Get Prometheus credentials:** stack page → click **Prometheus** →
   **Send Metrics**:
   - Copy "Remote Write Endpoint" → `GRAFANA_CLOUD_PROM_URL`
   - Copy "Username / Instance ID" (numeric, ~7 digits) → `GRAFANA_CLOUD_PROM_USER`
4. **Get Loki credentials:** stack page → **Loki** → **Send Logs**:
   - Copy "Endpoint" → `GRAFANA_CLOUD_LOKI_URL`
   - Copy "User" (numeric) → `GRAFANA_CLOUD_LOKI_USER`
5. **Generate the write token:** grafana.com top nav → **Access Policies**
   → **Create access policy**:
   - Name: `podcast-scraper-agent-prod-write` (or similar — easy to revoke later)
   - Realm: your stack
   - Scopes: check `metrics:write` AND `logs:write`
   - Save → click into the policy → **Add token** → name it →
     **Generate** → copy the `glc_eyJ...` value → `GRAFANA_CLOUD_API_KEY`
   - **The token is shown ONCE.** Save to your password manager
     immediately.
6. After staging vars + recreating `grafana-agent` (see [Operator hot-fix
   workflow](#operator-hot-fix-workflow)), verify within 2 min:
   - Grafana Cloud → **Explore** → **Prometheus** → query
     `up{component="api",env="prod"}` → expect a single series, value `1`
   - Grafana Cloud → **Explore** → **Loki** → query `{env="prod"}` →
     expect log lines from `api`, `viewer`, `grafana-agent`

### Sentry (one-time, per project)

1. Create free Sentry account at <https://sentry.io/signup/>.
2. Create **two** projects:
   - `podcast-scraper-api` — platform Python (FastAPI auto-detected)
   - `podcast-scraper-pipeline` — platform Python
3. For each project: **Settings** → **Client Keys (DSN)** → copy the
   **DSN** value (full `https://<key>@o<org>.ingest.<region>.sentry.io/<project>`).
4. Stage as `PODCAST_SENTRY_DSN_API` and `PODCAST_SENTRY_DSN_PIPELINE`
   in `.env`.
5. After api recreate, verify with the **Sentry validation ping** in
   the Smoke validation block above. Expect the event in the api
   project under `environment=prod` within ~1 min.

### Viewer Sentry (build-time DSN, runtime env)

The viewer SPA's Sentry DSN is baked into the bundle at build time
(`VITE_SENTRY_DSN_VIEWER` build-arg). The `env=` tag is injected at
*request time* by nginx `sub_filter` reading the container's
`PODCAST_ENV` env (so one viewer image serves prod + preprod with
correct tags).

To wire viewer Sentry (one-time):

1. Create a third Sentry project: `podcast-scraper-viewer` (platform
   JavaScript / Vue).
2. Copy the DSN.
3. Set GHA repo secret: `gh secret set VITE_SENTRY_DSN_VIEWER --repo
   chipi/podcast_scraper --app actions --body 'https://...'`.
4. Wait for the next push to `main` — `stack-test.yml`'s viewer
   publish step bakes the DSN into the new image.
5. Pull + restart on prod: `gh workflow run deploy-prod.yml`.

DSNs are write-only public tokens designed to ship with frontend code
— baking into the public GHCR image is the standard pattern.

---

## Tailscale operations

### Reaching the VPS

The VPS joins the tailnet at provision time (cloud-init runs
`tailscale up --auth-key=$TS_AUTHKEY --hostname=prod-podcast`). On
your laptop:

```bash
tailscale status               # confirm laptop is on the tailnet
ssh deploy@prod-podcast.<tailnet>.ts.net
# -OR by tailnet IP (rarely needed; MagicDNS handles names):
ssh deploy@$(tailscale ip -4 prod-podcast)
```

If the hostname doesn't resolve: a prior failed deploy may have left
an orphan device on the tailnet, with the live VPS auto-named
`prod-podcast-1` (or `-2`, etc.). Check `tailscale status | grep prod-podcast`
and use whatever the actual current name is, or run
`bash scripts/ops/resolve_prod_tailnet_host.sh` with
`PROD_TAILNET_FQDN` set to your canonical value (see [When the live hostname
has a numeric suffix](#tailscale-suffix-drift)).
Clean up orphans in the [Tailscale admin
console](https://login.tailscale.com/admin/machines) to reclaim the canonical
name.

### HTTPS over the tailnet

`tailscale serve` exposes the in-container viewer (on port 8080) as
HTTPS port 443 with an auto-issued TLS cert from Tailscale's CA.
Set up once on the VPS:

```bash
ssh deploy@prod-podcast.<tailnet>
sudo tailscale serve --bg 8080
sudo tailscale serve status   # confirm: https://prod-podcast.<tailnet> → 127.0.0.1:8080
```

After this, `https://prod-podcast.<tailnet>/` works from any tailnet
device, with a real cert (no browser warnings).

### Editing the ACL

ACL lives in `tailscale/policy.hujson` in the repo. `infra-apply.yml`
syncs it to the live tailnet via the terraform provider. Edit, open
PR, merge → next infra-apply run pushes the change.

For ad-hoc / urgent ACL changes (e.g. adding a new device), the admin
console at <https://login.tailscale.com/admin/acls> allows direct edits
— but those will be overwritten by the next `infra-apply` run unless
you also commit them to the file.

### Tailscale auth key vs API key

Both expire ≤ 90 days on Free plan. See [Credential rotation →
Tailscale credentials](#tailscale-credentials-free-plan-workaround)
for the rotation flow. Reminder: set a calendar event for ~80 days out.

---

## Hetzner operations

### Console access

If the VPS becomes unreachable over Tailscale (tailscaled crashed,
network down, etc.), Hetzner's web console gives serial access:

1. <https://console.hetzner.cloud> → Servers → `podcast-scraper-prod`
2. **Console** tab (top right) → opens a noVNC session
3. Log in with the `deploy` user via SSH key (your local `~/.ssh/id_ed25519`
   was injected by cloud-init) — you'll need to upload it via "Send key"
   button or use root login if the deploy user is broken
4. Last-resort: **Rescue** mode boots a recovery system to `chroot` the
   real disk

### Volume management

Optional Hetzner Volume for the corpus is created when
`volume_size_gb > 0` in `infra/terraform/variables.tf`. cloud-init
auto-detects `/mnt/HC_Volume_*` and symlinks it as
`/srv/podcast-scraper/corpus`. Verify via `mount | grep HC_Volume`.

If you upsize the volume:

1. Hetzner console → Volumes → resize (online, no downtime)
2. SSH in: `sudo resize2fs /dev/disk/by-id/scsi-0HC_Volume_<id>`
3. Confirm with `df -h /srv/podcast-scraper/corpus`

### Firewall

Hetzner's cloud firewall (managed via terraform) allows ONLY:

- inbound UDP 41641 (Tailscale WireGuard)
- inbound ICMP (ping)
- outbound: all (for image pulls, package updates, provider API calls)

Note: there's NO public TCP 80/443 rule. The viewer is reachable
ONLY over the tailnet (via `tailscale serve`). Adding public ingress
requires editing `infra/terraform/main.tf` AND removing/relaxing
`auth_basic` in `nginx-prod.conf.template` (keep auth on for any
public-internet exposure).

### API token rotation

See [Credential rotation → Hetzner API token](#hetzner-api-token).

---

## Operator hot-fix workflow

When you've made a local change to a compose file or non-image-baked
config (e.g. `compose/docker-compose.prod.yml`,
`compose/grafana-agent.yaml`, `nginx-prod.conf.template`) and want to
test it on prod **without** waiting for a full main-push +
publish + deploy cycle (~25 min).

> **Limit:** files COPY'd into the published image (api/viewer/pipeline
> Python source, baked nginx config, Vue bundle) require an image
> rebuild. This workflow only handles bind-mounted / overlay-defined
> files.

```bash
# 1. From your laptop, scp the updated file(s) into the right place on the VPS:
scp compose/docker-compose.prod.yml \
    compose/grafana-agent.yaml \
    docker/viewer/nginx-prod.conf.template \
    deploy@prod-podcast.<tailnet>:/srv/podcast-scraper/compose/
# (paths must match what the bind mounts in the YAML expect)

# 2. Recreate the affected service so it picks up the change.
#    --env-file is REQUIRED when running compose directly as deploy user;
#    see "Why --env-file?" in the FAQ.
ssh deploy@prod-podcast.<tailnet> 'bash -s' <<'REMOTE'
set -euo pipefail
cd /srv/podcast-scraper
COMPOSE="docker compose --env-file /srv/podcast-scraper/.env \
  -f compose/docker-compose.stack.yml \
  -f compose/docker-compose.prod.yml \
  -f compose/docker-compose.vps-prod.yml"
$COMPOSE up -d --force-recreate api grafana-agent viewer
$COMPOSE ps
REMOTE
```

After verifying the fix works on prod, **commit the same change to a
branch + open a PR** — otherwise the next time someone re-runs cloud-init
or `git pull`s on the VPS, your hot-fix gets blown away. There's no
auto-pull on prod, so you have a window, but don't trust it.

---

## FAQ / Troubleshooting

### "`curl http://127.0.0.1:8000/api/health` fails on the VPS but the app works"

The `api` service listens on `8000` **inside the container** only. Stock
compose uses `expose`, not `ports`, so nothing listens on the **host's**
`127.0.0.1:8000`. Use a check from the [API health checks by
context](#api-health-checks-by-context) table (container `exec`, viewer
port, or tailnet HTTPS). See [GitHub issue
745](https://github.com/chipi/podcast_scraper/issues/745).

### "Why `--env-file`?"

`docker compose -f compose/docker-compose.stack.yml ...` resolves the
project directory to **`dirname(first -f file)` = `/srv/podcast-scraper/compose/`**,
not `/srv/podcast-scraper/`. Compose's auto-load of `.env` searches
the project dir, so it looks at `compose/.env` (which doesn't exist)
and skips the actual file at `/srv/podcast-scraper/.env`.

The systemd unit avoids this because `EnvironmentFile=/srv/podcast-scraper/.env`
loads the env into the SERVICE environment, which the spawned compose
process inherits regardless of project-dir resolution.

When running compose directly (deploy / debugging / hot-fix), always
pass `--env-file /srv/podcast-scraper/.env` explicitly.

### "Pipeline fails with `PODCAST_CORPUS_HOST_PATH is missing a value`"

The api spawned a nested `docker compose run pipeline-llm` but couldn't
resolve `${PODCAST_CORPUS_HOST_PATH}` from `compose/docker-compose.prod.yml`.

Causes (in order of likelihood):

1. `.env` doesn't have `PODCAST_CORPUS_HOST_PATH=...` — check with
   `grep PODCAST_CORPUS_HOST_PATH /srv/podcast-scraper/.env`
2. The api container is running an OLD image that doesn't pass
   `--env-file` to the nested compose AND doesn't have the var in its
   env passthrough — pull/recreate api with the latest image
3. The api passthrough block in `compose/docker-compose.prod.yml`
   doesn't include `PODCAST_CORPUS_HOST_PATH:` — verify the file
   on prod matches the latest main

### "Pipeline fails with `OpenAI API key required for OpenAI providers`"

The pipeline container's `OPENAI_API_KEY` resolved to empty. Causes:

1. **Wrong variable name in `.env`** — the most common cause. Code
   expects `OPENAI_API_KEY` (with `_API_`); typos like `OPENAI_KEY`
   silently resolve to the empty default. Check with
   `grep -E '^OPENAI(_API)?_KEY=' /srv/podcast-scraper/.env`
2. The api container env doesn't have the key (passthrough missing /
   stale image) — see hot-fix workflow + recreate api
3. The key value itself is malformed (e.g. base64-encoded by accident)

### "Pipeline fails with `Docker pipeline jobs require top-level pipeline_install_extras`"

The corpus's `viewer_operator.yaml` doesn't declare
`pipeline_install_extras: ml` or `: llm`, AND the host has no
`PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS` env fallback set.

Fix (durable): in the viewer, **Sources** → **Operator** tab, add
`pipeline_install_extras: llm` to the YAML, **Save**. Now the field
is in the file.

Fix (host-wide): set `PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS=llm`
in `/srv/podcast-scraper/.env` and recreate the api. Falls back for
every corpus that omits the field.

The prod overlay sets the env default to `llm` out of the box;
this error only appears if someone explicitly unset it.

### "Viewer container shows `unhealthy` but the page loads fine"

The healthcheck probe (`wget` against `/`) gets 401 from `auth_basic`
in prod. Compose's older healthcheck used `curl -fsS` which both
isn't installed in `nginx:alpine` AND would fail on 401. The current
healthcheck (in `compose/docker-compose.stack.yml`) uses
`wget -qSO /dev/null` and accepts ANY `HTTP/` response as alive.
If you still see `unhealthy`, you're on an old viewer image — pull
the latest.

### "Grafana agent restarting / no data in Grafana Cloud"

In order of likelihood:

1. Wrong username for one of Prom / Loki — they have **separate**
   instance IDs in Grafana Cloud. Check both
   `GRAFANA_CLOUD_PROM_USER` and `GRAFANA_CLOUD_LOKI_USER` are set
   distinctly.
2. API key missing the right scope — must include both `metrics:write`
   AND `logs:write` to feed both endpoints with one token
3. URL has the wrong region — `prometheus-prod-65-prod-eu-west-2`
   vs `prometheus-prod-13-prod-us-central-0` etc. Re-copy from
   the stack page

Diagnose with:

```bash
ssh deploy@prod-podcast.<tailnet> \
  'docker logs --tail 50 compose-grafana-agent-1 | grep -iE "error|401|403|forbidden"'
```

### "Tailnet hostname won't resolve from my laptop"

```bash
tailscale status            # is the laptop on the tailnet?
tailscale up                # if not, log back in
tailscale ip -4 prod-podcast-1  # try variants if -1 / -2 suffix
```

If laptop is logged in and IP resolves but ssh hangs, the VPS may
have lost its tailscale registration (auth key expired before
re-up). Use Hetzner console (see Hetzner ops) to `tailscale up` again
with a fresh `TS_AUTHKEY`.

### "I can't sudo as deploy"

Cloud-init explicitly sets `sudo: false` for the `deploy` user. This
is intentional — deploy owns `/srv/podcast-scraper` and is in the
`docker` group, which covers everything operators normally need.
Operations requiring root (apt, systemctl, etc.) need the `root`
user via Hetzner console or via the same `~/.ssh/id_ed25519` key
which cloud-init also injected into root's `authorized_keys` as
emergency access.

### "Operator YAML on VPS keeps reverting"

If you edit `/srv/podcast-scraper/corpus/viewer_operator.yaml`
directly via SSH, then click **Save** in the viewer's Operator tab,
the viewer overwrites your on-disk edit with whatever's in the
textarea (which was loaded BEFORE your SSH edit and stayed cached).

Always edit via the viewer Operator tab, not on disk — viewer Save
preserves the textarea content verbatim.

### "Pipeline image just `Pull complete` then immediately exits"

Read the pipeline container's stdout (logged by the api factory):

```bash
ssh deploy@prod-podcast.<tailnet> \
  'ls -lt /srv/podcast-scraper/corpus/jobs/*/log.txt | head -3'
# then cat the most recent one
```

Common causes:

- LLM key missing/wrong name (see openai key FAQ)
- Profile in operator YAML names a provider whose key isn't set
- Corpus directory has a permission issue (run owner mismatch)

### "Stale Docker volume after env path changes"

`docker volume inspect compose_corpus_data` — if the `device:` field
doesn't match `PODCAST_CORPUS_HOST_PATH`, the volume was created with
old config. compose won't recreate volumes whose YAML changed. Fix:

```bash
ssh deploy@prod-podcast.<tailnet>
cd /srv/podcast-scraper
COMPOSE="docker compose --env-file /srv/podcast-scraper/.env \
  -f compose/docker-compose.stack.yml -f compose/docker-compose.prod.yml \
  -f compose/docker-compose.vps-prod.yml"
$COMPOSE down
docker volume rm compose_corpus_data
$COMPOSE up -d
```

The bind path content survives (host dir is the source of truth);
only the Docker-side volume metadata is rebuilt.

---

## Constraints to know

These are intentional design choices that look like bugs at first
glance. Knowing about them saves debugging time.

- **`deploy` user has no sudo** (cloud-init: `sudo: false`). Use root
  via Hetzner console for sudo-needed operations. See
  [FAQ → I can't sudo as deploy](#i-cant-sudo-as-deploy).
- **Direct shell `docker compose` MUST use `--env-file`** because the
  project dir resolves to `compose/`, not `/srv/podcast-scraper/`.
  Systemd-spawned compose is unaffected (different env-loading path).
- **Grafana Cloud has separate Prom + Loki user IDs.** Single
  `GRAFANA_CLOUD_USER` doesn't work — split into PROM_USER / LOKI_USER.
- **Viewer image is published once per main push** and used by BOTH
  codespace preprod AND the prod VPS. `PODCAST_ENV` is injected at
  runtime by nginx `sub_filter` — DON'T add `--build-arg
  VITE_PODCAST_ENV=...` back to stack-test.yml.
- **`pipeline_install_extras` is not in the profile YAMLs.** It's an
  operator-set field separate from profile. Default fallback comes
  from `PODCAST_DEFAULT_PIPELINE_INSTALL_EXTRAS=llm` env (set in the
  prod overlay).
- **No public ingress.** Hetzner firewall blocks all TCP. Viewer
  reachable only over tailnet via `tailscale serve`. Adding public
  exposure requires firewall + auth changes — don't add one without
  the other.
- **Auth keys (Tailscale) expire every 90 days max** on Free plan.
  Calendar reminder.
- **Profile dropdown is filtered to published images** via
  `PODCAST_AVAILABLE_PROFILES`. Don't add a profile to the allowlist
  whose backing pipeline image isn't published.
- **The published api image only ships `[server]` extras.** Pipeline
  runs MUST go through Docker job mode (`PODCAST_PIPELINE_EXEC_MODE=docker`,
  set in prod overlay). In-process pipeline runs would crash on
  missing `[llm]` deps.

---

## Cross-references

- [RFC-082 — design](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [`infra/`](https://github.com/chipi/podcast_scraper/tree/main/infra) — IaC code
- [`tailscale/policy.hujson`](https://github.com/chipi/podcast_scraper/blob/main/tailscale/policy.hujson) — ACL
- [`.github/workflows/deploy-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/deploy-prod.yml)
- [`.github/workflows/infra-apply.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/infra-apply.yml) — manual apply gate
- [`.github/workflows/backup-corpus-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/backup-corpus-prod.yml)
- [#714](https://github.com/chipi/podcast_scraper/issues/714) — account prereqs checklist
- [#723](https://github.com/chipi/podcast_scraper/issues/723) — Phase B cutover
- [#724](https://github.com/chipi/podcast_scraper/issues/724) — DR drill
