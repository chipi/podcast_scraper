# Prod runbook (always-on Hetzner VPS)

Operator-facing runbook for the production deploy defined in
[RFC-082](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md). The RFC
describes *what we decided*; this runbook describes *what to do today*.

Need the short version for daily ops? Use
[Prod operator cheat sheet](PROD_OPERATOR_CHEAT_SHEET.md).
**Other Docker Compose apps on the same VPS:** see
[VPS multi-app onboarding](VPS_MULTI_APP_ONBOARDING.md).
**Optional DGX Whisper primary (cost optimization):** [DGX_RUNBOOK](DGX_RUNBOOK.md) and profile
`cloud_with_dgx_whisper_primary` per [ADR-096](../adr/ADR-096-dgx-spark-prod-primary-with-fallback.md).

**How hosting fits together (diagrams + planes):** [Hosting and infrastructure](../architecture/HOSTING_AND_INFRASTRUCTURE.md). **Immutable decisions:** [ADR-079](../adr/ADR-079-opentofu-for-always-on-hosting-iac.md)–[ADR-083](../adr/ADR-083-tailscale-private-ingress-always-on-vps.md) (OpenTofu, state, drill workspace, app GitOps contract, tailnet ingress), [ADR-082](../adr/ADR-082-gitops-app-deploy-via-stack-test-and-gha.md) (stack-test gate and deploy nuance), [ADR-084](../adr/ADR-084-full-stack-docker-compose-topology.md)–[ADR-085](../adr/ADR-085-ephemeral-stack-test-integration-gate.md) (Compose + CI stack-test), [ADR-093](../adr/ADR-093-canonical-stack-contract-and-environment-adapters.md) (stack contract vs adapters). **Surface audit table:** [STACK_CONTRACT.md](STACK_CONTRACT.md). **CI workflow names:** [WORKFLOWS.md](../ci/WORKFLOWS.md).

## Steady-state playbook (routine prod)

For day-to-day prod (not DR drill, not manual corpus restore): **preflight** (secrets and
`PROD_TAILNET_FQDN`) → **deploy** (`deploy-prod.yml` → `infra/deploy/deploy.sh`) → **health**
(in-container `/api/health` on the api service — [API health checks by context](#api-health-checks-by-context))
→ **behavioral gate on main** (`stack-test.yml` before GHCR publish). Restoring corpus from
`snapshot.tgz` is **not** part of this path; see [Disaster recovery](#disaster-recovery) and
[STACK_CONTRACT.md](STACK_CONTRACT.md).

> **For the prerequisites checklist** (Hetzner account + Tailscale credentials —
> auth key + API access token on Free plan, see "Tailscale credentials" below
> for the why — sops/age + GHA secrets), see
> [#714](https://github.com/chipi/podcast_scraper/issues/714).
> All commands below assume those are done.

## Sections

1. [First-time bootstrap](#first-time-bootstrap) — includes [API health checks by context](#api-health-checks-by-context) (GH-745)
2. [Daily operations](#daily-operations)
3. [VPS access control (no HTTP Basic Auth)](#vps-access-control-no-http-basic-auth)
4. [Corpus migration from pre-prod (Codespace) to prod (VPS)](#corpus-migration)
5. [Rollback (deploy went red mid-way)](#rollback)
6. [Disaster recovery (VPS gone)](#disaster-recovery)
7. [Prod failover (stand up spare on DR row)](#prod-failover)
8. [Credential rotation](#credential-rotation)
9. [Environment variable reference](#environment-variable-reference)
10. [Observability setup walkthrough](#observability-setup-walkthrough) — includes [Sentry Slack routing (GH-725)](#sentry-slack-routing-prod-vs-pre-prod-gh-725) and [Grafana env filter (GH-726)](#grafana-env-filter-gh-726)
11. [Tailscale operations](#tailscale-operations)
12. [Hetzner operations](#hetzner-operations)
13. [Operator hot-fix workflow](#operator-hot-fix-workflow)
14. [FAQ / Troubleshooting](#faq-troubleshooting) — includes [corpus path (host vs `/app/output`)](#corpus-directory-host-vs-appoutput), [topic clusters](#topic-clusters-missing-after-a-successful-pipeline-run), [reprocess from transcripts](#reprocess-a-corpus-without-re-transcribing-audio), and [Cursor or automation cannot `ssh deploy@prod`](#cursor-or-automation-cannot-ssh-deployprod)
15. [Constraints to know](#constraints-to-know)

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
# PROD_SSH_PRIVATE_KEY — see [GitHub Actions SSH to prod](#github-actions-ssh-to-prod-prod_ssh_private_key)

# 4. Stage GHA repo secrets and variables
gh secret set OPERATOR_SSH_PUBLIC_KEY --repo chipi/podcast_scraper --body "$(cat ~/.ssh/id_ed25519.pub)"
gh variable set TAILNET_NAME            --repo chipi/podcast_scraper --body 'tail-xxxxx.ts.net'
# If OPERATOR_SSH_PUBLIC_KEY was previously a repo variable, add the same value as the secret above,
# then delete the variable so infra workflows use only the secret (GitHub masks secrets in logs).
# PROD_TAILNET_FQDN is set after first apply (it depends on the assigned hostname);
# default value is "prod-podcast.<TAILNET_NAME>".
```

### GitHub Actions SSH to prod (`PROD_SSH_PRIVATE_KEY`) {#github-actions-ssh-to-prod-prod_ssh_private_key}

`deploy-prod.yml`, `backup-corpus-prod.yml`, and **`prod-restore-corpus.yml`** run OpenSSH as `deploy@<prod>` **after**
the runner joins the tailnet. The Tailscale ACL allows **reachability** to port 22;
OpenSSH still requires a private key whose **public** half is listed in
`~deploy/.ssh/authorized_keys` on the VPS (Tailscale SSH is intentionally not used;
see `tailscale/policy.hujson` comments).

**One-time setup**

1. On a trusted machine, generate a **CI-only** Ed25519 key (empty passphrase):

   ```bash
   ssh-keygen -t ed25519 -f ./gha-prod-deploy -N "" -C "github-actions-prod-deploy"
   ```

2. SSH to prod using your **operator** key (the same public key OpenTofu passed as
   `TF_VAR_ssh_public_key` / `secrets.OPERATOR_SSH_PUBLIC_KEY`). As `deploy`, append **exactly one line**
   (the contents of `gha-prod-deploy.pub`) to `~/.ssh/authorized_keys`. Directory `~/.ssh` should be
   mode `700` and `authorized_keys` mode `600`.

3. Verify key auth before touching GitHub:

   ```bash
   ssh -i ./gha-prod-deploy -o BatchMode=yes deploy@<prod-tailnet-host> 'echo ok'
   ```

   Use the same hostname the workflows resolve (see [suffix drift](#tailscale-suffix-drift) if unsure).

4. Store the **private** PEM in a repo Actions secret (entire file, including `BEGIN OPENSSH PRIVATE KEY`
   / `END` lines):

   ```bash
   gh secret set PROD_SSH_PRIVATE_KEY --repo chipi/podcast_scraper --app actions < ./gha-prod-deploy
   ```

5. Remove or securely archive `gha-prod-deploy` on disk; do not commit it.

Workflows load this via the composite action `.github/actions/prod-ssh-key`, which exports
`SSH_PROD_IDENTITY` for `ssh -i "$SSH_PROD_IDENTITY" -o IdentitiesOnly=yes …`. Any **future** workflow
that SSHes to prod as `deploy@` should reuse `secrets.PROD_SSH_PRIVATE_KEY` and the same flags so
GitHub never relies on keys baked into the runner image.

**Rotating the CI key:** generate a new keypair, append the new public key to `authorized_keys` (keep
the old line until one green workflow run), update `PROD_SSH_PRIVATE_KEY`, re-run `deploy-prod.yml` or
`backup-corpus-prod.yml`, then delete the superseded public key line on the VPS.

### GitHub Actions deploy to DR drill (`DRILL_DEPLOY_SSH_PRIVATE_KEY`) {#github-actions-ssh-to-drill-drill_deploy_ssh_private_key}

**Drill workflow matrix, typed confirms, orchestrator, restore/destroy, and drill-only host checks:**
[DR_DRILL_RUNBOOK.md](DR_DRILL_RUNBOOK.md). This section stays focused on **`drill-deploy`** and
**`DRILL_DEPLOY_SSH_PRIVATE_KEY`** setup shared with prod-style deploys.

RFC-082 / #752: **`drill-deploy.yml`** mirrors **`deploy-prod.yml`** but targets the **drill** Hetzner
stack. After **`tailscale/github-action`** joins as **`tag:gha-deployer`**, the job SSHes
**`deploy@<resolved-drill-fqdn>`**, appends **`PODCAST_RELEASE=sha-<short>`** to
**`/srv/podcast-scraper/.env`**, runs **`/srv/podcast-scraper/infra/deploy/deploy.sh`**, then curls
**`https://<resolved>/api/health`**. Resolver: **`scripts/ops/resolve_drill_tailnet_host.sh`** with
**`vars.DRILL_TAILNET_FQDN`**. GitHub Environment: **`drill`**.

The composite **`.github/actions/prod-ssh-key`** is invoked with **`identity_env_name: SSH_DRILL_IDENTITY`**
so the workflow uses **`$SSH_DRILL_IDENTITY`** (prod workflows keep the default **`SSH_PROD_IDENTITY`**).

**One-time drill host setup (same authorized_keys story as prod above):**

1. SSH to the drill VPS with your **operator** key (only that key is present from cloud-init until you extend **`authorized_keys`**).
2. Append **exactly one line** (contents of **`gha-prod-deploy.pub`**, or a drill-only CI public key) to **`deploy@`** **`~/.ssh/authorized_keys`** (modes **`700`** / **`600`**).
3. Stage **`/srv/podcast-scraper/.env`**, then **`sudo rm /srv/podcast-scraper/.bootstrap-needs-env`** so Docker Compose can start (see cloud-init **`final_message`** on first boot).

**GitHub secret** (can reuse the same PEM file as prod if the same public key is on drill **`deploy@`**):

```bash
gh secret set DRILL_DEPLOY_SSH_PRIVATE_KEY --repo chipi/podcast_scraper --app actions < ./gha-prod-deploy
```

**Dispatch** — the workflow file must exist on **`main`** (merge your branch first). There is **no**
`make` target; use **`gh`**:

```bash
gh workflow run drill-deploy.yml -R chipi/podcast_scraper
gh run watch -R chipi/podcast_scraper "$(gh run list -R chipi/podcast_scraper --workflow=drill-deploy.yml -L1 --json databaseId -q '.[0].databaseId')"
```

**Local operator check** (optional; use **`ssh-add`** if your key is not already in an agent — see
the section **Cursor or automation cannot `ssh deploy@prod`** in this runbook for **`ssh-add -t 30m`**):

```bash
ssh-add -t 30m ~/.ssh/id_ed25519
ssh -o IdentitiesOnly=yes deploy@<your-drill-magicdns-host> 'echo ok'
```

**Related infra-only cleanup (local):** **`make delete-drill-hetzner-orphans`** sources **`infra/.env.drill.local`** and deletes orphan Hetzner objects by name after a failed drill **`tofu apply`**; see **`make drill-env`**. Not used for normal deploys.

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

**GitHub Actions:** `deploy-prod.yml`, `backup-corpus-prod.yml`, and **`prod-restore-corpus.yml`** join the
tailnet, run `scripts/ops/resolve_prod_tailnet_host.sh`, and use the resolved
FQDN for SSH (and the `/api/health` probe where applicable). Workflows still require
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

### Stage `.env` (workflow-staged from GH Secrets — #841)

As of #841, `/srv/podcast-scraper/.env` is rendered on the host by
`deploy-prod.yml` (and `prod-restore-corpus.yml`) from `PROD_*` GH Secrets
at deploy time. The operator's only step is **staging the 15 secrets once
in repo settings** — the workflow handles file creation, atomic delivery,
and permissions. No SSH, no manual `.env` editing.

#### Required PROD_* GH Secrets

Repo Settings → Secrets → Actions → New repository secret. Stage these
15 secrets with prod values from each provider's dashboard:

| Secret name | Purpose | Source |
| --- | --- | --- |
| `PROD_OPENAI_API_KEY` | LLM (OpenAI) | platform.openai.com → API keys |
| `PROD_ANTHROPIC_API_KEY` | LLM (Anthropic) | console.anthropic.com → API keys |
| `PROD_GEMINI_API_KEY` | LLM (Gemini) | aistudio.google.com → API keys |
| `PROD_MISTRAL_API_KEY` | LLM (Mistral) | console.mistral.ai → API keys |
| `PROD_DEEPSEEK_API_KEY` | LLM (DeepSeek) | platform.deepseek.com → API keys |
| `PROD_GROK_API_KEY` | LLM (Grok) | console.x.ai → API keys |
| `PROD_SENTRY_DSN_API` | Sentry (api project) | sentry.io → project settings → Client Keys (DSN) |
| `PROD_SENTRY_DSN_PIPELINE` | Sentry (pipeline project) | sentry.io → project settings → Client Keys (DSN) |
| `PROD_SENTRY_DSN_VIEWER` | Sentry (viewer SPA — baked into Vite bundle) | sentry.io → project settings → Client Keys (DSN) |
| `PROD_GRAFANA_CLOUD_API_KEY` | Grafana Cloud auth | grafana.com → My Account → Access Policies / Tokens |
| `PROD_GRAFANA_CLOUD_LOKI_URL` | Loki push URL | grafana.com → My Account → Details (per-stack) |
| `PROD_GRAFANA_CLOUD_LOKI_USER` | Loki tenant ID | grafana.com → My Account → Details (per-stack) |
| `PROD_GRAFANA_CLOUD_PROM_URL` | Prom remote_write URL | grafana.com → My Account → Details (per-stack) |
| `PROD_GRAFANA_CLOUD_PROM_USER` | Prom tenant ID | grafana.com → My Account → Details (per-stack) |
| `PROD_JOB_WEBHOOK_URL` | Outbound pipeline-completion webhook (optional) | your config |

Missing secret → empty value in `.env` → compose `${VAR:-}` default → that
feature is off (stack still starts). Stage incrementally as needed.

#### Verifying after a deploy

After the next `deploy-prod.yml` run, SSH in and inspect:

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net 'cat /srv/podcast-scraper/.env | wc -l && head -3 /srv/podcast-scraper/.env'
# expect: ≥21 lines (6 static + 15 secrets), mode 600 owner deploy:deploy
```

The systemd unit's `ExecStartPre` is now `test -f /srv/podcast-scraper/.env`
(positive check, #844). Service refuses to start until `.env` exists — same
protection as the prior `.bootstrap-needs-env` sentinel but self-explanatory.

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
| **Host loopback via viewer (nginx → api)** | `curl -fsS http://127.0.0.1:${VIEWER_PORT:-8080}/api/health` | `VIEWER_PORT` defaults to `8080` (`compose/docker-compose.stack.yml`). Same nginx path as the SPA shell. |
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

# 2. Viewer shell (no HTTP Basic Auth on the VPS; tailnet is the gate)
curl -fsS https://prod-podcast.tail-xxxxx.ts.net/ | head
curl -fsS https://prod-podcast.tail-xxxxx.ts.net/welcome | head

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
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f confirm=PROD_DEPLOY
gh run watch --repo chipi/podcast_scraper
```

Once a few green deploys have passed and you trust the loop, file the
follow-up PR that flips `deploy-prod.yml` to also auto-trigger on
`workflow_run: ["Stack test"]` (RFC-082 Decision 6 GitOps loop).

---

## Daily operations

### Cost visibility (#823 / #804)

After each pipeline run, inspect `corpus_manifest.json` → `cost_rollup.total_cost_usd` and
per-stage `by_stage` fields. When runs used billable providers but the rollup is **$0.00** while
`cost_appears_uninstrumented` is **true**, cost data was not recorded (check
`PRICING_ASSUMPTIONS_FILE` / `/app/config/pricing_assumptions.yaml` in pipeline containers).

```bash
python -m podcast_scraper.cli corpus-cost /srv/podcast-scraper/corpus --update-manifest
```

Re-aggregates from every `feeds/*/run_*/metrics.json` and refreshes `cost_rollup` (and
`produced_by` when missing). Structured per-call cost lines are emitted as JSON log objects with
`event_type: llm_cost` in pipeline logs (and stdout when JSONL metrics echo is enabled; see
Observability).

### Where to look first

| Symptom | Where |
| --- | --- |
| Viewer slow / unreachable | `gh run list --workflow deploy-prod.yml --limit 3` then Sentry → environment=prod |
| Pipeline run failing | Sentry → environment=prod, component=pipeline; viewer Library → Job logs |
| Deploy went red | GHA UI → Deploy to prod VPS → most recent run; api logs are dumped on health-check failure |
| "Did the alert fire because of X?" | Grafana Cloud → podcast-scraper folder → filter `env=prod` |

### Manual deploy

Typed confirm matches sibling prod mutators (`PROD_RESTORE`, `PROD_FAILOVER_STAND_UP`).

#### Pre-flight (operator laptop)

1. Confirm **Stack test** (or the target SHA) is green on `main`:
   `gh run list --workflow stack-test.yml --limit 3`
2. Note the image tag you intend to ship (`sha-<7>` from the green run, or blank = workflow SHA).
3. Confirm prod secrets/vars are staged (`TS_AUTHKEY`, `PROD_SSH_PRIVATE_KEY`, `PROD_TAILNET_FQDN`).
4. Optional rollback pin: keep the previous good `sha-<7>` from the last green deploy run.

#### Dispatch

```bash
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f confirm=PROD_DEPLOY \
  -f override_image_sha=                       # blank = deploy workflow SHA
# or pin to a specific image (rollback / hotfix):
gh workflow run deploy-prod.yml --repo chipi/podcast_scraper \
  -f confirm=PROD_DEPLOY \
  -f override_image_sha=abc1234
```

The workflow validates `override_image_sha` shape (`^[a-f0-9]{7,40}$`) and checks that
`:sha-<short>` manifests exist on GHCR **before** SSH. `deploy.sh` resets git to the same
ref and sets `PODCAST_IMAGE_TAG=sha-<short>` so compose files and images stay aligned.

#### Post-deploy smoke (automated + operator)

The workflow runs:

1. VPS-local `/api/health` inside the api container (`deploy.sh`)
2. External `/api/health` over Tailscale MagicDNS
3. Six-surface probe via `scripts/ops/post_deploy_smoke.sh` (health, Library episodes,
   Digest, Graph artifacts, topic-clusters, Search)

Operator spot-check in the viewer (over Tailscale): open **Library**, **Digest**, **Graph**, and run one **Search** query against the prod corpus path.

Watch for `corpus_version_warning` in `/api/health` or the viewer status bar when the on-disk corpus predates the server's minimum supported code version — reprocess per [Reprocess a corpus without re-transcribing audio](#reprocess-a-corpus-without-re-transcribing-audio) or `make reprocess-corpus-from-transcripts`.

**Local smoke (tailnet required):**

```bash
export PROD_TAILNET_FQDN=prod-podcast.<tailnet>.ts.net
make smoke-prod SMOKE_CORPUS_PATH=/app/output
```

See [Code/content compatibility](#codecontent-compatibility) for the full decision tree.

#### Rollback (same workflow, pinned SHA)

Re-dispatch with `override_image_sha=<previous-good-short-sha>`. See [Rollback](#rollback) for failure-mode detail.

#### Legacy cleanup

Commit `7c20b74` removed nginx HTTP Basic Auth from the VPS overlay. If `/etc/nginx/.htpasswd` (or a bind-mounted copy) still exists on an older host, it is inert — safe to delete:

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net 'sudo rm -f /etc/nginx/.htpasswd'
```

### Pipeline run via the viewer

Standard flow — open the viewer, hit Configuration → Run pipeline. Same
control plane as pre-prod. Profile dropdown is restricted to
`cloud_balanced,cloud_thin` (no ML profiles in prod, per RFC-082).

### Backup status

```bash
gh run list --workflow backup-corpus-prod.yml --repo chipi/podcast_scraper --limit 5
gh run list --workflow verify-backup-restore.yml --repo chipi/podcast_scraper --limit 5
gh release list --repo chipi/podcast_scraper-backup --limit 10 | grep snapshot-prod-
```

**Weekly compose restore verify (#798):** `verify-backup-restore.yml` runs **Sundays 04:00 UTC**
(plus `workflow_dispatch`). It downloads the newest compatible **`snapshot-prod-*`**, restores
into ephemeral Docker Compose on a GHA runner, runs **`post_deploy_smoke.sh`**, then tears down.
Failures go red + optional **`SMOKE_WEBHOOK_URL`** alert. Sister cadence: real-Hetzner DR drill
**Wednesdays 02:00 UTC** — see [DR drill runbook](DR_DRILL_RUNBOOK.md).

To download the latest matching **`snapshot-prod-*`** asset locally, print tarball
stats, and unpack under **`.tmp_backup_verify/`** (gitignored):

```bash
./scripts/ops/verify_prod_backup_snapshot.sh
```

Use **`./scripts/ops/verify_prod_backup_snapshot.sh --help`** for a specific tag
or **`--no-extract`** (list only, no unpack). Releases after RFC-084 also ship
**`snapshot.manifest.json`**; default restore picks the **newest compatible**
`snapshot-prod-*` (fail closed if none match — pin **`backup_tag`** / **`PODCAST_BACKUP_TAG`**).
See [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).

---

## VPS access control (no HTTP Basic Auth)

The VPS nginx overlay (`docker/viewer/nginx-prod.conf.template` via
`compose/docker-compose.vps-prod.yml`) does **not** enable `auth_basic`.
Reachability is **tailnet-only** (Hetzner firewall has no public TCP 80/443).
If you ever open public ingress, add a separate edge layer (for example
OAuth at a reverse proxy), not only Basic Auth on this nginx template.

---

## Corpus migration

One-time migration on cutover day. Use the newest **compatible** snapshot from the backup
repo rather than streaming files between hosts (more reliable; matches RFC-082
Decision 4).

**Preferred on prod:** run **`prod-restore-corpus.yml`** in GitHub Actions (confirm
**`PROD_RESTORE`**). On-host rehearsal or migration-only SSH:

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net
cd /srv/podcast-scraper
make restore-corpus-prod          # newest-compatible snapshot-prod-* → corpus/
# Pin a specific backup release (DR drills / audits):
#   PODCAST_BACKUP_TAG=<release-tag> make restore-corpus-prod
```

See [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md).

```bash
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

## Disaster recovery (VPS lost or unrecoverable) {#disaster-recovery}

Use this when the prod VPS is gone or stuck in a state that surgical fixes can't reach (cloud-init crashed before tailnet join, tofu state and Hetzner have drifted past reconciliation, server destroyed by an unintended apply cascade). 2026-05-29 incident playbook, distilled.

> **After any incident that triggers this section, write a post-incident
> review.** See [docs/incidents/README.md](../incidents/README.md) for
> the process, template, and prior reviews. PIRs are how we close the
> learning loop — without them, the same patterns repeat.

### What survives a VPS rebuild

| Artifact | Where it lives | Survives VPS destroy? |
| --- | --- | --- |
| IaaC code | this repo (`infra/`) | ✓ |
| GH Secrets (TS_AUTHKEY, OPERATOR_SSH_PUBLIC_KEY, etc.) | repo settings | ✓ |
| Encrypted tfstate (post-apply) | `infra/terraform/terraform.tfstate.enc` in main | ✓ if last apply's re-encrypt step succeeded AND was committed back; otherwise stale (see "State drift" below) |
| Corpus snapshots | `chipi/podcast_scraper-backup` releases (`snapshot-prod-YYYYMMDD`) | ✓ — restore via `prod-restore-corpus.yml` |
| Runtime `.env` (LLM keys, Sentry DSNs, Grafana creds) | on-disk `/srv/podcast-scraper/.env` only | ✗ **LOST** until #841 lands (GH Secrets-driven `.env` rendering); see "Recovering `.env` after destroy" below |
| Tailscale device records (offline node entries) | tailnet | ✗ — stale `prod-podcast-1`/`-2`/… entries linger and block MagicDNS hostname re-issue; clean via `Tailscale cleanup (gha-deployer devices)` workflow or expand to also delete `tag:prod` |

### Recovery sequence

**Pre-check**: confirm IaaC + secrets are intact before destruction.

```bash
# 1. Encrypted state present in main?
ls -la infra/terraform/terraform.tfstate.enc

# 2. Required GH Secrets staged? (operator dashboards or `gh secret list`)
gh secret list | grep -E "TS_AUTHKEY|PROD_SSH_PRIVATE_KEY|HCLOUD_TOKEN|OPERATOR_SSH_PUBLIC_KEY|TFSTATE_AGE_KEY"
```

**Wipe + rebuild** (single workflow run):

```bash
# Hetzner placement failure on cx43 in fsn1 is common during incidents.
# Override to nbg1 (Nuremberg) if needed; cax31 / cpx32 are server-type fallbacks.
gh workflow run "Infra apply (manual)" \
  -f confirm=WIPE_THEN_APPLY \
  -f mode=wipe-then-apply \
  -f override_location=nbg1
```

Wait for completion. Workflow does: API-enumerate-and-DELETE every Hetzner resource in the project + delete every `tag:prod`/`tag:dr-drill`/`tag:gha-deployer` Tailscale device + wipe local state + apply from scratch. Cloud-init bootstraps the new VPS.

**Capture state back to main** (CRITICAL — otherwise next run starts from stale state):

```bash
# Find the last infra-apply run id, download its terraform-state-after-apply artifact
RUN_ID=$(gh run list --workflow="Infra apply (manual)" --limit 1 --json databaseId -q '.[0].databaseId')
gh run download "$RUN_ID" -n terraform-state-after-apply -D /tmp/tfstate/
cp /tmp/tfstate/terraform.tfstate.enc infra/terraform/terraform.tfstate.enc
git add infra/terraform/terraform.tfstate.enc
git commit -m "infra(state): sync tfstate.enc post-wipe-then-apply <RUN_ID>"
git push
```

**Update `vars.PROD_TAILNET_FQDN` if hostname changed**: the new VPS may join the tailnet as `prod-podcast` (clean slot post-wipe) or `prod-podcast-2` (if an old device record lingered). Compare `gh variable list` against actual `tailscale status` and update if mismatched.

**Wait for tailnet join** (cloud-init takes 5–10 min on a fresh image; needs apt + docker + tailscale + repo clones):

```bash
while ! tailscale status | grep -q "^[0-9.]\+\s\+prod-podcast\s"; do sleep 30; done
ssh deploy@prod-podcast.<tailnet>.ts.net 'cloud-init status --wait && cloud-init status --long'
# Expect: status: done, errors: []
```

**Recover `.env`** — see next subsection.

**Restore corpus + deploy**:

```bash
gh workflow run "Prod restore corpus (backup → prod VPS)" \
  -f confirm=PROD_RESTORE \
  -f backup_tag=snapshot-prod-YYYYMMDD   # or leave empty for newest compatible

gh workflow run "Deploy to prod VPS" \
  -f confirm=PROD_DEPLOY \
  -f override_image_sha=<7-char SHA of the release you want>
```

**Verify**: tailnet `:443` and `:8443` (orrery) reachable; `/api/health` returns 200; Sentry shows a test event; Grafana receives metrics.

### Recovering `.env` after destroy

`.env` is workflow-staged from GH Secrets (#841), so it auto-rebuilds on the next `deploy-prod.yml` or `prod-restore-corpus.yml` run. No manual step. If you've never staged the secrets before, follow [Stage `.env` (workflow-staged from GH Secrets)](#stage-env-workflow-staged-from-gh-secrets--841) one-time.

### State drift gotchas

Things that bit on 2026-05-29 — read this before any `tofu apply` against prod.

**1. The post-apply re-encrypt step uploads `terraform.tfstate.enc` as a workflow artifact but does NOT commit it back to main.** Auto-commit-back is tracked in `infra-apply.yml`'s comments as a deferred follow-up. Until it lands, **every workflow run starts from main's encrypted state, which is whatever was last manually committed**. If you ran an apply but forgot to commit the artifact back, the next apply starts from stale state and recreates everything as orphans.

**2. `# forces replacement` in a tofu plan = destroy + recreate.** Read every plan output for these literal strings before approving:

- `# forces replacement` — destructive
- `must be replaced` — destructive
- `will be destroyed` — destructive
- `(sensitive value)` paired with any of the above — the diff is masked, you cannot see what changed; treat as hard stop and investigate the drift before applying

**3. `hcloud_ssh_key.operator.public_key` drift is the canonical trap.** If `OPERATOR_SSH_PUBLIC_KEY` GH Secret was rotated since the original apply but no apply has run in between, the next apply detects drift (`(sensitive value) # forces replacement` on the ssh_key), which cascades through `ssh_keys = [...] # forces replacement` on `hcloud_server.prod` → server destroyed and recreated. Mitigations: run `tofu plan` on a schedule (not yet wired); or move `ssh_keys` into `ignore_changes` on the server resource alongside the existing `user_data` ignore (tracked in #839 acceptance criteria).

**4. AGENTS.md rules 10 + 11** are the agent-side guardrails for the above. Recovery operators should re-read those before triggering `infra-apply.yml`.

### `infra-apply.yml` mode reference

| Mode | confirm string | When to use | What it does |
| --- | --- | --- | --- |
| `apply` | `APPLY` | Routine apply; state and reality are in sync | `tofu apply` from current state |
| `destroy-then-apply` | `DESTROY_THEN_APPLY` | Want a full rebuild but trust the current state | `tofu destroy` + `tofu apply` (state-driven destroy) |
| `wipe-then-apply` | `WIPE_THEN_APPLY` | State and reality have diverged; surgical fixes failed | API-enumerate-and-DELETE every Hetzner resource + delete prod-tagged Tailscale devices + wipe state file + apply from scratch |
| `wipe-only` | `WIPE_ONLY` | Want the wipe phase alone (e.g. step-by-step recovery with a pause before apply) | Same wipe as above, but no `tofu apply` after |

Inputs:

- `override_server_type` — e.g. `cpx32` when the default `cx43` has no placement capacity.
- `override_location` — e.g. `nbg1` (Nuremberg) or `hel1` (Helsinki) when the default `fsn1` (Falkenstein) is out of capacity.

The orphan-ID inputs (`orphan_server_id`, `orphan_ssh_key_ids`, `orphan_network_ids`, `orphan_firewall_ids`) are present for surgical cleanup but **prefer `wipe-then-apply` once divergence is suspected** — the orphan path leads to death-by-a-thousand-cuts as new orphans surface mid-recovery.

### Co-tenant tailnet publish rules {#co-tenant-tailscale-serve-rules}

The prod VPS hosts multiple apps that share **one** `tailscaled` daemon (one tailnet identity, one serve config). Each app publishes on its own port: podcast_scraper `:443 → 8080`, orrery `:8443 → 8090`, etc. The configs coexist without conflict, but they share global state.

**Rule: every co-tenant's wrapper script must use port-scoped `tailscale serve --https=<port> off`, never the global `tailscale serve reset`.** Reset wipes ALL ports for ALL co-tenants. The 2026-05-29 incident's "every podcast deploy kills orrery's `:8443`" sharp edge was the wrong-shape reset in `podcast-tailscale-serve.sh` — see #845 for the fix.

When adding a third co-tenant (Grafana, status page, anything else that wants tailnet exposure):

1. Pick a unique tailnet HTTPS port (not 443 / 8443 / anything already taken).
2. Write a wrapper at `infra/cloud-init/<app>-tailscale-serve.sh`, mirroring `orrery-tailscale-serve.sh`'s shape.
3. The wrapper does additive publish only, OR port-scoped clear-then-publish — never global reset.
4. Cloud-init `write_files` installs it root-owned; a parallel sudoers file gives `deploy@` narrow NOPASSWD invocation rights.
5. Wire it into the new app's deploy workflow as a post-compose-up step, same pattern as podcast's `deploy.sh` belt-and-suspenders.

The orchestration sibling — one wrapper per app, narrow sudo per script, additive serve config — scales linearly with co-tenant count without ever needing tailscaled-side coordination.

---

## Code/content compatibility {#codecontent-compatibility}

Operator framework for **code** (GHCR image / git tag on the VPS) vs **content** (corpus on disk from a prior pipeline run). Sibling automation lives in [GitHub #796](https://github.com/chipi/podcast_scraper/issues/796) (`produced_by`, `/api/health` preflight, CI matrix). This section is the manual decision tree; [GitHub #797](https://github.com/chipi/podcast_scraper/issues/797) tracks the docs + smoke script.

### Why this section exists

Prod deploy updates **code** every time you dispatch `deploy-prod.yml`. The **corpus** on `/srv/podcast-scraper/corpus` (or `PODCAST_CORPUS_HOST_PATH`) only changes when a pipeline run or restore rewrites it. Those clocks drift. A green `/api/health` only proves the API process is up — not that every viewer tab can read the artifacts it expects. Use this section before each deploy and after each smoke run.

### The risk class — four kinds of mismatch

| Kind | Example |
| --- | --- |
| **New required field** | Code reads `artifact.foo`; old GI/KG file lacks `foo` → `KeyError` or empty rail |
| **New artifact type** | Code reads `search/topic_clusters.json`; file never built → 404 / empty Intelligence panel |
| **Removed/renamed artifact** | Code still loads `*.bridge.json` but pipeline stopped writing bridges |
| **Format restructure** | `nodes: [...]` became `{nodes: {...}}`; parser returns 500 |

### What we defend with today

| Defense | Where |
| --- | --- |
| Per-artifact `schema_version` | GIL **2.0**, KG **1.2**, `corpus_manifest` **1.1.0**, `topic_clusters.json` **2** — bump this table when code changes |
| Read-time migrations | `src/podcast_scraper/migrations/gil_kg_identity_migrations.py` |
| Corpus-level stamp + health preflight | `corpus_manifest.produced_by` + `corpus_version_warning` in `/api/health` ([#796](https://github.com/chipi/podcast_scraper/issues/796)) |
| Post-deploy smoke | `scripts/ops/post_deploy_smoke.sh` wired in `deploy-prod.yml`; local: `make smoke-prod` |
| Dependency map | [`docs/architecture/CORPUS_ARTIFACTS_AND_SURFACES.md`](../architecture/CORPUS_ARTIFACTS_AND_SURFACES.md) |
| Release matrix | [`docs/COMPATIBILITY.md`](../COMPATIBILITY.md) |

### Decision tree before any deploy

| Question | Yes → | No → |
| --- | --- | --- |
| Are new viewer surfaces / endpoints reading artifacts the old pipeline did not produce? | **High risk** — verify files exist on disk or routes return empty cleanly | Likely low risk |
| Did the `schema_version` of any artifact bump? | Check migration helper + smoke that surface | Low risk |
| Is the corpus's last pipeline run date far from the deployed code? | **Compound risk** — prefer reprocess or restore | Close dates → lower risk |
| Recent backup snapshot exists? | Rollback path exists ([Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md)) | **Do not deploy** — fix backups first |
| Does post-deploy smoke hit every surface that reads corpus data? | Real coverage (`make smoke-prod` or GHA step) | `/api/health` alone is **not** sufficient |

### Forward compatibility (rollback is asymmetric)

Rolling **code back** to an older image does not undo corpus mutations from the newer code (new files, bumped `schema_version`, optional fields now required by old readers). Treat **code rollback** and **corpus restore** as separate levers:

- **Code rollback:** re-dispatch `deploy-prod.yml` with `override_image_sha=<prior-sha>` (see [Rollback](#rollback)).
- **Corpus rollback:** `prod-restore-corpus.yml` or `make restore-corpus-prod` from a snapshot **before** the bad deploy.

**Mitigation when shipping:** deprecated fields stay readable for at least one release; release notes state the oldest code tag still safe to roll back to ([COMPATIBILITY.md](../COMPATIBILITY.md)).

### Worked example — v2.6.0 first prod deploy (2026-05-23)

| Check | Outcome |
| --- | --- |
| New surfaces vs old artifacts | Viewer tabs read GI/KG/bridge/search artifacts the v2.5-era pipeline already wrote |
| Schema bumps in the 18-day window | None required for deploy |
| Corpus vs code age | Snapshot from previous day; corpus actively ingested |
| Backup | `snapshot-prod-*` release available |
| Smoke | `/api/health` + Library/Digest/Graph/Search routes returned structured 200s |

Result: **v2.6.0 / sha-4edfee5** deployed successfully. Follow-up hardening: [#796](https://github.com/chipi/podcast_scraper/issues/796), [#797](https://github.com/chipi/podcast_scraper/issues/797).

### Pointers

- Framework + smoke tooling: [#797](https://github.com/chipi/podcast_scraper/issues/797)
- Automated contract: [#796](https://github.com/chipi/podcast_scraper/issues/796)
- Artifact ↔ surface map: [CORPUS_ARTIFACTS_AND_SURFACES.md](../architecture/CORPUS_ARTIFACTS_AND_SURFACES.md)
- Compatibility matrix: [COMPATIBILITY.md](../COMPATIBILITY.md)
- Corpus restore: [Corpus snapshot manifest and restore](CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md)
- Reprocess without re-transcription: [Reprocess a corpus without re-transcribing audio](#reprocess-a-corpus-without-re-transcribing-audio)

### Inspecting compatibility locally

Before dispatching deploy, or when the viewer shows a yellow **corpus version** banner, check the on-disk corpus against the running server:

```bash
# On the VPS (server default output_dir is the corpus mount)
curl -fsS https://prod-podcast.tail-xxxxx.ts.net/api/health | jq \
  '.code_version, .min_supported_corpus_code_version, .corpus_code_version, .corpus_version_warning'

# Viewer-entered path (same query the status bar uses when a corpus path is set)
curl -fsS --get 'https://prod-podcast.tail-xxxxx.ts.net/api/health' \
  --data-urlencode 'path=/srv/podcast-scraper/corpus' | jq \
  '.corpus_produced_by, .corpus_version_warning'
```

Off-host, against a corpus directory you can read:

```bash
make corpus-compat-check CORPUS_DIR=/path/to/corpus
```

Exit code **0** when `produced_by.code_version` meets `min_supported_corpus_code_version`; **1** when a warning would appear in `/api/health`.

### Single-feed vs multi-feed `produced_by`

`corpus_manifest.produced_by` is written only on **multi-feed finalize** (`write_corpus_manifest` after a batch with two or more feeds). Single-feed pipeline runs may leave a manifest with `tool_version` but no `produced_by` object. `/api/health` then emits `corpus_version_warning` and the viewer shows the status-bar banner until you:

1. Run a multi-feed batch that rewrites the manifest, or
2. Use `make reprocess-corpus-from-transcripts CORPUS_DIR=…` (see below).

Legacy corpora without any manifest still warn; readers fall back to per-artifact `schema_version` and migration helpers.

### Post-deploy smoke inventory

`scripts/ops/post_deploy_smoke.sh` (GHA step in `deploy-prod.yml`, local `make smoke-prod`) probes **six** surfaces when `--corpus-path` is set:

| # | Route | Viewer surface |
| --- | --- | --- |
| 1 | `GET /api/health` | Status bar / subsystem flags |
| 2 | `GET /api/corpus/episodes` | Library |
| 3 | `GET /api/corpus/digest` | Digest |
| 4 | `GET /api/artifacts` | Graph (GI/KG/bridge files) |
| 5 | `GET /api/corpus/topic-clusters` | Graph topic overlay |
| 6 | `GET /api/search` | Search |

Health-only mode (no `--corpus-path`) is intentional for stacks without a mounted corpus. Prod deploy passes `SMOKE_CORPUS_PATH` from repo vars when configured.

### Viewer warning banner

When `corpus_version_warning` is non-null, the viewer renders **`data-testid="corpus-version-warning-banner"`** above the status bar. The message is informational — routes may still return 200 with empty panels. Treat it as **compound risk**: deploy succeeded but content may be stale relative to code. Prefer reprocess or restore before relying on Digest / Graph / Search for operator decisions.

### When to reprocess vs restore vs roll back code

| Situation | Action |
| --- | --- |
| Code deployed; corpus age OK but missing new artifact types | `make reprocess-corpus-from-transcripts CORPUS_DIR=…` then re-run smoke |
| Corpus mutated by newer code; rolling code back | Restore snapshot **before** the bad deploy (`prod-restore-corpus.yml`), **then** code rollback with pinned `override_image_sha` |
| `corpus_version_warning` only (semver below minimum) | Reprocess first; restore only if reprocess fails or disk is corrupt |
| Smoke fails on `/api/artifacts` or topic-clusters | Check pipeline last run + index build; do not treat green health as sufficient |

### Release checklist tie-in

`scripts/pre_release_check.py` and `scripts/tools/create_release_notes_draft.py` both require a row for the shipping version in [`docs/COMPATIBILITY.md`](../COMPATIBILITY.md). CI job **`test-corpus-version-compat`** runs current server code against the N-1 fixture corpus (`tests/integration/server/test_corpus_version_compat.py`).

**Validation checklist:** step-by-step local + CI + drill/prod tiers — [Prod compat validation guide](PROD_COMPAT_VALIDATION.md).

---

## Disaster recovery

**Weekly automated drills (#799):** `drill-exercise.yml` runs **Wednesdays 02:00 UTC** (full
provision → deploy → restore → smoke → destroy). Check **`gh run list --workflow drill-exercise.yml`**
for the latest green run. Interpret failures via [DR drill runbook](DR_DRILL_RUNBOOK.md).

**Weekly backup restore proof (#798):** `verify-backup-restore.yml` runs **Sundays 04:00 UTC** on
a GHA runner (no Hetzner cost). See [Backup status](#backup-status).

If the Hetzner instance is irrecoverable (account issue, hardware failure,
accidental `tofu destroy`):

```bash
# 1. Re-provision (~5–10 min)
cd infra
./tofu apply              # same hostname, same Tailscale registration
                          # (cloud-init re-runs the bootstrap)

# 2. Restore corpus (~3–5 min for ~20 MB snapshot)
#    Preferred: prod-restore-corpus.yml (confirm PROD_RESTORE).
#    On-host: ssh deploy@prod-podcast.tail-xxxxx.ts.net \
#      'cd /srv/podcast-scraper && make restore-corpus-prod'

# 3. Re-stage host-side `.env` (operator's responsibility)
#    See "First-time bootstrap → Stage the host-side `.env`".

# 4. Verify (~5 min)
#    See "Smoke validation".
```

**Total wall time: ~15–20 min** assuming the operator knows the runbook.
Corpus is recoverable to within ~24 h of pre-disaster state (last
`backup-corpus-prod.yml` run).

[#724](https://github.com/chipi/podcast_scraper/issues/724) tracks the
end-to-end DR drill that calibrates these numbers against reality.
Complete readiness ([#751](https://github.com/chipi/podcast_scraper/issues/751)) and use [DR drill runbook](DR_DRILL_RUNBOOK.md)
before scheduling that drill.

---

## Prod failover (stand up spare on DR row) {#prod-failover}

When prod is degraded but the VPS is still reachable enough to keep
serving — or when you want a hot spare ready for a planned cutover —
**dispatch `prod-failover-stand-up.yml`**
to bring up a spare on the DR drill VPS row without destroying it.

Phases (RFC-083):

| Phase | Automated? | What runs |
| --- | --- | --- |
| A — Provision | ✅ | `drill-infra-plan` + `drill-infra-apply` (no-op if spare already up) |
| B — Deploy | ✅ | `drill-deploy` (pinned to current image SHA unless overridden) |
| C — Restore | ✅ | `drill-restore-corpus` (newest `snapshot-prod-*` with sibling `snapshot.manifest.json`) |
| D — Validate | ✅ | freeze ingestion → `drill-e2e` → `drill-stack-playwright` |
| **E — Cutover** | ❌ **MANUAL** | DNS flip off-band (operator's laptop, see below) |
| **F — Failback** | ❌ **MANUAL** | Reverse DNS flip + spare teardown (see below) |

ADR-089 prohibits the orchestrator from composing `drill-exercise` or
`drill-infra-destroy`. Spare decommission is a separate manual
`drill-infra-destroy` dispatch with its own `DRILL_DESTROY` confirm
(ADR-091).

### Phase A–D: dispatch the workflow

1. **Decide the image to deploy.** Default is the workflow's own SHA.
   Override only if you need to pin an older known-good tag.
2. **Decide the backup tag.** Leave blank for newest
   `snapshot-prod-*` that has a sibling `snapshot.manifest.json`
   (ADR-092 / RFC-084). Override only for replay scenarios.
3. **Dispatch the workflow:**

   ```bash
   gh workflow run prod-failover-stand-up.yml \
     -f confirm=PROD_FAILOVER_STAND_UP \
     -f override_image_sha= \
     -f backup_tag= \
     -f backup_repo=chipi/podcast_scraper-backup
   ```

4. **Watch the run.** Phases A–D take ~8–12 min in aggregate. If any
   phase fails the spare stays up (no auto-destroy) so you can SSH and
   triage. The `finalize` step prints `::notice::` with the spare's
   tailnet FQDN on success.

5. **Verify ingestion is frozen on the spare.** The `freeze-ingestion`
   job strips `scheduled_jobs` from `corpus/viewer_operator.yaml` on
   the spare (RFC-083 §6 dual-writer guard). The api is **not** restarted
   in this step — the spare is not serving traffic during stand-up
   validation, so the existing process can keep running with its
   pre-freeze config. APScheduler reads the key-less yaml on the api's
   next natural restart (cutover in phase E, or any host/compose
   redeploy in the post-D window). Verify the on-disk yaml on the spare:

   ```bash
   ssh deploy@<DRILL_TAILNET_FQDN> \
     'grep -c "^scheduled_jobs:" /srv/podcast-scraper/corpus/viewer_operator.yaml || true'
   # expect: 0   ← key absent (the ``|| true`` keeps grep's exit 1 from tripping set -e)
   ```

### Phase E (manual): cutover

> **Do not cut over until the workflow's `finalize` step prints
> "Spare validated".** A failed D phase means the spare cannot serve.

The cutover is a **DNS flip** (ADR-090). The exact mechanism depends on
where `podcast.tail-xxxxx.ts.net` (or whichever name your operator
clients use) is resolved.

1. **Pre-cutover snapshot of prod.** Prod is still up; trigger one
   final corpus backup so the spare has the freshest data when you
   flip over. From the operator's laptop:

   ```bash
   gh workflow run backup-corpus.yml \
     -f confirm=PROD_BACKUP \
     -f release_tag=snapshot-prod-pre-cutover-$(date -u +%Y%m%dT%H%M%SZ)
   ```

   Then dispatch `prod-failover-stand-up.yml` again with the new tag
   in `backup_tag=` so the spare is one snapshot fresher. (Optional;
   skip if drift is acceptable.)

2. **Freeze ingestion on prod.** Mirror the spare's state on prod
   itself so two stacks don't both write to corpus:

   ```bash
   ssh deploy@<PROD_TAILNET_FQDN> '
     cd /srv/podcast-scraper
     cp -p corpus/viewer_operator.yaml corpus/viewer_operator.yaml.before-cutover
     python3 -c "
   import yaml, sys
   p = \"corpus/viewer_operator.yaml\"
   d = yaml.safe_load(open(p)) or {}
   d.pop(\"scheduled_jobs\", None)
   yaml.safe_dump(d, open(p, \"w\"), sort_keys=False)
   "
     docker compose -f compose/docker-compose.prod.yml restart api
   '
   ```

3. **Flip the DNS / client target.** Update wherever clients resolve
   prod to point at the spare's tailnet FQDN. Typical paths:

   - **Tailscale serve / funnel rule on prod VPS:** disable the rule
     so traffic stops landing there.
   - **Operator DNS shortcut / bookmark:** repoint to the spare's
     MagicDNS name.
   - **External DNS** (if any): change the CNAME / A record. Watch
     the TTL — flips with TTL > 60s can leave clients pinned to prod
     for several minutes.

4. **Verify the spare is now the primary.** From the operator's
   laptop:

   ```bash
   # 4a. Spare answers on the operator-facing URL
   curl -fsS https://<OPERATOR_URL>/api/health | jq .

   # 4b. Re-enable scheduled_jobs on the spare so it starts ingesting
   ssh deploy@<DRILL_TAILNET_FQDN> '
     cd /srv/podcast-scraper
     # restore the YAML from the pre-failover backup (the freeze
     # script wrote *.preserved-by-failover.<UTC-stamp>)
     cp -p corpus/viewer_operator.yaml.preserved-by-failover.* \
           corpus/viewer_operator.yaml
     docker compose -f compose/docker-compose.prod.yml restart api
     curl -fsS http://127.0.0.1:8080/api/scheduled-jobs | jq ".jobs | length"
   '
   ```

The spare is now the primary. Prod is a "warm reserve" with frozen
ingestion.

### Phase F (manual): failback

Failback is the cutover in reverse. The spare keeps running while you
catch prod up.

1. **Freshen prod from spare's corpus.** Dispatch
   `backup-corpus.yml` against the spare (run from the operator's
   laptop, with the spare's FQDN in the env override), then
   `prod-restore-corpus.yml` against prod.

   > 2.6 `backup-corpus.yml` targets prod by hardcoded FQDN; for v2.6
   > failback, copy the corpus by `rsync` between spare and prod on
   > the tailnet:
   > `rsync -av --delete deploy@<DRILL_TAILNET_FQDN>:/srv/podcast-scraper/corpus/ deploy@<PROD_TAILNET_FQDN>:/srv/podcast-scraper/corpus/`

2. **Freeze ingestion on the spare** (mirror Phase E step 2 against
   the drill FQDN).

3. **Flip DNS back to prod.** Re-enable prod's tailscale serve rule;
   update operator DNS / bookmark back to prod.

4. **Verify prod is primary again** (mirror Phase E step 4 against the
   prod FQDN).

5. **Decommission the spare** by dispatching
   `drill-infra-destroy.yml` with `confirm=DRILL_DESTROY`. This is
   a separate workflow per ADR-091.

### Failover constraints (RFC-083)

- The spare reuses the **DR drill VPS row** of the canonical stack
  contract (ADR-093) — same Tailscale auth, same
  `DRILL_DEPLOY_SSH_PRIVATE_KEY`, same MagicDNS FQDN.
- The orchestrator is **idempotent at the VPS layer**: re-dispatching
  with the spare already up is a no-op for phase A.
- The orchestrator **NEVER** auto-destroys the spare. ADR-089 forbids
  composing `drill-infra-destroy` into the failover workflow.
- The freeze-ingestion guard is **not optional** — without it the
  spare and prod will both fire `scheduled_jobs` against the same
  upstreams (dual-writer corruption).

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
| `PODCAST_DEFAULT_CORPUS_PATH` | path (optional) | Viewer-only: nginx injects `/app/output` so the SPA matches the api corpus mount | status bar empty on first load; catalog APIs not called until you paste `/app/output` |
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

> **Operator note:** if your existing prod VPS is already running and Grafana
> shipping is healthy, harmonizing Grafana env var names in repo templates is a
> no-op for that live host. It only takes effect on bootstrap/re-provision paths
> (for example new VPS from cloud-init, DR rebuild, or explicit host-metrics
> Alloy bootstrap updates).

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

### LLM cost panels (#804)

Pipeline runs with cloud profiles emit structured JSON log lines (`event_type: llm_cost`; Loki
`| json` when Grafana Agent or docker log shipping is enabled). Import
`config/grafana/grafana-dashboard-llm-cost.json` into the Grafana Cloud **podcast-scraper**
folder. Per-run soft caps and per-run Sentry cost alerts are configured via profile fields
`cost_soft_cap_usd_per_run`, `cost_soft_cap_action`, and `cost_daily_alert_usd` (override with
`COST_SOFT_CAP_USD_PER_RUN`, `COST_SOFT_CAP_ACTION`, `COST_DAILY_ALERT_USD` env vars when needed).

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

#### Grafana dashboards and alert rules (env filter) {#grafana-env-filter-gh-726}

When prod and pre-prod both ship metrics and logs to the same Grafana
Cloud stack, any panel or alert that omits **`env`** will aggregate
both environments. Track importable JSON and operator steps in GitHub
[issue #726](https://github.com/chipi/podcast_scraper/issues/726).

**Repo dashboards (`config/grafana/`):** Re-import or overwrite panels
from git after changes. Each JSON dashboard in the podcast-scraper set
includes a template variable **`env`** (default **`prod`**; choose
**`preprod`** for Codespaces — these values match `PODCAST_ENV` in
`compose/grafana-agent.yaml` `external_labels`, not the prose spelling
"pre-prod"). Prometheus panels use **`env="$env"`** in selectors; Loki
panels already used the same pattern.

**Grafana Cloud alert rules (not stored in this repo):** Edit every
rule whose query touches podcast-scraper metrics and add **`env="prod"`**
to the PromQL selector so pre-prod traffic cannot fire prod alerts.
Log-based alerts should filter on **`{env="prod", ...}`** the same way.

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

### Sentry Slack routing (prod vs pre-prod, GH-725) {#sentry-slack-routing-prod-vs-pre-prod-gh-725}

**Decision (RFC-082 Open Question 3):** [Option
B](https://github.com/chipi/podcast_scraper/issues/725) — keep a **single
DSN per component** and split noise in **Sentry** using the
**`environment`** tag already set by `init_sentry()` from `PODCAST_ENV`
(`prod` on the VPS `.env`; `preprod` in the default Codespace — see
`.devcontainer/devcontainer.json`). Option A (separate prod DSNs) is
documented in the issue for teams that prefer separate Sentry projects.

**Prod-only Slack path (Sentry UI, per project: api and pipeline):**

1. Ensure Slack is installed under **Settings → Integrations → Slack**
   for the Sentry org (or use the project-level Slack integration).
2. Open **Alerts → Create Alert** (Issue alert) for `podcast-scraper-api`.
3. Set **When** to the issue volume you want (e.g. new issue, or regressed).
4. Under **If**, add a filter on **event environment** (wording varies by
   Sentry version: e.g. **The event's environment attribute** or **An
   event's tags** with key `environment`) **equals** `prod` (must match
   `PODCAST_ENV` on the VPS exactly).
5. Under **Then**, choose **Send a notification via an integration** →
   Slack → channel `#podcast-prod-alerts` (or your prod channel).
6. Repeat for `podcast-scraper-pipeline` if pipeline issues should also
   notify Slack.

**Pre-prod path:** create a second alert (or use the same rule with a
different **Then** branch if your Sentry plan supports it) with
**If** `environment` **equals** `preprod` targeting `#podcast-preprod-alerts`,
or **do not** attach Slack (pre-prod stays Sentry-email / UI only).

**Acceptance checks (operator):**

1. From prod: run the **Sentry validation ping** in [Smoke
   validation](#smoke-validation) and confirm only the **prod** Slack
   route fires.
2. From a Codespace (default `PODCAST_ENV=preprod`): send a test event
   (same Python snippet with `init_sentry("api")`) and confirm it does
   **not** hit the prod-only rule (it should match the pre-prod rule or
   stay unrouted).

Sentry product reference: [Issue
alerts](https://docs.sentry.io/product/alerts/) and filter conditions on
event attributes.

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
5. Pull + restart on prod: `gh workflow run deploy-prod.yml --repo chipi/podcast_scraper -f confirm=PROD_DEPLOY`.

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

On **new or reprovisioned** VPS hosts, cloud-init installs
`/usr/local/sbin/podcast-tailscale-serve.sh` and `podcast-scraper.service`
runs it after `docker compose up` (and clears serve on `compose down`).
The canonical script source is `infra/cloud-init/podcast-tailscale-serve.sh`
(injected by OpenTofu into `prod.user-data`); deploy and drill CI can copy it
onto the host with `sudo install …` when `/etc/sudoers.d/99-podcast-deploy-tailscale-serve`
includes that `install` rule.
Existing servers created before that change keep whatever serve state they
had; run the manual steps below once if MagicDNS HTTPS still fails.

Set up manually on older hosts (or to verify):

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
requires editing `infra/terraform/main.tf` and an explicit edge security
design (do not rely on nginx Basic Auth alone).

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

### Corpus directory (host vs `/app/output` in containers) {#corpus-directory-host-vs-appoutput}

**Source of truth on disk:** the host directory in
`PODCAST_CORPUS_HOST_PATH` inside `/srv/podcast-scraper/.env`. Always
confirm before runbooks, backups, or one-off CLI:

```bash
grep '^PODCAST_CORPUS_HOST_PATH=' /srv/podcast-scraper/.env
```

The [cheat sheet](PROD_OPERATOR_CHEAT_SHEET.md) must not hard-code a corpus
path without telling operators to run that `grep`.

**Inside `api` / pipeline containers** the same tree is mounted at
`/app/output` (default `serve --output-dir` in `docker/api/Dockerfile`).

Compose may implement `corpus_data` as a **named volume** whose data
directory still reflects that host path from the initial bind
definition in `compose/docker-compose.prod.yml` — validate with
`docker inspect <api-container>` mounts if you need to see Docker's
view.

### Topic clusters missing after a successful pipeline run {#topic-clusters-missing-after-a-successful-pipeline-run}

**Expected behaviour:** `POST /api/jobs` runs
`full_incremental_pipeline` only. Finalize calls `maybe_index_corpus`
when `vector_search` is true in the profile (see
`src/podcast_scraper/workflow/orchestration.py`), but nothing in that
path runs `topic-clusters`. So `search/topic_clusters.json` can be
absent even when `search/vectors.faiss` exists.

**Profiles:** `cloud_thin` sets `vector_search: false` — no FAISS index
from the pipeline; clustering cannot run until an indexing-capable
profile has built `vectors.faiss`. See profile YAMLs under
`config/profiles/`.

**What to run:** follow [Prod operator cheat sheet — Topic clusters](PROD_OPERATOR_CHEAT_SHEET.md#topic-clusters-manual-maintenance)
(host venv with `.[search]`, or `docker exec` into the running `api`
container). Reference clustering design in
[RFC-075](../rfc/RFC-075-corpus-topic-clustering.md) and the API in
[Server guide — `GET /api/corpus/topic-clusters`](SERVER_GUIDE.md).

**Troubleshooting:**

- **`topic-clusters` exits with missing FAISS:** ensure
  `$HOST_CORPUS/search/vectors.faiss` exists (or
  `/app/output/search/vectors.faiss` in-container).
- **404 on `GET /api/corpus/topic-clusters`:** artifact not built or wrong
  `path` query versus server default.
- **Host `python3 -m venv` fails (`ensurepip`):** install
  `python3.12-venv` (or distribution equivalent) with sudo once, or use
  the `docker exec` route in the cheat sheet.
- **Artifact owned by root** after an in-container write: `chown` back
  to `deploy` (and corpus group) so backups and host tools stay consistent.

### Reprocess a corpus without re-transcribing audio {#reprocess-a-corpus-without-re-transcribing-audio}

Use this when extraction/summarization/index logic changed and you want to
recompute derived outputs from existing transcript files.

**Safe rule:** keep `transcripts/`; rebuild downstream artifacts only.

1. Resolve corpus root from host env:

   ```bash
   grep '^PODCAST_CORPUS_HOST_PATH=' /srv/podcast-scraper/.env
   ```

2. Optional rollback snapshot (derived outputs only), then remove stale
   derived layers (`search/`, old `metadata/` dirs / old run outputs). Keep
   `transcripts/` intact.

3. Re-run pipeline with **both** flags:

   - `--skip-existing` (reuse transcript files already on disk)
   - `--no-transcribe-missing` (do not invoke Whisper/audio transcription)

   Example (host Python route):

   ```bash
   ssh deploy@prod-podcast.<tailnet>.ts.net
   cd /srv/podcast-scraper
   HOST_CORPUS=$(grep '^PODCAST_CORPUS_HOST_PATH=' .env | cut -d= -f2-)
   .venv/bin/python -m podcast_scraper.cli \
     --config "$HOST_CORPUS/viewer_operator.yaml" \
     --feeds-spec "$HOST_CORPUS/feeds.spec.yaml" \
     --output-dir "$HOST_CORPUS" \
     --skip-existing \
     --no-transcribe-missing
   ```

4. Optional post-step: rebuild topic clusters if the index exists:

   ```bash
   .venv/bin/python -m podcast_scraper.cli topic-clusters --output-dir "$HOST_CORPUS"
   ```

For a copy/paste operator sequence (including backup and cleanup commands),
see [Prod operator cheat sheet — Reprocess from existing transcripts (no re-transcription)](PROD_OPERATOR_CHEAT_SHEET.md#reprocess-from-existing-transcripts-no-re-transcription).

**Makefile shortcut (host checkout with venv):**

```bash
make reprocess-corpus-from-transcripts CORPUS_DIR=/srv/podcast-scraper/corpus
```

Requires `CORPUS_DIR` pointing at the corpus parent (contains `feeds.spec.yaml` and `transcripts/`).

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

The healthcheck probe (`wget` against `/`) accepts any `HTTP/` status
line as alive (`compose/docker-compose.stack.yml`). If you still see
`unhealthy`, you're on an old viewer image — pull the latest.

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

### "Cursor or automation cannot `ssh deploy@prod`"

First confirm **Tailscale** on the machine that runs the command:

```bash
tailscale status | head -5
```

You should see `prod-podcast-1` (or similar) as **active**. SSH to the VPS is
intended **over the tailnet** only; inbound SSH on the public Hetzner IPv4 is
not exposed in the default firewall.

Then confirm **OpenSSH has a usable key** for that shell:

```bash
ssh-add -l
```

If this prints **"The agent has no identities."**, no key is loaded for that
process. Keys added only in another app (for example Terminal.app) do not
always end up on the **same** `SSH_AUTH_SOCK` that a Cursor agent subprocess
inherits.

**Time-limited agent load (recommended for IDE / agent `ssh` to prod):** in
**Cursor’s integrated terminal** for this workspace, load the operator private
key into `ssh-agent` with a lifetime so it is not left loaded indefinitely:

```bash
ssh-add -t 30m ~/.ssh/id_ed25519
ssh-add -l
```

After **30 minutes** the key is removed from the agent again; `ssh-add -l` will
go back to empty and agent-driven `ssh` will fail until you re-run `ssh-add
-t …`. That matches “it worked earlier today, then stopped.”

If your operator key is not `~/.ssh/id_ed25519`, substitute the path that
matches `TF_VAR_ssh_public_key` / `secrets.OPERATOR_SSH_PUBLIC_KEY`.

**macOS Keychain persistence (optional):** if you prefer the key to survive
agent restarts until reboot:

```bash
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
ssh-add -l
```

Sanity check:

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/<operator-private-key> \
  deploy@prod-podcast-1.<tailnet> 'echo ok'
```

Replace `<tailnet>` with your MagicDNS suffix (for example `tail6d0ed4.ts.net`).
If this fails but `tailscale status` shows prod as active, you are using the
wrong private key file for `deploy@`.

**GitHub Actions** does not use your laptop `ssh-agent`; it uses
**`PROD_SSH_PRIVATE_KEY`** and `.github/actions/prod-ssh-key` (see
[GitHub Actions SSH to prod](#github-actions-ssh-to-prod-prod_ssh_private_key)).

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
- **The published api image only ships `[dev]` extras.** Pipeline
  runs MUST go through Docker job mode (`PODCAST_PIPELINE_EXEC_MODE=docker`,
  set in prod overlay). In-process pipeline runs would crash on
  missing `[llm]` deps.

---

## Cross-references

- [RFC-082 — design](../rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [RFC-087 — public edge + multi-compose (Draft)](../rfc/RFC-087-vps-public-edge-multi-compose.md)
- [Prod operator cheat sheet](PROD_OPERATOR_CHEAT_SHEET.md)
- [VPS multi-app onboarding](VPS_MULTI_APP_ONBOARDING.md)
- [`infra/`](https://github.com/chipi/podcast_scraper/tree/main/infra) — IaC code
- [`tailscale/policy.hujson`](https://github.com/chipi/podcast_scraper/blob/main/tailscale/policy.hujson) — ACL
- [`.github/workflows/deploy-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/deploy-prod.yml)
- [`.github/workflows/infra-apply.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/infra-apply.yml) — manual apply gate
- [`.github/workflows/backup-corpus-prod.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/backup-corpus-prod.yml)
- [`.github/workflows/prod-restore-corpus.yml`](https://github.com/chipi/podcast_scraper/blob/main/.github/workflows/prod-restore-corpus.yml) — manual prod corpus restore from backup releases (confirm **PROD_RESTORE**)
- [#714](https://github.com/chipi/podcast_scraper/issues/714) — account prereqs checklist
- [#723](https://github.com/chipi/podcast_scraper/issues/723) — Phase B cutover
- [#724](https://github.com/chipi/podcast_scraper/issues/724) — DR drill
- [#751](https://github.com/chipi/podcast_scraper/issues/751) — DR drill prerequisites (before #724)
