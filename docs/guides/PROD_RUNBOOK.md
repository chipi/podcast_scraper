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

1. [First-time bootstrap](#first-time-bootstrap)
2. [Daily operations](#daily-operations)
3. [Basic-auth credentials](#basic-auth-credentials)
4. [Corpus migration from pre-prod (Codespace) to prod (VPS)](#corpus-migration)
5. [Rollback (deploy went red mid-way)](#rollback)
6. [Disaster recovery (VPS gone)](#disaster-recovery)
7. [Credential rotation](#credential-rotation)

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
# 1Password as `sops/podcast-scraper/tofu-state-age-key`.

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

### Stage the host-side `.env` (one-time, post-apply)

The `.env` holds runtime secrets (provider API keys, Grafana credentials,
Sentry DSN). Cloud-init drops a sentinel file `/srv/podcast-scraper/.bootstrap-needs-env`;
the systemd unit refuses to start while it exists.

```bash
ssh deploy@prod-podcast.tail-xxxxx.ts.net   # over Tailscale
sudo install -o deploy -g deploy -m 600 /dev/stdin /srv/podcast-scraper/.env <<'ENV'
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
GRAFANA_CLOUD_PROM_URL=...
GRAFANA_CLOUD_LOKI_URL=...
GRAFANA_CLOUD_USER=...
GRAFANA_CLOUD_API_KEY=...
PODCAST_SENTRY_DSN_API=...
PODCAST_SENTRY_DSN_PIPELINE=...
PODCAST_ENV=prod
PODCAST_AVAILABLE_PROFILES=cloud_balanced,cloud_thin
PODCAST_DEFAULT_PROFILE=cloud_balanced
ENV

# Stage the basic-auth password file too (see #Basic-auth credentials below).
sudo htpasswd -bcB /srv/podcast-scraper/.htpasswd <user> <pass>
sudo chown root:docker /srv/podcast-scraper/.htpasswd
sudo chmod 640         /srv/podcast-scraper/.htpasswd

# Release the systemd gate.
sudo rm /srv/podcast-scraper/.bootstrap-needs-env
sudo systemctl restart podcast-scraper.service
```

### Smoke validation

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
3. Update 1Password entry `Hetzner Cloud / podcast-scraper-prod / api-token`.
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
3. Update 1Password (entry: `Tailscale / podcast-scraper / authkey`).
4. Revoke the old auth key in the Tailscale admin.
5. Trigger one workflow_dispatch run of `deploy-prod.yml` to confirm.

**Rotating `TS_API_KEY`:**

1. [admin/settings/keys](https://login.tailscale.com/admin/settings/keys) → API access tokens → Generate new.
2. `gh secret set TS_API_KEY --repo chipi/podcast_scraper --app actions --body '<tskey-api-...>'`
3. Update 1Password (entry: `Tailscale / podcast-scraper / api-key`).
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
4. Update 1Password (keep the OLD key for ~30 days as a safety net).
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
