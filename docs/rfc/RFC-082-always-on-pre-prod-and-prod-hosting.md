# RFC-082: Production hosting (always-on VPS, IaC + GitOps)

> **Status:** Draft. Picks up where RFC-081 ended.
>
> **Naming:** RFC-081 covers **pre-prod** (Codespaces, auto-suspend, ad-hoc validation surface). This RFC covers **prod** — the always-on host that real corpus building runs against. Pre-prod and prod coexist; pre-prod stays as a free fallback / smoke surface.

## Abstract

RFC-081 shipped pre-prod on GitHub Codespaces. It works at hobby scale
($0/month) but auto-suspends, has a 60-120 hr/mo compute cap, and
exposes an unstable forwarded URL that home-automation and scheduled
callers can't subscribe to.

This RFC defines **prod**: an always-on Hetzner VPS, on the operator's
existing Tailscale tailnet, deployed via GitOps (Terraform-managed
infra + GHA-driven app deploy), with the same observability + backup
contract as pre-prod. The published GHCR image set
(`api`, `viewer`, `pipeline-llm`) is reused **unchanged** — same
`:main` and `:sha-<short>` tags, same compose files. The only deltas
are host-side glue: provider provisioning, Tailscale auth, deploy
trigger, and bind-mount path.

The infra is described entirely in `infra/` (Terraform + cloud-init +
deploy script) so the same definition can rebuild the VPS from scratch
into a clean state in <30 min.

## Problem Statement

Pre-prod (Codespaces) caveats that motivate prod:

- **Auto-suspend latency.** Tapping the bookmarked URL on a phone wakes
  the codespace cold (~30 s). Acceptable for a deliberate check, painful
  for "did the alert fire because of X?" reactive workflows.
- **Compute quota.** 60–120 core-hours / month is a couple hours / day.
  A real ingestion run touching 100+ episodes can chew through hours
  per session; multiple sessions / week start pushing the cap.
- **No persistent always-on URL.** The Codespaces forwarded URL contains
  the codespace name, which is short-lived. External integrations
  (Home Assistant, Slack apps, an iOS shortcut) prefer a stable URL.
- **No service-to-service auth surface.** All operator actions require
  a GitHub UI session. No headless-callable endpoint without a
  separate auth wall.

## Goals

1. **Always-on viewer + api** with sub-second response from a stable
   URL on the operator's tailnet.
2. **Existing tailnet identity** for human + headless callers.
   Operator's iOS/macOS Tailscale clients reach the viewer at
   `prod-podcast.<tailnet>.ts.net` with TLS, no per-session re-auth.
   Home Assistant + Shortcuts on the same tailnet hit `/api/jobs`
   directly.
3. **Lift-and-shift, not redesign.** Reuse the GHCR `:main` tags +
   compose files + grafana-agent.yaml unchanged. Operator who knows
   pre-prod can read prod in 30 minutes.
4. **Same observability surface.** grafana-agent + Sentry continue to
   work; existing dashboards / Slack alerts don't break.
5. **Same backup contract.** `backup-corpus.yml` continues to push
   daily snapshots to `chipi/podcast_scraper-backup` releases; only
   the source path changes.
6. **Cost ceiling: ≤ $20/month all-in** for hobby scale.
7. **Infrastructure-as-code, end-to-end.** The VPS, its network
   surface, its Tailscale registration, the OS bootstrap, and the
   deploy trigger are all defined in `infra/` and applied via PR.
   Re-creating prod from scratch is a `terraform apply` + a
   `git push` away.
8. **GitOps deploy loop.** Push to main → CI green → image published
   → deploy fires automatically → host pulls and restarts. No SSH-
   from-the-laptop steps in the normal flow.

## Non-Goals

- Multi-region / HA — single VPS is fine.
- Multi-tenant identity layer — collaborator access is granted by
  adding their tailnet identity to the ACL.
- Public-internet open ports — everything sits inside the tailnet.
- Production-grade hardening (S3 log shipping, secrets rotation
  cadence, blue/green deploys). This is "stable enough to rely on",
  not "five nines".
- Replacing the api's job-factory model. Pipeline still runs as
  one-shot subprocesses spawned by `docker compose run pipeline-llm`.
- Replacing GHCR. We continue to publish from the same stack-test
  workflow.
- Kubernetes. One operator + a handful of services + a hobby corpus
  doesn't need it.

## Use Cases (delta vs. pre-prod)

In addition to RFC-081's use cases, prod enables:

1. **Phone tap on a stable URL** wakes the viewer in <2 s (no cold
   start). Operator opens the bookmark from a notification and lands
   on a live page.
2. **Home Assistant subscribes to `/api/jobs` events** without GitHub
   auth. A failed pipeline can flash a smart-light or trigger a
   Shortcut.
3. **Cron-style ingestion** (optional, see Open Question 4) without
   an operator opening a tab. A scheduled feed-sweep at 04:00 local
   without keeping a codespace warm.
4. **Collaborator UAT** without granting `codespace` repo permission.
   Tailscale ACL handles invite + revoke at the network level.

## Design

The design is **pre-prod minus Codespaces, plus Hetzner + Tailscale +
IaC + a GitOps deploy loop**:

```text
[ Reused unchanged from pre-prod ]
  - GHCR image set: api, viewer, pipeline-llm
  - compose/docker-compose.stack.yml + docker-compose.prod.yml
  - grafana-agent.yaml + Grafana Cloud
  - Sentry init wired in api + pipeline subprocess
  - backup-corpus.yml uploads to chipi/podcast_scraper-backup
  - viewer's mobile control plane (Library / Search / Configuration)

[ New for prod ]
  1. infra/ directory: Terraform + cloud-init + deploy script
  2. Hetzner CX43 provisioned via Terraform (see Decision 1)
  3. Tailscale daemon on the VPS, auto-joins tailnet via auth key
  4. GHA workflow deploy-prod.yml fires on stack-test success → main
  5. Host bind-mount: /srv/podcast-scraper/corpus
  6. Backup-corpus.yml points SSH at the VPS over Tailscale
```

### Decision 1 — Hosting target

**Hetzner CX43** (8 vCPU shared Intel/AMD, 16 GB RAM, 160 GB SSD,
20 TB traffic, **€15.11/mo** as of May 2026, EU). Operator-confirmed
EU geography + want predictable flat billing.

CX43 is the cheapest 8-core option in Hetzner's lineup — about **25 %
less than the same-shape CAX31** (ARM Ampere, €19.95/mo) and roughly
half the cost of the dedicated-AMD CCX23 (€31.49/mo, broke ceiling).
CX43 + 50 GB Volume (~€2.86/mo) totals ~€18/mo, well under the $20
ceiling.

> **Pricing volatility caveat.** Hetzner pushed a 30–37 % hike on
> 2026-04-01 and renames the CX line periodically (CX22 / CX32 /
> CX42 → CX23 / CX33 / CX43 in 2024). The Hetzner pricing page
> renders prices client-side so cached/scraped figures (e.g.
> costgoat) lag reality. **Re-check at
> [hetzner.com/cloud/cost-optimized](https://www.hetzner.com/cloud/cost-optimized)
> before every apply.** Earlier RFC drafts referenced stale
> €15.99 CAX31 and €11.99 CX43 figures; both have since increased.

**CX (shared Intel/AMD) vs. CAX (ARM Ampere) for this stack.** The
prod image set is cloud-thin (`INSTALL_EXTRAS=llm`): no torch /
spaCy / FAISS / Whisper / llama-cpp. The arm64-native-wheel concern
is therefore irrelevant — both architectures work. The trade-off
collapses to:

| Dimension | CX43 (chosen) | CAX31 (alternative) |
| --- | --- | --- |
| Monthly price (May 2026) | **€15.11** | €19.95 (~25 % more) |
| Per-core perf for ffmpeg | OK; depends on noisy-neighbor luck | Roughly equal; Ampere has good codec instructions |
| Per-core perf for Python / HTTP | Roughly equal | Roughly equal |
| Native Python wheels | All amd64-native (zero risk) | All arm64 wheels exist; multi-arch publish required (#712 / #737, already merged) |
| Noisy-neighbor risk | Higher — x86 shared cloud is in heavier demand | Lower — ARM cloud demand has historically been quieter |
| GHCR storage | Single-arch | Multi-arch ≈ 2× (free for public repos, no impact) |
| CI publish wall-time | Native | ~2× longer under QEMU emulation on amd64 runners (already paid by #737) |

The two genuine reasons to prefer CAX31 over CX43:

1. **Anti-jitter premium** — €4.84/mo (~25 %) for less variability
   from noisy x86 neighbors. Worth it only if shared-CPU jitter
   actually shows up in measured ffmpeg / Whisper wall-times.
2. **Energy efficiency / philosophy** — Ampere is more
   power-efficient and Hetzner's ARM fleet runs greener. Real but
   soft factor.

For a single-operator hobby workload (ffmpeg, cloud-LLM SDK calls,
static viewer, grafana-agent), neither outweighs the 25 % cost
delta. **Pick CX43; escalate only on measured signal.**

**Alternatives kept on the bench (only if CX43 falls short):**

| Trigger | Move to | Specs | Price (May 2026) |
| --- | --- | --- | --- |
| Want the anti-jitter premium of ARM Ampere | **CAX31** | 8 vCPU ARM Ampere, 16 GB, 160 GB | €19.95/mo |
| Shared-CPU jitter visible in measured ffmpeg / Whisper wall-times on CX43 | **CCX23** | 4 dedicated AMD vCPU, 16 GB, 160 GB | ~€31.49/mo (re-verify) |

Switching CX43 → CAX31 is a single Terraform variable change
(`server_type = "cax31"`) — multi-arch publish is already merged
in #712 / #737, so no CI work is needed. CCX23 breaks the $20
ceiling and should only be picked if measured numbers demand it.

**Storage add-on:** a separate Hetzner Volume (~€0.057/GB/month;
re-verify) for the corpus bind-mount, so the VPS image can be
replaced without touching corpus data. ~€2.86/mo for 50 GB.
Recommended once corpus grows past ~20 GB on the boot disk.

### Decision 2 — Auth wall: Tailscale

Operator-confirmed: Tailscale already in use; this is the path of
least resistance.

**What this gives:**

- VPS joins the tailnet on boot via a Tailscale auth key (rotated
  per-machine, scoped to a tag like `tag:prod`).
- Operator phones / laptops already on the tailnet reach
  `prod-podcast.<tailnet-name>.ts.net` with no extra login.
- Tailscale's MagicDNS provides a stable hostname; `tailscale cert`
  issues a Let's Encrypt-backed TLS cert for that hostname (no Caddy
  / Traefik / Cloudflare needed).
- ACLs in `tailscale/policy.hujson` (in this repo) declare which
  tag-or-user can reach `tag:prod`. Collaborator access = add their
  identity, push to main, Tailscale picks up the change.
- Home Assistant on the same tailnet calls
  `https://prod-podcast.../api/jobs` directly, no extra auth layer.

**What this gives up vs. Cloudflare option:**

- No public-internet stable URL. Slack apps / external services
  can't directly call the api unless they're on the tailnet (HA can;
  Slack can't). For Slack-driven workflows, the chain becomes
  Slack → GHA workflow_dispatch (which IS reachable) → GHA SSHes the
  VPS over Tailscale → triggers the api. One extra hop.

The Slack-direct-to-api path was speculative anyway; the Slack →
GHA → api fan-out is fine for hobby scale.

> **Free-plan auth model.** OAuth clients (the typical "service account"
> path on Tailscale) are gated to Premium+ tiers. On Personal Free we
> use **two separate credentials**:
>
> | Credential | Purpose |
> | --- | --- |
> | `TS_AUTHKEY` | Device-level auth: joins the GHA runner to the tailnet (used by `tailscale/github-action@v2`); also passed to cloud-init for the VPS's own `tailscale up`. |
> | `TS_API_KEY` | Tailscale management API: terraform's `tailscale` provider uses this to sync the ACL from `tailscale/policy.hujson` and create per-server auth keys. |
>
> Both are **Tailscale Personal API access tokens / auth keys** (different
> tabs of the same admin panel: [admin/settings/keys](https://login.tailscale.com/admin/settings/keys)).
> Both expire ≤ 90 days on Free plan and need calendar-driven rotation.
> See PROD_RUNBOOK.md "Tailscale credentials (Free-plan workaround)" for
> rotation procedure. If the operator ever upgrades to Premium+, both
> can be replaced with a single OAuth client and the workflows / terraform
> provider switch back to `oauth-client-id` / `oauth_client_id` (one
> diff, no architecture change).

### Decision 3 — Deploy mechanism: push from GHA (over Tailscale)

A new `.github/workflows/deploy-prod.yml`:

```yaml
on:
  workflow_run:
    workflows: ["Stack test"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: tailscale/github-action@v2
        with:
          # Free-plan substitute for OAuth (Premium+ feature). See note below
          # the snippet on the two-credential model.
          authkey: ${{ secrets.TS_AUTHKEY }}
          tags:    tag:gha-deployer
      - run: |
          ssh deploy@prod-podcast.<tailnet>.ts.net "cd /srv/podcast-scraper && \
            docker compose -f compose/docker-compose.stack.yml \
                           -f compose/docker-compose.prod.yml \
              pull && \
            docker compose -f compose/docker-compose.stack.yml \
                           -f compose/docker-compose.prod.yml \
              up -d --remove-orphans"
```

The runner ephemerally joins the tailnet (`tag:gha-deployer`),
reaches the VPS over MagicDNS, runs the same `compose pull && up`
that pre-prod's `start.sh` does. ACL allows `tag:gha-deployer →
tag:prod:22` only.

This mirrors today's `deploy-codespace.yml` pattern — same trigger,
same observability (workflow log surfaces deploy outcome), same
single source of truth ("what's running on prod" = "last successful
deploy-prod workflow run").

### Decision 4 — Persistence + backup

**Bind-mount path: `/srv/podcast-scraper/corpus`** on the VPS (Linux
convention). The compose prod overlay reads it from
`PODCAST_CORPUS_HOST_PATH` env (already wired for pre-prod; just
exported differently in the host's `.env`).

**Backup mechanism: GHA-side, same as pre-prod.** The
`backup-corpus-prod.yml` workflow joins the tailnet using the same
`TS_AUTHKEY` device-auth credential as the deploy workflow (Free-plan
substitute for OAuth — see Decision 2 note), reaches the VPS, tarballs
the corpus, uploads to `chipi/podcast_scraper-backup` releases. The
single change vs. pre-prod's backup is the SSH target hostname —
`prod-podcast.<...>` instead of the codespace name.

This keeps backup logic in code-review (`.github/workflows/...`)
rather than as a host-side cron that escapes git visibility.

**Optional: Hetzner Volume snapshots.** If corpus moves to a
detached Volume, Hetzner's automatic snapshot feature is a
zero-config secondary safety net (€0.0143/GB/month, post-2026-04-01
pricing). Out of scope for this RFC unless corpus grows large
enough to matter.

### Decision 5 — IaC: Terraform with the hetznercloud provider

`infra/` directory in this repo, structured as:

```text
infra/
├── terraform/
│   ├── main.tf              # provider, server, network, firewall, volume
│   ├── tailscale.tf         # tailscale_tailnet_key, ACL update if managed
│   ├── outputs.tf           # tailnet hostname, IPv4
│   ├── variables.tf
│   ├── backend.tf           # remote state — local file, encrypted in repo via sops + age
│   └── README.md
├── cloud-init/
│   └── prod.user-data       # bootstraps Docker + tailscale + deploy user + pulls repo
├── deploy/
│   └── deploy.sh            # called by deploy-prod.yml after SSH; idempotent
└── README.md                # runbook: bootstrap, replace, rollback
```

**Provider stack:**

- **OpenTofu** (Terraform fork) — open-source, drop-in compatible
  with the `hetznercloud/hcloud` provider, also supports the
  `tailscale/tailscale` provider. Avoids the BSL-license question on
  HashiCorp Terraform.
- **`hetznercloud/hcloud` provider** — declares the CX43 server,
  attached Volume, firewall rules (default-deny inbound; allow
  outbound), SSH key registration.
- **`tailscale/tailscale` provider** — declares the auth key (so
  Terraform produces the per-server key cloud-init uses), and
  optionally manages ACL syncs from `tailscale/policy.hujson`.
- **cloud-init `user_data`** baked into the Hetzner instance config
  — installs Docker, Tailscale, creates `deploy` user, pulls
  `chipi/podcast_scraper` to `/srv/podcast-scraper`, copies a
  per-host `.env` from a known location (operator-managed).

**State storage: repo-encrypted with sops + age (free, zero new vendor).**
OpenTofu state lives in `infra/terraform/terraform.tfstate.enc`, encrypted
with [age](https://github.com/FiloSottile/age) via [sops](https://github.com/getsops/sops).
The age private key is stored in the operator's 1Password; sops decrypts
in-memory before each `tofu` invocation.

```bash
# One-time setup (operator's laptop)
brew install sops age
age-keygen -o ~/.config/sops/age/keys.txt   # save the public key in .sops.yaml
# Save the private key contents to 1Password as "tofu-state-age-key"

# Per-operation
sops -d infra/terraform/terraform.tfstate.enc > /tmp/state.tfstate
TF_STATE=/tmp/state.tfstate tofu plan
sops -e /tmp/state.tfstate > infra/terraform/terraform.tfstate.enc
shred /tmp/state.tfstate   # never let plaintext touch the repo
```

A wrapper `infra/tofu` script automates the decrypt → run → re-encrypt
loop so operators don't manage the dance manually.

**Trade-offs vs. alternatives:**

| Dimension | sops + age (chosen) | Hetzner Object Storage | Terraform Cloud |
| --- | --- | --- | --- |
| Cost | $0 | ~€0.50/mo | $0 free tier (5 users) |
| Vendor surface | none new | one (already on Hetzner) | new |
| State in git history | yes (encrypted blob) | no | no |
| Remote locking | no — fine for team-of-1 | yes | yes |
| Restore-after-laptop-loss | restore age key from 1Password | re-auth API token | re-login |

For team-of-1 hobby scale, the locking concern doesn't apply and
`git diff` showing an encrypted blob is genuinely useful as a "state
changed" signal. If we ever go multi-operator, swap to Object Storage
— migration is a one-time `sops -d | tofu state push`.

**Bootstrap flow:**

1. Operator clones repo locally, runs `cd infra/terraform && tofu apply`.
2. OpenTofu provisions Hetzner CX43 + attaches Volume + registers
   tailscale auth key.
3. cloud-init runs on first boot: installs Docker, Tailscale, joins
   tailnet, pulls repo, copies operator-staged `.env`, runs
   `start.sh` (same script as pre-prod, picks up at `compose pull`).
4. Stack comes up; tailscale assigns MagicDNS hostname.
5. Operator verifies `https://prod-podcast.<tailnet>.ts.net/api/health`
   from phone.

**Replacement flow:** `tofu destroy && tofu apply` — fresh VPS, same
hostname (Tailscale assigns based on machine name in code), corpus
restored from latest backup-repo release. <30 min wall.

### Decision 6 — GitOps loop

The **single source of truth** for what's running on prod is **main**.
Two flows touch it:

**App deploy** (most common):

```text
PR opens → CI on PR → green → review → squash-merge to main
  → stack-test workflow_run on main → green
  → publish job pushes new GHCR images
  → deploy-prod workflow_run → tailnet SSH → docker compose pull && up
  → deploy log lands in GHA UI
```

Same shape as pre-prod's chain, just pointed at the VPS.

**Infra deploy** (rare):

```text
PR changes infra/** → CI on PR (terraform fmt + validate + plan)
  → review the plan output → squash-merge to main
  → infra-apply workflow_dispatch (manually triggered, NOT auto)
  → tofu apply → state diff applied to Hetzner / Tailscale
```

Infra changes are gated by manual dispatch (not auto on merge) so a
typo in `main.tf` can't accidentally `terraform destroy` prod. The
PR's `terraform plan` output in CI gives a preview; the manual
dispatch is the explicit approval.

### Layer-2 observability — unchanged

Same `compose/grafana-agent.yaml` mounts + same `${GRAFANA_CLOUD_*}`
env vars. The agent scrapes `api:8000/metrics` and ships to Grafana
Cloud. Sentry init in api + pipeline is unchanged. Operator
dashboards / alerts work unchanged.

The only delta: `external_labels.env` flips to `prod` so dashboards
can filter pre-prod vs prod cleanly. One-line change in the host's
`.env` (a Terraform-managed `local-file` resource).

### Layer-3 control plane — unchanged

Viewer's mobile-friendly Library / Search / Configuration tabs work
the same. The auth wall change (Codespaces port forward → Tailscale
MagicDNS) is transparent to the SPA. The api binds to the same
internal port; nginx proxies the same way.

### Layer-4 notifications — minor delta

The api's outbound webhook emitter (env-configured, defaults off)
becomes more useful in prod because Home Assistant on the same
tailnet can subscribe to `/api/jobs` event posts directly. No code
change required; just operator-side configuration of the webhook URL.

Slack continues to receive notifications via the GHA → Slack route
already wired in pre-prod. No api → Slack direct path needed.

## Operator runbooks

### First-time bootstrap

One-shot setup that takes prod from "nothing" to "viewer reachable
on the tailnet". ~30-45 min wall.

**Pre-bootstrap (one-time, on operator's laptop):**

```bash
# 1. Tools
brew install opentofu sops age
gh auth login                               # codespace + repo:write

# 2. Tailscale credentials for the GHA deployer (Free-plan workaround;
#    see Decision 2 note for the why). Two creds, both at:
#    Tailscale admin → Settings → Keys
#      a) Auth keys tab → Generate (Reusable, Ephemeral, Pre-approved,
#         Tags: tag:gha-deployer). Save tskey-auth-... as TS_AUTHKEY.
#      b) API access tokens tab → Generate. Save tskey-api-... as TS_API_KEY.
#    Save both in 1Password.

# 3. Hetzner Cloud project + API token
#    Hetzner console → Cloud → New project → API tokens → Generate
#    Scope: read+write
#    Save HCLOUD_TOKEN in 1Password.

# 4. age key for sops
age-keygen -o ~/.config/sops/age/keys.txt
#    Copy the public key into infra/.sops.yaml (commit-safe)
#    Save the private key contents to 1Password.

# 5. Stage GHA secrets (one-time, via gh)
gh secret set HCLOUD_TOKEN          --repo chipi/podcast_scraper --app actions --body '<token>'
gh secret set TS_AUTHKEY            --repo chipi/podcast_scraper --app actions --body '<tskey-auth-...>'
gh secret set TS_API_KEY            --repo chipi/podcast_scraper --app actions --body '<tskey-api-...>'
gh secret set TFSTATE_AGE_KEY       --repo chipi/podcast_scraper --app actions --body '<age-private-key>'
```

**First `tofu apply`:**

```bash
cd infra/terraform
export HCLOUD_TOKEN=$(op read 'op://Personal/Hetzner Cloud/api-token')
export TF_VAR_tailscale_api_key=$(op read 'op://Personal/Tailscale/podcast-scraper/api-key')
../tofu init
../tofu plan -out=plan.bin
../tofu apply plan.bin
# Outputs: prod-podcast.<tailnet>.ts.net hostname, IPv4, ssh fingerprint
```

**Stage the host-side `.env` (one-time after first apply):**

The `.env` file holds runtime secrets (provider API keys, Grafana
credentials, Sentry DSN). It's NOT in Terraform state — staged
separately so a `tofu destroy && apply` doesn't leak it. Cloud-init
bootstraps a placeholder; operator overwrites it before stack starts.

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net  # over Tailscale
# On the VPS:
sudo install -o deploy -g deploy -m 600 /dev/stdin /srv/podcast-scraper/.env <<'ENV'
OPENAI_API_KEY=...
GEMINI_API_KEY=...
GRAFANA_CLOUD_PROM_URL=...
GRAFANA_CLOUD_LOKI_URL=...
GRAFANA_CLOUD_USER=...
GRAFANA_CLOUD_API_KEY=...
PODCAST_SENTRY_DSN_API=...
PODCAST_SENTRY_DSN_PIPELINE=...
PODCAST_ENV=prod
PODCAST_RELEASE=...
ENV
sudo systemctl restart podcast-scraper.service  # picks up new .env
```

The first `compose up` happens after this; cloud-init waits for the
`.env` file to exist before starting the stack.

**Smoke validation (post-bootstrap):**

```bash
# 1. Tailnet reachability
curl -fsS https://prod-podcast.<tailnet>.ts.net/api/health | jq .
# Expected: {"status":"ok","feeds_api":true,...}

# 2. grafana-agent shipping (logs should mention WAL + "Replaying WAL"
#    + remote_write target reachable)
ssh deploy@prod-podcast.<tailnet>.ts.net 'docker logs compose-grafana-agent-1 --tail 20'

# 3. Sentry validation ping
ssh deploy@prod-podcast.<tailnet>.ts.net \
  'docker exec compose-api-1 python -c "
from podcast_scraper.utils.sentry_init import init_sentry
import sentry_sdk; init_sentry(\"api\")
sentry_sdk.capture_message(\"prod bootstrap validation ping\", level=\"info\")
"'
# Check sentry.io within ~1 min for the event under environment=prod

# 4. Grafana Cloud query (~30 s after agent's first scrape)
#    Open https://<org>.grafana.net → Explore → Prometheus
#    Query:  up{component="api",env="prod"}
#    Expected: 1 series, value=1
```

### Corpus migration from pre-prod (Codespace) to prod (VPS)

One-time migration on cutover day. Use the most recent backup-repo
release (cleanest path) rather than `gh codespace cp` between hosts
(brittle; transcripts may diverge between codespace and VPS during
sync).

```bash
# On the VPS, as the deploy user:
cd /srv/podcast-scraper
make restore-corpus               # Makefile target; pulls latest snapshot.tgz
                                  # from chipi/podcast_scraper-backup,
                                  # untars into /srv/podcast-scraper/corpus/

# Verify
ls -la corpus/feeds/ | head
find corpus -name '*.gi.json' | wc -l
docker compose restart api viewer  # so api re-scans the corpus

# In the viewer (over Tailscale): Library tab should now show all
# episodes from the snapshot.
```

After this, future backups come from the VPS instead of the codespace
(see Backup mechanism in Decision 4).

### Rollback (deploy went red mid-way)

`deploy-prod.yml` runs `docker compose pull && docker compose up -d`.
Three failure modes + how to recover:

**Failure 1: image pull failed (network blip / GHCR auth issue).**

Symptom: `pull` step in workflow exits non-zero.
Recovery: re-run the workflow. No state has changed yet on the host;
old containers still running with old images.

**Failure 2: pull succeeded but `compose up` rolls a container that
won't start.**

Symptom: workflow's compose-up step exits non-zero. New image is on
disk but old container is gone (compose recreates by default).
Recovery: SSH in, manually pin the old image:

```bash
ssh deploy@prod-podcast.<tailnet>.ts.net
cd /srv/podcast-scraper
PODCAST_IMAGE_TAG=sha-<previous-good-short-sha> \
  docker compose -f compose/docker-compose.stack.yml \
                 -f compose/docker-compose.prod.yml \
    up -d --remove-orphans
```

The `:sha-<short>` tags from the publish job are the rollback target;
they're never garbage-collected.

**Failure 3: stack is up but functionally broken** (e.g., api 200s
but the Library is empty due to a corpus path bug).

Symptom: workflow green, but operator notices the bug from the viewer.
Recovery: same as Failure 2 — pin the previous `:sha-<short>`. Then
file a bug + ship a fix-forward via the hotfix path.

### Disaster recovery (VPS gone)

If the Hetzner instance is irrecoverable (account issue, hardware
failure, accidental `tofu destroy` — Tofu has a 5-second cool-down
prompt but mistakes happen):

```bash
# 1. Re-provision (~5-10 min)
cd infra/terraform
../tofu apply              # same hostname, same Tailscale registration

# 2. Restore corpus (~3-5 min for 18 MB snapshot)
ssh deploy@prod-podcast.<tailnet>.ts.net 'cd /srv/podcast-scraper && make restore-corpus'

# 3. Re-stage the .env (operator's responsibility; not in Tofu state)
#    — see "First-time bootstrap" → "Stage the host-side .env"

# 4. Verify (~5 min)
#    — see "Smoke validation" steps 1-4
```

**Total wall time: ~15-20 min** assuming the operator knows the
runbook. The corpus is recoverable to within ~24 h of pre-disaster
state (last `backup-corpus.yml` run).

If the corpus loss matters more than the speed of recovery,
optionally enable Hetzner Volume snapshots (€0.0143/GB/month) for a
secondary safety net with ~hour-level RPO.

### Image set and pull behavior

Same image set as pre-prod (`api`, `viewer`, `pipeline-llm`); the
`pipeline` (ml) variant is **NOT** pulled to prod for the same
license-clean reason as pre-prod. Profiles that require local ML
(`airgapped*`) are not deployable to prod; the operator dropdown
filter (`PODCAST_AVAILABLE_PROFILES`) is set to `cloud_balanced,cloud_thin`
exactly as in pre-prod.

First-deploy image pull on a fresh VPS is ~3 GB total (~1.5 GB
pipeline-llm + ~700 MB api + ~50 MB viewer + ~250 MB grafana-agent).
Bandwidth allowance on Hetzner CX43 is 20 TB/mo; first-deploy pull
is negligible against that. Subsequent deploys reuse layers and pull
~50-200 MB per release.

## Security

- **Tailnet-only ingress.** No public open ports on the VPS; firewall
  default-deny inbound except Tailscale (UDP 41641) + outbound. Even
  the SSH port is reachable only over Tailscale.
- **Per-server Tailscale auth keys** (rotated via Terraform on each
  apply). Compromise of one key only affects one ephemeral
  registration.
- **Tailscale ACLs in repo** (`tailscale/policy.hujson`) — adding a
  collaborator is a PR. Revoke = remove + apply. No host-side state.
- **Host hardening checklist** (cloud-init):
  - SSH password auth disabled.
  - Auto-updates enabled (`unattended-upgrades`).
  - Root login disabled; deploy user has Docker group, no sudo.
  - Docker daemon socket readable only by `deploy` group.
  - Pipeline-llm spawning continues via api's
    `/var/run/docker.sock` mount (same as pre-prod).
- **Secrets:** loaded from a host-side `.env` file (mode 600, owned
  by `deploy`). Same shape as Codespaces secrets — `OPENAI_API_KEY`,
  `GEMINI_API_KEY`, `GRAFANA_CLOUD_*`, `PODCAST_SENTRY_DSN_*`,
  `TAILSCALE_AUTH_KEY` (for re-registration). `.env` not committed;
  staged manually on first bootstrap (operator drops it into the
  `deploy` user's home dir before `start.sh` runs).
- **Backup repo (`chipi/podcast_scraper-backup`)** stays private.
  Same retention as pre-prod.

## Costs

All EUR figures reflect Hetzner's May 2026 list price; re-verify
against [hetzner.com/cloud](https://www.hetzner.com/cloud/) before
every apply (cached / scraped figures lag the live page).

| Component | Plan | Cost / mo |
| --- | --- | --- |
| Hetzner CX43 (chosen, 8 vCPU shared Intel/AMD, 16 GB) | flat | €15.11 (~$16) |
| Hetzner Volume 50 GB (optional) | flat | ~€2.86 (~$3) |
| sops + age state storage (in-repo, no vendor) | — | $0 |
| Tailscale Personal | up to 100 devices | $0 |
| GitHub (CI / Packages public) | included | $0 |
| Grafana Cloud Free | unchanged | $0 |
| Sentry Free | unchanged | $0 |
| **Pipeline LLM calls** | metered (operator's keys) | varies — same as pre-prod |
| **Total fixed (CX43, no Volume)** | | **~$16/mo** |
| **Total fixed (CX43 + 50 GB Volume)** | | **~$19/mo** |

CX43 + 50 GB Volume sits comfortably under the $20 ceiling with ~$1
of headroom. Any further upgrade — CAX31 (€19.95 ARM premium),
CCX23 (€31.49 dedicated AMD), or a larger Volume — breaks the
ceiling and needs an explicit budget revision.

## Phased Rollout

This RFC's own rollout has two sub-phases:

**Phase A — Lift-and-shift basic.** VPS up, Tailscale registered,
deploy via push-from-GHA, corpus restored from latest
`chipi/podcast_scraper-backup` release. Codespaces stays available
as fallback. Goal: prove the same image set + compose works on
the chosen VPS.

**Phase B — Switch the bookmark.** Operator updates their phone
bookmark from the Codespaces forwarded URL to the new
`prod-podcast.<tailnet>.ts.net`. Backup workflow flips its source
from codespace to VPS. Pre-prod stays alive (free) as a smoke
surface for risky changes.

## Alternatives Considered

- **Cloudflare Tunnel + Access** instead of Tailscale. Public stable
  URL + service-token auth. Rejected: operator already runs
  Tailscale; no need to add a second auth wall.
- **Stay on Codespaces with a paid plan.** Cheapest for compute, but
  doesn't solve the auto-suspend or stable-URL pain. Punts the
  problem.
- **Fly.io / Railway / Render.** Higher abstraction, simpler push-
  to-deploy. Rejected: less control over disk + compose; per-second
  billing makes monthly cost less predictable; tailnet registration
  on those platforms is fiddlier.
- **Self-hosted on a home machine + dynamic DNS.** Free; fragile;
  uptime depends on operator's home network and power. Doesn't meet
  "phone-friendly always-on" bar.
- **Kubernetes / k3s.** Massive overkill for one operator + a handful
  of services + a hobby corpus. Skip.
- **No IaC, just SSH + ad-hoc commands.** Faster to ship initially.
  Rejected: rebuilding the VPS in 6 months becomes a memory test.
  Terraform pays for itself the first time we replace the host.
- **Pulumi / Crossplane / NixOS.** Each is a fine alternative for
  IaC. Terraform/OpenTofu chosen for the largest provider ecosystem
  and the lowest learning-curve cliff.

## Decisions made

The original "Open Questions" set has been resolved (see RFC-082
discussion thread). Recording here so the implementation has explicit
direction:

1. **Hetzner CX43** (€15.11/mo, 8 vCPU shared Intel/AMD, 16 GB,
   160 GB) is the chosen target — cheapest 8-core option in the
   lineup, ~25 % less than the same-shape CAX31 (ARM Ampere,
   €19.95/mo). Multi-arch GHCR publish (#712 / #737) is already
   merged, so a future flip to CAX31 is just a Terraform variable
   change (`server_type = "cax31"`) if the anti-jitter premium
   ever justifies the €5/mo. Fall back to CCX23 (4 dedicated AMD
   vCPU, ~€31.49 — breaks the $20 ceiling) only if measured
   shared-CPU jitter on CX43 demands it. **Earlier RFC drafts
   referenced stale €15.99 CAX31 / €11.99 CX43 figures from
   cached costgoat data** — the live Hetzner pricing page is
   authoritative; re-verify before each apply.
2. **No detached Volume on day one.** 80 GB boot disk is enough for
   early use. Add a Volume on first VPS replacement when we already
   need to migrate the corpus anyway.
3. **Scheduled cron feed sweep is a general capability, not VPS-only.**
   Out of scope for RFC-082. Tracked as
   [#708](https://github.com/chipi/podcast_scraper/issues/708) —
   design is API-level (apscheduler in the api process, config in
   `viewer_operator.yaml` or a sibling `schedules.yaml`) so it works
   on both pre-prod and prod with no host-side cron.
4. **Tailscale ACL via Terraform** (`tailscale_acl` resource).
   ACL changes ship as PRs. Aligns with the broader "maximize Tofu
   coverage" principle.
5. **Codespace pre-prod stays indefinitely.** $0 while stopped; acts
   as a free fallback / smoke surface for risky changes.
6. **State storage: sops + age in-repo, encrypted.** Free; no new
   vendor; encrypted blob in `git diff` is a useful "state changed"
   signal. age private key in 1Password. Migration to Object Storage
   is a one-time `sops -d | tofu state push` if we ever go
   multi-operator.

## Open Questions (remaining)

The decisions above leave a smaller residual set:

1. **Bootstrap secret-staging UX.** Currently the `.env` is staged
   manually post-`tofu apply` via SSH (see Operator runbooks). An
   alternative is to put each secret into 1Password CLI and have
   cloud-init pull them on first boot via `op` CLI. More moving
   parts; not clearly worth it for one operator. Defer; revisit if
   secret rotation cadence picks up.
2. **Sentry release tag** in prod. `PODCAST_RELEASE` should ideally
   carry the deployed image's `:sha-<short>` so Sentry events group
   by release. Two paths: (a) `deploy-prod.yml` writes the SHA into
   the host's `.env` before `compose up`; (b) the api reads the
   image digest at startup and self-tags. (a) is simpler. To wire
   during implementation.
3. **Sentry alert routing.** Pre-prod Sentry events go to the same
   Slack channel as prod by default (currently single-DSN-per-
   component). Splitting requires either two DSNs (one per env) or a
   Sentry-side filter rule. Not blocking; revisit when first prod
   incident shows whether the noise is a problem.

## References

- [RFC-081 (pre-prod)](RFC-081-pre-prod-environment-and-control-plane.md) — what we're lifting from.
- [docs/wip/CODESPACE_PREPROD_RUNBOOK.md](../wip/CODESPACE_PREPROD_RUNBOOK.md) — operator-facing notes from pre-prod; many of the same gotchas apply (corpus bind mount, image pull on first deploy, agent yaml comment foot-guns).
- [Hetzner Cloud pricing](https://www.hetzner.com/cloud/) — CX / CCX line.
- [Tailscale GitHub Actions integration](https://tailscale.com/kb/1276/tailscale-github-action) — ephemeral runner-to-tailnet auth via OAuth.
- [Tailscale MagicDNS + cert](https://tailscale.com/kb/1153/enabling-https) — TLS without Let's Encrypt boilerplate.
- [hetznercloud/hcloud Terraform provider](https://registry.terraform.io/providers/hetznercloud/hcloud/latest) — IaC surface.
- [tailscale/tailscale Terraform provider](https://registry.terraform.io/providers/tailscale/tailscale/latest) — auth key + ACL management.
- [OpenTofu](https://opentofu.org/) — open-source Terraform fork.
