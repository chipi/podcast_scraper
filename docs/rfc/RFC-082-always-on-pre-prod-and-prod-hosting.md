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
  2. Hetzner CX32 (or CCX13) provisioned via Terraform
  3. Tailscale daemon on the VPS, auto-joins tailnet via auth key
  4. GHA workflow deploy-prod.yml fires on stack-test success → main
  5. Host bind-mount: /srv/podcast-scraper/corpus
  6. Backup-corpus.yml points SSH at the VPS over Tailscale
```

### Decision 1 — Hosting target

**Hetzner CX32** (4 vCPU shared, 8 GB RAM, 80 GB SSD, ~€7.89/mo, EU).
Operator-confirmed EU geography + want predictable flat billing.

Upgrade path: **CCX13** (2 dedicated vCPU, 8 GB, 80 GB, ~€13.99/mo)
if shared-CPU jitter shows up in pipeline ffmpeg stages. Both are
within the cost ceiling.

Optional: a separate Hetzner Volume (€0.0476/GB/month) for the corpus
bind-mount, so the VPS image can be replaced without touching corpus
data. ~€2.50/mo for 50 GB. Recommended once corpus grows past ~20 GB
on the boot disk.

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
          oauth-client-id: ${{ secrets.TAILSCALE_OAUTH_CLIENT_ID }}
          oauth-secret:    ${{ secrets.TAILSCALE_OAUTH_SECRET }}
          tags:            tag:gha-deployer
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
`backup-corpus.yml` workflow uses Tailscale OAuth (same pattern as
the deploy workflow) to reach the VPS, tarballs the corpus, uploads
to `chipi/podcast_scraper-backup` releases. The single change vs.
pre-prod's backup is the SSH target hostname — `prod-podcast.<...>`
instead of the codespace name.

This keeps backup logic in code-review (`.github/workflows/...`)
rather than as a host-side cron that escapes git visibility.

**Optional: Hetzner Volume snapshots.** If corpus moves to a
detached Volume, Hetzner's automatic snapshot feature is a
zero-config secondary safety net (~€0.0119/GB/month). Out of scope
for this RFC unless corpus grows large enough to matter.

### Decision 5 — IaC: Terraform with the hetznercloud provider

`infra/` directory in this repo, structured as:

```text
infra/
├── terraform/
│   ├── main.tf              # provider, server, network, firewall, volume
│   ├── tailscale.tf         # tailscale_tailnet_key, ACL update if managed
│   ├── outputs.tf           # tailnet hostname, IPv4
│   ├── variables.tf
│   ├── backend.tf           # remote state — Hetzner Storage Box or local-encrypted
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
- **`hetznercloud/hcloud` provider** — declares the CX32 server,
  attached Volume, firewall rules (default-deny inbound; allow
  outbound), SSH key registration.
- **`tailscale/tailscale` provider** — declares the auth key (so
  Terraform produces the per-server key cloud-init uses), and
  optionally manages ACL syncs from `tailscale/policy.hujson`.
- **cloud-init `user_data`** baked into the Hetzner instance config
  — installs Docker, Tailscale, creates `deploy` user, pulls
  `chipi/podcast_scraper` to `/srv/podcast-scraper`, copies a
  per-host `.env` from a known location (operator-managed).

**State storage:** OpenTofu state lives in a Hetzner Object Storage
bucket (S3-compatible, ~€0.50/mo for tiny state files), encrypted
client-side with a passphrase that the operator stores in 1Password.
Avoids the "where does state live" foot-gun without dragging in
Terraform Cloud.

**Bootstrap flow:**

1. Operator clones repo locally, runs `cd infra/terraform && tofu apply`.
2. OpenTofu provisions Hetzner CX32 + attaches Volume + registers
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

| Component | Plan | Cost / mo |
|---|---|---|
| Hetzner CX32 (or CCX13) | flat | €7.89 (~$9) or €13.99 (~$15) |
| Hetzner Volume 50 GB (optional) | flat | ~€2.50 |
| Hetzner Object Storage (state) | flat | ~€0.50 |
| Tailscale Personal | up to 100 devices | $0 |
| GitHub (CI / Packages public) | included | $0 |
| Grafana Cloud Free | unchanged | $0 |
| Sentry Free | unchanged | $0 |
| **Pipeline LLM calls** | metered (operator's keys) | varies — same as pre-prod |
| **Total fixed** | | **~$10-19/mo** |

Within the $20/mo ceiling even on the dedicated-CPU CCX13 + 50 GB
Volume + state storage.

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

## Open Questions

1. **Hetzner instance size.** CX32 (€7.89, shared CPU) for cost-
   floor, or CCX13 (€13.99, dedicated CPU) to remove jitter from
   ffmpeg / preprocessing? Recommendation: start CX32, upgrade to
   CCX13 if pipeline wall-time is sensitive (post-validation
   measurement decides).
2. **Detached corpus Volume from day one, or only when corpus grows?**
   Volume is €2.50/mo and decouples corpus survival from VPS
   lifecycle. Recommendation: skip on day one (boot disk has 80 GB,
   plenty for early use), add on first VPS replacement.
3. **Scheduled cron feed sweep** (Open Question 4 from prior draft).
   Worth implementing in this RFC's scope, or defer? Recommendation:
   defer — out of scope unless a clear use case appears.
4. **Tailscale ACL management** — manage via Terraform (the
   `tailscale_acl` resource), or hand-edit on the Tailscale admin
   console? Recommendation: Terraform, so the ACL change has a PR
   trail. Costs ~30 min of one-time setup.
5. **Codespace lifecycle** post-cutover. Recommendation: keep
   indefinitely as a smoke / fallback surface. $0 while stopped.
6. **Hetzner Object Storage vs other state backends** for OpenTofu.
   Cheapest option that keeps state out of git. Alternative: encrypted
   state file committed to a private branch (cheaper but messier).
   Recommendation: Object Storage.

## References

- [RFC-081 (pre-prod)](RFC-081-pre-prod-environment-and-control-plane.md) — what we're lifting from.
- [docs/wip/CODESPACE_PREPROD_RUNBOOK.md](../wip/CODESPACE_PREPROD_RUNBOOK.md) — operator-facing notes from pre-prod; many of the same gotchas apply (corpus bind mount, image pull on first deploy, agent yaml comment foot-guns).
- [Hetzner Cloud pricing](https://www.hetzner.com/cloud/) — CX / CCX line.
- [Tailscale GitHub Actions integration](https://tailscale.com/kb/1276/tailscale-github-action) — ephemeral runner-to-tailnet auth via OAuth.
- [Tailscale MagicDNS + cert](https://tailscale.com/kb/1153/enabling-https) — TLS without Let's Encrypt boilerplate.
- [hetznercloud/hcloud Terraform provider](https://registry.terraform.io/providers/hetznercloud/hcloud/latest) — IaC surface.
- [tailscale/tailscale Terraform provider](https://registry.terraform.io/providers/tailscale/tailscale/latest) — auth key + ACL management.
- [OpenTofu](https://opentofu.org/) — open-source Terraform fork.
