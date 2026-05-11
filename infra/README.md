# `infra/` — production VPS infrastructure-as-code

OpenTofu configuration that provisions the always-on production Hetzner VPS,
registers it with the operator's Tailscale tailnet, and prepares it to run the
podcast_scraper stack. Implements [RFC-082](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md).

## Layout

```text
infra/
├── .sops.yaml                  # age public-key config for state encryption
├── .gitignore                  # excludes plaintext state, plans, .terraform/
├── tofu                        # wrapper script — handles sops decrypt/encrypt
├── README.md                   # this file
├── terraform/
│   ├── *.tf                    # OpenTofu config (server, network, Tailscale)
│   ├── terraform.drill.ci.tfvars  # non-secret drill defaults for GHA (#752)
│   ├── terraform.tfstate.enc # sops+age-encrypted state, default workspace (committed after prod apply)
│   └── terraform.tfstate.enc.drill  # optional; encrypted state for workspace `drill` (#752)
├── cloud-init/
│   └── prod.user-data          # first-boot bootstrap (Docker, Tailscale, repo pull, optional Alloy)
└── deploy/
    └── deploy.sh               # called by .github/workflows/deploy-prod.yml
```

## Quickstart

> **Full operator runbook:** see [PROD_RUNBOOK.md](../docs/guides/PROD_RUNBOOK.md).
> Account-prerequisite checklist: [#714](https://github.com/chipi/podcast_scraper/issues/714).

One-time setup (per operator):

1. `brew install opentofu sops age`
2. `age-keygen -o ~/.config/sops/age/keys.txt`
3. Copy the public key (the `# public key:` comment) into [`.sops.yaml`](.sops.yaml), replacing the `age1PLACEHOLDER…` value.
4. Save the **private** key to 1Password as `sops/podcast-scraper/tofu-state-age-key`.
5. Stage `HCLOUD_TOKEN`, `TS_AUTHKEY` (device-join), `TS_API_KEY` (terraform's tailscale provider; Free-plan substitute for OAuth — see [PROD_RUNBOOK.md "Tailscale credentials"](../docs/guides/PROD_RUNBOOK.md)), `TFSTATE_AGE_KEY` in repo Actions secrets (see [#714](https://github.com/chipi/podcast_scraper/issues/714)).

Per-operation:

```bash
cd infra
export HCLOUD_TOKEN=$(op read 'op://Personal/Hetzner Cloud/podcast-scraper-prod/api-token')
export TF_VAR_tailscale_api_key=$(op read 'op://Personal/Tailscale/podcast-scraper/api-key')
# Optional — host OS metrics to Grafana Cloud (all three must be set or cloud-init skips Alloy):
# export TF_VAR_grafana_cloud_metrics_remote_write_url='https://prometheus-prod-…/api/prom/push'
# export TF_VAR_grafana_cloud_metrics_username='…'
# export TF_VAR_grafana_cloud_metrics_password=$(op read '…')
./tofu init
./tofu plan
./tofu apply
```

The `tofu` wrapper auto-decrypts state before invocation and re-encrypts after.

## DR drill workspace (OpenTofu, GitHub #752)

Use a **separate OpenTofu workspace** (e.g. `drill`) and a **Hetzner API token scoped only to the drill project** so `tofu apply` cannot touch production resources.

1. `cd infra/terraform && tofu workspace new drill` (once).
2. Copy [terraform/terraform.drill.tfvars.example](terraform/terraform.drill.tfvars.example) to a **gitignored** `terraform.drill.tfvars` (or export matching `TF_VAR_*`).
3. Set **`manage_tailscale_acl = false`** for the drill workspace so this state does **not** manage `tailscale_acl` (the tailnet-wide ACL must stay owned by the default/prod workspace only — two states must not fight the same Tailscale API object).
4. Set **`tailscale_advertise_tags`** to your drill tag (e.g. `["tag:dr-drill"]`) and **`tailnet_hostname`** distinct from prod (e.g. `dr-podcast`), matching [tailscale/policy.hujson](../tailscale/policy.hujson) `tagOwners` / ACLs.
5. **`hcloud_environment_label = "drill"`** so the Hetzner server label reflects the stack.
6. Encrypted state for workspace **`drill`** is stored separately as **`terraform.tfstate.enc.drill`** (sops+age, same key as `TFSTATE_AGE_KEY` unless you rotate a drill-only key). Commit that file when you want CI to plan against existing drill state; otherwise the first plan starts empty. **Never** commit plaintext under `terraform.tfstate.d/`.

### GitHub Actions (drill, #752)

| Workflow | Purpose |
| -------- | ------- |
| [`.github/workflows/drill-infra-plan.yml`](../.github/workflows/drill-infra-plan.yml) | On PRs touching `infra/**` or `tailscale/policy.hujson`, runs `tofu fmt -check`, `validate`, and `tofu plan` in workspace **`drill`** (posts a second PR comment next to `infra-ci.yml`). |
| [`.github/workflows/drill-infra-apply.yml`](../.github/workflows/drill-infra-apply.yml) | Manual `workflow_dispatch` with confirm **`APPLY`**, GitHub Environment **`drill`**, `tofu apply`, re-encrypt **`terraform.tfstate.enc.drill`**, upload artifact. |
| [`.github/workflows/drill-e2e.yml`](../.github/workflows/drill-e2e.yml) | Manual **`DRILL_SMOKE`** confirm, environment **`drill`**: join tailnet, resolve **`DRILL_TAILNET_FQDN`**, HTTPS **`/api/health`** on the drill host (read-only). Requires ACL **`tag:gha-deployer` → `tag:dr-drill`** (see `tailscale/policy.hujson`) applied via prod **`infra-apply.yml`**. |
| [`.github/workflows/deploy-drill.yml`](../.github/workflows/deploy-drill.yml) | Manual **`workflow_dispatch`**, environment **`drill`**: same steps as **`deploy-prod.yml`** (Tailscale join, SSH as **`deploy@`**, **`infra/deploy/deploy.sh`**, external **`/api/health`**). Uses **`DRILL_TAILNET_FQDN`** and secret **`DRILL_DEPLOY_SSH_PRIVATE_KEY`** (public key must be in **`deploy@`** `authorized_keys` on the drill host). |

**Secrets (repository):** `HCLOUD_TOKEN_DRILL` (Hetzner token scoped to the drill project only), plus the same `TS_API_KEY`, `TFSTATE_AGE_KEY`, and **`OPERATOR_SSH_PUBLIC_KEY`** (Actions **secret**, same value as prod infra workflows; masks the pubkey in workflow logs) as prod unless you document drill-only keys. For app deploys to the drill VPS, add **`DRILL_DEPLOY_SSH_PRIVATE_KEY`** (same PEM as **`PROD_SSH_PRIVATE_KEY`** is fine if the same CI public key is authorized for **`deploy@` on the drill host**). **Variables:** reuse `TAILNET_NAME` from prod. Set **`DRILL_TAILNET_FQDN`** (e.g. `dr-podcast.tail-xxxx.ts.net`, matching `tailnet_hostname` in drill tfvars) for **`drill-e2e.yml`** and **`deploy-drill.yml`**.

Prod / default workspace keeps **`manage_tailscale_acl = true`** (default) and `tailscale_advertise_tags = ["tag:prod"]`.

### Host metrics (Grafana Alloy on the VPS)

When all of `TF_VAR_grafana_cloud_metrics_remote_write_url`, `TF_VAR_grafana_cloud_metrics_username`, and `TF_VAR_grafana_cloud_metrics_password` are non-empty, first-boot cloud-init installs **Grafana Alloy** from `apt.grafana.com`, enables `prometheus.exporter.unix` (node-style host metrics), and **remote_writes** to Grafana Cloud using basic auth. Credentials land in `/etc/alloy/grafana-cloud.env` (mode 0600); River config is `/etc/alloy/config.alloy`.

**Existing prod server:** `hcloud_server.prod` ignores drift on `user_data` (see `main.tf`), so changing these variables does **not** re-run cloud-init on the current instance. Alloy appears only on a **new** server (replace/recreate) or via a **manual** install mirroring the same steps. Remove `lifecycle.ignore_changes` only with an explicit ops decision (data loss / tailnet identity risk is documented in `main.tf`).

## State encryption model

State lives at `terraform/terraform.tfstate.enc` (sops+age, committed) for the **default**
OpenTofu workspace. Workspace **`drill`** uses **`terraform.tfstate.enc.drill`** when managed
through CI or operators following `infra/README.md` § DR drill. Plaintext
**never** touches the repo or shell history. The wrapper:

1. Decrypts `.enc` → `terraform/terraform.tfstate` (plaintext, for tofu only).
2. Runs the requested subcommand against that plaintext file.
3. Re-encrypts back to `.enc` if state changed (sha256 comparison).
4. Shreds the plaintext on any exit (trap, including Ctrl-C).

If the wrapper crashes mid-flight, check `git status infra/terraform/` and
confirm no plaintext `.tfstate` is staged. If it is, **do NOT commit** — re-run
`cd infra && ./tofu version` (a no-op invocation) to re-trigger the trap and
clean up.

## Cross-references

- [RFC-082 — Decision 5: IaC](../docs/rfc/RFC-082-always-on-pre-prod-and-prod-hosting.md)
- [PROD_RUNBOOK.md](../docs/guides/PROD_RUNBOOK.md) — full operator runbook
- [#716](https://github.com/chipi/podcast_scraper/issues/716) — this scaffolding ticket
- [#719](https://github.com/chipi/podcast_scraper/issues/719) — `infra-ci.yml` (PR plan) and `infra-apply.yml` (manual apply) workflows
- [#752](https://github.com/chipi/podcast_scraper/issues/752) — recurring DR drill CI (`drill-infra-plan.yml`, `drill-infra-apply.yml`, workspace `drill`, prod-only `tailscale_acl`)
