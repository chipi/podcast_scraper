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
│   ├── prod.user-data                 # first-boot bootstrap (Docker, Tailscale, repo pull, optional Alloy)
│   └── podcast-tailscale-serve.sh     # canonical ``tailscale serve`` helper (also injected into prod.user-data)
└── deploy/
    └── deploy.sh               # called by deploy-prod.yml and drill-deploy.yml
```

## Quickstart

> **Full operator runbook:** see [PROD_RUNBOOK.md](../docs/guides/PROD_RUNBOOK.md).
> **DR drill only (GitHub Actions, confirms, orchestrator):** [DR_DRILL_RUNBOOK.md](../docs/guides/DR_DRILL_RUNBOOK.md).
> **Architecture narrative (how hosting fits together):** [Hosting and infrastructure](../docs/architecture/HOSTING_AND_INFRASTRUCTURE.md).
> **Stack contract (compose, health, gates by surface):** [STACK_CONTRACT.md](../docs/guides/STACK_CONTRACT.md).
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

**Public SSH (drill break-glass):** `terraform.drill.ci.tfvars` sets **`hcloud_inbound_ssh_troubleshoot_cidrs`** to `0.0.0.0/0` and `::/0` so the Hetzner firewall allows **TCP/22** on the drill VPS in addition to Tailscale UDP/41641. That lets you run `ssh deploy@<server public IPv4>` (same **`OPERATOR_SSH_PUBLIC_KEY`** as cloud-init) when the node has not joined the tailnet yet. Prod leaves this list **empty** (default). Narrow the CIDRs in a private drill tfvars if you keep a long-lived drill VM.

### GitHub Actions (drill, #752)

| Workflow | Purpose |
| -------- | ------- |
| [`.github/workflows/drill-infra-plan.yml`](../.github/workflows/drill-infra-plan.yml) | On PRs touching `infra/**` or `tailscale/policy.hujson`, runs `tofu fmt -check`, `validate`, and `tofu plan` in workspace **`drill`** (posts a second PR comment next to `infra-ci.yml`). Also **`workflow_call`** from **`drill-exercise.yml`**. |
| [`.github/workflows/drill-infra-apply.yml`](../.github/workflows/drill-infra-apply.yml) | Manual **`APPLY`**, Environment **`drill`**: restores artifact **`terraform-state-after-apply-drill`** when present (retry same run), **`tofu apply`**, re-encrypt **`terraform.tfstate.enc.drill`**, upload **`terraform-state-after-apply-drill`** ( **`drill-exercise`** then **`drill-tfstate-bridge`** publishes **`drill-tfstate-for-teardown`** for destroy). **`workflow_call`** with **`skip_confirm: true`**. |
| [`.github/workflows/drill-infra-destroy.yml`](../.github/workflows/drill-infra-destroy.yml) | Manual **`DRILL_DESTROY`**, environment **`drill`**: downloads **`drill-tfstate-for-teardown`** (from **`drill-exercise`** **`drill-tfstate-bridge`**) or **`terraform-state-after-apply-drill`**, then **`tofu destroy`**, **Hetzner `hcloud` sweep**, **Tailscale device DELETE** (`tag:dr-drill` or drill `tailnet_hostname`); re-encrypt state; upload artifact. **`workflow_call`** with **`skip_confirm: true`**. |
| [`.github/workflows/drill-e2e.yml`](../.github/workflows/drill-e2e.yml) | Manual **`DRILL_SMOKE`** confirm, environment **`drill`**: join tailnet, resolve **`DRILL_TAILNET_FQDN`**, run **`scripts/ops/post_deploy_smoke.sh`** over **HTTPS** (same six API probes as prod; corpus path **`/srv/podcast-scraper/corpus`**). Requires ACL **`tag:gha-deployer` → `tag:dr-drill:443`** (HTTPS) and **`DRILL_DEPLOY_SSH_PRIVATE_KEY`**. Also **`workflow_call`** from **`drill-exercise.yml`**. |
| [`.github/workflows/drill-stack-playwright.yml`](../.github/workflows/drill-stack-playwright.yml) | Manual **`DRILL_STACK_PLAYWRIGHT`**, environment **`drill`**: join tailnet, run **`tests/stack-test/stack-viewer.spec.ts`** against **HTTPS** on the resolved drill host (browser + API + corpus **`/app/output`**). **`STACK_TEST_INSECURE_TLS=1`** for MagicDNS TLS. Also **`workflow_call`** with **`skip_confirm: true`** from **`drill-exercise.yml`** (after **`drill-e2e`**). |
| [`.github/workflows/drill-restore-corpus.yml`](../.github/workflows/drill-restore-corpus.yml) | Manual **`DRILL_RESTORE`** confirm, environment **`drill`**: download **`snapshot.tgz`** from the backup repo (default **`chipi/podcast_scraper-backup`**), restore **`corpus/`** on the drill host, recreate **api** + **viewer**. Optional secret **`BACKUP_REPO_TOKEN`** for a private backup repo. Also **`workflow_call`** from **`drill-exercise.yml`**. |
| [`.github/workflows/drill-exercise.yml`](../.github/workflows/drill-exercise.yml) | Manual **`DRILL_FULL_CYCLE`** (alias **`DRILL_EXERCISE`**) or **Wednesdays 02:00 UTC** schedule (#799): **`drill-infra-plan`** → **`drill-infra-apply`** → **`drill-tfstate-bridge`** (artifact **`drill-tfstate-for-teardown`**) → **`drill-deploy`** → **`drill-restore-corpus`** → **`drill-e2e`** → **`drill-stack-playwright`**, then **always** **`drill-infra-destroy`**; **`assert-post-conditions`** fails on any red step or Hetzner orphans. |
| [`.github/workflows/drill-deploy.yml`](../.github/workflows/drill-deploy.yml) | Manual **`workflow_dispatch`**, environment **`drill`**: same steps as **`deploy-prod.yml`** (Tailscale join, SSH as **`deploy@`**, **`infra/deploy/deploy.sh`**, external **`/api/health`**). Uses **`DRILL_TAILNET_FQDN`** and secret **`DRILL_DEPLOY_SSH_PRIVATE_KEY`** (public key must be in **`deploy@`** `authorized_keys` on the drill host). Also **`workflow_call`** from **`drill-exercise.yml`**. |

**Secrets (repository):** `HCLOUD_TOKEN_DRILL` (Hetzner token scoped to the drill project only), plus the same `TS_API_KEY`, `TFSTATE_AGE_KEY`, and **`OPERATOR_SSH_PUBLIC_KEY`** (Actions **secret**, same value as prod infra workflows; masks the pubkey in workflow logs) as prod unless you document drill-only keys. First-boot **`deploy@`** `authorized_keys` is built only from **`OPERATOR_SSH_PUBLIC_KEY`** (see `infra/cloud-init/prod.user-data`). For app deploys to the drill VPS, add **`DRILL_DEPLOY_SSH_PRIVATE_KEY`** (same PEM as **`PROD_SSH_PRIVATE_KEY`** is fine when its public half matches **`OPERATOR_SSH_PUBLIC_KEY`**). Optional **`BACKUP_REPO_TOKEN`** (same as **`backup-corpus-prod.yml`**) when **`chipi/podcast_scraper-backup`** is private so **`drill-restore-corpus.yml`** can download **`snapshot.tgz`**. **Variables:** reuse `TAILNET_NAME` from prod. Set **`DRILL_TAILNET_FQDN`** (e.g. `dr-podcast.tail-xxxx.ts.net`, matching `tailnet_hostname` in drill tfvars) for **`drill-e2e.yml`**, **`drill-deploy.yml`**, **`drill-restore-corpus.yml`**, **`drill-stack-playwright.yml`**, and **`drill-exercise.yml`**.

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
- [DR_DRILL_RUNBOOK.md](../docs/guides/DR_DRILL_RUNBOOK.md) — drill workflows, confirms, orchestrator
- [#716](https://github.com/chipi/podcast_scraper/issues/716) — this scaffolding ticket
- [#719](https://github.com/chipi/podcast_scraper/issues/719) — `infra-ci.yml` (PR plan) and `infra-apply.yml` (manual apply) workflows
- [#752](https://github.com/chipi/podcast_scraper/issues/752) — recurring DR drill CI (`drill-infra-plan.yml`, `drill-infra-apply.yml`, `drill-tfstate-bridge` + `drill-tfstate-for-teardown`, `drill-infra-destroy.yml`, workspace `drill`, prod-only `tailscale_acl`)
